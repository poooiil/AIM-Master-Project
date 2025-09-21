import torch
import torch.nn as nn
import os

from model.mdm import MDM
from transformers import BertTokenizer, BertModel, pipeline
from transformers import AutoTokenizer, AutoModel
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class my_MDM(MDM):

    def __init__(self, modeltype, njoints, nfeats, num_actions, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", legacy=False, data_rep='rot6d', dataset='amass', clip_dim=512,
                 arch='trans_enc', emb_trans_dec=False, clip_version=None, args=None, **kargs):

        super(my_MDM, self).__init__(modeltype, njoints, nfeats, num_actions, translation, pose_rep, glob, glob_rot,
                         latent_dim, ff_size, num_layers, num_heads, dropout,
                         ablation, activation, legacy, data_rep, dataset, clip_dim,
                         arch, emb_trans_dec, clip_version, **kargs)
        self.is_multi = True
        self.args = args

        self.multi_person = MultiPersonBlock(arch=self.args.multi_arch,
                                             fn_type=self.args.multi_func,
                                             num_layers=self.args.multi_num_layers,
                                             latent_dim=self.latent_dim,
                                             input_feats=self.input_feats,
                                             predict_6dof=self.args.predict_6dof)

        if self.arch == 'trans_enc':
            assert 0 < self.args.multi_backbone_split <= self.num_layers
            print(f'CUTTING BACKBONE AT LAYER [{self.args.multi_backbone_split}]')
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=self.activation)
            del self.seqTransEncoder
            self.seqTransEncoder_start = nn.TransformerEncoder(seqTransEncoderLayer,
                                                               num_layers=self.args.multi_backbone_split)
            self.seqTransEncoder_end = nn.TransformerEncoder(seqTransEncoderLayer,
                                                             num_layers=self.num_layers - self.args.multi_backbone_split)
        else:
            raise ValueError('Supporting only trans_enc arch.')

        # bert model
        # self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # self.bert_model = BertModel.from_pretrained("bert-base-uncased").cuda()

        # bart model
        self.bert_tokenizer =AutoTokenizer.from_pretrained("distilroberta-base")
        self.bert_model =  AutoModel.from_pretrained("distilroberta-base")
        for p in self.bert_model.parameters():
            p.requires_grad = False
        self.feature_extraction = pipeline('feature-extraction', model="distilroberta-base", tokenizer="distilroberta-base")

        self.embed_bert = nn.Linear(768, 512)
        if self.args.multi_mdm_freeze:
            self.freeze_block(self.input_process)
            self.freeze_block(self.sequence_pos_encoder)
            self.freeze_block(self.seqTransEncoder_start)
            self.freeze_block(self.seqTransEncoder_end)
            self.freeze_block(self.embed_timestep)
            if 'text' in self.cond_mode:
                print("Load text~!~!!~!~!~")
                self.freeze_block(self.embed_text)
                # self.freeze_block(self.feature_extraction)
            self.freeze_block(self.output_process)


    def encode_bert(self, raw_text):
        import numpy as np
        device = next(self.parameters()).device
        features = []
        for text in raw_text:
            encoded_input = self.bert_tokenizer(text, padding=True, truncation=True, return_tensors='pt')
            encoded_input = encoded_input.to(device)
            # with torch.no_grad(): # We don't need to compute gradients here, so we disable them.
            model_output = self.bert_model(**encoded_input)
            # model_output = model_output.to(device)
            embeddings = model_output.last_hidden_state.mean(dim=1).squeeze()# .to(device)
            features.append(embeddings.cpu().numpy().astype(float))
            # features_i = self.feature_extraction(text)
            # features.append(np.mean(np.array(features_i[0]), axis=0))
        # print("text feature:", np.array(features).shape) (64, 768)
        
        return torch.from_numpy(np.array(features)).to(torch.float32).to(device)# .unsqueeze(0)
        
        
    def forward(self, x, timesteps, y=None):
        x1, x2 = torch.split(x,[1, x.shape[2]-1], dim=2)
        # print("检查输入数据：", x1)
        # print("检查输入数据值和：", torch.sum(x1))
        # print("输入数据拆分的格式:",x1.shape, x2.shape) # 输入数据拆分的格式: torch.Size([64, 138, 1, 121]) torch.Size([64, 138, 1, 121])
        canon, x = torch.split(x1,[1, x1.shape[-1]-1], dim=-1)
        canon_other, x_other = torch.split(x2,[1, x2.shape[-1]-1], dim=-1)

        # Build embedding vector
        emb = self.embed_timestep(timesteps)  # [1, bs, d]

        force_mask = y.get('uncond', False)
        force_no_com = y.get('no_com', False)  # FIXME - note that this feature not working for com_only - which is ok
        if 'text' in self.cond_mode:
            enc_text = self.encode_bert(y['text']) # torch.Size([64, 768])
            # print("emb_text:::", enc_text.shape)
            # print("mask_Cond:::", self.mask_cond(enc_text, force_mask=force_mask).shape)
            emb += self.embed_bert(self.mask_cond(enc_text, force_mask=force_mask))
            # print("emb:::", emb.shape) torch.Size([1, 64, 512])
            # enc_text = self.encode_text(y['text'])
            # emb += self.embed_text(self.mask_cond(enc_text, force_mask=force_mask))
        if 'action' in self.cond_mode:
            action_emb = self.embed_action(y['action'])
            emb += self.mask_cond(action_emb, force_mask=force_mask)

        # Embed motion to latent space (frame by frame)
        x = self.input_process(x) #[seqlen, bs, d]
        x_other = self.input_process(x_other)

        low_x, low_x_other = x, x_other

        # adding the timestep embed
        xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
        xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
        x_other = torch.cat((emb, x_other), axis=0)
        x_other = self.sequence_pos_encoder(x_other)

        mid = self.seqTransEncoder_start(xseq)[1:]
        mid_other = self.seqTransEncoder_start(x_other)[1:]
        cur_canon = canon if self.args.predict_6dof else None
        other_canon = canon_other if self.args.predict_6dof else None

        delta_x, delta_x_other, canon_out, canon_other_out = self.multi_person(low_cur=low_x, low_other=low_x_other,cur=mid, other=mid_other, cur_canon=cur_canon,
                                               other_canon=other_canon, text_emb = emb)
        if force_no_com:
            output_x = self.seqTransEncoder(xseq)[1:]  # [seqlen, bs, d]
            output_other = self.seqTransEncoder(x_other)[1:]  # [seqlen, bs, d]
        else:
            if 'out_cur' in self.multi_person.fn_type:
                mid += delta_x
                mid_other += delta_x_other
            elif 'out_cond' in self.multi_person.fn_type:
                mid[0] += delta_x[0]
                mid_other[0] += delta_x_other[0]
            if self.args.multi_backbone_split < self.num_layers:
                output_x = self.seqTransEncoder_end(mid)# [1:]
                output_other = self.seqTransEncoder_end(mid_other)# [1:]
            elif self.args.multi_backbone_split == self.num_layers:
                output_x = mid# [1:]
                output_other = mid_other# [1:]

        output_x = self.output_process(output_x)  # [bs, njoints, nfeats, nframes]
        output_other = self.output_process(output_other)
        output_1 = torch.cat((canon_out, output_x), dim=-1)
        output_2 = torch.cat((canon_other_out, output_other), dim=-1)
        output = torch.cat([output_1, output_2], dim=2)
        return output

    def trainable_parameters(self):
        return [p for name, p in self.named_parameters() if p.requires_grad]
        # return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def multi_parameters(self):
        return [p for name, p in self.multi_person.named_parameters() if p.requires_grad]

    def freeze_block(self, block):
        block.eval()
        for p in block.parameters():
            p.requires_grad = False



class MultiPersonBlock(nn.Module):
    def __init__(self, arch, fn_type, num_layers, latent_dim, input_feats, predict_6dof):
        super().__init__()
        self.arch = arch
        self.fn_type = fn_type
        self.predict_6dof = predict_6dof
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        self.num_heads = 4
        self.ff_size = 1024
        self.dropout = 0.1
        self.activation = 'gelu'
        self.input_feats = input_feats
        if self.predict_6dof:
            self.canon_agg = nn.Linear(9*2, self.latent_dim)
            # self.canon_agg = nn.Linear(self.input_feats*2, self.latent_dim)
            self.canon_out = nn.Linear(self.latent_dim, 9)
            # self.canon_out = nn.Linear(self.latent_dim, self.input_feats)
        if 'in_both' in self.fn_type:
            self.aggregation = nn.Linear(self.latent_dim*2, self.latent_dim)
        if self.arch == 'trans_enc':
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=self.activation)
            self.model = nn.TransformerEncoder(seqTransEncoderLayer,
                                               num_layers=self.num_layers)
        else:
            raise NotImplementedError()

    def forward(self, other, cur=None, cur_canon=None, other_canon=None):

        if 'in_both' in self.fn_type:
            assert other is not None
            x = self.aggregation(torch.concatenate((cur, other), dim=-1))
        else:
            x = other

        if self.predict_6dof:
            assert cur_canon is not None and other_canon is not None
            cur_canon = cur_canon.squeeze(-1).permute(2, 0, 1)[..., :9]
            other_canon = other_canon.squeeze(-1).permute(2, 0, 1)[..., :9]
            canon = self.canon_agg(torch.concatenate((cur_canon, other_canon), dim=-1))
            x = torch.concatenate((canon, x), dim=0)

        out = self.model(x)
        if self.predict_6dof:
            canon, out = torch.split(out, [1, out.shape[0] - 1], dim=0)
            canon = self.canon_out(canon).permute(1, 2, 0).unsqueeze(-1)
            pad = torch.zeros([canon.shape[0], 138-9, 1, 1], device=canon.device, dtype=canon.dtype)
            canon = torch.cat((canon, pad), axis=1)
        else:
            canon = None

        return out, canon
    


class MultiPersonBlock(nn.Module):
    def __init__(self, arch, fn_type, num_layers, latent_dim, input_feats, predict_6dof):
        super().__init__()
        self.arch = arch
        self.fn_type = fn_type
        self.predict_6dof = predict_6dof
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        self.num_heads = 4
        self.ff_size = 1024
        self.dropout = 0.1
        self.activation = 'gelu'
        self.input_feats = input_feats
        if self.predict_6dof:
            self.canon_agg = nn.Linear(9*2, self.latent_dim)
            # self.canon_agg = nn.Linear(9*2, self.latent_dim)
            # self.canon_agg = nn.Linear(self.input_feats*2, self.latent_dim)
            self.canon_out = nn.Linear(self.latent_dim, 9)
            # self.canon_out = nn.Linear(self.latent_dim, self.input_feats)
        if 'in_both' in self.fn_type:
            self.aggregation = nn.Linear(self.latent_dim*2, self.latent_dim)
        if self.arch == 'trans_enc':
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=self.activation)
            self.model = nn.TransformerEncoder(seqTransEncoderLayer,
                                               num_layers=self.num_layers)
        self.cross_attention = CrossAttention(embed_size=512, heads=4) 
        self.con_global = nn.Linear(self.latent_dim, 9)
        self.avg_pooling = nn.AdaptiveAvgPool1d(output_size=1)
        self.max_pooling = nn.AdaptiveMaxPool1d(output_size=1)

    def forward(self, low_cur=None, low_other=None, other=None, cur=None, cur_canon=None, other_canon=None, text_emb =None):     
        low_x, low_x_other = low_cur + cur, low_other + other
        # 交换concat顺序计算向量
        x = self.aggregation(torch.concatenate((low_x, low_x_other), dim=-1))
        x_other = self.aggregation(torch.concatenate((low_x_other, low_x), dim=-1))
        # print("COMDMD中的x维度:", x.shape) torch.Size([120, 64, 512])
        x_out = self.model(x)# torch.Size([120, 64, 512])
        x_other_out= self.model(x_other)# torch.Size([120, 64, 512])
        # print("文本向量的emb:", text_emb.shape) torch.Size([1, 64, 512])
        # cross attention 
        x_hat = self.cross_attention(x_out, x_out, text_emb)
        x_other_hat = self.cross_attention(x_other_out, x_other_out, text_emb)
        # print("交叉注意力之后的维度：", x_hat.shape) # torch.Size([120, 64, 512])
        

        # 添加D的信息
        cur_canon = cur_canon.squeeze(-1).permute(2, 0, 1)[..., :9]
        other_canon = other_canon.squeeze(-1).permute(2, 0, 1)[..., :9]
        x_canon = self.canon_agg(torch.concatenate((cur_canon, other_canon), dim=-1))
        x_other_canon =  self.canon_agg(torch.concatenate((other_canon, cur_canon), dim=-1))
        # print("聚合后的x_canon:", x_canon.shape) # torch.Size([1, 64, 512])


        # 池化层，
        x_pool, x_other_pool = self.avg_pooling(x_hat.permute(1,2,0)) + self.max_pooling(x_hat.permute(1,2,0)), self.avg_pooling(x_other_hat.permute(1,2,0)) + self.max_pooling(x_other_hat.permute(1,2,0))
        # print("池化的维度：", x_pool.shape) # torch.Size([64, 512, 1])
        x_pool, x_other_pool = x_pool.permute(2,0,1), x_other_pool.permute(2,0,1)# torch.Size([1, 64, 512])
        d = x_pool + x_canon
        other_d = x_other_pool + x_other_canon
        # print("聚合后的x_d", d.shape) # torch.Size([1, 64, 512])
        d_out = self.canon_out(d)
        d_other = self.canon_out(other_d)
        # print("最后输出的d_out", d_out.shape) # torch.Size([1, 64, 9])
        

        pad = torch.zeros([d_out.shape[1], 138-9, 1, 1], device=d_out.device, dtype=d_out.dtype)
        canon_d = torch.cat((d_out.squeeze().unsqueeze(2).unsqueeze(3), pad), axis=1)
        canon_d_other = torch.cat((d_other.squeeze().unsqueeze(2).unsqueeze(3), pad), axis=1)
        # print("输出的D", canon_d.shape) # 输出的D torch.Size([64, 138, 1, 1])
        
        return x_hat, x_other_hat, canon_d, canon_d_other
    

class MultiPersonBlock(nn.Module):
    def __init__(self, arch, fn_type, num_layers, latent_dim, input_feats, predict_6dof):
        super().__init__()
        self.arch = arch
        self.fn_type = fn_type
        self.predict_6dof = predict_6dof
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        self.num_heads = 4
        self.ff_size = 1024
        self.dropout = 0.1
        self.activation = 'gelu'
        self.input_feats = input_feats
        if self.predict_6dof:
            self.canon_agg = nn.Linear(9*2, self.latent_dim)
            # self.canon_agg = nn.Linear(9*2, self.latent_dim)
            # self.canon_agg = nn.Linear(self.input_feats*2, self.latent_dim)
            self.canon_out = nn.Linear(self.latent_dim, 9)
            # self.canon_out = nn.Linear(self.latent_dim, self.input_feats)
        if 'in_both' in self.fn_type:
            self.aggregation = nn.Linear(self.latent_dim*2, self.latent_dim)
        if self.arch == 'trans_enc':
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=self.activation)
            self.model = nn.TransformerEncoder(seqTransEncoderLayer,
                                               num_layers=self.num_layers)
        self.cross_attention = CrossAttention(embed_size=512, heads=4) 
        self.con_global = nn.Linear(self.latent_dim, 9)
        self.avg_pooling = nn.AdaptiveAvgPool1d(output_size=1)
        self.max_pooling = nn.AdaptiveMaxPool1d(output_size=1)

    def forward(self, low_cur=None, low_other=None, other=None, cur=None, cur_canon=None, other_canon=None, text_emb =None):     
        low_x, low_x_other = low_cur + cur, low_other + other
        # 交换concat顺序计算向量
        x = self.aggregation(torch.concatenate((low_x, low_x_other), dim=-1))
        x_other = self.aggregation(torch.concatenate((low_x_other, low_x), dim=-1))
        # print("COMDMD中的x维度:", x.shape) torch.Size([120, 64, 512])
        x_out = self.model(x)# torch.Size([120, 64, 512])
        x_other_out= self.model(x_other)# torch.Size([120, 64, 512])
        # print("文本向量的emb:", text_emb.shape) torch.Size([1, 64, 512])
        # cross attention 
        x_hat = self.cross_attention(x_out, x_out, text_emb)
        x_other_hat = self.cross_attention(x_other_out, x_other_out, text_emb)
        # print("交叉注意力之后的维度：", x_hat.shape) # torch.Size([120, 64, 512])
        

        # 添加D的信息
        cur_canon = cur_canon.squeeze(-1).permute(2, 0, 1)[..., :9]
        other_canon = other_canon.squeeze(-1).permute(2, 0, 1)[..., :9]
        x_canon = self.canon_agg(torch.concatenate((cur_canon, other_canon), dim=-1))
        x_other_canon =  self.canon_agg(torch.concatenate((other_canon, cur_canon), dim=-1))
        # print("聚合后的x_canon:", x_canon.shape) # torch.Size([1, 64, 512])


        # 池化层，
        x_pool, x_other_pool = self.avg_pooling(x_hat.permute(1,2,0)) + self.max_pooling(x_hat.permute(1,2,0)), self.avg_pooling(x_other_hat.permute(1,2,0)) + self.max_pooling(x_other_hat.permute(1,2,0))
        # print("池化的维度：", x_pool.shape) # torch.Size([64, 512, 1])
        x_pool, x_other_pool = x_pool.permute(2,0,1), x_other_pool.permute(2,0,1)# torch.Size([1, 64, 512])
        d = x_pool + x_canon
        other_d = x_other_pool + x_other_canon
        # print("聚合后的x_d", d.shape) # torch.Size([1, 64, 512])
        d_out = self.canon_out(d)
        d_other = self.canon_out(other_d)
        # print("最后输出的d_out", d_out.shape) # torch.Size([1, 64, 9])
        

        pad = torch.zeros([d_out.shape[1], 138-9, 1, 1], device=d_out.device, dtype=d_out.dtype)
        canon_d = torch.cat((d_out.squeeze().unsqueeze(2).unsqueeze(3), pad), axis=1)
        canon_d_other = torch.cat((d_other.squeeze().unsqueeze(2).unsqueeze(3), pad), axis=1)
        # print("输出的D", canon_d.shape) # 输出的D torch.Size([64, 138, 1, 1])
        
        return x_hat, x_other_hat, canon_d, canon_d_other
    

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(CrossAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query):
        ori_v = values
        values, keys, query = values.permute(1,0,2),keys.permute(1,0,2), query.permute(1,0,2)
        query = torch.repeat_interleave(query, repeats=values.shape[1],dim=1)
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Attention mechanism
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        out =  out.permute(1,0,2) + ori_v
        return out
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import sys, os
path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(path)

from model.rotation2xyz import Rotation2xyz

class my_MDM(nn.Module):
    def __init__(self, modeltype, njoints, nfeats, num_actions, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", legacy=False, data_rep='rot6d', dataset='amass', clip_dim=512,
                 arch='trans_enc', emb_trans_dec=False, clip_version=None, args=None, **kargs):
        super().__init__()

        self.legacy = legacy
        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_actions = num_actions
        self.data_rep = data_rep
        self.dataset = dataset

        self.args = args

        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation
        self.clip_dim = clip_dim
        self.action_emb = kargs.get('action_emb', None)

        self.input_feats = self.njoints * self.nfeats

        self.normalize_output = kargs.get('normalize_encoder_output', False)

        self.cond_mode = kargs.get('cond_mode', 'no_cond')
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        self.arch = arch
        self.gru_emb_dim = self.latent_dim if self.arch == 'gru' else 0
        self.input_process = InputProcess(self.data_rep, self.input_feats+self.gru_emb_dim, self.latent_dim)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.emb_trans_dec = emb_trans_dec

        if self.arch == 'trans_enc':
            print("TRANS_ENC init")
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=self.activation)

            self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                         num_layers=self.num_layers)
        else:
            raise ValueError('Please choose correct architecture [trans_enc, trans_dec, gru]')

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        self.multi_person = MultiPersonBlock(arch=self.args.multi_arch,
                                             fn_type=self.args.multi_func,
                                             num_layers=self.args.multi_num_layers,
                                             latent_dim=self.latent_dim,
                                             input_feats=self.input_feats,
                                             predict_6dof=self.args.predict_6dof)
       
            # if 'text' in self.cond_mode:
        self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
        print('EMBED TEXT')
        print('Loading CLIP...')
        self.clip_version = clip_version
        self.clip_model = self.load_and_freeze_clip(clip_version)


        self.output_process = OutputProcess(self.data_rep, self.input_feats, self.latent_dim, self.njoints,
                                            self.nfeats)

        self.rot2xyz = Rotation2xyz(device='cpu', dataset=self.dataset)

        self.freeze_block(self.input_process)
        self.freeze_block(self.sequence_pos_encoder)
        self.freeze_block(self.seqTransEncoder)
        self.freeze_block(self.embed_timestep)
        # if 'text' in self.cond_mode:
        self.freeze_block(self.embed_text)
        self.freeze_block(self.output_process)

        self.emb_dict = {}

    def freeze_block(self, block):
        block.eval()
        for p in block.parameters():
            p.requires_grad = False
    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                                jit=False)  # Must set jit=False for training
        clip.model.convert_weights(
            clip_model)  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    def encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        device = next(self.parameters()).device
        max_text_len = 20 if self.dataset in ['humanml', 'kit','self_humanml','film'] else None  # Specific hardcoding for humanml dataset
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2 # start_token + 20 + end_token
            assert context_length < default_context_length
            texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device) # [bs, context_length] # if n_tokens > context_length -> will truncate
            # print('texts', texts.shape)
            zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], dtype=texts.dtype, device=texts.device)
            texts = torch.cat([texts, zero_pad], dim=1)
            # print('texts after pad', texts.shape, texts)
        else:
            texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length] # if n_tokens > 77 -> will truncate
        return self.clip_model.encode_text(texts).float()

    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, nfeats, 1, max_frames+1], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        x1, x2 = torch.split(x,[1, x.shape[2]-1], dim=2)
        # print("检查输入数据：", x1)
        # print("检查输入数据值和：", torch.sum(x1))
        # print("输入数据拆分的格式:",x1.shape, x2.shape) # 输入数据拆分的格式: torch.Size([64, 138, 1, 121]) torch.Size([64, 138, 1, 121])
        canon, x = torch.split(x1,[1, x1.shape[-1]-1], dim=-1)
        canon_other, x_other = torch.split(x2,[1, x2.shape[-1]-1], dim=-1)
        # print("D的维度：", canon.shape)
        # print("MOTIO的维度：", x.shape)

        # 验证的时候
        # print("检查采样：", y['text'])
        enc_text = self.encode_text(y['text']) # 

        # print("text emb 的维度：", enc_text.shape)

        # has_nan_x, has_nan_canon = torch.isnan(x).any(), torch.isnan(canon).any()
        # has_nan_x_other, has_nan_canon_other = torch.isnan(x_other).any(), torch.isnan(canon_other).any()
        # has_nan_text = torch.isnan(enc_text).any()

        # print("Has NaN--1 values?", has_nan_x, has_nan_canon, has_nan_x_other, has_nan_canon_other,has_nan_text )


        # timestep emb
        emb = self.embed_timestep(timesteps)  # [1, bs, d]
        # t_emb + text_emb
        emb += self.embed_text(enc_text)

        # self.emb_dict['text_emb'] = self.embed_text(enc_text)
        
        #print(self.emb_dict['text_emb'].shape)

        # has_nan_emb = torch.isnan(emb).any()
        # print("Has NaN--2 values?",has_nan_emb)
        # force_mask = y.get('uncond', False)
        # if 'text' in self.cond_mode:
        
        x = self.input_process(x)
        x_other = self.input_process(x_other)

        # self.emb_dict['lower_emb_person_1'] = x
        # self.emb_dict['lower_emb_person_2'] = x_other
        # print(x.shape)
        low_x, low_x_other = x, x_other

        # has_nan_x, has_nan_x_other = torch.isnan(x).any(), torch.isnan(x_other).any()
        # print("Has NaN--3 values?",has_nan_x, has_nan_x_other)

        # adding the timestep embed
        xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
        xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
        x_other = torch.cat((emb, x_other), axis=0)
        x_other = self.sequence_pos_encoder(x_other)

        mid = self.seqTransEncoder(xseq)[1:]
        mid_other = self.seqTransEncoder(x_other)[1:]

        # self.emb_dict['pretrained_motion_emb_person_1'] = mid
        # self.emb_dict['pretrained_motion_emb_person_2'] = mid_other

        # print(mid.shape)

        # has_nan_mid, has_nan_mid_other = torch.isnan(mid).any(), torch.isnan(mid_other).any()
        # print("Has NaN--4 values?",has_nan_mid, has_nan_mid_other)

        delta_x, delta_x_other, canon_out, canon_other_out = self.multi_person(low_cur=low_x, low_other=low_x_other,cur=mid, other=mid_other, cur_canon=canon,
                                               other_canon=canon_other, text_emb = emb)
        # print("修正值：", delta_x)
        # print("修正值的和：", torch.sum(delta_x))
        # has_nan_delta_x, has_nan_canon_out = torch.isnan(delta_x).any(), torch.isnan(canon_out).any()
        # print("Has NaN--5 values?",has_nan_delta_x, has_nan_canon_out)

        # motion的修正量
        mid_out = mid + delta_x
        mid_other_out = mid_other + delta_x_other 

        # self.emb_dict['output_motion_emb_person_1'] = mid_out
        # self.emb_dict['output_motion_emb_person_2'] = mid_other_out
        self.emb_dict['name'] = self.args.input_text
        # print("差值：",self.emb_dict['pretrained_motion_emb_person_1']-self.emb_dict['output_motion_emb_person_1'])
        # print("求和：", torch.sum(self.emb_dict['pretrained_motion_emb_person_1']-self.emb_dict['output_motion_emb_person_1']))
        # assert 1==2
        # np.save("D:\Code\priorMDM-main\\temporary_folder\\test_our_350\samples\\emb\\{}.npy".format(self.args.text_prompt), self.emb_dict)
        # assert 1==2
        # print("输出D1 的维度：",canon_out.shape)
        output_x = self.output_process(mid_out)  # [bs, njoints, nfeats, nframes]
        output_x_other =  self.output_process(mid_other_out)
        # print("查看output的维度：", output_x.shape) #  torch.Size([64, 138, 1, 120])
        # has_nan_out =  torch.isnan(output).any()
        # print("Has NaN--6 values?",has_nan_out)

        # print("输出X1 的维度：",output.shape)
        output_x = torch.cat((canon_out, output_x), dim=-1)
        output_x_other = torch.cat((canon_other_out, output_x_other), dim=-1)

        # 拼接
        final_output = torch.cat([output_x, output_x_other], dim=2)
        # print("模型的最终输出，拼接的维度：",final_output.shape) #  torch.Size([64, 138, 2, 121])
        # assert 1==2
        return final_output

    def _apply(self, fn):
        super()._apply(fn)
        self.rot2xyz.smpl_model._apply(fn)


    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        self.rot2xyz.smpl_model.train(*args, **kwargs)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class InputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        if self.data_rep == 'rot_vel':
            self.velEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats)

        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            x = self.poseEmbedding(x)  # [seqlen, bs, d]
            return x
        elif self.data_rep == 'rot_vel':
            first_pose = x[[0]]  # [1, bs, 150]
            first_pose = self.poseEmbedding(first_pose)  # [1, bs, d]
            vel = x[1:]  # [seqlen-1, bs, 150]
            vel = self.velEmbedding(vel)  # [seqlen-1, bs, d]
            return torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, d]
        else:
            raise ValueError


class OutputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim, njoints, nfeats):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        if self.data_rep == 'rot_vel':
            self.velFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        nframes, bs, d = output.shape
        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            output = self.poseFinal(output)  # [seqlen, bs, 150]
        elif self.data_rep == 'rot_vel':
            first_pose = output[[0]]  # [1, bs, d]
            first_pose = self.poseFinal(first_pose)  # [1, bs, 150]
            vel = output[1:]  # [seqlen-1, bs, d]
            vel = self.velFinal(vel)  # [seqlen-1, bs, 150]
            output = torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, 150]
        else:
            raise ValueError
        output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        return output


class EmbedAction(nn.Module):
    def __init__(self, num_actions, latent_dim):
        super().__init__()
        self.action_embedding = nn.Parameter(torch.randn(num_actions, latent_dim))

    def forward(self, input):
        idx = input[:, 0].to(torch.long)  # an index array must be long
        output = self.action_embedding[idx]
        return output
    


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
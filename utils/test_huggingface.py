from transformers import pipeline, BertTokenizer, BertModel
import numpy as np

# bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# bert_model = BertModel.from_pretrained("bert-base-uncased")



# direct encoding of the sample sentence
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
encoded_seq = tokenizer.encode("i am sentence")

# your approach
feature_extraction = pipeline('feature-extraction', model="distilroberta-base", tokenizer="distilroberta-base")
features = feature_extraction(["One person is sitting, the other comes toward him and they hug.", "Hello world."])
features_2 = feature_extraction("Hello world.")
# Compare lengths of outputs
print("encode_seq::",encoded_seq) # 5
# Note that the output has a weird list output that requires to index with 0.
# print("feature::",np.mean(np.array(features[0]), axis=0)) # 5
print("feature_2::",np.array(features).shape)
# input data
source_path = "../data"
train_source = "train_data.json"
dev_source = "dev_data.json"
test_source = "test1_data_postag.json"

# output data
data_path = "./data"
train_name = "train_data.pkl"
dev_name = "dev_data.pkl"
test_name = "test1_data_postag.pkl"
vocabulary_name = "vocabulary.pkl"
weight_name = "weight.pkl"

# args
seq_len = 320
pad_str = '<pad>'
unk_str = '<unk>'
batch_size = 64
embed_dim = 128

# seed
seed = 11

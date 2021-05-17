# path
source_path = "../../data"
data_path = "../data"
save_path = "./models"
log_path = "./log"

train_source = "train_data.json"
train_name = "train_data.pkl"
dev_source = "dev_data.json"
dev_name = "dev_data.pkl"
# test_source = "test1_data_postag_with_predicate.json"
# test_name = "test1_data_postag_with_predicate.pkl"

word_vocab_name = "word_vocab.pkl"
char_vocab_name = "char_vocab.pkl"
pos_vocab_name = "pos_vocab.pkl"
tag_vocab_name = "tag_vocab.pkl"
weight_name = "weight.pkl"

device = 2
seed = 11

char_embed_dim = 64
word_embed_dim = 64
pos_embed_dim = 64
sentence_length = 320
hidden_dim = 128
encoding_type = 'bieso'
batch_size = 64
epochs = 64
learning_rate = 1e-3
weight_decay = 0
patience = 10


num_layers = 4
inner_size = 256
key_size = 64
value_size = 64
num_head = 4
dropout = 0.1

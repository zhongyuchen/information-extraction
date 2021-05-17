# path
source_path = "../../data"
data_path = "../data"
save_path = "./model.pth"
log_path = "./log"

train_name = "train_data.pkl"
dev_name = "dev_data.pkl"
# test_name = "test1_data_postag_with_predicate.pkl"

word_vocab_name = "word_vocab.pkl"
char_vocab_name = "char_vocab.pkl"
pos_vocab_name = "pos_vocab.pkl"
tag_vocab_name = "tag_vocab.pkl"
weight_name = "weight.pkl"

device = 3
seed = 11

char_embed_dim = 64
word_embed_dim = 64
pos_embed_dim = 64
seq_len = 320
hidden_dim = 128
dropout = 0.5
encoding_type = 'bieso'
learning_rate = 1e-3
weight_decay = 0


epochs = 64

patience = 10

print_every = 1000
validate_every = 0

num_layers = 4
inner_size = 256
key_size = 64
value_size = 64
num_head = 4
dropout = 0.1


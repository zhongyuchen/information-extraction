# path
data_path = "../data"
save_path = "./model.pth"
log_path = "./log"

# data
train_name = "train_data.pkl"
dev_name = "dev_data.pkl"
test_name = "test1_data_postag.pkl"
vocabulary_name = "vocabulary.pkl"
weight_name = "weight.pkl"

# args
num_layers = 1
embed_dim = 128
hidden_dim = 256
class_num = 50
dropout = 0.5

# learning
seed = 11
learning_rate = 1e-3
weight_decay = 0
device = 0
epochs = 64
patience = 10

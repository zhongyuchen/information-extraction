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

# cnn args
class_num = 50
embed_dim = 128
kernel_sizes = (3, 4, 5)
kernel_num = 128
in_channels = 1
dropout = 0.5
static = False

# learning
seed = 11
learning_rate = 1e-3
weight_decay = 0
device = 0
epochs = 64
batch_size = 16
patience = 10

import os


class Config:
    # path
    data_path = "./data"
    save_path = "./model_log"

    train_source = "train_data.json"
    train_name = "train_data.pkl"
    dev_source = "dev_data.json"
    dev_name = "dev_data.pkl"
    test_source = "test1_data_postag.json"
    test_name = "test1_data_postag.pkl"

    vocabulary_name = "vocabulary.pkl"

    # cnn
    class_num = 50
    embed_dim = 128
    kernel_sizes = (3, 4, 5)
    kernel_num = 128
    in_channels = 1
    dropout = 0.5
    static = False
    sentence_length = 320

    # lstm
    num_layers = 1
    hidden_dim = 256

    # Adam
    lr = 1e-3
    weight_decay = 0
    # early stop
    patience = 20

    # train
    device = [0, 1, 2]
    # device = 'cpu'
    epoch = 128
    batch_size = 16
    print_every = 100
    validate_every = 0

    # task
    task_name = "cnn"

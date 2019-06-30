
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

    word_vocab_name = "word_vocab.pkl"
    char_vocab_name = "char_vocab.pkl"
    pos_vocab_name = "pos_vocab.pkl"
    spo_vocab_name = "spo_vocab.pkl"
    tag_vocab_name = "tag_vocab.pkl"

    embed_dim = 128
    hidden_dim = 256
    batch_size = 16
    epoch = 64
    # device = [0, 1, 2]
    device = 'cpu'

    lr = 0.015
    momentum = 0.9
    patience = 20

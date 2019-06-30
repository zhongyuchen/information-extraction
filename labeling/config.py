
class Config:
    debug = False

    # path
    source_path = "../data"
    data_path = "./data"
    save_path = "./models"

    train_source = "train_data.json"
    train_name = "train_data.pkl"
    dev_source = "dev_data.json"
    dev_name = "dev_data.pkl"
    test_source = "test1_data_postag_with_predicate.json"
    test_name = "test1_data_postag_with_predicate.pkl"

    word_vocab_name = "word_vocab.pkl"
    char_vocab_name = "char_vocab.pkl"
    pos_vocab_name = "pos_vocab.pkl"
    #spo_vocab_name = "spo_vocab.pkl"
    tag_vocab_name = "tag_vocab.pkl"

    device = [5]
    # device = 'cpu'

    char_embed_dim = 64
    word_embed_dim = 64
    pos_embed_dim = 64
    sentence_length = 320
    hidden_dim = 128
    #dropout = 0.5
    encoding_type = 'bieso'
    batch_size = 64
    epoch = 64
    lr = 1e-3
    weight_decay = 0
    patience = 10

    print_every = 1000
    validate_every = 0

    task_names = ['bilstm_crf', 'trans_crf']
    
    num_layers = 4
    inner_size = 256
    key_size = 64
    value_size = 64
    num_head = 4
    dropout = 0.1

    ensemble_models = ['best_AdvSeqLabel_f_2019-06-28-21-16-23', 
                        'best_TransformerSeqLabel_f_2019-06-29-11-26-33']


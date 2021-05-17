
class Config:
    debug = False

    # path
    source_path = "../data"
    data_path = "./data"
    save_path = "./models"
    bert_path = "../bert_model.pt"
    bert_data_path = "./data_bert"
    bert_folder = "../chinese_L-12_H-768_A-12"
    bert_vocab_path = "../chinese_L-12_H-768_A-12/vocab.txt"

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

    # Adam for 5 models
    # lr = 1e-3
    # weight_decay = 0
    # early stop
    patience = 10

    # BertAdam for bert
    weight_decay = 0.01
    lr = 5e-5
    warmup_proportion = 0.1
    local_rank = -1
    update_every = 1  # gradient_accumulation_steps = 1

    # train
    device = [2]
    # device = 'cpu'
    epoch = 3.0
    batch_size = 16
    print_every = 1000
    validate_every = 0

    task_names = ['cnn', 'rnn', 'lstm', 'lstm_maxpool', 'rcnn', 'bert']

    ensemble_models = ['best_CNN_f1_2019-06-25-22-40-27',
                       'best_LSTM_f1_2019-06-25-22-40-39',
                       'best_LSTM_maxpool_f1_2019-06-25-22-40-45',
                       'best_RCNN_f1_2019-06-25-22-40-51',
                       'best_BertForMultiLabelSequenceClassification_f1_2019-06-26-23-26-14']
    ensemble_batch = 16
    prob_path = './prob'

    predict_batch = 32
    predict_path = '../data'
    predict_name = 'test1_data_postag_with_predicate.json'


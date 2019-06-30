from fastNLP import Trainer
from fastNLP import Adam
from fastNLP import EarlyStopCallback
from fastNLP import Tester
from fastNLP.modules.encoder.bert import BertModel

import os
import pickle
import sys

from config import Config
from model import CNN, LSTM, LSTM_maxpool, RCNN, RNN
from utils import TimingCallback, BCEWithLogitsLoss, F1_score

import fitlog
from fastNLP.core.callback import FitlogCallback
fitlog.commit(__file__)             # auto commit your codes
fitlog.add_hyper_in_file(__file__)  # record your hyperparameters


def train(config, task_name):
    train_data = pickle.load(open(os.path.join(config.data_path, config.train_name), "rb"))
    # debug
    if config.debug:
        train_data = train_data[0:30]
    dev_data = pickle.load(open(os.path.join(config.data_path, config.dev_name), "rb"))
    # test_data = pickle.load(open(os.path.join(config.data_path, config.test_name), "rb"))
    vocabulary = pickle.load(open(os.path.join(config.data_path, config.vocabulary_name), "rb"))

    # load w2v data
    # weight = pickle.load(open(os.path.join(config.data_path, config.weight_name), "rb"))

    if task_name == "lstm":
        text_model = LSTM(vocab_size=len(vocabulary), embed_dim=config.embed_dim,
                          output_dim=config.class_num, hidden_dim=config.hidden_dim,
                          num_layers=config.num_layers, dropout=config.dropout)
    elif task_name == "lstm_maxpool":
        text_model = LSTM_maxpool(vocab_size=len(vocabulary), embed_dim=config.embed_dim,
                         		  output_dim=config.class_num, hidden_dim=config.hidden_dim,
                         		  num_layers=config.num_layers, dropout=config.dropout)
    elif task_name == "cnn":
        text_model = CNN(vocab_size=len(vocabulary), embed_dim=config.embed_dim,
                         class_num=config.class_num, kernel_num=config.kernel_num,
                         kernel_sizes=config.kernel_sizes, dropout=config.dropout,
                         static=config.static, in_channels=config.in_channels)
    elif task_name == "rnn":
        text_model = RNN(vocab_size=len(vocabulary), embed_dim=config.embed_dim,
                         output_dim=config.class_num, hidden_dim=config.hidden_dim,
                         num_layers=config.num_layers, dropout=config.dropout)
    # elif task_name == "cnn_w2v":
    #     text_model = CNN_w2v(vocab_size=len(vocabulary), embed_dim=config.embed_dim,
    #                          class_num=config.class_num, kernel_num=config.kernel_num,
    #                          kernel_sizes=config.kernel_sizes, dropout=config.dropout,
    #                          static=config.static, in_channels=config.in_channels,
    #                          weight=weight)
    elif task_name == "rcnn":
        text_model = RCNN(vocab_size=len(vocabulary), embed_dim=config.embed_dim,
                          output_dim=config.class_num, hidden_dim=config.hidden_dim,
                          num_layers=config.num_layers, dropout=config.dropout)
    #elif task_name == "bert":
    #    text_model = BertModel.from_pretrained(config.bert_path)

    optimizer = Adam(lr=config.lr, weight_decay=config.weight_decay)
    timing = TimingCallback()
    early_stop = EarlyStopCallback(config.patience)
    logs = FitlogCallback(dev_data)
    f1 = F1_score(pred='output', target='target')

    trainer = Trainer(train_data=train_data, model=text_model, loss=BCEWithLogitsLoss(),
                      batch_size=config.batch_size, check_code_level=-1,
                      metrics=f1, metric_key='f1', n_epochs=config.epoch,
                      dev_data=dev_data, save_path=config.save_path,
                      print_every=config.print_every, validate_every=config.validate_every,
                      optimizer=optimizer, use_tqdm=False,
                      device=config.device, callbacks=[timing, early_stop, logs])
    trainer.train()

    # test result
    tester = Tester(dev_data, text_model, metrics=f1,
                    device=config.device, batch_size=config.batch_size,)
    tester.test()


def log_config(config, task_name):
    fitlog.add_other(task_name, name="model_name")
    fitlog.add_other(config.class_num, name="class_num")
    fitlog.add_other(str(config.kernel_sizes), name="kernel_sizes")
    fitlog.add_other(config.kernel_num, name="kernel_num")
    fitlog.add_other(config.in_channels, name="in_channels")
    fitlog.add_other(config.dropout, name="dropout")
    fitlog.add_other(config.static, name="static")
    fitlog.add_other(config.sentence_length, name="sentence_length")
    fitlog.add_other(config.num_layers, name="num_layers")
    fitlog.add_other(config.hidden_dim, name="hidden_dim")
    fitlog.add_other(config.lr, name="lr")
    fitlog.add_other(config.weight_decay, name="weight_decay")
    fitlog.add_other(config.patience, name="patience")
    fitlog.add_other(config.epoch, name="epoch")
    fitlog.add_other(config.batch_size, name="batch_size")
    fitlog.add_other(str(config.device), name="device")


if __name__ == "__main__":
    config = Config()
    # print configs
    #with open('config.py', 'r') as f:
    #    for line in f:
    #        print(line, end='')
    
    argv = sys.argv
    task_name = argv[1]
    assert task_name in config.task_names

    log_config(config, task_name)

    train(config, task_name)
    fitlog.finish()


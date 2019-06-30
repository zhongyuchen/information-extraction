from fastNLP import Trainer
from fastNLP import Adam, SGD
from fastNLP import EarlyStopCallback
from fastNLP import SpanFPreRecMetric
from fastNLP import NLLLoss
from fastNLP import Tester

import os
import pickle

from config import Config
from model import BiLSTM_CRF
# from ref.model import CNN, CNN_w2v, RNN, LSTM, LSTM_maxpool, RCNN
from utils import TimingCallback


def train(config):
    train_data = pickle.load(open(os.path.join(config.data_path, config.train_name), "rb"))
    # debug
    train_data = train_data[0:100]
    dev_data = pickle.load(open(os.path.join(config.data_path, config.dev_name), "rb"))
    print(len(train_data), len(dev_data))
    # test_data = pickle.load(open(os.path.join(config.data_path, config.test_name), "rb"))
    # load w2v data
    # weight = pickle.load(open(os.path.join(config.data_path, config.weight_name), "rb"))

    word_vocab = pickle.load(open(os.path.join(config.data_path, config.word_vocab_name), "rb"))
    char_vocab = pickle.load(open(os.path.join(config.data_path, config.char_vocab_name), "rb"))
    pos_vocab = pickle.load(open(os.path.join(config.data_path, config.pos_vocab_name), "rb"))
    spo_vocab = pickle.load(open(os.path.join(config.data_path, config.spo_vocab_name), "rb"))
    tag_vocab = pickle.load(open(os.path.join(config.data_path, config.tag_vocab_name), "rb"))
    print('word vocab', len(word_vocab))
    print('char vocab', len(char_vocab))
    print('pos vocab', len(pos_vocab))
    print('spo vocab', len(spo_vocab))
    print('tag vocab', len(tag_vocab))

    model = BiLSTM_CRF(config.batch_size, len(word_vocab), len(char_vocab), len(pos_vocab), len(spo_vocab),
                 config.embed_dim, config.hidden_dim, tag_vocab.idx2word, dropout=0.5)

    optimizer = SGD(lr=config.lr, momentum=config.momentum)
    timing = TimingCallback()
    early_stop = EarlyStopCallback(config.patience)
    loss = NLLLoss()
    metrics = SpanFPreRecMetric(tag_vocab)
    # accuracy = AccuracyMetric(pred='output', target='target')

    trainer = Trainer(train_data=train_data, model=model, loss=loss, metrics=metrics,
                      batch_size=config.batch_size, n_epochs=config.epoch,
                      dev_data=dev_data, save_path=config.save_path,
                      check_code_level=-1,
                      print_every=100, validate_every=0,
                      optimizer=optimizer, use_tqdm=False,
                      device=config.device, callbacks=[timing, early_stop])
    trainer.train()

    # test result
    # tester = Tester(test_data, text_model, metrics=accuracy)
    # tester.test()


if __name__ == "__main__":
    config = Config()
    # print configs
    with open('config.py', 'r') as f:
        for line in f:
            print(line, end='')
    train(config)

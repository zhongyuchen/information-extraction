from fastNLP import Trainer
from fastNLP import Adam, SGD
from fastNLP import EarlyStopCallback
from fastNLP import SpanFPreRecMetric
from fastNLP import NLLLoss
from fastNLP import Tester

import os
import pickle
import sys

from config import Config
from model_trans import AdvSeqLabel, TransformerSeqLabel
# from ref.model import CNN, CNN_w2v, RNN, LSTM, LSTM_maxpool, RCNN
from utils import TimingCallback, get_schemas

import fitlog
from fastNLP.core.callback import FitlogCallback
fitlog.commit(__file__)             # auto commit your codes
fitlog.add_hyper_in_file(__file__)  # record your hyperparameters


def train(config, task_name):
    train_data = pickle.load(open(os.path.join(config.data_path, config.train_name), "rb"))
    # debug
    if config.debug:
        train_data = train_data[0:100]
    dev_data = pickle.load(open(os.path.join(config.data_path, config.dev_name), "rb"))
    print(len(train_data), len(dev_data))
    # test_data = pickle.load(open(os.path.join(config.data_path, config.test_name), "rb"))
    # load w2v data
    # weight = pickle.load(open(os.path.join(config.data_path, config.weight_name), "rb"))

    word_vocab = pickle.load(open(os.path.join(config.data_path, config.word_vocab_name), "rb"))
    char_vocab = pickle.load(open(os.path.join(config.data_path, config.char_vocab_name), "rb"))
    pos_vocab = pickle.load(open(os.path.join(config.data_path, config.pos_vocab_name), "rb"))
    # spo_vocab = pickle.load(open(os.path.join(config.data_path, config.spo_vocab_name), "rb"))
    tag_vocab = pickle.load(open(os.path.join(config.data_path, config.tag_vocab_name), "rb"))
    print('word vocab', len(word_vocab))
    print('char vocab', len(char_vocab))
    print('pos vocab', len(pos_vocab))
    # print('spo vocab', len(spo_vocab))
    print('tag vocab', len(tag_vocab))

    schema = get_schemas(config.source_path)
    
    if task_name == 'bilstm_crf':
        model = AdvSeqLabel(char_init_embed=(len(char_vocab), config.char_embed_dim),
                        word_init_embed=(len(word_vocab), config.word_embed_dim),
                        pos_init_embed=(len(pos_vocab), config.pos_embed_dim),
                        spo_embed_dim=len(schema),
                        sentence_length=config.sentence_length,
                        hidden_size=config.hidden_dim,
                        num_classes=len(tag_vocab),
                        dropout=config.dropout, 
                        id2words=tag_vocab.idx2word,
                        encoding_type=config.encoding_type)
    elif task_name == 'trans_crf':
        model = TransformerSeqLabel(char_init_embed=(len(char_vocab), config.char_embed_dim),
                                    word_init_embed=(len(word_vocab), config.word_embed_dim),
                                    pos_init_embed=(len(pos_vocab), config.pos_embed_dim),
                                    spo_embed_dim=len(schema),
                                    num_classes=len(tag_vocab),
                                    id2words=tag_vocab.idx2word,
                                    encoding_type=config.encoding_type,
                                    num_layers=config.num_layers,
                                    inner_size=config.inner_size,
                                    key_size=config.key_size,
                                    value_size=config.value_size,
                                    num_head=config.num_head,
                                    dropout=config.dropout)

    optimizer = Adam(lr=config.lr, weight_decay=config.weight_decay)
    timing = TimingCallback()
    early_stop = EarlyStopCallback(config.patience)
    # loss = NLLLoss()
    logs = FitlogCallback(dev_data)
    metrics = SpanFPreRecMetric(tag_vocab, pred='pred', seq_len='seq_len', target='tag')
    
    train_data.set_input('tag')
    dev_data.set_input('tag')
    dev_data.set_target('seq_len')
    #print(train_data.get_field_names())
    trainer = Trainer(train_data=train_data, model=model, 
                      # loss=loss, 
                      metrics=metrics, metric_key='f',
                      batch_size=config.batch_size, n_epochs=config.epoch,
                      dev_data=dev_data, save_path=config.save_path,
                      check_code_level=-1,
                      print_every=config.print_every, validate_every=config.validate_every,
                      optimizer=optimizer, use_tqdm=False,
                      device=config.device, callbacks=[timing, early_stop, logs])
    trainer.train()

    # test result
    tester = Tester(dev_data, model, metrics=metrics, device=config.device,
        batch_size=config.batch_size)
    tester.test()


def log_config(config, task_name):
    fitlog.add_other(task_name, name="model_name")
    fitlog.add_other(str(config.device), name="device")
    fitlog.add_other(config.char_embed_dim, name="char_embed_dim")
    fitlog.add_other(config.word_embed_dim, name="word_embed_dim")
    fitlog.add_other(config.pos_embed_dim, name="pos_embed_dim")
    fitlog.add_other(config.sentence_length, name="sentence_length")
    fitlog.add_other(config.hidden_dim, name="hidden_dim")
    fitlog.add_other(config.dropout, name="dropout")
    fitlog.add_other(config.encoding_type, name="encoding_type")
    fitlog.add_other(config.epoch, name="epoch")
    fitlog.add_other(config.batch_size, name="batch_size")
    fitlog.add_other(config.lr, name="lr")
    fitlog.add_other(config.weight_decay, name="weight_decay")
    fitlog.add_other(config.patience, name="patience")

    fitlog.add_other(config.num_layers, name="num_layers")
    fitlog.add_other(config.inner_size, name="inner_size")
    fitlog.add_other(config.key_size, name="key_size")
    fitlog.add_other(config.value_size, name="value_size")
    fitlog.add_other(config.num_head, name="num_head")


if __name__ == "__main__":
    config = Config()
    # print configs
    # with open('config.py', 'r') as f:
    #     for line in f:
    #         print(line, end='')

    argv = sys.argv
    task_name = argv[1]
    assert task_name in config.task_names

    log_config(config, task_name)
    train(config, task_name)
    fitlog.finish()

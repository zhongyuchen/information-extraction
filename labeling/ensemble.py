from config import Config
from model_trans import AdvSeqLabel, TransformerSeqLabel
from dataset import dump_data
from utils import TimingCallback, get_schemas

import os
import torch
import pickle
import sys
import numpy as np

from fastNLP import Tester, Batch, SequentialSampler
from fastNLP import Trainer
from fastNLP import Adam, SGD
from fastNLP import EarlyStopCallback
from fastNLP import SpanFPreRecMetric
from fastNLP import NLLLoss
from fastNLP import Tester


def load_models(config):
    dev_data = pickle.load(open(os.path.join(config.data_path, config.dev_name), "rb"))

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

    bilstm_crf = AdvSeqLabel(char_init_embed=(len(char_vocab), config.char_embed_dim),
                        word_init_embed=(len(word_vocab), config.word_embed_dim),
                        pos_init_embed=(len(pos_vocab), config.pos_embed_dim),
                        spo_embed_dim=len(schema),
                        sentence_length=config.sentence_length,
                        hidden_size=config.hidden_dim,
                        num_classes=len(tag_vocab),
                        dropout=config.dropout, 
                        id2words=tag_vocab.idx2word,
                        encoding_type=config.encoding_type)

    state_dict = torch.load(os.path.join(config.save_path, config.ensemble_models[0])).state_dict()
    bilstm_crf.load_state_dict(state_dict)

    trans_crf = TransformerSeqLabel(char_init_embed=(len(char_vocab), config.char_embed_dim),
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

    state_dict = torch.load(os.path.join(config.save_path, config.ensemble_models[1])).state_dict()
    trans_crf.load_state_dict(state_dict)

    return bilstm_crf, trans_crf


def test_each(config, models):
    dev_data = pickle.load(open(os.path.join(config.data_path, config.dev_name), "rb"))
    metrics = SpanFPreRecMetric(tag_vocab, pred='pred', seq_len='seq_len', target='tag')
    for model_name, model in zip(config.ensemble_models, models):
        print(model_name)
        tester = Tester(dev_data, model, metrics=metrics, device=config.device, batch_size=config.batch_size)
        tester.test()


def _format_eval_results(results):
    _str = ''
    for metric_name, metric_result in results.items():
        _str += metric_name + ': '
        _str += ", ".join([str(key) + "=" + str(value) for key, value in metric_result.items()])
        _str += '\n'
    return _str[:-1]


def dump_model_result(config, model):
    tag_vocab = pickle.load(open(os.path.join(config.data_path, config.tag_vocab_name), 'rb'))
    metrics = SpanFPreRecMetric(tag_vocab, pred='pred', seq_len='seq_len', target='tag')
    dev_data = pickle.load(open(os.path.join(config.data_path, config.dev_name), "rb"))
    data_iterator = Batch(dev_data, config.batch_size, sampler=SequentialSampler(), as_numpy=False)
    model.cuda()

    eval_results = {}
    dev_data.set_input('tag')
    dev_data.set_target('seq_len')
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(data_iterator):
            print('batch', i)
            #if i > 10:
            #    break
            char = batch_x['char'].cuda()
            word = batch_x['word'].cuda()
            pos = batch_x['pos'].cuda()
            spo = batch_x['spo'].cuda()
            seq_len = batch_x['seq_len'].cuda()

            tag = batch_y['tag'].cuda()
            
            #pred = model(char, word, pos, spo, seq_len, tag)
            pred = model.predict(char, word, pos, spo, seq_len)  # labels?
            #labels = idx2label(pred['pred'], tag_vocab.idx2word)
            #print(pred)
            #print(tag)
            #exit()
            metrics({'pred': pred['pred'].cuda(), 'seq_len':seq_len}, {'tag': batch_y['tag'].cuda()})
        eval_result = metrics.get_metric()
        metric_name = metrics.__class__.__name__
        eval_results[metric_name] = eval_result

    print("[tester] \n{}".format(_format_eval_results(eval_results)))


def ensemble(config, models, weight=[1,1]):
    tag_vocab = pickle.load(open(os.path.join(config.data_path, config.tag_vocab_name), 'rb'))
    metrics = SpanFPreRecMetric(tag_vocab, pred='pred', seq_len='seq_len', target='tag')
    dev_data = pickle.load(open(os.path.join(config.data_path, config.dev_name), "rb"))
    data_iterator = Batch(dev_data, config.batch_size, sampler=SequentialSampler(), as_numpy=False)
    models[0].cuda()
    models[1].cuda()

    eval_results = {}
    dev_data.set_input('tag')
    dev_data.set_target('seq_len')
    weight = torch.tensor(weight).float().cuda()
    weight_sum = torch.sum(weight)
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(data_iterator):
            print('batch', i)
            #if i > 10:
            #    break
            char = batch_x['char'].cuda()
            word = batch_x['word'].cuda()
            pos = batch_x['pos'].cuda()
            spo = batch_x['spo'].cuda()
            seq_len = batch_x['seq_len'].cuda()

            tag = batch_y['tag'].cuda()
            
            #pred = model(char, word, pos, spo, seq_len, tag)
            
            pred = models[0].predict(char, word, pos, spo, seq_len)  # labels?
            pred['pred'] += models[1].predict(char, word, pos, spo, seq_len)['pred']
            pred['pred'] /= weight_sum
            #labels = idx2label(pred['pred'], tag_vocab.idx2word)
            #print(pred)
            #print(tag)
            #exit()
            metrics({'pred': pred['pred'].cuda(), 'seq_len':seq_len}, {'tag': batch_y['tag'].cuda()})
        eval_result = metrics.get_metric()
        metric_name = metrics.__class__.__name__
        eval_results[metric_name] = eval_result

    print("[tester] \n{}".format(_format_eval_results(eval_results)))


if __name__ == "__main__":
    config = Config()
    bilstm_crf, trans_crf = load_models(config)
    models = [bilstm_crf, trans_crf]

    # test_each(config, models)
    # for model_name, model in zip(config.ensemble_models, models):
    #     print(model_name)
    #     dump_model_result(config, model)
    
    ensemble(config, models, weight=[3,1])

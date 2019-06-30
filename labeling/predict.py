from config import Config
from model_trans import AdvSeqLabel, TransformerSeqLabel
from dataset import dump_data
from utils import TimingCallback, get_schemas
from ensemble import load_models

import os
import torch
import json
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


def char2text(batch, idx2word):
    texts = []
    for sentence in batch:
        text = ""
        for c in sentence:
            #print(c)
            text += idx2word[c.item()]
        texts.append(text)
    return texts


def idx2label(batch, idx2word):
    texts = []
    for sentence in batch:
        text = []
        for c in sentence:
            text.append(idx2word[c.item()])
        texts.append(text)
    return texts

def idx2spo(schema, batch):
    spo = []
    for s in batch:
        spo.append(np.argmax(s))
    return spo


def _label2spo(label, text):
    sub = []
    obj = []
    for i, l in enumerate(label):
        if l == 'O':
            continue
        if l[0] == 'S':
            if 'SUB' in l:
                sub.append(text[i])
            elif 'OBJ' in l:
                obj.append(text[i]) 
        if l[0] == 'B':
            start_idx = i
        if l[0] == 'I':
            continue
        if l[0] == 'E':
            end_idx = i
            if 'SUB' in l:
                sub.append(text[start_idx: end_idx+1])
            elif 'OBJ' in l:
                obj.append(text[start_idx: end_idx+1])
    return sub, obj


def label2spo(labels, texts, result, spos):
    for label, text, spo in zip(labels, texts, spos):
        sub, obj = _label2spo(label, text)
        if text not in result:
            result[text] = []
        for s in sub:
            for o in obj:
                result[text].append({"predicate": spo, "subject": s, "object": o})


def data2dic(config):
    dic = {}
    with open(os.path.join(config.source_path, config.dev_source), 'rb') as f:
        for line in f:
            d = json.loads(line)
            dic[d['text']] = []
            for spo in d['spo_list']:
                idx = schema[spo['subject_type'] + spo['predicate'] + spo['object_type']]
                dic[d['text']].append({"predicate": idx, "subject": spo['subject'], "object": spo['object']})
            
    return dic

def _f1(spo1, spo2):
    tp = 0  # 1, 1
    fp = 0  # 1, 0
    fn = 0  # 0, 1
    for s1 in spo1:
        if s1 in spo2:
            tp += 1
            spo2.remove(s1)
        else:
            fp += 1
    fn += len(spo2)
    return tp, fp, fn


def f1(result, target):
    tp = 0  # 1, 1
    fp = 0  # 1, 0
    fn = 0  # 0, 1
    for text in result:
        if text in target:
            tp_temp, fp_temp, fn_temp = _f1(result[text], target[text])
            tp += tp_temp
            fp += fp_temp
            fn += fn_temp
            del target[text]
        else:
            fp += len(result[text])
    for text in target:
        fn += len(target[text])
    precision = tp * 1.0 / (tp + fp)
    recall = tp * 1.0 / (tp + fn)
    f1 = 2.0 * precision * recall / (precision + recall)
    print('f1', f1, 'precision', precision, 'recall', recall)


def predict(config, model):
    tag_vocab = pickle.load(open(os.path.join(config.data_path, config.tag_vocab_name), 'rb'))
    metrics = SpanFPreRecMetric(tag_vocab, pred='pred', seq_len='seq_len', target='tag')
    dev_data = pickle.load(open(os.path.join(config.data_path, config.dev_name), "rb"))
    char_vocab = pickle.load(open(os.path.join(config.data_path, config.char_vocab_name), "rb"))

    data_iterator = Batch(dev_data, config.batch_size, sampler=SequentialSampler(), as_numpy=False)
    model.cuda()

    schema = get_schemas(config.source_path)

    eval_results = {}
    dev_data.set_input('tag')
    dev_data.set_target('seq_len')
    result = {}
    with torch.no_grad():
        for i, (batch_x, _) in enumerate(data_iterator):
            print('batch', i)
            #if i > 10:
            #    break
            char = batch_x['char'].cuda()
            word = batch_x['word'].cuda()
            pos = batch_x['pos'].cuda()
            spo = batch_x['spo'].cuda()
            seq_len = batch_x['seq_len'].cuda()
            
            #pred = model(char, word, pos, spo, seq_len, tag)
            pred = model.predict(char, word, pos, spo, seq_len)  # labels?

            texts = char2text(char.cpu().data, char_vocab.idx2word)
            labels = idx2label(pred['pred'].cpu().data, tag_vocab.idx2word)
            spos = idx2spo(schema, spo.cpu().data)
            result = label2spo(labels, texts, result, spos)
            #print(pred)
            #print(tag)
            #exit()
            # metrics({'pred': pred['pred'].cuda(), 'seq_len':seq_len}, {'tag': batch_y['tag'].cuda()})
        # eval_result = metrics.get_metric()
        # metric_name = metrics.__class__.__name__
        # eval_results[metric_name] = eval_result

    return result
    # print("[tester] \n{}".format(_format_eval_results(eval_results)))


if __name__ == "__main__":
    config = Config()
    bilstm_crf, trans_crf = load_models(config)
    print('result')
    result = predict(config, bilstm_crf)
    print('dev data')
    dev_data = data2dic(config)
    print('f1')
    f1(result, dev_data)

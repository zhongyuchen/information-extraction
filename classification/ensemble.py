from config import Config
from model import CNN, LSTM, LSTM_maxpool, RCNN
from model_bert import BertForMultiLabelSequenceClassification
from utils import F1_score
from dataset_bert import get_schemas
from dataset import dump_data

import os
import torch
import pickle
import numpy as np

from fastNLP import Tester, Batch, SequentialSampler
#from fastNLP.core.utils import _move_dict_value_to_device


def load_models(config):
    # train_data = pickle.load(open(os.path.join(config.data_path, config.train_name), "rb"))
    # debug
    # if config.debug:
    #     train_data = train_data[0:30]
    # dev_data = pickle.load(open(os.path.join(config.data_path, config.dev_name), "rb"))
    # test_data = pickle.load(open(os.path.join(config.data_path, config.test_name), "rb"))
    vocabulary = pickle.load(open(os.path.join(config.data_path, config.vocabulary_name), "rb"))

    # load w2v data
    # weight = pickle.load(open(os.path.join(config.data_path, config.weight_name), "rb"))

    cnn = CNN(vocab_size=len(vocabulary), embed_dim=config.embed_dim,
                     class_num=config.class_num, kernel_num=config.kernel_num,
                     kernel_sizes=config.kernel_sizes, dropout=config.dropout,
                     static=config.static, in_channels=config.in_channels)
    state_dict = torch.load(os.path.join(config.save_path, config.ensemble_models[0])).state_dict()
    cnn.load_state_dict(state_dict)

    lstm = LSTM(vocab_size=len(vocabulary), embed_dim=config.embed_dim,
                      output_dim=config.class_num, hidden_dim=config.hidden_dim,
                      num_layers=config.num_layers, dropout=config.dropout)
    state_dict = torch.load(os.path.join(config.save_path, config.ensemble_models[1])).state_dict()
    lstm.load_state_dict(state_dict)

    lstm_mxp = LSTM_maxpool(vocab_size=len(vocabulary), embed_dim=config.embed_dim,
                     		  output_dim=config.class_num, hidden_dim=config.hidden_dim,
                     		  num_layers=config.num_layers, dropout=config.dropout)
    state_dict = torch.load(os.path.join(config.save_path, config.ensemble_models[2])).state_dict()
    lstm_mxp.load_state_dict(state_dict)

    rcnn = RCNN(vocab_size=len(vocabulary), embed_dim=config.embed_dim,
                      output_dim=config.class_num, hidden_dim=config.hidden_dim,
                      num_layers=config.num_layers, dropout=config.dropout)
    state_dict = torch.load(os.path.join(config.save_path, config.ensemble_models[3])).state_dict()
    rcnn.load_state_dict(state_dict)

    schemas = get_schemas(config.source_path)
    state_dict = torch.load(os.path.join(config.save_path, config.ensemble_models[4])).state_dict()
    bert = BertForMultiLabelSequenceClassification.from_pretrained(config.bert_folder, state_dict=state_dict, num_labels=len(schemas))
    bert.load_state_dict(state_dict)

    return cnn, lstm, lstm_mxp, rcnn, bert


def test_each(config, models):
    dev_data = pickle.load(open(os.path.join(config.data_path, config.dev_name), "rb"))
    f1 = F1_score(pred='output', target='target')
    #for model_name, model in zip(config.ensemble_models[:-1], models[:-1]):
    #    print(model_name)
    #    tester = Tester(dev_data, model, metrics=f1, device=config.device, batch_size=config.batch_size)
    #    tester.test()

    dev_data = pickle.load(open(os.path.join(config.bert_data_path, config.dev_name), "rb"))
    print(config.ensemble_models[-1])
    tester = Tester(dev_data, models[-1], metrics=f1, device=config.device, batch_size=config.batch_size)
    tester.test()


def _format_eval_results(results):
    _str = ''
    for metric_name, metric_result in results.items():
        _str += metric_name + ': '
        _str += ", ".join([str(key) + "=" + str(value) for key, value in metric_result.items()])
        _str += '\n'
    return _str[:-1]


def ensemble(config, models, sum_prob=False, weight=[1,1,1,1,1]):
    f1 = F1_score(pred='output', target='target')
    f1.tp.cuda()
    f1.fp.cuda()
    f1.fn.cuda()

    dev_data = pickle.load(open(os.path.join(config.data_path, config.dev_name), "rb"))
    bert_dev_data = pickle.load(open(os.path.join(config.bert_data_path, config.dev_name), "rb"))

    data_iterator = Batch(dev_data, config.ensemble_batch, sampler=SequentialSampler(), as_numpy=False)
    bert_data_iterator = Batch(bert_dev_data, config.ensemble_batch, sampler=SequentialSampler(), as_numpy=False)

    for model in models:
        model.cuda()

    eval_results = {}
    weight = torch.tensor(weight)
    weight.cuda()
    weight_sum = torch.sum(weight).float()
    with torch.no_grad():
        for i, ((batch_x, batch_y), (bert_batch_x, bert_batch_y)) in enumerate(zip(data_iterator, bert_data_iterator)):
            print('batch', i)
            #if i > 10:
            #    break
            # batch
            text = batch_x['text'].cuda()
            target = batch_y['target'].cuda()
            # bert batch
            input_ids = bert_batch_x['input_ids'].cuda()
            token_type_ids = bert_batch_x['token_type_ids'].cuda()
            attention_mask  = bert_batch_x['attention_mask'].cuda()
            label_id = bert_batch_y['label_id'].cuda()

            #assert torch.equal(target, label_id)
            
            pred = models[-1](input_ids, token_type_ids, attention_mask)
            pred['output'] *= weight[-1]
            #if not sum_prob:
            #    pred['output'][pred['output'] >= 0.5] = 1.0 * weight[-1]
            #    pred['output'][pred['output'] < 0.5] = 0.0
            #    for i, model in enumerate(models[:-1]):
            #        temp = model(text)['output']
            #        temp[temp >= 0.5] = 1.0 * weight[i]
            #        temp[temp < 0.5] = 0.0
            #        pred['output'] += temp
            #else:
            for i, model in enumerate(models[:-1]):
                pred['output'] += model(text)['output'] * weight[i]
            pred['output'] /= weight_sum

            #bert_batch_y['label_id'].cuda()
            f1({'output': pred['output'].cuda()}, {'label_id': bert_batch_y['label_id'].cuda()})
        eval_result = f1.get_metric()
        metric_name = f1.__class__.__name__
        eval_results[metric_name] = eval_result

    print("[ensemble] \n{}".format(_format_eval_results(eval_results)))


def dump_one_model_prob(path, name, dev_data, model, data_iterator):
    name += '.pkl'
    model.cuda()

    prob = []
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(data_iterator):
            print('batch', i)
            # batch
            text = batch_x['text'].cuda()
            # target = batch_y['target'].cuda()

            #assert torch.equal(target, label_id)

            pred = model(text)['output']
            prob.append(pred)
    #prob = torch.tensor(prob)

    dump_data(path, name, prob)


def dump_bert_model_prob(path, name, dev_data, model, data_iterator):
    name += '.pkl'
    model.cuda()

    prob = []
    with torch.no_grad():
        for i, (bert_batch_x, bert_batch_y) in enumerate(data_iterator):
            print('batch', i)
            # bert batch
            input_ids = bert_batch_x['input_ids'].cuda()
            token_type_ids = bert_batch_x['token_type_ids'].cuda()
            attention_mask  = bert_batch_x['attention_mask'].cuda()
            # label_id = bert_batch_y['label_id'].cuda()

            #assert torch.equal(target, label_id)

            pred = model(input_ids, token_type_ids, attention_mask)['output']
            prob.append(pred)
    #prob = torch.tensor(prob)
    dump_data(path, name, prob)


def dump_all_models_prob(config, models):
    dev_data = pickle.load(open(os.path.join(config.data_path, config.dev_name), "rb"))
    bert_dev_data = pickle.load(open(os.path.join(config.bert_data_path, config.dev_name), "rb"))

    data_iterator = Batch(dev_data, config.ensemble_batch, sampler=SequentialSampler(), as_numpy=False)
    bert_data_iterator = Batch(bert_dev_data, config.ensemble_batch, sampler=SequentialSampler(), as_numpy=False)

    for i, model in enumerate(models[:-1]):
        dump_one_model_prob(config.prob_path, config.ensemble_models[i], dev_data, model, data_iterator)
    dump_bert_model_prob(config.prob_path, config.ensemble_models[-1], bert_dev_data, models[-1], bert_data_iterator)



if __name__ == "__main__":
    config = Config()
    cnn, lstm, lstm_mxp, rcnn, bert = load_models(config)
    models = [cnn, lstm, lstm_mxp, rcnn, bert]
    # test_each(config, models)
    ensemble(config, models, weight=[1,10,30,9,100])

    # dump_all_models_prob(config, models)

#[ensemble] sum prob, then take those with avg >= 0.5 
#F1_score: f1=tensor(0.8864), recall=tensor(0.8583), precision=tensor(0.9163)
#[ensemble] prob >= 0.5, take those with >= 3
#F1_score: f1=tensor(0.8873), recall=tensor(0.8625), precision=tensor(0.9135)

#[ensemble] [1,3,4,2,5]
#F1_score: f1=tensor(0.8908), recall=tensor(0.8657), precision=tensor(0.9175)

#[ensemble] [1,4,8,3,10]
#F1_score: f1=tensor(0.8927), recall=tensor(0.8681), precision=tensor(0.9188)

#[ensemble] [1,5,10,4,20]
#F1_score: f1=tensor(0.8947), recall=tensor(0.8728), precision=tensor(0.9178)

#[ensemble] [1,10,30,9,50]
#F1_score: f1=tensor(0.8940), recall=tensor(0.8716), precision=tensor(0.9175)

#[ensemble] [1,10,30,9,100]
#F1_score: f1=tensor(0.8970), recall=tensor(0.8778), precision=tensor(0.9172)

#in the end take [0,0,0,0,1]
#F1_score: f1=0.90 (bert)

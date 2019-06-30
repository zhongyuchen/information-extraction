from config import Config
from model import CNN, LSTM, LSTM_maxpool, RCNN
from model_bert import BertForMultiLabelSequenceClassification
from utils import F1_score, get_schemas_list
from ensemble import load_models, test_each

import os
import torch
import pickle
import json

from fastNLP import Tester, Batch, SequentialSampler


def prob2spo(prob, schema):
    spo_list = []
    for i in range(len(prob)):
        if prob[i] >= 0.5:
            spo_list.append(schema[i])
    return spo_list


def predict(config, models, weight):
    test_data = pickle.load(open(os.path.join(config.data_path, config.test_name), "rb"))
    bert_test_data = pickle.load(open(os.path.join(config.bert_data_path, config.test_name), "rb"))

    data_iterator = Batch(test_data, config.predict_batch, sampler=SequentialSampler(), as_numpy=False)
    bert_data_iterator = Batch(bert_test_data, config.predict_batch, sampler=SequentialSampler(), as_numpy=False)

    for model in models:
        model.cuda()

    schema = get_schemas_list(config.source_path)
    weight = torch.tensor(weight).float()
    weight.cuda()
    weight_sum = torch.sum(weight)

    read_data = []
    with open(os.path.join(config.source_path, config.test_source), 'rb') as f:
    	for line in f:
    	    read_data.append(json.loads(line))

    spo_list = []
    with torch.no_grad():
        for i, ((batch_x, _), (bert_batch_x, _)) in enumerate(zip(data_iterator, bert_data_iterator)):
            print('batch', i)
            #if i >= 5:
            #    break
            # batch
            text = batch_x['text'].cuda()
            # target = batch_y['target'].cuda()
            # bert batch
            input_ids = bert_batch_x['input_ids'].cuda()
            token_type_ids = bert_batch_x['token_type_ids'].cuda()
            attention_mask  = bert_batch_x['attention_mask'].cuda()
            # label_id = bert_batch_y['label_id'].cuda()

            # assert torch.equal(target, label_id)

            pred = models[-1](input_ids, token_type_ids, attention_mask)
            pred['output'] *= weight[-1]
            for i, model in enumerate(models[:-1]):
                pred['output'] += model(text)['output'] * weight[i]
            pred['output'] /= weight_sum

            for prob in pred['output']:
            	spo_list.append(prob2spo(prob, schema))
                
    with open(os.path.join(config.predict_path, config.predict_name), 'w') as f:
        for data, spo in zip(read_data, spo_list):
            data["spo_list"] = spo
            f.write(json.dumps(data, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    config = Config()
    cnn, lstm, lstm_mxp, rcnn, bert = load_models(config)
    models = [cnn, lstm, lstm_mxp, rcnn, bert]
    # test_each(config, models)
    predict(config, models, [0,0,0,0,1])


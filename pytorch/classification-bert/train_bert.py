from fastNLP import Trainer
from fastNLP import EarlyStopCallback
from fastNLP import Tester

import os
import pickle
import sys
import random
import numpy as np
import torch

from config import Config
from model_bert import BertForMultiLabelSequenceClassification
from utils import TimingCallback, BCEWithLogitsLoss, F1_score
from dataset_bert import get_schemas
from utils import BertAdam

import fitlog
from fastNLP.core.callback import FitlogCallback
fitlog.commit(__file__)             # auto commit your codes
fitlog.add_hyper_in_file(__file__)  # record your hyperparameters

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def train(config, task_name):
    train_data = pickle.load(open(os.path.join(config.bert_data_path, config.train_name), "rb"))
    print(train_data[0])
    # debug
    if config.debug:
        train_data = train_data[0:30]
    dev_data = pickle.load(open(os.path.join(config.bert_data_path, config.dev_name), "rb"))
    print(dev_data[0])
    # test_data = pickle.load(open(os.path.join(config.bert_data_path, config.test_name), "rb"))

    schemas = get_schemas(config.source_path)
    state_dict = torch.load(config.bert_path)
    # print(state_dict)
    text_model = BertForMultiLabelSequenceClassification.from_pretrained(config.bert_folder, state_dict=state_dict, num_labels=len(schemas))

    # optimizer
    param_optimizer = list(text_model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    num_train_optimization_steps = int(len(train_data) / config.batch_size / config.update_every) * config.epoch
    if config.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    optimizer = BertAdam(lr=config.lr,
                         warmup=config.warmup_proportion,
                         t_total=num_train_optimization_steps).construct_from_pytorch(optimizer_grouped_parameters)

    timing = TimingCallback()
    early_stop = EarlyStopCallback(config.patience)
    logs = FitlogCallback(dev_data)
    f1 = F1_score(pred='output', target='label_id')

    trainer = Trainer(train_data=train_data, model=text_model, loss=BCEWithLogitsLoss(),
                      batch_size=config.batch_size, check_code_level=-1,
                      metrics=f1, metric_key='f1', n_epochs=int(config.epoch),
                      dev_data=dev_data, save_path=config.save_path,
                      print_every=config.print_every, validate_every=config.validate_every,
                      update_every=config.update_every,
                      optimizer=optimizer, use_tqdm=False,
                      device=config.device, callbacks=[timing, early_stop, logs])
    trainer.train()

    # test result
    tester = Tester(dev_data, text_model, metrics=f1,
                    device=config.device, batch_size=config.batch_size,)
    tester.test()


def log_config(config, task_name):
    fitlog.add_other(task_name, name="model_name")
    # irrelavant
    fitlog.add_other(config.class_num, name="class_num")
    fitlog.add_other(str(config.kernel_sizes), name="kernel_sizes")
    fitlog.add_other(config.kernel_num, name="kernel_num")
    fitlog.add_other(config.in_channels, name="in_channels")
    fitlog.add_other(config.dropout, name="dropout")
    fitlog.add_other(config.static, name="static")
    # irrelavant above
    fitlog.add_other(config.sentence_length, name="sentence_length")
    fitlog.add_other(config.num_layers, name="num_layers")  # irrelavant
    fitlog.add_other(config.hidden_dim, name="hidden_dim")  # irrelavant
    fitlog.add_other(config.lr, name="lr")
    fitlog.add_other(config.weight_decay, name="weight_decay")  # irrelavant
    fitlog.add_other(config.patience, name="patience")
    fitlog.add_other(config.epoch, name="epoch")
    fitlog.add_other(config.batch_size, name="batch_size")
    fitlog.add_other(str(config.device), name="device")
    fitlog.add_other(config.warmup_proportion, name="warmup_proportion")
    fitlog.add_other(config.local_rank, name="local_rank")
    fitlog.add_other(config.update_every, name="update_every")


if __name__ == "__main__":
    config = Config()

    task_name = "bert"
    assert task_name in config.task_names

    log_config(config, task_name)

    train(config, task_name)
    fitlog.finish()

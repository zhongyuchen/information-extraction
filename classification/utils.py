from fastNLP import Callback
from fastNLP.core.losses import LossBase
from fastNLP.core.metrics import MetricBase
from fastNLP.core.optimizer import Optimizer
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.optimizer import required

from pytorch_pretrained_bert.optimization import BertAdam as BertAdam_pytorch

import time
import os
import json

start_time = time.time()


def get_schemas(data_path):
    schemas = {}
    with open(os.path.join(data_path, "all_50_schemas"), 'rb') as f:
        for i, line in enumerate(f):
            spo = json.loads(line)
            schemas[spo['subject_type'] + spo['predicate'] + spo['object_type']] = i
    return schemas


def get_schemas_list(data_path):
    schemas = []
    with open(os.path.join(data_path, "all_50_schemas"), 'rb') as f:
        for line in f:
            #print(line)
            #print(type(line))
            spo = json.loads(line)
            schemas.append(spo)
    return schemas


class TimingCallback(Callback):
    def on_epoch_end(self):
        print('Sum Time: {:d}s\n\n'.format(round(time.time() - start_time)))


# Loss
class BCEWithLogitsLoss(LossBase):
    def __init__(self, pred=None, target=None):
        super(BCEWithLogitsLoss, self).__init__()
        self._init_param_map(pred=pred, target=target)

    def get_loss(self, pred, target):
        #print(pred)
        #print(target)
        loss = F.binary_cross_entropy_with_logits(pred, target.float())
        return loss

# Metric
class F1_score(MetricBase):
    def __init__(self, pred=None, target=None):
        super(F1_score, self).__init__()
        self._init_param_map(pred=pred, target=target)
        self.tp = torch.tensor(0).float()
        self.fp = torch.tensor(0).float()
        self.fn = torch.tensor(0).float()

    def evaluate(self, pred, target):
        pred[pred < 0.5] = 0
        pred[pred >= 0.5] = 1

        self.tp += torch.sum((pred == 1) * (target == 1), (1, 0))
        self.fp += torch.sum((pred == 1) * (target == 0), (1, 0))
        self.fn += torch.sum((pred == 0) * (target == 1), (1, 0))

    def get_metric(self, reset=True):
        precision = self.tp * 1.0 / (self.tp + self.fp) if self.tp or self.fp else 0
        recall = self.tp * 1.0 / (self.tp + self.fn) if self.tp or self.fn else 0
        f1 = 2.0 * precision * recall / (precision + recall) if precision or recall else 0
        if reset:
            self.tp = torch.tensor(0).float()
            self.fp = torch.tensor(0).float()
            self.fn = torch.tensor(0).float()
        return {'f1': f1, 'recall': recall, 'precision': precision}


# Optimizer
class BertAdam(Optimizer):
    def __init__(self, lr=required, warmup=-1, t_total=-1, schedule='warmup_linear', b1=0.9, b2=0.999, e=1e-6, weight_decay=0.01, max_grad_norm=1.0, model_params=None):
        # model_params has to be None
        super(BertAdam, self).__init__(model_params, lr=lr, warmup=warmup, t_total=t_total, schedule=schedule, b1=b1, b2=b2, e=e, weight_decay=weight_decay, max_grad_norm=max_grad_norm)

    def construct_from_pytorch(self, model_params):
        if self.model_params is None:
            return BertAdam_pytorch(model_params, **self.settings)
        else:
            return BertAdam_pytorch(self._get_require_grads_param(self.model_params), **self.settings)


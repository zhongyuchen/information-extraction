from fastNLP import Callback
from fastNLP.core.losses import LossBase
from fastNLP.core.metrics import MetricBase
import numpy as np
import torch
import torch.nn.functional as F

import time

start_time = time.time()


class TimingCallback(Callback):
    def on_epoch_end(self):
        print('Sum Time: {:d}s\n\n'.format(round(time.time() - start_time)))


class BCEWithLogitsLoss(LossBase):
    def __init__(self, pred=None, target=None):
        super(BCEWithLogitsLoss, self).__init__()
        self._init_param_map(pred=pred, target=target)

    def get_loss(self, pred, target):
        # print(pred.shape)
        # print(target.shape)
        loss = F.binary_cross_entropy_with_logits(pred, target)
        return loss


class F1_score(MetricBase):
    def __init__(self, pred=None, target=None):
        super(F1_score, self).__init__()
        self._init_param_map(pred=pred, target=target)
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def evaluate(self, pred, target):
        pred[pred < 0.5] = 0
        pred[pred >= 0.5] = 1

        self.tp += torch.sum((pred == 1) * (target == 1), (1, 0))
        self.fp += torch.sum((pred == 1) * (target == 0), (1, 0))
        self.fn += torch.sum((pred == 0) * (target == 1), (1, 0))

    def get_metric(self, reset=True):
        precision = self.tp * 1.0 / (self.tp + self.fp)
        recall = self.tp * 1.0 / (self.tp + self.fn)
        f1 = 2.0 * precision * recall / (precision + recall)
        if reset:
            self.tp = 0
            self.fp = 0
            self.fn = 0
        return {'f1': f1, 'recall': recall, 'precision': precision}

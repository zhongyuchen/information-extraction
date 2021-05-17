import os
import pickle
import sys
import torch
from config import *
from model import CNN
import time
import random
import numpy as np
from tensorboardX import SummaryWriter
import torch.nn.functional as F

sys.path.append('../')
from dataset import Dataset


def train():
    model.train()
    best_score, best_epoch = {'f1': 0., 'recall': 0., 'precision': 0.}, 0
    batch_cnt = 0
    start_time = time.time()
    for e in range(1, epochs + 1):
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_data):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.binary_cross_entropy_with_logits(output, target)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            batch_cnt += 1
            writer.add_scalar('train/batch_loss', loss.item() / len(data), batch_cnt)
            
        train_loss /= len(train_data.dataset)
        writer.add_scalar('train/epoch_loss', train_loss, e)
        print('Train Epoch: {}\tLoss: {:.6f}'.format(e, train_loss))
        score = dev(e)

        if score['f1'] > best_score['f1']:
            torch.save(model.state_dict(), save_path)
            best_score, best_epoch = score, e
        print('best f1', best_score['f1'], 'in epoch', best_epoch)
        # early stop
        if e - best_epoch >= patience:
            print('Early stop at epoch', e)
            writer.add_scalar('train/early_stop', e, 0)
            break
    # duration
    writer.add_scalar('train/duration', (time.time() - start_time) / 60, 0)

    
def dev(e):
    model.eval()
    tp = torch.tensor(0).float()
    fp = torch.tensor(0).float()
    fn = torch.tensor(0).float()
    dev_loss = 0
    with torch.no_grad():
        for data, target in dev_data:
            data, target = data.to(device), target.to(device)
            pred = model(data)
            dev_loss += F.binary_cross_entropy_with_logits(pred, target)
            pred[pred < 0.5] = 0
            pred[pred >= 0.5] = 1
            tp += torch.sum((pred == 1) * (target == 1), (1, 0))
            fp += torch.sum((pred == 1) * (target == 0), (1, 0))
            fn += torch.sum((pred == 0) * (target == 1), (1, 0))
            
    # loss and score
    dev_loss /= len(dev_data.dataset)
    precision = tp * 1.0 / (tp + fp) if tp or fp else 0
    recall = tp * 1.0 / (tp + fn) if tp or fn else 0
    f1 = 2.0 * precision * recall / (precision + recall) if precision or recall else 0
    score = {'f1': f1, 'recall': recall, 'precision': precision}
    # writer
    writer.add_scalar('dev/loss', dev_loss, e)
    writer.add_scalar('dev/f1', score['f1'], e)
    writer.add_scalar('dev/recall', score['recall'], e)
    writer.add_scalar('dev/precision', score['precision'], e)
    
    print('Dev set: Average loss: {:.4f}, f1: {:.4f}'.format(dev_loss, score['f1']))
    model.train()
    return score


if __name__ == "__main__":
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    train_data = pickle.load(open(os.path.join(data_path, train_name), "rb"))
    dev_data = pickle.load(open(os.path.join(data_path, dev_name), "rb"))
    vocabulary = pickle.load(open(os.path.join(data_path, vocabulary_name), "rb"))
    print('dataset', len(train_data), len(dev_data))

    # load w2v data
    weight = pickle.load(open(os.path.join(data_path, weight_name), "rb"))
    
    # model
    train_device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = CNN(vocab_size=len(vocabulary), embed_dim=embed_dim,
                 class_num=class_num, kernel_num=kernel_num,
                 kernel_sizes=kernel_sizes, dropout=dropout,
                 static=static, in_channels=in_channels,
                 weight=weight)
    model.to(train_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # train
    writer = SummaryWriter(log_dir=log_path)
    train()
    writer.close()

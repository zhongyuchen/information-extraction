import tensorflow as tf
import csv
import json
import numpy as np
import os
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default=None, help="The input data dir.")
parser.add_argument("--test_file", default=None, help="Test data file")
parser.add_argument("--evaluate_dir", default=None, help="The output directory where the model checkpoints will be written.")
parser.add_argument("--evaluate_file", default=None, help="Output file name")
args = parser.parse_args()


def flag_count(a, flag_a, b, flag_b):
    count = 0
    for i in range(len(a)):
        if a[i] == flag_a and b[i] == flag_b:
            count += 1
    return count


def predict_result(divide):
    schemas = {}
    with open(os.path.join(args.data_dir, "all_50_schemas"), 'r') as f:
        for i, line in enumerate(f):
            spo = json.loads(line)
            schemas[spo['predicate']+ spo['subject_type']] = i

    data = []
    with open(os.path.join(args.evaluate_dir, args.evaluate_file), 'r') as f:
        for i, line in enumerate(f):
            d = []
            l = line.split('\t')
            for j in l:
                d.append(float(j))
            if len(d) != len(schemas):
                print('WRONG')
            data.append(np.array(d))
    data = np.array(data)

    data[data>=divide] = 1
    data[data<divide] = 0

    test = []
    with open(os.path.join(args.data_dir, args.test_file), 'r') as f:
        for i, line in enumerate(f):
            t = np.zeros(len(schemas))
            d = json.loads(line)
            for spo in d['spo_list']:
                t[schemas[spo['predicate'] + spo['subject_type']]] = 1
            test.append(t)
    test = np.array(test)
    
    if len(data) != len(test):
        print('WRONG lenght!')
    
    tp = 0.
    fp = 0.
    fn = 0.
    data = torch.tensor(data)
    test = torch.tensor(test)
    print(data.shape, type(data))
    print(test.shape, type(test))
    tp += torch.sum((data == 1) * (test == 1), (1, 0))
    fp += torch.sum((data == 1) * (test == 0), (1, 0))
    fn += torch.sum((data == 0) * (test == 1), (1, 0))
    print(tp, fp, fn)
    tp = tp.double()
    fp = fp.double()
    fn = fn.double()
    #for i in range(len(data)):
    #    tp += flag_count(data[i], 1, test[i], 1)
    #    fp += flag_count(data[i], 1, test[i], 0)
    #    fn += flag_count(data[i], 0, test[i], 1)
    # print(tp + fp)
    precision = tp * 1.0 / (tp + fp)
    recall = tp * 1.0 / (tp + fn)
    print(precision, recall)
    f1 = 2.0 * precision * recall / (precision + recall)
    print(f1)
    # print('---Evaluate---')
    # print('divided by', divide)
    # print('tp:', tp, 'fp:', fp, 'fn:', fn)
    # print('precision:', precision)
    # print('recall:', recall)
    # print('f1:', f1)
    return f1, recall, precision


if __name__ == "__main__":
    max_i = 0.
    max_f1, recall, precision = 0., 0., 0.
    for i in np.arange(0.46, 0.56, 0.01):
        f1, r, p = predict_result(i)
        if f1 > max_f1:
            max_i = i
            max_f1, recall, precision = f1, r, p
    print('Max f1 score:', max_f1, 'recall:', recall, 'precision:', precision, 'divided by:', max_i)
    # Max f1 score: 0.8994610096147692 recall: 0.9245913720948595 precision: 0.875660584305514 divided by: 0.47000000000000003

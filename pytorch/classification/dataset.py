import pickle
import os
import json
from fastNLP import Vocabulary

from config import *

import torch
import numpy as np
import random
import jieba
    
    
class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        data, target = [], []
        for (d, t) in dataset:
            data.append(d)
            target.append(t)
        self.data = torch.tensor(data).long()
        self.target = torch.tensor(target).float()
        self.len = len(self.data)
        
    def __getitem__(self, index):
        return self.data[index], self.target[index]

    def __len__(self):
        return self.len
    
    
def process_class(schemas, spo_list):
    # label -> multi-hot
    target = [0] * len(schemas)
    for spo in spo_list:
        spo_concat = spo['subject_type'] + spo['predicate'] + spo['object_type']
        target[schemas[spo_concat]] = 1
    return target


def get_dataset(path, name):
    dataset = []
    with open(os.path.join(path, name), 'rb') as f:
        for _, line in enumerate(f):
            dic = json.loads(line)
#             data = list(dic['text'])
            data = jieba.lcut(dic['text'])
            if 'spo_list' in dic:
                target = process_class(schemas, dic['spo_list'])
            else:
                target = [0] * len(schemas)
            dataset.append((data, target))

    print(len(dataset))
    print(dataset[0])
    return dataset


def get_vocab(dataset):
    vocabulary = Vocabulary(unknown=unk_str, padding=pad_str)
    for data, _ in dataset:
        vocabulary.add_word_lst(data)
    print('vocab', len(vocabulary))
    print('pad', vocabulary.to_index(pad_str))

    return vocabulary


def get_dataloader(vocabulary, dataset, shuffle):
    pad = vocabulary.to_index(pad_str)
    dataset_index = []
    for data, target in dataset:
        data = data[:seq_len]
        data = [pad_str] * (seq_len - len(data)) + data
        data_index = [vocabulary.to_index(word) for word in data]
        dataset_index.append((data_index, target))
    dataset_torch = Dataset(dataset_index)
    print(dataset_torch[0])
    dataloader = torch.utils.data.DataLoader(dataset_torch, batch_size=batch_size, shuffle=shuffle, drop_last=False, **kwargs)
    
    return dataloader


def dump_data(path, name, data):
    pickle.dump(data, open(os.path.join(path, name), "wb"))


def main():
    # preprocess
    train_data = get_dataset(source_path, train_source)
    dev_data = get_dataset(source_path, dev_source)
    test_data = get_dataset(source_path, test_source)

    # vocabs
    vocabulary = get_vocab(train_data)

    # to index
    train_dataloader = get_dataloader(vocabulary, train_data, shuffle=True)
    dev_dataloader = get_dataloader(vocabulary, dev_data, shuffle=False)
    test_dataloader = get_dataloader(vocabulary, test_data, shuffle=False)

    # dump
    dump_data(data_path, train_name, train_dataloader)
    dump_data(data_path, dev_name, dev_dataloader)
    dump_data(data_path, test_name, test_dataloader)

    dump_data(data_path, vocabulary_name, vocabulary)


if __name__ == "__main__":
    os.makedirs(data_path, exist_ok=True)
    
    # seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # dataloader args
    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    
    # schemas
    schemas = {}
    with open(os.path.join(source_path, "all_50_schemas"), 'rb') as f:
        for i, line in enumerate(f):
            spo = json.loads(line)
            schemas[spo['subject_type'] + spo['predicate'] + spo['object_type']] = i
            
    main()

import pickle
import os
import json
from fastNLP import DataSet
from fastNLP import Vocabulary
from config import *
from tagging import tagging
import torch
import numpy as np
import random
import jieba.posseg as posseg


class Dataset(torch.utils.data.Dataset):
    def __init__(self, char, word, pos, spo, sentence_length, tag):
        self.char = torch.tensor(char).long()
        self.word = torch.tensor(word).long()
        self.pos = torch.tensor(pos).long()
        self.spo = torch.tensor(spo).long()
        self.seq_len = torch.tensor(sentence_length).long()
        
        self.tag = torch.tensor(tag).long()

        self.len = len(self.char)
        
    def __getitem__(self, index):
        return self.char[index], self.word[index], self.pos[index], self.spo[index], self.seq_len[index], self.tag[index]

    def __len__(self):
        return self.len


def get_schemas(path):
    schemas = {}
    with open(os.path.join(path, "all_50_schemas"), 'rb') as f:
        for i, line in enumerate(f):
            spo = json.loads(line)
            schemas[spo['subject_type'] + spo['predicate'] + spo['object_type']] = i
    return schemas


# def process_postag(postag, text):
#     word, pos = [], []
#     for tag in postag:
#         word += [tag['word']] * len(tag['word'])
#         pos += [tag['pos']] * len(tag['word'])
#     return word, pos

def process_postag(text):
    word, pos = [], []
    for w, p in posseg.lcut(text):
        word += [w] * len(w)
        pos += [p] * len(w)

    return word, pos


def one_hot(idx, length):
    l = [0] * length
    l[idx] = 1
    return l


def process_data(path, name):
    print('Processing', name)

    schemas = get_schemas(path)

    # input
    word_data, pos_data, char_data, spo_data = [], [], [], []
    # target
    tag_data = []
    with open(os.path.join(path, name), 'rb') as f:
        for i, line in enumerate(f):
#             if i > 10:
#                 break
            dic = json.loads(line)
            spo_set = set()
            for spo in dic['spo_list']:
                spo_set.add(spo['subject_type'] + spo['predicate'] + spo['object_type'])
            word, pos = process_postag(dic['text'])
            if len(word) == 0 or len(pos) == 0 or len(spo_set) == 0:
                continue  # no postag or no spo
            char = list(dic['text'])
#             print(len(word) ,len(pos) ,len(char))
            assert len(word) == len(pos) == len(char)
            for spo in spo_set:
                word_data.append(word)  # word: 影业
                pos_data.append(pos)  # pos: n
                char_data.append(char)  # characters
                spo_data.append(one_hot(schemas[spo], len(schemas)))  # spo concat: subject_type + predicate + object_type
                tag_data.append(tagging(spo, dic['text'], dic['spo_list']))  # for test, return all 'O'

    dataset = {
        "word": word_data,
        "pos": pos_data,
        "char": char_data,
        "spo": spo_data,  # each one is a spo concat one hot
        "tag": tag_data
    }
    print('Len', len(dataset['word']))
    print(dataset['word'][0], dataset['pos'][0], dataset['char'][0], dataset['spo'][0], dataset['tag'][0])

    return dataset


def _get_vocab(data_list):
    vocab = Vocabulary(unknown=unk_str, padding=pad_str)
    for l in data_list:
        vocab.add_word_lst(l)
    vocab.build_vocab()
    print('vocab', len(vocab))
    return vocab

    
def get_vocab(dataset):
    word_vocab = _get_vocab(dataset['word'])
    char_vocab = _get_vocab(dataset['char'])
    pos_vocab = _get_vocab(dataset['pos'])
    tag_vocab = _get_vocab(dataset['tag'])

    return word_vocab, char_vocab, pos_vocab, tag_vocab


def _to_index(vocab, data_list):
    new_data_list = []
    pad = vocab.to_index(pad_str)
    for data in data_list:
        l = [vocab.to_index(w) for w in data]
        l = l[:seq_len]
        l += [pad] * (seq_len - len(l))
        new_data_list.append(l)
    print(new_data_list[0])
    return new_data_list


def to_index(word_vocab, char_vocab, pos_vocab, tag_vocab, dataset, shuffle):
    sentence_length = [len(l) for l in dataset['char']]
    word_index = _to_index(word_vocab, dataset['word'])
    char_index = _to_index(char_vocab, dataset['char'])
    pos_index = _to_index(pos_vocab, dataset['pos'])
    
    spo_index = dataset['spo']

    tag_index = _to_index(tag_vocab, dataset['tag'])
    
    dataset_torch = Dataset(char_index, word_index, pos_index, spo_index, sentence_length, tag_index)
    print(dataset_torch[0])
    for d in dataset_torch[0]:
        print(d.size())
    dataloader = torch.utils.data.DataLoader(dataset_torch, batch_size=batch_size, shuffle=shuffle, drop_last=False, **kwargs)
    
    return dataloader


def dump_data(path, name, data):
    pickle.dump(data, open(os.path.join(path, name), "wb"))


def main():
    # preprocess
    train_data = process_data(source_path, train_source)
    dev_data = process_data(source_path, dev_source)
#     test_data = process_data(source_path, test_source)

    # vocabs
    word_vocab, char_vocab, pos_vocab, tag_vocab = get_vocab(train_data)

    # to index
    train_data = to_index(word_vocab, char_vocab, pos_vocab, tag_vocab, train_data, shuffle=True)
    dev_data = to_index(word_vocab, char_vocab, pos_vocab, tag_vocab, dev_data, shuffle=False)
#     test_data = to_index(word_vocab, char_vocab, pos_vocab, tag_vocab, test_data, shuffle=False)

    # dump
    dump_data(data_path, train_name, train_data)
    dump_data(data_path, dev_name, dev_data)
#     dump_data(data_path, test_name, test_data)

    dump_data(data_path, word_vocab_name, word_vocab)
    dump_data(data_path, char_vocab_name, char_vocab)
    dump_data(data_path, pos_vocab_name, pos_vocab)
    dump_data(data_path, tag_vocab_name, tag_vocab)


if __name__ == "__main__":
    os.makedirs(data_path, exist_ok=True)
    
    # seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # dataloader args
    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    main()

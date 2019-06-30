import pickle
import os
import json
from fastNLP import DataSet
from fastNLP import Vocabulary

from config import Config
from tagging import tagging


def process_postag(postag, text):
    word, pos, flag = [], [], False
    for tag in postag:
        word += [tag['word']] * len(tag['word'])
        pos += [tag['pos']] * len(tag['word'])
    if len(postag) == 0:
        word = list(text)
        pos = ['n'] * len(text)
        flag = True
    # if type(word[0]) == type(list):
    #     print(postag)
    #     print(text)
    #     exit()
    return word, pos, flag


def process_data(data_path, data_name, train=False, test=False):
    print('Processing', data_name)

    schemas = {}
    with open(os.path.join(data_path, "all_50_schemas"), 'rb') as f:
        for i, line in enumerate(f):
            spo = json.loads(line)
            schemas[spo['subject_type'] + spo['predicate'] + spo['object_type']] = i

    # input
    word_data, pos_data, char_data, spo_data = [], [], [], []
    # target
    tag_data = []
    with open(os.path.join(data_path, data_name), 'rb') as f:
        for i, line in enumerate(f):
            # if i > 10:
            #     break
            dic = json.loads(line)
            spo_set = set()
            for spo in dic['spo_list']:
                spo_set.add(spo['subject_type'] + spo['predicate'] + spo['object_type'])
            word, pos, flag = process_postag(dic['postag'], dic['text'])
            if train and flag:
                continue  # ignore samples with empty postag
            if flag:
                spo_set = set()
            # at least 1 class for a text
            if len(spo_set) == 0:
                spo_set.add('无')
            char = list(dic['text'])
            # print(len(word) ,len(pos) ,len(char))
            assert len(word) == len(pos) == len(char)
            for spo in spo_set:
                word_data.append(word)  # word: 影业
                pos_data.append(pos)  # pos: n
                char_data.append(char)  # characters
                spo_data.append(spo)  # spo concat: subject_type + predicate + object_type
                if not test:
                    tag_data.append(tagging(spo, dic['text'], dic['spo_list']))

    data_dict = {
        "word": word_data,
        "pos": pos_data,
        "char": char_data,
        "spo": spo_data,  # each one is not a list!
        "tag": tag_data
    }
    dataset = DataSet(data=data_dict)
    print('Len', len(dataset))
    print('Sample', dataset[0])

    return dataset


def get_vocab(dataset):
    word_vocab = Vocabulary(unknown='<unk>', padding='<pad>')
    dataset.apply(lambda x: [word_vocab.add(word) for word in x['word']])
    word_vocab.build_vocab()
    print('word vocab', len(word_vocab))

    char_vocab = Vocabulary(unknown='<unk>', padding='<pad>')
    dataset.apply(lambda x: [char_vocab.add(char) for char in x['char']])
    char_vocab.build_vocab()
    print('char vocab', len(char_vocab))

    pos_vocab = Vocabulary(unknown='<unk>', padding='<pad>')
    dataset.apply(lambda x: [pos_vocab.add(pos) for pos in x['pos']])
    pos_vocab.build_vocab()
    print('pos vocab', len(pos_vocab))

    spo_vocab = Vocabulary(unknown='<unk>', padding='<pad>')
    dataset.apply(lambda x: spo_vocab.add(x['spo']))
    spo_vocab.build_vocab()
    print('spo vocab', len(spo_vocab))

    tag_vocab = Vocabulary(unknown='<unk>', padding='<pad>')
    dataset.apply(lambda x: [tag_vocab.add(tag) for tag in x['tag']])
    # tag_vocab.add_word('start')
    # tag_vocab.add_word('end')
    tag_vocab.build_vocab()
    print('tag_vocab', len(tag_vocab))

    return word_vocab, char_vocab, pos_vocab, spo_vocab, tag_vocab


def to_index(word_vocab, char_vocab, pos_vocab, spo_vocab, tag_vocab, dataset):
    dataset.apply(lambda x: [word_vocab.to_index(word) for word in x['word']], new_field_name='word')
    dataset.apply(lambda x: [char_vocab.to_index(char) for char in x['char']], new_field_name='char')
    dataset.apply(lambda x: [pos_vocab.to_index(pos) for pos in x['pos']], new_field_name='pos')
    dataset.apply(lambda x: spo_vocab.to_index(x['spo']), new_field_name='spo')

    dataset.apply(lambda x: [tag_vocab.to_index(tag) for tag in x['tag']], new_field_name='tag')

    dataset.set_input('word', 'char', 'pos', 'spo')
    dataset.set_target('tag')

    return dataset


def dump_data(path, name, data):
    pickle.dump(data, open(os.path.join(path, name), "wb"))


def dump(config):
    # preprocess
    train_data = process_data(config.data_path, config.train_source, train=True)
    dev_data = process_data(config.data_path, config.dev_source)
    # test_data = process_data(config.data_path, config.test_source, test=True)

    # vocabs
    word_vocab, char_vocab, pos_vocab, spo_vocab, tag_vocab = get_vocab(train_data)

    # to index
    train_data = to_index(word_vocab, char_vocab, pos_vocab, spo_vocab, tag_vocab, train_data)
    dev_data = to_index(word_vocab, char_vocab, pos_vocab, spo_vocab, tag_vocab, dev_data)
    # test_data = to_index(word_vocab, char_vocab, pos_vocab, spo_vocab, tag_vocab, test_data)

    # dump
    dump_data(config.data_path, config.train_name, train_data)
    dump_data(config.data_path, config.dev_name, dev_data)
    # dump_data(config.data_path, config.test_name, test_data)

    dump_data(config.data_path, config.word_vocab_name, word_vocab)
    dump_data(config.data_path, config.char_vocab_name, char_vocab)
    dump_data(config.data_path, config.pos_vocab_name, pos_vocab)
    dump_data(config.data_path, config.spo_vocab_name, spo_vocab)
    dump_data(config.data_path, config.tag_vocab_name, tag_vocab)


if __name__ == "__main__":
    config = Config()
    dump(config)

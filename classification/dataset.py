import pickle
import os
import json
from fastNLP import DataSet
from fastNLP import Vocabulary

from config import Config


def process_class(schemas, spo_list):
    # label -> multi-hot
    target = [0] * len(schemas)
    for spo in spo_list:
        spo_concat = spo['subject_type'] + spo['predicate'] + spo['object_type']
        target[schemas[spo_concat]] = 1
    return target


def process_data(data_path, data_name, test=False, bert=False, input_name='text', target_name='target'):
    print('Processing', data_name)

    schemas = {}
    with open(os.path.join(data_path, "all_50_schemas"), 'rb') as f:
        for i, line in enumerate(f):
            spo = json.loads(line)
            schemas[spo['subject_type'] + spo['predicate'] + spo['object_type']] = i

    # input
    text = []
    # target
    target = []
    with open(os.path.join(data_path, data_name), 'rb') as f:
        for line in f:
            dic = json.loads(line)
            if bert:
                text.append(dic['text'])
            else:
                text.append(list(dic['text']))
            if not test:
                target.append(process_class(schemas, dic['spo_list']))

    if not test:
        data_dict = {
            input_name: text,
            target_name: target
        }
    else:
        data_dict = {input_name: text}
    dataset = DataSet(data=data_dict)
    print('Len', len(dataset))
    print('Sample', dataset[0])
    #exit()
    return dataset


def get_vocab(dataset):
    vocabulary = Vocabulary(unknown='<unk>', padding='<pad>')
    dataset.apply(lambda x: [vocabulary.add(char) for char in x['text']])
    vocabulary.build_vocab()
    print('vocab', len(vocabulary))

    return vocabulary


def to_index(vocabulary, dataset, seq_len, test=False):
    print('to index')
    dataset.apply(lambda x: [vocabulary.to_index(char) for char in x['text']], new_field_name='text')
    pad = vocabulary.to_index('<pad>')
    dataset.apply(lambda x: [pad] * (seq_len - len(x['text'])) + x['text'], new_field_name='text')
    print('Sample', dataset[0])

    dataset.set_input('text')
    if not test:
        dataset.set_target('target')

    return dataset


def dump_data(path, name, data):
    pickle.dump(data, open(os.path.join(path, name), "wb"))


def dump(config):
    # preprocess
    train_data = process_data(config.source_path, config.train_source)
    dev_data = process_data(config.source_path, config.dev_source)
    test_data = process_data(config.source_path, config.test_source, test=True)

    # vocabs
    vocabulary = get_vocab(train_data)

    # to index
    train_data = to_index(vocabulary, train_data, config.sentence_length)
    dev_data = to_index(vocabulary, dev_data, config.sentence_length)
    test_data = to_index(vocabulary, test_data, config.sentence_length, test=True)

    # dump
    dump_data(config.data_path, config.train_name, train_data)
    dump_data(config.data_path, config.dev_name, dev_data)
    dump_data(config.data_path, config.test_name, test_data)

    dump_data(config.data_path, config.vocabulary_name, vocabulary)


if __name__ == "__main__":
    config = Config()
    dump(config)

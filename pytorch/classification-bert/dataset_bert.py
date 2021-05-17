from dataset import process_class, process_data, dump_data
from config import Config
import os
import json
import pickle

from fastNLP import DataSet
from pytorch_pretrained_bert.tokenization import BertTokenizer


def get_schemas(data_path):
    schemas = {}
    with open(os.path.join(data_path, "all_50_schemas"), 'rb') as f:
        for i, line in enumerate(f):
            spo = json.loads(line)
            schemas[spo['subject_type'] + spo['predicate'] + spo['object_type']] = i
    return schemas


def get_input_examples(data_path, data_name, test=False, input_name='text_a', target_name='label'):
    dataset = process_data(data_path, data_name, test, bert=True, input_name=input_name, target_name=target_name)
    schemas = get_schemas(data_path)
    if test:
        dataset.apply(lambda x: process_class(schemas, []), new_field_name=target_name)
    print(dataset[0])
    #exit()
    return dataset


def convert_examples_to_features(dataset, max_seq_length, tokenizer):
    dataset.apply(lambda x: tokenizer.tokenize(x['text_a']), new_field_name='tokens_a')
    dataset.apply(lambda x: x['tokens_a'][:(max_seq_length - 2)] if len(x['tokens_a']) > (max_seq_length - 2) else x['tokens_a'], new_field_name='tokens_a')
    
    dataset.apply(lambda x: ["[CLS]"] + x['tokens_a'] + ["[SEP]"], new_field_name='tokens')

    dataset.apply(lambda x: [0] * len(x['tokens']), new_field_name='segment_ids')

    dataset.apply(lambda x: tokenizer.convert_tokens_to_ids(x['tokens']), new_field_name='input_ids')
    dataset.apply(lambda x: [1] * len(x['input_ids']), new_field_name='input_mask')
	
    dataset.apply(lambda x: [0] * (max_seq_length - len(x['input_ids'])), new_field_name='padding')
    # input_ids, input_mask, segment_ids
    dataset.apply(lambda x: x['input_ids'] + x['padding'], new_field_name='input_ids')
    dataset.apply(lambda x: x['input_mask'] + x['padding'], new_field_name='input_mask')
    dataset.apply(lambda x: x['segment_ids'] + x['padding'], new_field_name='segment_ids')

    for x in dataset:
        assert len(x['input_ids']) == max_seq_length
        assert len(x['input_mask']) == max_seq_length
        assert len(x['segment_ids']) == max_seq_length

    # input
    dataset.rename_field('input_mask', 'attention_mask')
    dataset.rename_field('segment_ids', 'token_type_ids')
    # target
    dataset.rename_field('label', 'label_id')

    # delete fields
    dataset.delete_field('text_a')
    dataset.delete_field('tokens_a')
    dataset.delete_field('tokens')
    dataset.delete_field('padding')

    print(dataset[0])

    dataset.set_input('input_ids', 'token_type_ids', 'attention_mask')
    dataset.set_target('label_id')

    return dataset


def dump(config):
    # input examples
    train_data = get_input_examples(config.source_path, config.train_source)
    dev_data = get_input_examples(config.source_path, config.dev_source)
    test_data = get_input_examples(config.source_path, config.test_source, test=True)

    tokenizer = BertTokenizer.from_pretrained(config.bert_vocab_path, do_lower_case=False)

    train_data = convert_examples_to_features(train_data, config.sentence_length, tokenizer)
    dev_data = convert_examples_to_features(dev_data, config.sentence_length, tokenizer)
    test_data = convert_examples_to_features(test_data, config.sentence_length, tokenizer)

    dump_data(config.bert_data_path, config.train_name, train_data)
    dump_data(config.bert_data_path, config.dev_name, dev_data)
    dump_data(config.bert_data_path, config.test_name, test_data)


if __name__ == "__main__":
    config = Config()
    dump(config)

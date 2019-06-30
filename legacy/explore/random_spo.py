# -*- coding: utf-8 -*-
import json
import random

SCHEMA_DIR = 'data/all_50_schemas'
TEST_DATA_DIR = 'data/test1_data_postag.json'
ORIGINAL_RESULT = 'fix_encoding/result.json'
NEW_RESULT = 'result.json'


schemas = []
with open(SCHEMA_DIR, 'r') as f:
	for line in f:
		schemas.append(json.loads(line))
SCHE_SIZE = len(schemas)


def random_spo(postag):
	spo = schemas[random.randint(0, SCHE_SIZE - 1)]
	spo['object'] = postag[random.randint(0, len(postag) - 1)]['word']
	spo['subject'] = postag[random.randint(0, len(postag) - 1)]['word']
	return spo


def fill_random_spo():
	test = []
	with open(TEST_DATA_DIR, 'r') as f:
		for line in f:
			test.append(json.loads(line))

	result = []
	with open(ORIGINAL_RESULT, 'r') as f:
		for i, line in enumerate(f):
			d = json.loads(line)
			if len(d['spo_list']) == 0 and len(test[i]['postag']) != 0:
				d['spo_list'].append(random_spo(test[i]['postag']))
			result.append(d)

	with open(NEW_RESULT, 'w') as f:
		for r in result:
			f.write(json.dumps(r, ensure_ascii=False) + '\n')


if __name__ == "__main__":
	# fill empty spo_list with one random spo
	fill_random_spo()

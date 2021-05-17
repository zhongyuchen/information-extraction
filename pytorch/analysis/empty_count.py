import json
from matplotlib import pyplot as plt
import numpy as np


TRAIN_DIR = '../data/train_data.json'
DEV_DIR = '../data/dev_data.json'
TEST_DIR = '../data/test1_data_postag.json'


def empty_count(filepath, test=False):
	count = {
		'postag': 0,
		'text': 0,
		'spo_list': 0
	}
	with open(filepath, 'r') as f:
		for line in f:
			l = json.loads(line)
			if len(l['postag']) == 0:
				count['postag'] += 1
			if len(l['text']) == 0:
				count['text'] += 1
			if not test:
				if len(l['spo_list']) == 0:
					count['spo_list'] += 1
	print(count)


if __name__ == "__main__":
	empty_count(TRAIN_DIR)
	empty_count(DEV_DIR)
	empty_count(TEST_DIR, test=True)

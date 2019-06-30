import json
from matplotlib import pyplot as plt
import numpy as np


TRAIN_DIR = '../data/train_data.json'
DEV_DIR = '../data/dev_data.json'


def spo_count(filepath):
	length = np.zeros(51)
	unique_length = np.zeros(51)
	with open(filepath, 'r') as f:
		for line in f:
			l = json.loads(line)
			length[len(l['spo_list'])] += 1
			unique = set()
			for spo in l['spo_list']:
				unique.add(spo['predicate'] + spo['subject_type'])
			unique_length[len(unique)] += 1

	return length * 1.0 / np.sum(length), unique_length * 1.0 / np.sum(unique_length)


if __name__ == "__main__":
	l1, ul1 = spo_count(TRAIN_DIR)
	l2, ul2 = spo_count(DEV_DIR)
	print('train data:')
	print('spo percentage:', l1)
	print('unique spo percentage:', ul1)
	print('dev data:')
	print('spo percentage:', l2)
	print('unique spo percentage:', ul2)
	plt.subplot(121)
	plt.plot(l1[0:10])
	plt.title('spo count percentage in train data')
	plt.subplot(122)
	plt.plot(l2[0:10])
	plt.title('spo count percentage in dev data')
#plt.subplot(223)
#	plt.plot(ul1)
#	plt.title('unique spo count percentage in train data')
#	plt.subplot(224)
#	plt.plot(ul2)
#	plt.title('unique spo count percentage in dev data')
	plt.show()
	
# 0,	1,	  2,	3
# 0.00, 0.43, 0.33, 0.12, 0.06, 0.03, 0.01, 0.01, 0.01, 0.00, 0.00
# 0.00, 0.43, 0.33, 0.12, 0.06, 0.03, 0.01, 0.01, 0.00, 0.00, 0.00

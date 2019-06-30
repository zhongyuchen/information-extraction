import json


def max_length(filename):
	length = -1
	with open(filename, 'r') as f:
		for line in f:
			d = json.loads(line)
			if len(d['text']) > length:
				length = len(d['text'])
	return length

if __name__ == "__main__":
	# max length of text in data files
	print(max_length('data/train_data.json')) # 300
	print(max_length('data/dev_data.json')) # 299
	print(max_length('data/test_data_postag.json')) # 300

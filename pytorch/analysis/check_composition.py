import json

TEST_DIR = 'data/test1_data_postag.json'
RESULT_DIR = 'fix_encoding/result.json'
ERROR_DIR = 'error.txt'

def check_concat_postag():
	test = []
	with open(TEST_DIR, 'r') as f:
		test = [json.loads(line) for line in f]

	for t in test:
		concat_text = ""
		for p in t['postag']:
			concat_text += p['word']
		if concat_text != t['text']:
			print(t['text'])
			print(concat_text)


def check_composition():
	with open(RESULT_DIR, 'r') as f:
		for i, line in enumerate(f):
			result = json.loads(line)
			for spo in result['spo_list']:
				if spo['subject'] not in result['text']:
					print(result['text'])
				if spo['object'] not in result['text']:
					print(result['text'])


def compose(word, word_list):
	l = []
	for w in word_list:
		if w in word:
			l.append(w)

	for i in range(len(l)):
		s = ""
		for j in range(i, len(l)):
			s += l[j]
			if s == word:
				return True

	return False


def real_check_composition():
	# the result might not be composed by the given words!
	test = []
	with open(TEST_DIR, 'r') as f:
		test = [json.loads(line) for line in f]

	words = [[p['word'] for p in t['postag']] for t in test]

	e = []
	with open(RESULT_DIR, 'r') as f:
		for i, line in enumerate(f):
			result = json.loads(line)
			for spo in result['spo_list']:
				if not compose(spo['subject'], words[i]) or not compose(spo['object'], words[i]):
					e.append(str(i + 1) + "," + spo['subject'] + "," + spo['object'] + "," + result['text'])

	with open(ERROR_DIR, 'w') as f:
		for er in e:
			f.write(er + '\n')


if __name__ == "__main__":
	# check_concat_postag()
	# check_composition()
	real_check_composition()

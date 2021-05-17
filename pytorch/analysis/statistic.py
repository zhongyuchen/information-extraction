# -*- coding: utf-8 -*-
import json
import re


def add_item_offset(token, sentence):
    """Get the start and end offset of a token in a sentence"""
    s_pattern = re.compile(re.escape(token), re.I)
    token_offset_list = []
    for m in s_pattern.finditer(sentence):
        token_offset_list.append((m.group(), m.start(), m.end()))
    return token_offset_list


def cal_item_pos(target_offset, idx_list):
    """Get the index list where the token is located"""
    target_idx = []
    for target in target_offset:
        start, end = target[1], target[2]
        cur_idx = []
        for i, idx in enumerate(idx_list):
            if idx >= start and idx < end:
                cur_idx.append(i)
        if len(cur_idx) > 0:
            target_idx.append(cur_idx)
    return target_idx


def tagging(p, sentence, postag, spo_list):
	# extended BIEO tagging (7 kinds of tagging)
	word_list = [item['word'] for item in postag]

	token_idx_list = []
	start_idx = 0
	for word in word_list:
	    if start_idx >= len(sentence):
	        break
	    token_idx_list.append(start_idx)
	    start_idx += len(word)

	mark_list = ['O'] * len(token_idx_list)
	for spo in spo_list:
	    predicate = spo['predicate']
	    if predicate != p:
	        continue
	    sub = spo['subject']
	    obj = spo['object']
	    s_idx_list = cal_item_pos(add_item_offset(sub, sentence), token_idx_list)
	    o_idx_list = cal_item_pos(add_item_offset(obj, sentence), token_idx_list)
	    if len(s_idx_list) == 0 or len(o_idx_list) == 0:
	        continue
	    for s_idx in s_idx_list:
	        if len(s_idx) == 1:
	            mark_list[s_idx[0]] = 'B-SUB'
	        elif len(s_idx) == 2:
	            mark_list[s_idx[0]] = 'B-SUB'
	            mark_list[s_idx[1]] = 'E-SUB'
	        else:
	            mark_list[s_idx[0]] = 'B-SUB'
	            mark_list[s_idx[-1]] = 'E-SUB'
	            for idx in range(1, len(s_idx) - 1):
	                mark_list[s_idx[idx]] = 'I-SUB'
	    for o_idx in o_idx_list:
	        if len(o_idx) == 1:
	            mark_list[o_idx[0]] = 'B-OBJ'
	        elif len(o_idx) == 2:
	            mark_list[o_idx[0]] = 'B-OBJ'
	            mark_list[o_idx[1]] = 'E-OBJ'
	        else:
	            mark_list[o_idx[0]] = 'B-OBJ'
	            mark_list[o_idx[-1]] = 'E-OBJ'
	            for idx in range(1, len(o_idx) - 1):
	                mark_list[o_idx[idx]] = 'I-OBJ'

	# word tagging -> character tagging
	tag_list = []
	for i, word in enumerate(word_list):
		if len(word) == 1:
			tag_list.append(mark_list[i])
			continue

		if mark_list[i] == 'B-SUB':
			start, middle, end = 'B-SUB', 'I-SUB', 'I-SUB'
			if i + 1 >= len(word_list) or mark_list[i + 1] != 'I-SUB' and mark_list[i + 1] != 'E-SUB':
				end = 'E-SUB'
		elif mark_list[i] == 'B-OBJ':
			start, middle, end = 'B-OBJ', 'I-OBJ', 'I-OBJ'
			if i + 1 >= len(word_list) or mark_list[i + 1] != 'I-OBJ' and mark_list[i + 1] != 'E-OBJ':
				end = 'E-OBJ'
		elif mark_list[i] == 'E-SUB':
			start, middle, end = 'I-SUB', 'I-SUB', 'E-SUB'
		elif mark_list[i] == 'E-OBJ':
			start, middle, end = 'I-OBJ', 'I-OBJ', 'E-OBJ'
		else:
			start, middle, end = mark_list[i], mark_list[i], mark_list[i]

		tag_list.append(start)
		for j in range(1, len(word) - 1):
			tag_list.append(middle)
		tag_list.append(end)

	return tag_list


def count0(path, save_path):
	# 导出含有predicate是'改编自'的数据
	data = []
	with open(path, 'r') as f:
		for line in f:
			data.append(json.loads(line))

	result = []
	for i, d in enumerate(data):
		r = {
			'text': d['text'],
			'spo_list': []
		}
		for spo in d['spo_list']:
			if spo['predicate'] == '改编自':
				r['spo_list'].append(spo)
		if len(r['spo_list']) != 0:
			result.append(r)

	with open(save_path, 'w') as f:
		for r in result:
			f.write(json.dumps(r, ensure_ascii=False) + '\n')

	return


def count1(path, save_path):
	# 导出含有predicate是'改编自'，而且subject和object一样的数据
	data = []
	with open(path, 'r') as f:
		for line in f:
			data.append(json.loads(line))

	result = []
	for i, d in enumerate(data):
		flag = False
		for spo in d['spo_list']:
			if spo['predicate'] == '改编自' and spo['subject'] == spo['object']:
				flag = True
				break
		if flag:
			result.append(d)

	with open(save_path, 'w') as f:
		for r in result:
			f.write(json.dumps(r, ensure_ascii=False) + '\n')

	return


def count2(path, save_path):
	# 导出text含有'改编自'，但是spo_list没有'改编自'关系
	data = []
	with open(path, 'r') as f:
		for line in f:
			data.append(json.loads(line))

	result = []
	for i, d in enumerate(data):
		if '改编自' in d['text']:
			flag = False
			for spo in d['spo_list']:
				if spo['predicate'] == '改编自':
					flag = True
					break
			if not flag:
				result.append(d['text'])

	with open(save_path, 'w') as f:
		for r in result:
			f.write(json.dumps(r, ensure_ascii=False) + '\n')

	return


def gen_tag(path, save_path):
	# 导入数据文件，导出每条数据每个spo对应的tagging list
	result = []
	with open(path, 'r') as f:
		for i, line in enumerate(f):
			d = json.loads(line)
			postag = d['postag']
			text = d['text']
			spo_list = d['spo_list']
			for spo in spo_list:
				result.append({
					'mark_list': tagging(spo['predicate'], text, postag, spo_list),
					'text': text,
					'spo': spo
				})

	with open(save_path, 'w') as f:
		for i, r in enumerate(result):
			f.write(json.dumps(r, ensure_ascii=False) + '\n')


if __name__ == "__main__":
	# count0('../data/train_data.json', 'train_data.p')
	# count0('../data/dev_data.json', 'dev_data.p')
	# count1('../data/train_data.json', 'train_data_1.p')
	# count2('../data/train_data.json', 'train_data_2.p')
	gen_tag('train_data_1.p', 'train_data_1.res')

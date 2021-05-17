import re
import json


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


def tagging(spo_concat, sentence, postag, spo_list):
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
		# 2 predicate share the same name
		if spo['subject_type'] + spo['predicate'] + spo['object_type'] != spo_concat:
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


if __name__ == "__main__":
	d = {
		'postag': [{'word': '如何', 'pos': 'r'}, {'word': '演', 'pos': 'v'}, {'word': '好', 'pos': 'a'}, {'word': '自己', 'pos': 'r'}, {'word': '的', 'pos': 'u'}, {'word': '角色', 'pos': 'n'}, {'word': '，', 'pos': 'w'}, {'word': '请', 'pos': 'v'}, {'word': '读', 'pos': 'v'}, {'word': '《', 'pos': 'w'}, {'word': '演员自我修养', 'pos': 'nw'}, {'word': '》', 'pos': 'w'}, {'word': '《', 'pos': 'w'}, {'word': '喜剧之王', 'pos': 'nw'}, {'word': '》', 'pos': 'w'}, {'word': '周星驰', 'pos': 'nr'}, {'word': '崛起', 'pos': 'v'}, {'word': '于', 'pos': 'p'}, {'word': '穷困潦倒', 'pos': 'a'}, {'word': '之中', 'pos': 'f'}, {'word': '的', 'pos': 'u'}, {'word': '独门', 'pos': 'n'}, {'word': '秘笈', 'pos': 'n'}], 
		'text': '如何演好自己的角色，请读《演员自我修养》《喜剧之王》周星驰崛起于穷困潦倒之中的独门秘笈', 
		'spo_list': [{'predicate': '主演', 'object_type': '人物', 'subject_type': '影视作品', 'object': '周星驰', 'subject': '喜剧之王'}]
	}
	d = {"postag": [{"word": "《", "pos": "w"}, {"word": "端脑", "pos": "nw"}, {"word": "》", "pos": "w"}, {"word": "改编", "pos": "v"}, {"word": "自有", "pos": "v"}, {"word": "妖气", "pos": "n"}, {"word": "同名", "pos": "vn"}, {"word": "漫画", "pos": "n"}, {"word": "《", "pos": "w"}, {"word": "端脑", "pos": "nw"}, {"word": "》", "pos": "w"}, {"word": "，", "pos": "w"}, {"word": "是", "pos": "v"}, {"word": "由", "pos": "p"}, {"word": "搜狐视频", "pos": "nz"}, {"word": "、", "pos": "w"}, {"word": "有", "pos": "v"}, {"word": "妖气", "pos": "n"}, {"word": "、", "pos": "w"}, {"word": "留白", "pos": "v"}, {"word": "影视", "pos": "n"}, {"word": "出品", "pos": "n"}, {"word": "，", "pos": "w"}, {"word": "于中中", "pos": "nr"}, {"word": "执导", "pos": "v"}, {"word": "，", "pos": "w"}, {"word": "朱元冰", "pos": "nr"}, {"word": "、", "pos": "w"}, {"word": "蒋依依", "pos": "nr"}, {"word": "、", "pos": "w"}, {"word": "杨奇煜", "pos": "nr"}, {"word": "、", "pos": "w"}, {"word": "黄一琳", "pos": "nr"}, {"word": "、", "pos": "w"}, {"word": "谢佳见", "pos": "nr"}, {"word": "、", "pos": "w"}, {"word": "赵奕欢", "pos": "nr"}, {"word": "等人", "pos": "n"}, {"word": "主演", "pos": "v"}, {"word": "的", "pos": "u"}, {"word": "科幻", "pos": "n"}, {"word": "悬疑", "pos": "vn"}, {"word": "网络", "pos": "n"}, {"word": "剧", "pos": "n"}], "text": "《端脑》改编自有妖气同名漫画《端脑》，是由搜狐视频、有妖气、留白影视出品，于中中执导，朱元冰、蒋依依、杨奇煜、黄一琳、谢佳见、赵奕欢等人主演的科幻悬疑网络剧", "spo_list": [{"predicate": "主演", "object_type": "人物", "subject_type": "影视作品", "object": "蒋依依", "subject": "端脑"}, {"predicate": "主演", "object_type": "人物", "subject_type": "影视作品", "object": "朱元冰", "subject": "端脑"}, {"predicate": "主演", "object_type": "人物", "subject_type": "影视作品", "object": "赵奕欢", "subject": "端脑"}, {"predicate": "主演", "object_type": "人物", "subject_type": "影视作品", "object": "黄一琳", "subject": "端脑"}, {"predicate": "导演", "object_type": "人物", "subject_type": "影视作品", "object": "于中中", "subject": "端脑"}, {"predicate": "主演", "object_type": "人物", "subject_type": "影视作品", "object": "杨奇煜", "subject": "端脑"}, {"predicate": "改编自", "object_type": "作品", "subject_type": "影视作品", "object": "端脑", "subject": "端脑"}, {"predicate": "主演", "object_type": "人物", "subject_type": "影视作品", "object": "谢佳见", "subject": "端脑"}]}
	postag = d['postag']
	text = d['text']
	spo_list = d['spo_list']
	spo_concat = d['spo_list'][0]['subject_type'] + d['spo_list'][0]['predicate'] + d['spo_list'][0]['object_type']

	mark_list = tagging(spo_concat, text, postag, spo_list)
	for i in range(len(mark_list)):
		print(mark_list[i], text[i])

	# postag = [{"word": "南迦帕尔巴特峰", "pos": "ns"}, {"word": "，", "pos": "w"}, {"word": "8125米", "pos": "m"}]
	# text = "南迦帕尔巴特峰，8125米"
	# spo_list = [{"predicate": "海拔", "object_type": "Number", "subject_type": "地点", "object": "8125米", "subject": "南迦帕尔巴特峰"}]
	# p = spo_list[0]['predicate']

	# mark_list = tagging(p, text, postag, spo_list)
	# for i in range(len(mark_list)):
	# 	print(mark_list[i], text[i])

	# with open('train_data.json', 'r') as f:
	# 	for i, line in enumerate(f):
	# 		d = json.loads(line)
	# 		postag = d['postag']
	# 		text = d['text']
	# 		spo_list = d['spo_list']
	# 		for spo in spo_list:
	# 			mark_list = tagging(spo['predicate'], text, postag, spo_list)



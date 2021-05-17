import re
import json


def find(sub,sent):
    res=[i.start() for i in re.finditer(re.compile(re.escape(sub), re.I),sent)]
    return res

def tagging(spo_concat, sentence, postag, spo_list):
    word_list = [item['word'] for item in postag]

    mark_list = ["O"]*len(sentence)
    for spo in spo_list:
        if spo['subject_type'] + spo['predicate'] + spo['object_type'] != spo_concat:
            continue
        sub = spo['subject']
        obj = spo['object']
        s_idx_list=find(sub,sentence)
        o_idx_list=find(obj,sentence)
        if sub==obj:
            o_idx_list=[x for i,x in enumerate(s_idx_list) if i%2==1]
            
        for s_idx in s_idx_list:
            if len(sub) == 1:
                mark_list[s_idx] = 'B-SUB'
            elif len(sub) == 2:
                mark_list[s_idx] = 'B-SUB'
                mark_list[s_idx+1] = 'E-SUB'
            else:
                mark_list[s_idx] = 'B-SUB'
                mark_list[s_idx+len(sub)-1] = 'E-SUB'
                for idx in range(1, len(sub) - 1):
                    mark_list[s_idx+idx] = 'I-SUB'
        for o_idx in o_idx_list:
            if len(obj) == 1:
                mark_list[o_idx] = 'B-OBJ'
            elif len(obj) == 2:
                mark_list[o_idx] = 'B-OBJ'
                mark_list[o_idx+1] = 'E-OBJ'
            else:
                mark_list[o_idx] = 'B-OBJ'
                mark_list[o_idx+len(obj)-1] = 'E-OBJ'
                for idx in range(1, len(obj) - 1):
                    mark_list[o_idx+idx] = 'I-OBJ'

    return mark_list


if __name__ == "__main__":
	d = {"postag": [{"word": "《", "pos": "w"}, {"word": "端脑", "pos": "nw"}, {"word": "》", "pos": "w"}, {"word": "改编", "pos": "v"}, {"word": "自有", "pos": "v"}, {"word": "妖气", "pos": "n"}, {"word": "同名", "pos": "vn"}, {"word": "漫画", "pos": "n"}, {"word": "《", "pos": "w"}, {"word": "端脑", "pos": "nw"}, {"word": "》", "pos": "w"}, {"word": "，", "pos": "w"}, {"word": "是", "pos": "v"}, {"word": "由", "pos": "p"}, {"word": "搜狐视频", "pos": "nz"}, {"word": "、", "pos": "w"}, {"word": "有", "pos": "v"}, {"word": "妖气", "pos": "n"}, {"word": "、", "pos": "w"}, {"word": "留白", "pos": "v"}, {"word": "影视", "pos": "n"}, {"word": "出品", "pos": "n"}, {"word": "，", "pos": "w"}, {"word": "于中中", "pos": "nr"}, {"word": "执导", "pos": "v"}, {"word": "，", "pos": "w"}, {"word": "朱元冰", "pos": "nr"}, {"word": "、", "pos": "w"}, {"word": "蒋依依", "pos": "nr"}, {"word": "、", "pos": "w"}, {"word": "杨奇煜", "pos": "nr"}, {"word": "、", "pos": "w"}, {"word": "黄一琳", "pos": "nr"}, {"word": "、", "pos": "w"}, {"word": "谢佳见", "pos": "nr"}, {"word": "、", "pos": "w"}, {"word": "赵奕欢", "pos": "nr"}, {"word": "等人", "pos": "n"}, {"word": "主演", "pos": "v"}, {"word": "的", "pos": "u"}, {"word": "科幻", "pos": "n"}, {"word": "悬疑", "pos": "vn"}, {"word": "网络", "pos": "n"}, {"word": "剧", "pos": "n"}], "text": "《端脑》改编自有妖气同名漫画《端脑》，是由搜狐视频、有妖气、留白影视出品，于中中执导，朱元冰、蒋依依、杨奇煜、黄一琳、谢佳见、赵奕欢等人主演的科幻悬疑网络剧", "spo_list": [{"predicate": "主演", "object_type": "人物", "subject_type": "影视作品", "object": "蒋依依", "subject": "端脑"}, {"predicate": "主演", "object_type": "人物", "subject_type": "影视作品", "object": "朱元冰", "subject": "端脑"}, {"predicate": "主演", "object_type": "人物", "subject_type": "影视作品", "object": "赵奕欢", "subject": "端脑"}, {"predicate": "主演", "object_type": "人物", "subject_type": "影视作品", "object": "黄一琳", "subject": "端脑"}, {"predicate": "导演", "object_type": "人物", "subject_type": "影视作品", "object": "于中中", "subject": "端脑"}, {"predicate": "主演", "object_type": "人物", "subject_type": "影视作品", "object": "杨奇煜", "subject": "端脑"}, {"predicate": "改编自", "object_type": "作品", "subject_type": "影视作品", "object": "端脑", "subject": "端脑"}, {"predicate": "主演", "object_type": "人物", "subject_type": "影视作品", "object": "谢佳见", "subject": "端脑"}]}
	postag = d['postag']
	text = d['text']
	spo_list = d['spo_list']
	spo_concat = d['spo_list'][0]['subject_type'] + d['spo_list'][0]['predicate'] + d['spo_list'][0]['object_type']

	mark_list = tagging(spo_concat, text, postag, spo_list)
	assert len(mark_list) == len(text)
	for i in range(len(mark_list)):
		print(mark_list[i], text[i])

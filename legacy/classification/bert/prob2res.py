import tensorflow as tf
import csv
import json
import numpy as np
import copy
import argparse
import os
#import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", default=None, help="output dir")
parser.add_argument("--output_file", default=None, help="output file")
parser.add_argument("--prob_file", default=None, help="Output file name")
parser.add_argument("--threshold", default=0.5, help="threshold")
args = parser.parse_args()


def transform(divide):
    data = []
    with open(os.path.join(args.output_dir, args.prob_file), 'r') as f:
        for i, line in enumerate(f):
            d = []
            l = line.split('\t')
            for j in l:
                d.append(float(j))
            data.append(np.array(d))
    data = np.array(data)

    data[data>divide] = 1
    data[data<=divide] = 0

    #pickle.dump(data, open(os.path.join(args.output_dir, args.output_file), 'wb'))

    label_list = ['丈夫', '上映时间', '专业代码', '主持人', '主演', '主角', '人口数量', '作曲', '作者', '作词', '修业年限', '出品公司', '出版社', '出生地', '出生日期',
                            '创始人', '制片人', '占地面积', '号', '嘉宾', '国籍', '妻子', '字', '官方语言', '导演', '总部地点', '成立日期', '所在城市', '所属专辑', '改编自',
                                            '朝代', '歌手', '母亲', '毕业院校', '民族', '气候', '注册资本', '海拔', '父亲', '目', '祖籍', '简称', '编剧', '董事长', '身高', '连载网站',
                                                            '邮政编码', '面积', '首都']
    to_index = np.arange(data.shape[1])
    with open(os.path.join(args.output_dir, args.output_file), 'w') as f:
        for line in data:
            l = to_index[line == 1]
            labels = []
            for i in range(len(l)):
                labels.append(label_list[i])
            f.write(' '.join(labels) + '\n')


if __name__ == "__main__":
    # check()
    #for i in np.arange(0., 1., 0.1):
    #    predict_result(i)
    transform(float(args.threshold))


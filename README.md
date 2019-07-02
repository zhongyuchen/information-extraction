# Information Extraction 

[2019 Language and Intelligence Challenge](http://lic2019.ccf.org.cn/): Information Extraction 

## Prerequisites

* Install required packages by:
```angular2
pip install -r requirements.txt
```

## Data

* sample schema:

```
{"object_type": "地点", "predicate": "祖籍", "subject_type": "人物"}
```

* sample data, with `postag` and `text` as input and `spo_list` as output:

```
{"postag": [{"word": "《", "pos": "w"}, {"word": "星空黑夜传奇", "pos": "nw"}, {"word": "》", "pos": "w"}, {"word": "是", "pos": "v"}, {"word": "连载", "pos": "v"}, {"word": "于", "pos": "p"}, {"word": "起点中文网", "pos": "nz"}, {"word": "的", "pos": "u"}, {"word": "网络", "pos": "n"}, {"word": "小说", "pos": "n"}, {"word": "，", "pos": "w"}, {"word": "作者", "pos": "n"}, {"word": "是", "pos": "v"}, {"word": "啤酒", "pos": "n"}, {"word": "的", "pos": "u"}, {"word": "罪孽", "pos": "n"}], "text": "《星空黑夜传奇》是连载于起点中文网的网络小说，作者是啤酒的罪孽", "spo_list": [{"predicate": "连载网站", "object_type": "网站", "subject_type": "网络小说", "object": "起点中文网", "subject": "星空黑夜传奇"}, {"predicate": "作者", "object_type": "人物", "subject_type": "图书作品", "object": "啤酒的罪孽", "subject": "星空黑夜传奇"}]}
```

## Idea

* Train multi-label classification model: predict predicate.
* Train sequence labeling model: input text and predicate, output text labeling.
* Extract SPO from sequence labeling result.

## Implementation

Check `report/PRML-final-project-doc-2019.pdf` for details.

### Multi-label Classification

* CNN, BiRNN, BiLSTM, BiLSTM with max pooling and RCNN
* BERT

### Sequence Labeling

* Encoder: BiLSTM and Transformer
* Decoder: CRF

## Result

### Multi-label Classification

![classification](pic/classification_result.png)

### Sequence Labeling

![labeling](pic/labeling_result.png)

## fitlog usage

* Initialize fitlog in `classification` folder:
```
cd classification/
fitlog init
fitlog log logs
```
* Initialize fitlog in `labeling` folder:
```
cd labeling/
fitlog init
fitlog log logs
```

## Author

Zhongyu Chen

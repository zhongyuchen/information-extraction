# information-extraction

[2019 Language and Intelligence Challenge](http://lic2019.ccf.org.cn/): Information Extraction 

## Prerequisites

* Install required packages by:
```angular2
pip install -r requirements.txt
```

## Data
Download data: initialize and update the `information-extraction-data` git submodule by `git submodule init` and `git submodule update`, and then unzip the data files
* sample schema:
    ```
    {"object_type": "地点", "predicate": "祖籍", "subject_type": "人物"}
    ```
* sample data, with `postag` and `text` as input and `spo_list` as output:
    ```
    {
        "postag": [
            {"word": "一直", "pos": "d"}, 
            {"word": "陪", "pos": "v"}, 
            {"word": "我", "pos": "r"}, 
            {"word": "到", "pos": "p"}, 
            {"word": "现在", "pos": "t"}, 
            {"word": "是", "pos": "v"}, 
            {"word": "歌手", "pos": "n"}, 
            {"word": "马健涛", "pos": "nr"}, 
            {"word": "原创", "pos": "v"}, 
            {"word": "的", "pos": "u"}, 
            {"word": "歌曲", "pos": "n"}
        ], 
        "text": "一直陪我到现在是歌手马健涛原创的歌曲", 
        "spo_list": [
            {"predicate": "歌手", "object_type": "人物", "subject_type": "歌曲", "object": "马健涛", "subject": "一直陪我到现在"}
        ]
    }
    ```

## Baseline

* [baidu/information-extraction](https://github.com/baidu/information-extraction)

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

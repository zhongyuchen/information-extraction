from gensim.models import word2vec
import os
import pickle
import random
import torch
import numpy as np
from config import *
from dataset import Dataset


def get_texts(train_loader, vocabulary):
    texts = []
    length = 0
    for data_batch, _ in train_loader:
        for data in data_batch:
            data = data.tolist()
            text = []
            for i in data:
                text.append(vocabulary.to_word(i))
            length += len(text)
            texts.append(text)
    
    print('texts', length)
    return texts


def get_weights(model, vocabulary):
    weights = np.zeros((len(vocabulary), embed_dim))
    for i in range(len(vocabulary)):
        if vocabulary.to_word(i) == pad_str or vocabulary.to_word(i) == unk_str:
            continue
        weights[i] = model.wv[vocabulary.to_word(i)]
    return weights


def main():
    vocabulary = pickle.load(open(os.path.join(data_path, vocabulary_name), 'rb'))
    print('<unk>', vocabulary.to_index(unk_str))
    train_loader = pickle.load(open(os.path.join(data_path, train_name), 'rb'))
    texts = get_texts(train_loader, vocabulary)
    model = word2vec.Word2Vec(window=64, min_count=1, size=embed_dim)
    model.build_vocab(texts)
    model.train(texts, total_examples=model.corpus_count, epochs=model.epochs)
    weights = get_weights(model, vocabulary)
    pickle.dump(weights, open(os.path.join(data_path, weight_name), 'wb'))


if __name__ == "__main__":
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    main()
    
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
    for (char, word, pos, spo, sentence_length, tag) in train_loader:
        for w in word:
            data = w.tolist()
            text = []
            for i in data:
                text.append(vocabulary.to_word(i))
            texts.append(text)
    print('texts', len(texts))
    return texts


def get_weights(model, vocabulary):
    weights = np.zeros((len(vocabulary), word_embed_dim))
    for i in range(len(vocabulary)):
        if vocabulary.to_word(i) == pad_str or vocabulary.to_word(i) == unk_str:
            continue
        weights[i] = model.wv[vocabulary.to_word(i)]
    return weights


def main():
    vocabulary = pickle.load(open(os.path.join(data_path, word_vocab_name), 'rb'))
    print('<unk>', vocabulary.to_index(unk_str))
    train_loader = pickle.load(open(os.path.join(data_path, train_name), 'rb'))
    texts = get_texts(train_loader, vocabulary)
    model = word2vec.Word2Vec(window=64, min_count=1, size=word_embed_dim)
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
    
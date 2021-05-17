import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import sys
sys.path.append('../')
from mask import seq_len_to_mask
from crf import allowed_transitions
from crf import ConditionalRandomField as CRF
from fastNLP.modules import encoder



class Transformer_CRF(nn.Module):
    def __init__(self, char_init_embed, word_init_embed, pos_init_embed, spo_embed_dim, num_classes,
        num_layers, inner_size, key_size, value_size, num_head, dropout=0.1,
        id2words=None, encoding_type='bieso', weight=None):
        super().__init__()
        
        # self.Embedding = nn.Embedding(init_embed)
        #print(char_init_embed)
        self.char_embed = nn.Embedding(char_init_embed[0], char_init_embed[1])
        self.word_embed = nn.Embedding(word_init_embed[0], word_init_embed[1])
        self.word_embed.weight.data.copy_(torch.from_numpy(weight))
        self.pos_embed = nn.Embedding(pos_init_embed[0], pos_init_embed[1])
        # spo embed size: 50
        self.embed_dim = self.char_embed.embedding_dim + self.word_embed.embedding_dim + self.pos_embed.embedding_dim + spo_embed_dim

        self.norm1 = torch.nn.LayerNorm(self.embed_dim)
        self.transformer = encoder.TransformerEncoder(num_layers=num_layers, 
            model_size=self.embed_dim, inner_size=inner_size, 
            key_size=key_size, value_size=value_size, num_head=num_head, dropout=dropout)
        self.Linear1 = nn.Linear(self.embed_dim, self.embed_dim // 3)
        self.norm2 = torch.nn.LayerNorm(self.embed_dim // 3)
        self.relu = torch.nn.LeakyReLU()
        self.drop = torch.nn.Dropout(dropout)
        self.Linear2 = nn.Linear(self.embed_dim // 3, num_classes)
        self.Linear = nn.Linear(self.embed_dim, num_classes)
        
        if id2words is None:
            self.Crf = CRF(num_classes, include_start_end_trans=False)
        else:
            self.Crf = CRF(num_classes, include_start_end_trans=False,
                            allowed_transitions=allowed_transitions(id2words, encoding_type=encoding_type))
    
    def _decode(self, x):
        """
        :param torch.FloatTensor x: [batch_size, max_len, tag_size]
        :return torch.LongTensor, [batch_size, max_len]
        """
        tag_seq, _ = self.Crf.viterbi_decode(x, self.mask)
        return tag_seq
    
    def _internal_loss(self, x, y):
        """
        Negative log likelihood loss.
        :param x: Tensor, [batch_size, max_len, tag_size]
        :param y: Tensor, [batch_size, max_len]
        :return loss: a scalar Tensor

        """
        x = x.float()
        y = y.long()
        assert x.shape[:2] == y.shape
        assert y.shape == self.mask.shape
        total_loss = self.Crf(x, y, self.mask)
        return torch.mean(total_loss)
    
    def _make_mask(self, x, seq_len):
        batch_size, max_len = x.size(0), x.size(1)
        mask = seq_len_to_mask(seq_len)
#         mask = mask.view(batch_size, max_len)
        mask = mask.to(x).float()
        return mask

    def _forward(self, char, word, pos, spo, seq_len, tag=None):
        """
        :param torch.LongTensor words: [batch_size, mex_len]
        :param torch.LongTensor seq_len:[batch_size, ]
        :param torch.LongTensor target: [batch_size, max_len]
        :return y: If truth is None, return list of [decode path(list)]. Used in testing and predicting.
                   If truth is not None, return loss, a scalar. Used in training.
        """
        
        char = char.long()
        #word = word.long()
        #pos = pos.long()
        #spo = spo.long()
        seq_len = seq_len.long()
        self.mask = self._make_mask(char, seq_len)
        
        # seq_len = seq_len.long()
        tag = tag.long() if tag is not None else None
        
        #if next(self.parameters()).is_cuda:
        #    char = char.cuda()
        #    self.mask = self.mask.cuda()
        
        # x = self.Embedding(words)
        char = self.char_embed(char)
        word = self.word_embed(word)
        pos = self.pos_embed(pos)
        #print(spo)
        #print(self.zeros)
        spo = spo.unsqueeze(1).repeat(1, char.shape[1], 1).float()
        #print(char.shape)
        #print(word.shape)
        #print(pos.shape)
        #print(spo.shape)
        x = torch.cat((char, word, pos, spo), dim=2)
        #print(x.shape)

        x = self.norm1(x)
        # [batch_size, max_len, char_embed_dim + word_embed_dim + pos_embed_dim + spo_embed_dim ]
        
        x = self.transformer(x, seq_mask=self.mask)
        
        #x = self.Linear1(x)
        #x = self.norm2(x)
        #x = self.relu(x)
        #x = self.drop(x)
        #x = self.Linear2(x)
        x = self.Linear(x)
        if tag is not None:
            return self._internal_loss(x, tag)
        else:
            return self._decode(x)
        #return {"pred": self._decode(x)}

    def forward(self, char, word, pos, spo, seq_len, tag):
        """
        
        :param torch.LongTensor words: [batch_size, mex_len]
        :param torch.LongTensor seq_len: [batch_size, ]
        :param torch.LongTensor target: [batch_size, max_len], 目标
        :return torch.Tensor: a scalar loss
        """
        return self._forward(char, word, pos, spo, seq_len, tag)

    def predict(self, char, word, pos, spo, seq_len):
        """
        
        :param torch.LongTensor words: [batch_size, mex_len]
        :param torch.LongTensor seq_len: [batch_size, ]
        :return torch.LongTensor: [batch_size, max_len]
        """
        return self._forward(char, word, pos, spo, seq_len)


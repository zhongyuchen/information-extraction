import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
sys.path.append('../')
from mask import seq_len_to_mask
# from fastNLP.modules.utils import seq_len_to_mask
from crf import allowed_transitions
from crf import ConditionalRandomField as CRF


class BiLSTM_CRF(nn.Module):
    """
    别名：:class:`fastNLP.models.AdvSeqLabel`  :class:`fastNLP.models.sequence_labeling.AdvSeqLabel`

    更复杂的Sequence Labelling模型。结构为Embedding, LayerNorm, 双向LSTM(两层)，FC，LayerNorm，DropOut，FC，CRF。
    
    :param tuple(int,int),torch.FloatTensor,nn.Embedding,numpy.ndarray init_embed: Embedding的大小(传入tuple(int, int),
        第一个int为vocab_zie, 第二个int为embed_dim); 如果为Tensor, Embedding, ndarray等则直接使用该值初始化Embedding
    :param int hidden_size: LSTM的隐层大小
    :param int num_classes: 有多少个类
    :param float dropout: LSTM中以及DropOut层的drop概率
    :param dict id2words: tag id转为其tag word的表。用于在CRF解码时防止解出非法的顺序，比如'BMES'这个标签规范中，'S'
        不能出现在'B'之后。这里也支持类似与'B-NN'，即'-'前为标签类型的指示，后面为具体的tag的情况。这里不但会保证
        'B-NN'后面不为'S-NN'还会保证'B-NN'后面不会出现'M-xx'(任何非'M-NN'和'E-NN'的情况。)
    :param str encoding_type: 支持"BIO", "BMES", "BEMSO", 只有在id2words不为None的情况有用。
    """
    
    def __init__(self, char_init_embed, word_init_embed, pos_init_embed, spo_embed_dim, sentence_length, 
        hidden_size, num_classes, dropout=0.3, id2words=None, encoding_type='bieso', weight=None):
        
        super().__init__()
        
        # self.Embedding = nn.Embedding(init_embed)
#         print(char_init_embed)
        self.char_embed = nn.Embedding(char_init_embed[0], char_init_embed[1])
        self.word_embed = nn.Embedding(word_init_embed[0], word_init_embed[1])
        # word2vec
        self.word_embed.weight.data.copy_(torch.from_numpy(weight))
        self.pos_embed = nn.Embedding(pos_init_embed[0], pos_init_embed[1])
        # spo embed size: 50
        self.embed_dim = self.char_embed.embedding_dim + self.word_embed.embedding_dim + self.pos_embed.embedding_dim + spo_embed_dim
        # sentence length
        #self.sen_len = sentence_length
        #self.zeros = torch.zeros(self.sen_len, dtype=torch.long)

        self.norm1 = torch.nn.LayerNorm(self.embed_dim)
        self.Rnn = nn.LSTM(input_size=self.embed_dim, hidden_size=hidden_size, num_layers=2,
                            dropout=dropout, bidirectional=True, batch_first=True)
        self.Linear1 = nn.Linear(hidden_size * 2, hidden_size * 2 // 3)
        self.norm2 = torch.nn.LayerNorm(hidden_size * 2 // 3)
        self.relu = torch.nn.LeakyReLU()
        self.drop = torch.nn.Dropout(dropout)
        self.Linear2 = nn.Linear(hidden_size * 2 // 3, num_classes)
        
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
        mask = seq_len_to_mask(seq_len, max_len)
#         print(seq_len)
#         print(seq_len.size())
#         print(x)
#         print(x.size())
#         print(mask.size())
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
        
#         x, _ = self.Rnn(x, seq_len=seq_len)
        x, _ = self.Rnn(x)
        
        x = self.Linear1(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.Linear2(x)
        if tag is not None:
            return self._internal_loss(x, tag)
        else:
            return self._decode(x)

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
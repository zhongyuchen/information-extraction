import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from fastNLP.modules import ConditionalRandomField as CRF
from fastNLP.core.utils import seq_len_to_mask


def _is_transition_allowed(encoding_type, from_tag, from_label, to_tag, to_label):
    """

    :param str encoding_type: 支持"BIO", "BMES", "BEMSO"。
    :param str from_tag: 比如"B", "M"之类的标注tag. 还包括start, end等两种特殊tag
    :param str from_label: 比如"PER", "LOC"等label
    :param str to_tag: 比如"B", "M"之类的标注tag. 还包括start, end等两种特殊tag
    :param str to_label: 比如"PER", "LOC"等label
    :return: bool，能否跃迁
    """
    if to_tag == 'start' or from_tag == 'end':
        return False
    encoding_type = encoding_type.lower()
    if encoding_type == 'bieso':
        if from_tag == 'start':
            return to_tag in ['b', 's', 'o']
        elif from_tag == 'b':
            return to_tag in ['i', 'e'] and from_label == to_label
        elif from_tag == 'i':
            return to_tag in ['i', 'e'] and from_label == to_label
        elif from_tag in ['e', 's', 'o']:
            return to_tag in ['b', 's', 'end', 'o']
        else:
            raise ValueError("Unexpect tag type {}. Expect only 'B', 'I', 'E', 'S', 'O'.".format(from_tag))
    else:
        raise ValueError("Only support BIESO encoding type, got {}.".format(encoding_type))


def allowed_transitions(id2target, encoding_type='BIESO', include_start_end=True):
    """
    别名：:class:`fastNLP.modules.allowed_transitions`  :class:`fastNLP.modules.decoder.crf.allowed_transitions`

    给定一个id到label的映射表，返回所有可以跳转的(from_tag_id, to_tag_id)列表。

    :param dict id2target: key是label的indices，value是str类型的tag或tag-label。value可以是只有tag的, 比如"B", "M"; 也可以是
        "B-NN", "M-NN", tag和label之间一定要用"-"隔开。一般可以通过Vocabulary.idx2word得到id2label。
    :param str encoding_type: 支持"bio", "bmes", "bmeso"。
    :param bool include_start_end: 是否包含开始与结尾的转换。比如在bio中，b/o可以在开头，但是i不能在开头；
        为True，返回的结果中会包含(start_idx, b_idx), (start_idx, o_idx), 但是不包含(start_idx, i_idx);
        start_idx=len(id2label), end_idx=len(id2label)+1。为False, 返回的结果中不含与开始结尾相关的内容
    :return: List[Tuple(int, int)]], 内部的Tuple是可以进行跳转的(from_tag_id, to_tag_id)。
    """
    num_tags = len(id2target)
    start_idx = num_tags
    end_idx = num_tags + 1
    encoding_type = encoding_type.lower()
    allowed_trans = []
    id_label_lst = list(id2target.items())
    if include_start_end:
        id_label_lst += [(start_idx, 'start'), (end_idx, 'end')]

    def split_tag_label(from_label):
        from_label = from_label.lower()
        if from_label in ['start', 'end']:
            from_tag = from_label
            from_label = ''
        else:
            from_tag = from_label[:1]
            from_label = from_label[2:]
        return from_tag, from_label

    for from_id, from_label in id_label_lst:
        if from_label in ['<pad>', '<unk>']:
            continue
        from_tag, from_label = split_tag_label(from_label)
        for to_id, to_label in id_label_lst:
            if to_label in ['<pad>', '<unk>']:
                continue
            to_tag, to_label = split_tag_label(to_label)
            if _is_transition_allowed(encoding_type, from_tag, from_label, to_tag, to_label):
                allowed_trans.append((from_id, to_id))
    return allowed_trans


# class BiLSTM_CRF(nn.Module):
#     def __init__(self, idx2tag):
#         super(BiLSTM_CRF, self).__init__()
#
#         self.crf = CRF(len(idx2tag), allowed_transitions=allowed_transitions(idx2tag))
#
#         self.embedding = nn.Embedding(vocab_size, embed_dim)
#         self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, bidirectional=True, dropout=dropout)
#         self.fc = nn.Linear(hidden_dim * 2, output_dim)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, word, char, pos, spo):
#
#
#         input = input.permute(1, 0)
#         embeds = self.embedding(input)
#         embeds = self.dropout(embeds)
#         # self.lstm.flatten_parameters()
#         output, (hidden, _) = self.lstm(embeds)
#
#         hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
#         hidden = self.dropout(hidden)
#
#         output = self.fc(hidden.squeeze(0))
#         return {"output": output}


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

    def __init__(self, batch_size, word_vocab_size, char_vocab_size, pos_vocab_size, spo_vocab_size,
                 embed_dim, hidden_dim, id2words, dropout=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.word_embeds = nn.Embedding(word_vocab_size, embed_dim)
        self.char_embeds = nn.Embedding(char_vocab_size, embed_dim)
        self.pos_embeds = nn.Embedding(pos_vocab_size, embed_dim)
        self.spo_embeds = nn.Embedding(spo_vocab_size, embed_dim)
        self.norm1 = torch.nn.LayerNorm(embed_dim)
        self.Rnn = nn.LSTM(embed_dim, hidden_dim, num_layers=2,
                           dropout=dropout, bidirectional=True, batch_first=True)
        self.Linear1 = nn.Linear(hidden_dim * 2, hidden_dim * 2 // 3)
        self.norm2 = torch.nn.LayerNorm(hidden_dim * 2 // 3)
        self.relu = torch.nn.LeakyReLU()
        self.drop = torch.nn.Dropout(dropout)
        self.Linear2 = nn.Linear(hidden_dim * 2 // 3, len(id2words))

        self.Crf = CRF(len(id2words), allowed_transitions=allowed_transitions(id2words))

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
        mask = mask.view(batch_size, max_len)
        mask = mask.to(x).float()
        return mask

    def _forward(self, word, char, pos, spo, target=None):
        """
        :param torch.LongTensor words: [batch_size, mex_len]
        :param torch.LongTensor seq_len:[batch_size, ]
        :param torch.LongTensor target: [batch_size, max_len]
        :return y: If truth is None, return list of [decode path(list)]. Used in testing and predicting.
                   If truth is not None, return loss, a scalar. Used in training.
        """
        word = word.long()
        seq_len = torch.tensor(self.batch_size).long()
        print(seq_len, seq_len.dim())
        self.mask = self._make_mask(word, seq_len)

        # seq_len = seq_len.long()
        target = target.long() if target is not None else None

        if next(self.parameters()).is_cuda:
            word = word.cuda()
            self.mask = self.mask.cuda()

        word = self.word_embeds(word)
        char = self.char_embeds(char)
        pos = self.pos_embeds(pos)
        spo = self.spo_embeds(spo)

        x = torch.cat((word, char, pos, spo), 1)
        # x = self.Embedding(x)
        x = self.norm1(x)
        # [batch_size, max_len, word_emb_dim]

        x, _ = self.Rnn(x, seq_len=seq_len)

        x = self.Linear1(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.Linear2(x)
        if target is not None:
            return {"loss": self._internal_loss(x, target)}
        else:
            return {"pred": self._decode(x)}

    def forward(self, word, char, pos, spo):
        """

        :param torch.LongTensor words: [batch_size, mex_len]
        :param torch.LongTensor seq_len: [batch_size, ]
        :param torch.LongTensor target: [batch_size, max_len], 目标
        :return torch.Tensor: a scalar loss
        """
        return self._forward(word, char, pos, spo)

    def predict(self, word, char, pos, spo):
        """

        :param torch.LongTensor words: [batch_size, mex_len]
        :param torch.LongTensor seq_len: [batch_size, ]
        :return torch.LongTensor: [batch_size, max_len]
        """
        return self._forward(word, char, pos, spo)

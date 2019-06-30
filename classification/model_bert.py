import torch
from torch import nn
# from torch.nn import CrossEntropyLosss

import os
import json

from fastNLP.modules.encoder.bert import BertModel, BertEmbeddings, BertEncoder, BertPooler, BertLayerNorm

CONFIG_FILE = 'bert_config.json'


# fastNLP formated class based on BertModel
class BertForMultiLabelSequenceClassification(nn.Module):
    """BERT(Bidirectional Embedding Representations from Transformers).

    如果你想使用预训练好的权重矩阵，请在以下网址下载.
    sources::

    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz",


    用预训练权重矩阵来建立BERT模型::

        model = BertModel.from_pretrained("path/to/weights/directory")

    用随机初始化权重矩阵来建立BERT模型::

        model = BertModel()

    :param int vocab_size: 词表大小，默认值为30522，为BERT English uncase版本的词表大小
    :param int hidden_size: 隐层大小，默认值为768，为BERT base的版本
    :param int num_hidden_layers: 隐藏层数，默认值为12，为BERT base的版本
    :param int num_attention_heads: 多头注意力头数，默认值为12，为BERT base的版本
    :param int intermediate_size: FFN隐藏层大小，默认值是3072，为BERT base的版本
    :param str hidden_act: FFN隐藏层激活函数，默认值为``gelu``
    :param float hidden_dropout_prob: FFN隐藏层dropout，默认值为0.1
    :param float attention_probs_dropout_prob: Attention层的dropout，默认值为0.1
    :param int max_position_embeddings: 最大的序列长度，默认值为512，
    :param int type_vocab_size: 最大segment数量，默认值为2
    :param int initializer_range: 初始化权重范围，默认值为0.02
    """

    def __init__(self, vocab_size=30522,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02, num_labels=50, **kwargs):
        super(BertForMultiLabelSequenceClassification, self).__init__()
        self.embeddings = BertEmbeddings(vocab_size, hidden_size, max_position_embeddings,
                                         type_vocab_size, hidden_dropout_prob)
        self.encoder = BertEncoder(num_hidden_layers, hidden_size, num_attention_heads,
                                   attention_probs_dropout_prob, hidden_dropout_prob, intermediate_size,
                                   hidden_act)
        self.pooler = BertPooler(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.initializer_range = initializer_range

        self.apply(self.init_bert_weights)

    def init_bert_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]

        # return encoded_layers, pooled_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return {"output": logits}

    @classmethod
    def from_pretrained(cls, pretrained_model_dir, state_dict=None, *inputs, **kwargs):
        # Load config
        config_file = os.path.join(pretrained_model_dir, CONFIG_FILE)
        config = json.load(open(config_file, "r"))
        # config = BertConfig.from_json_file(config_file)
        # logger.info("Model config {}".format(config))
        # Instantiate model.
        model = cls(*inputs, **config, **kwargs)
        if state_dict is None:
            weights_path = os.path.join(pretrained_model_dir, MODEL_WEIGHTS)
            state_dict = torch.load(weights_path)

        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(model, prefix='' if hasattr(model, 'bert') else 'bert.')
        if len(missing_keys) > 0:
            print("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            print("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        return model

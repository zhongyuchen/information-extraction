import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

    
class RCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, output_dim, hidden_dim, num_layers, dropout, weight):
        super(RCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(weight))
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, bidirectional=True, dropout=dropout)
        self.linear = nn.Linear(2 * hidden_dim + embed_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # input = input.permute(1, 0, 2)
        embeds = self.embedding(text)
        embeds = embeds.permute(1, 0, 2)
        # embeds = self.dropout(embeds)
        # self.lstm.flatten_parameters()
        output, (hidden, _) = self.lstm(embeds)

        output = torch.cat((output, embeds), 2)
        output = output.permute(1, 0, 2)
        output = self.linear(output).permute(0, 2, 1)

        pool = F.max_pool1d(output, output.size(2)).squeeze(2)
        # hidden = self.dropout(hidden)
        # pool = self.dropout(pool)

        # output = self.fc(hidden.squeeze(0))
        output = self.fc(pool)
        return output
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

    
class LSTM_pooling(nn.Module):
    def __init__(self, vocab_size, embed_dim, output_dim, hidden_dim, num_layers, dropout, weight):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(weight))
        
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        text = text.permute(1, 0)
        embeds = self.embedding(text)
        embeds = self.dropout(embeds)
        output, (hidden, _) = self.lstm(embeds)

        pool = F.max_pool1d(output.permute(1, 2, 0), output.size(0)).squeeze(2)
        pool = self.dropout(pool)

        output = self.fc(pool)
        return output

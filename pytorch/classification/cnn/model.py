import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, class_num, kernel_num, kernel_sizes,
                 dropout, static, in_channels, weight):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = kernel_num
        self.static = static

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.embed.weight.data.copy_(torch.from_numpy(weight))
        self.convs = nn.ModuleList([nn.Conv2d(self.in_channels, self.out_channels, (kernel_size, embed_dim)) for kernel_size in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_sizes) * kernel_num, class_num)

    def forward(self, text):
        x = self.embed(text)
        if self.static:
            x = Variable(x)

        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        output = self.fc(x)
        return output

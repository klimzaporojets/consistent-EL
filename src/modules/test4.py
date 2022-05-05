import torch
import torch.nn as nn


## Simple GRU implementation

class MyGRU(nn.Module):
    def __init__(self, dim_input, config):
        super(MyGRU, self).__init__()
        self.dim_input = dim_input
        self.dim_hidden = 64
        self.input = nn.Linear(self.dim_input + self.dim_hidden, self.dim_hidden)
        self.update = nn.Linear(self.dim_input + self.dim_hidden, self.dim_hidden)
        self.reset = nn.Linear(self.dim_input + self.dim_hidden, self.dim_hidden)
        self.dim = self.dim_hidden

    def forward(self, inputs):
        # print('inputs:', inputs.size()) 
        batch, length, _ = inputs.size()
        h_prev = torch.Tensor(batch, self.dim_hidden).zero_().cuda()
        outputs = []
        for pos in range(length):
            x = torch.cat((inputs[:, pos, :], h_prev), 1)
            z = torch.sigmoid(self.update(x))
            r = torch.sigmoid(self.reset(x))
            y = torch.cat((inputs[:, pos, :], h_prev * r), 1)
            h_next = (1 - z) * h_prev + z * torch.tanh(self.input(y))
            outputs.append(h_next.unsqueeze(1))
            h_prev = h_next
        outputs = torch.cat(outputs, 1)
        return outputs


class Test4(nn.Module):
    def __init__(self, dim_input, config):
        super(Test4, self).__init__()
        self.fw = MyGRU(dim_input, config)
        self.bw = MyGRU(dim_input, config)
        self.dim = self.fw.dim + self.bw.dim
        # self.fw = nn.GRU(dim_input, 64, batch_first=True)
        # self.bw = nn.GRU(dim_input, 64, batch_first=True)
        # self.dim = 128

    def forward(self, inputs, seqlens):
        f = self.fw(inputs)
        b = self.bw(inputs.flip(1))
        out = torch.cat((f, b.flip(1)), 2)
        return out

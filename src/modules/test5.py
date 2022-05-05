import torch
import torch.nn as nn
import torch.nn.functional as F

# from modules.misc import LayerNorm
# from models.misc.misc import LayerNorm
from modules.misc.misc import LayerNorm


class AverageLayer1(nn.Module):
    def __init__(self, w):
        super(AverageLayer1, self).__init__()
        self.w = w

    def forward(self, inputs):
        outputs = inputs
        denom = 1

        for i in range(1, self.w + 1):
            l = F.pad(inputs[:, i:, :], (0, 0, 0, i, 0, 0), "constant", 0)
            r = F.pad(inputs[:, 0:-i, :], (0, 0, i, 0, 0, 0), "constant", 0)
            outputs = l + outputs + r
            denom += 2

        return outputs / denom


class AverageLayer2(nn.Module):
    def __init__(self, w):
        super(AverageLayer2, self).__init__()
        self.avg = nn.AvgPool1d(2 * w + 1, stride=1, padding=w)

    def forward(self, inputs):
        return self.avg(inputs.permute(0, 2, 1)).permute(0, 2, 1)


class MemoryBank(nn.Module):
    def __init__(self, dim):
        super(MemoryBank, self).__init__()
        self.b0 = AverageLayer2(0)
        self.b1 = AverageLayer2(1)
        self.b2 = AverageLayer2(2)
        self.b3 = AverageLayer2(4)
        self.b4 = AverageLayer2(8)
        self.b5 = AverageLayer2(16)
        self.b6 = AverageLayer2(32)
        self.b7 = AverageLayer2(64)
        self.key = nn.Linear(128, dim, bias=False)
        self.value = nn.Linear(128, dim, bias=False)

    def forward(self, inputs):
        m0 = self.b0(inputs).unsqueeze(-2)
        m1 = self.b1(inputs).unsqueeze(-2)
        m2 = self.b2(inputs).unsqueeze(-2)
        m3 = self.b3(inputs).unsqueeze(-2)
        m4 = self.b4(inputs).unsqueeze(-2)
        m5 = self.b5(inputs).unsqueeze(-2)
        m6 = self.b6(inputs).unsqueeze(-2)
        m7 = self.b7(inputs).unsqueeze(-2)
        M = torch.cat((m0, m1, m2, m3, m4, m5, m6, m7), -2)
        # print('inputs:', inputs.size())
        # print('M:', M.size())
        k = self.key(inputs)
        v = self.value(M)
        scores = torch.matmul(v, k.unsqueeze(-1)).squeeze(-1)
        scores = torch.softmax(scores, dim=-1).unsqueeze(-2)
        # print('scores:', scores.size())
        # print(scores[0,0,:,:])
        out = torch.matmul(scores, M).squeeze(-2)
        return out


class MyBlock(nn.Module):
    def __init__(self):
        super(MyBlock, self).__init__()
        self.memory = MemoryBank(32)
        self.linear1 = nn.Linear(128, 128)
        self.linear2 = nn.Linear(128, 128)
        self.ln = LayerNorm(128)

    def forward(self, inputs):
        h = self.memory(inputs)
        h = self.linear1(h)
        h = torch.tanh(h)
        h = self.linear2(h)
        return self.ln(h + inputs)


class BN(nn.Module):
    def __init__(self, dim):
        super(BN, self).__init__()
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, inputs):
        return self.bn(inputs.permute(0, 2, 1)).permute(0, 2, 1)


class Test5(nn.Module):
    def __init__(self, dim_input, config):
        super(Test5, self).__init__()
        self.dim = 128

        layers = [
            nn.Linear(dim_input, self.dim),
            MyBlock(),
            MyBlock(),
            MyBlock(),
            MyBlock(),
            MyBlock()
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs, seqlens):
        outputs = self.layers(inputs)
        return outputs

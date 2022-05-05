import math

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils


class VecLinear(nn.Module):
    def __init__(self, num_input, num_output):
        super(VecLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(num_output, num_input))
        self.reset_parameters()

    def reset_parameters(self):
        print("reset")
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input):
        # print("before:", input.mean(), input.std())
        output = torch.matmul(self.weight, input)
        # print("after:", output.mean(), output.std())
        return output


class EntNet(nn.Module):
    def __init__(self):
        super(EntNet, self).__init__()
        self.num_cell = 16
        self.dim_cell = 32
        self.fc1 = VecLinear(self.num_cell, self.num_cell)
        self.fc2 = nn.Linear(self.dim_cell, self.dim_cell)
        self.dim = self.num_cell * self.dim_cell

    def forward(self, inputs):
        vecs = inputs.view(inputs.size()[:-1] + (self.num_cell, self.dim_cell))
        ents = self.fc1(vecs)
        outputs = self.fc2(ents)
        outputs = outputs.view(*inputs.size())
        return outputs


class EntRNN(nn.Module):

    def __init__(self, dim_input, dim_output, right_to_left):
        super(EntRNN, self).__init__()
        self.input = nn.Linear(dim_input, dim_output)
        self.forget = nn.Linear(dim_input, dim_output)
        from torchqrnn import ForgetMult
        self.gate = ForgetMult()
        self.right_to_left = right_to_left

    def forward(self, inputs, mask):
        if self.right_to_left:
            inputs = inputs.flip(1)
            mask = mask.flip(1)
        x = self.input(inputs)
        f = self.forget(inputs)
        f = f + mask * 10000
        from torchqrnn import ForgetMult
        outputs = ForgetMult()(torch.sigmoid(f), torch.tanh(x))
        if self.right_to_left:
            outputs = outputs.flip(1)
        return outputs


class Layer(nn.Module):
    def __init__(self, dim_input, dim_output):
        super(Layer, self).__init__()
        self.fw = EntRNN(dim_input, dim_output, False)
        self.bw = EntRNN(dim_input, dim_output, True)
        self.dim = dim_output * 2

    def forward(self, inputs, mask):
        f = self.fw(inputs, mask)
        b = self.bw(inputs, mask)
        y = torch.cat((f, b), -1)
        return y


class LayerRNN(nn.Module):
    def __init__(self, dim_input, dim_output):
        super(LayerRNN, self).__init__()
        self.rnn = nn.LSTM(dim_input, dim_output, bidirectional=True, num_layers=1)
        self.dim = dim_output * 2

    def forward(self, inputs, seqlens):
        packed_inputs = rnn_utils.pack_padded_sequence(inputs, seqlens, batch_first=True)
        packed_outputs, _ = self.rnn(packed_inputs)
        outputs, _ = rnn_utils.pad_packed_sequence(packed_outputs, batch_first=True)
        return outputs


class ParallelRNN(nn.Module):
    def __init__(self, dim_input, dim_output):
        super(ParallelRNN, self).__init__()
        self.rnn = nn.LSTM(dim_input, dim_output, bidirectional=True, num_layers=1)
        self.dim = dim_output * 2

    def forward(self, inputs, seqlens):
        b, c, l, d = inputs.size()
        parallel_seqlens = seqlens.unsqueeze(1).expand(b, c).contiguous().view(-1)
        parallel_inputs = inputs.view(-1, l, d)

        packed_inputs = rnn_utils.pack_padded_sequence(parallel_inputs, parallel_seqlens, batch_first=True)
        packed_outputs, _ = self.rnn(packed_inputs)
        outputs, _ = rnn_utils.pad_packed_sequence(packed_outputs, batch_first=True)

        outputs = outputs.view(b, c, l, -1)
        return outputs


def to_cells(tensor, num_cell, dim_cell):
    b, l, _ = tensor.size()
    tensor = tensor.view(b, l, num_cell, dim_cell).permute(0, 2, 1, 3)
    tensor = tensor.contiguous()
    return tensor


def from_cells(tensor):
    b, num_cell, l, dim_cell = tensor.size()
    tensor = tensor.permute(0, 2, 1, 3)
    tensor = tensor.contiguous()
    tensor = tensor.view(b, l, num_cell * dim_cell)
    return tensor


class Test3(nn.Module):
    def __init__(self, dim_input, config):
        super(Test3, self).__init__()
        self.num_cell = config['num_cell']
        self.dim_cell = config['dim_cell']
        self.dim = self.num_cell * self.dim_cell * 2

        self.dense = nn.Linear(dim_input, self.num_cell * self.dim_cell)
        self.layer1 = ParallelRNN(self.dim_cell, self.dim_cell)
        self.dropout1 = nn.Dropout(config['dropout'])
        self.fc1 = nn.Linear(self.dim, self.dim)
        self.layer2 = ParallelRNN(self.layer1.dim, self.dim_cell)

    def forward(self, inputs, seqlens):
        inp = self.dense(inputs)

        inp = to_cells(inp, self.num_cell, self.dim_cell)

        y1 = self.layer1(inp, seqlens)
        y1 = self.dropout1(y1)

        # print(y1.size())
        y1 = from_cells(y1)
        # print(y1.size())
        y1 = self.fc1(y1)
        y1 = to_cells(y1, self.num_cell, 2 * self.dim_cell)
        # print(y1.size())

        y2 = self.layer2(y1, seqlens)

        y = from_cells(y2)

        return y

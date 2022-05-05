import torch
import torch.nn as nn


class RNNMaxpool(nn.Module):

    def __init__(self, dim_input, dim_output):
        super(RNNMaxpool, self).__init__()
        self.rnn = nn.LSTM(dim_input, int(dim_output / 2), num_layers=1, bidirectional=True, batch_first=True)

    def forward(self, inputs):
        outputs, _ = self.rnn(inputs)
        maxpool, _ = torch.max(outputs, 1)
        return maxpool


class CNNMaxpool(nn.Module):

    def __init__(self, dim_input, config):
        super(CNNMaxpool, self).__init__()
        self.cnns = nn.ModuleList([nn.Conv1d(dim_input, config['dim'], k) for k in config['kernels']])
        self.dim_output = config['dim'] * len(config['kernels'])
        self.max_kernel = max(config['kernels'])

    def forward(self, inputs):
        # print('chars:', inputs.size(), inputs[0,0,:,:])

        inp = inputs.view(inputs.size(0) * inputs.size(1), inputs.size(2), inputs.size(3))
        inp = inp.transpose(1, 2)
        outputs = []
        for cnn in self.cnns:
            tmp = cnn(inp)
            # print('tmp[0]:', tmp.size(), tmp[0,:,0])
            # print('tmp[1]:', tmp.size(), tmp[0,:,1])
            maxpool, _ = torch.max(tmp, -1)
            outputs.append(maxpool)
        outputs = torch.cat(outputs, -1)
        result = outputs.view(inputs.size(0), inputs.size(1), -1)
        return result

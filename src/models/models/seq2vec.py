import torch
import torch.nn as nn


class CNNMaxpool(nn.Module):

    def __init__(self, dim_input, config):
        super(CNNMaxpool, self).__init__()
        self.cnns = nn.ModuleList([nn.Conv1d(dim_input, config['dim'], k) for k in config['kernels']])
        self.dim_output = config['dim'] * len(config['kernels'])
        self.max_kernel = max(config['kernels'])

    def forward(self, inputs):
        inp = inputs.view(inputs.size(0) * inputs.size(1), inputs.size(2), inputs.size(3))
        inp = inp.transpose(1, 2)
        outputs = []
        for cnn in self.cnns:
            tmp = cnn(inp)
            maxpool, _ = torch.max(tmp, -1)
            outputs.append(maxpool)
        outputs = torch.cat(outputs, -1)
        result = outputs.view(inputs.size(0), inputs.size(1), -1)
        return result

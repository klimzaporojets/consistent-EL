import torch.nn as nn


def actfnc(name):
    if name == 'tanh':
        return nn.Tanh()
    elif name == 'relu':
        return nn.ReLU()
    else:
        raise BaseException("no such activation function:", name)


class CNN1(nn.Module):

    def __init__(self, dim_input, config):
        super(CNN1, self).__init__()
        layers = []
        dim = dim_input
        for _ in range(config['layers']):
            layers.append(nn.Conv1d(dim, config['dim'], config['w'] * 2 + 1, padding=config['w']))
            layers.append(actfnc(config['actfnc']))
            if config['batchnorm']:
                layers.append(nn.BatchNorm1d(config['dim']))
            if 'dropout' in config:
                layers.append(nn.Dropout(config['dropout']))
            dim = config['dim']
        self.layers = nn.Sequential(*layers)
        self.dim = dim

    def forward(self, inputs, seqlens):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.layers(inputs)
        outputs = outputs.permute(0, 2, 1)
        return outputs


class DilatedBlock(nn.Module):
    def __init__(self, dim_input, config):
        super(DilatedBlock, self).__init__()
        d = 1
        w = 1

        layers = []
        dim = dim_input
        for _ in range(config['depth']):
            layers.append(nn.Conv1d(dim, config['dim'], 3, dilation=d, padding=d))
            layers.append(actfnc(config['actfnc']))
            if config['batchnorm']:
                layers.append(nn.BatchNorm1d(config['dim']))
            if 'dropout' in config:
                layers.append(nn.Dropout(config['dropout']))
            dim = config['dim']
            d = d * 2
        self.layers = nn.Sequential(*layers)
        self.dim = dim

    def forward(self, inputs):
        outputs = self.layers(inputs)
        return outputs


class CNN2(nn.Module):
    def __init__(self, dim_input, config):
        super(CNN2, self).__init__()
        layers = []
        dim = dim_input
        for _ in range(config['layers']):
            block = DilatedBlock(dim, config)
            layers.append(block)
            dim = block.dim
        self.layers = nn.Sequential(*layers)
        self.dim = dim

    def forward(self, inputs, seqlens):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.layers(inputs)
        outputs = outputs.permute(0, 2, 1)
        return outputs


class DilatedBlockX(nn.Module):
    def __init__(self, config):
        super(DilatedBlockX, self).__init__()
        self.dim = config['dim']
        d = 1
        w = 1

        layers = []
        for _ in range(config['depth']):
            layers.append(nn.Conv1d(self.dim, self.dim, 3, dilation=d, padding=d))
            layers.append(actfnc(config['actfnc']))
            if config['batchnorm']:
                layers.append(nn.BatchNorm1d(self.dim))
            if 'dropout' in config:
                layers.append(nn.Dropout(config['dropout']))
            d = d * 2
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs):
        outputs = self.layers(inputs) + inputs
        return outputs


class CNN3(nn.Module):
    def __init__(self, dim_input, config):
        super(CNN3, self).__init__()
        self.dim = config['dim']
        self.linear = nn.Linear(dim_input, self.dim)
        layers = []
        for _ in range(config['layers']):
            block = DilatedBlockX(config)
            layers.append(block)
            dim = block.dim
        self.layers = nn.Sequential(*layers)
        self.dim = dim

    def forward(self, inputs, seqlens):
        inputs = self.linear(inputs)
        inputs = inputs.permute(0, 2, 1)
        outputs = self.layers(inputs)
        outputs = outputs.permute(0, 2, 1)
        return outputs

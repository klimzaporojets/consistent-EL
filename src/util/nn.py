import torch.nn as nn


def create_activation_function(name):
    if name == 'relu':
        return nn.ReLU()
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'glu':
        return nn.GLU()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    else:
        raise BaseException("no such activation function:", name)


class ResidualBlock(nn.Module):
    def __init__(self, block):
        super(ResidualBlock, self).__init__()
        self.block = block

    def forward(self, x):
        return x + self.block(x)


def create_module(dim_input, dim_output, config):
    if config['type'] == 'linear':
        return nn.Linear(dim_input, dim_output)
    elif config['type'] == 'ffnn':
        last = dim_input
        layers = []
        for dim in config['dims']:
            layers.append(nn.Linear(last, dim))
            layers.append(create_activation_function(config['actfnc']))
            if 'dropout' in config:
                layers.append(nn.Dropout(config['dropout']))
            last = dim
        layers.append(nn.Linear(last, dim_output))
        return nn.Sequential(*layers)
    else:
        raise BaseException("unknown module type:", config['type'])


## try dim_hidden = 4 x dim_input
def create_ffn(dim_input, config):
    module = nn.Sequential()
    dim_output = dim_input

    if 'dim' in config:
        dim_hidden = config['dim']

        layers = []
        layers.append(nn.Linear(dim_input, dim_hidden))
        layers.append(create_activation_function(config['actfnc']))

        if config.get('residual', False):
            layers.append(nn.Linear(dim_hidden, dim_input))
            if 'dropout' in config:
                layers.append(nn.Dropout(config['dropout']))
            module = ResidualBlock(nn.Sequential(*layers))
        else:
            if 'dropout' in config:
                layers.append(nn.Dropout(config['dropout']))
            module = nn.Sequential(*layers)
            dim_output = dim_hidden

    return module, dim_output

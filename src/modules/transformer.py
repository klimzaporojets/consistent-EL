import math

import torch
import torch.nn as nn
# from pytorch_pretrained_bert.modeling import BertEncoder
from torch.nn.parameter import Parameter

from util.sequence import get_mask_from_sequence_lengths


class objectview(object):
    def __init__(self, d):
        self.__dict__ = d


class Transformer(nn.Module):

    def __init__(self, dim_input, config):
        super(Transformer, self).__init__()
        config = objectview(config)

        self.linear = nn.Linear(dim_input, config.hidden_size)
        # self.encoder = BertEncoder(config)
        self.encoder = None
        self.dim = config.hidden_size * config.num_hidden_layers

    def forward(self, inputs, seqlens):
        batchsize, maxlen, _ = inputs.size()
        mask = get_mask_from_sequence_lengths(seqlens, maxlen)
        mask = mask.unsqueeze(1).unsqueeze(2)
        mask = mask.float()
        mask = (1.0 - mask) * -10000.0

        inputs = self.linear(inputs)
        output = self.encoder(inputs, mask)
        output = torch.cat(output, 2)
        return output


class TransformerPE(nn.Module):

    def __init__(self, dim_input, config):
        super(TransformerPE, self).__init__()
        config = objectview(config)

        self.dim_pos = config.dim_position_embedding
        self.position_encoding = Parameter(torch.Tensor(500, self.dim_pos))
        nn.init.kaiming_uniform_(self.position_encoding, a=math.sqrt(5))
        self.linear = nn.Linear(dim_input + self.dim_pos, config.hidden_size)
        # self.encoder = BertEncoder(config)
        self.encoder = None
        self.dim = config.hidden_size * config.num_hidden_layers

    def forward(self, inputs, seqlens):
        batchsize, maxlen, _ = inputs.size()
        mask = get_mask_from_sequence_lengths(seqlens, maxlen)
        mask = mask.unsqueeze(1).unsqueeze(2)
        mask = mask.float()
        mask = (1.0 - mask) * -10000.0

        pos = self.position_encoding.unsqueeze(0).expand(batchsize, 500, self.dim_pos)
        pos = pos[:, :maxlen, :]
        inputs = torch.cat((inputs, pos), 2)

        inputs = self.linear(inputs)
        output = self.encoder(inputs, mask)
        output = torch.cat(output, 2)
        return output

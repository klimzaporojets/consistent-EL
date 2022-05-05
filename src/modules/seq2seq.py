import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

from modules.cnn1 import CNN1, CNN2, CNN3
from modules.masked_qrnn import MaskedBiQRNN
from modules.qrnn import BiQRNN
# from modules.mixedrnn import MixingRNN2
from modules.seq2seqs.augmented_lstm import AugmentedLSTM
from modules.test1 import Test1
from modules.test2 import Test2
from modules.test3 import Test3
from modules.test4 import Test4
from modules.test5 import Test5
from modules.transformer import Transformer


def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    # seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


seq2seq_modules = {}


def seq2seq_register(name, factory):
    print("register", name, factory)
    seq2seq_modules[name] = factory


def seq2seq_create(dim_input, config):
    # kzaporoj - TODO - leave it like this with different options?
    if config['type'] == 'none':
        return LayerNone(dim_input, config)
    elif config['type'] == 'layers':
        return RNNLayers(dim_input, config)
    elif config['type'] == 'residual':
        return RNNResidual(dim_input, config)
    elif config['type'] == 'lstm' or config['type'] == 'gru':
        return Seq2seq(dim_input, config)
    elif config['type'] == 'qrnn':
        return BiQRNN(dim_input, config)
    elif config['type'] == 'qrnn2':
        return MaskedBiQRNN(dim_input, config)
    elif config['type'] == 'transformer':
        return Transformer(dim_input, config)
    elif config['type'] == 'test1':
        return Test1(dim_input, config)
    elif config['type'] == 'test2':
        return Test2(dim_input, config)
    elif config['type'] == 'test3':
        return Test3(dim_input, config)
    elif config['type'] == 'test4':
        return Test4(dim_input, config)
    elif config['type'] == 'test5':
        return Test5(dim_input, config)
    elif config['type'] == 'cnn1':
        return CNN1(dim_input, config)
    elif config['type'] == 'cnn2':
        return CNN2(dim_input, config)
    elif config['type'] == 'cnn3':
        return CNN3(dim_input, config)
    elif config['type'] == 'stacked-lstm':
        return StackedLSTM(dim_input, config)
    elif config['type'] == 'multi-out-rnn':
        return MultiOutRNN(dim_input, config)
    elif config['type'] == 'concat-layers-rnn':
        return ConcatLayersRNN(dim_input, config)
    elif config['type'] == 'res-rnn':
        return ResRNN(dim_input, config)
    elif config['type'] == 'split-rnn':
        return SplitRNN(dim_input, config)
    elif config['type'] == 'multitask-1':
        return RNNMultitask1(dim_input, config)
    elif config['type'] == 'augmented-lstm':
        return AugmentedLSTM(dim_input, config)
    elif config['type'] in seq2seq_modules:
        return seq2seq_modules[config['type']](dim_input, config)
    else:
        raise BaseException("no such type", config['type'])


class LayerNone(nn.Module):

    def __init__(self, dim_input, config):
        super(LayerNone, self).__init__()
        self.dim_output = dim_input

    def forward(self, inputs, seqlens, indices=None):
        return inputs


class Seq2seq(nn.Module):

    def __init__(self, dim_input, config):
        super(Seq2seq, self).__init__()
        if 'i_dp' in config:
            self.idp = nn.Dropout(config['i_dp'])
        else:
            self.idp = nn.Sequential()

        if config['type'] == 'lstm':
            self.rnn = nn.LSTM(dim_input, config['dim'], bidirectional=True, num_layers=config['layers'],
                               dropout=config['dropout'], batch_first=True)
        elif config['type'] == 'gru':
            self.rnn = nn.GRU(dim_input, config['dim'], bidirectional=True, num_layers=config['layers'],
                              dropout=config['dropout'], batch_first=True)

        print("WARNING:WDROP COMMENTED OUT")
        # if 'wdrop' in config:
        #     wdrop_params = [k for k,v in self.rnn.named_parameters() if k.startswith('weight_hh')]
        #     print("WDROP ENABLED:", config['wdrop'], wdrop_params)
        #     self.rnn = WeightDrop(self.rnn, wdrop_params, config['wdrop'])

        self.dim_output = config['dim'] * 2

        self.concat_input_output = config['concat_input_output']
        if self.concat_input_output:
            self.dim_output += dim_input

        self.dim = self.dim_output

    def forward(self, inputs, seqlens, indices=None):
        # print('inputs:', inputs.size())
        inputs = self.idp(inputs)
        # print('inputs device in seq2seq: ', inputs.device)
        # print('seqlens device in seq2seq: ', seqlens.device)
        # (kzaporoj 21/12/2020) - adding .cpu() , for some reason gives
        #   - RuntimeError: 'lengths' argument should be a 1D CPU int64 tensor, but got 1D cuda:0 Long tensor
        #   I checked this erros and seems like .cpu() on the lengthsshould solve it according to:
        #
        packed_inputs = rnn_utils.pack_padded_sequence(inputs, seqlens.cpu(), batch_first=True)
        # the previous call (without .cpu()) was:
        # packed_inputs = rnn_utils.pack_padded_sequence(inputs, seqlens, batch_first=True)
        packed_outputs, _ = self.rnn(packed_inputs)
        outputs, _ = rnn_utils.pad_packed_sequence(packed_outputs, batch_first=True)

        if self.concat_input_output:
            outputs = torch.cat((outputs, inputs), -1)

        return outputs


class StackedLSTM(nn.Module):

    def __init__(self, dim_input, config):
        super(StackedLSTM, self).__init__()
        self.rnn1 = nn.LSTM(dim_input, config['dim'], bidirectional=True, num_layers=1, batch_first=True)
        self.dropout1 = nn.Dropout(config['dropout'])
        self.rnn2 = nn.LSTM(config['dim'] * 2, config['dim'], bidirectional=True, num_layers=1, batch_first=True)
        self.dim = config['dim'] * 2

    def forward(self, inputs, seqlens):
        # print('inputs:', inputs.size())
        packed_inputs1 = rnn_utils.pack_padded_sequence(inputs, seqlens, batch_first=True)
        packed_outputs1, _ = self.rnn1(packed_inputs1)
        outputs1, _ = rnn_utils.pad_packed_sequence(packed_outputs1, batch_first=True)

        inputs2 = self.dropout1(outputs1)

        packed_inputs2 = rnn_utils.pack_padded_sequence(inputs2, seqlens, batch_first=True)
        packed_outputs2, _ = self.rnn2(packed_inputs2)
        outputs2, _ = rnn_utils.pad_packed_sequence(packed_outputs2, batch_first=True)

        return outputs1 + outputs2


class ResLayer(nn.Module):

    def __init__(self, config):
        super(ResLayer, self).__init__()
        dim = config['dim']

        if config['rnn'] == 'lstm':
            self.rnn = nn.LSTM(dim * 2, dim, bidirectional=True, num_layers=1, batch_first=True)
        elif config['rnn'] == 'gru':
            self.rnn = nn.GRU(dim * 2, dim, bidirectional=True, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(config['dropout'])
        from modules.misc import LayerNorm
        self.out = LayerNorm(dim * 2) if 'ln' in config else nn.Sequential()

    def forward(self, inputs, seqlens, indices=None):
        # print('inputs:', inputs.size())
        packed_inputs = rnn_utils.pack_padded_sequence(inputs, seqlens, batch_first=True)
        packed_outputs, _ = self.rnn(packed_inputs)
        outputs, _ = rnn_utils.pad_packed_sequence(packed_outputs, batch_first=True)
        outputs = inputs + self.dropout(outputs)
        return self.out(outputs)


class ResRNN(nn.Module):

    def __init__(self, dim_input, config):
        super(ResRNN, self).__init__()
        self.dim_output = config['dim'] * 2

        if dim_input != self.dim_output:
            self.pre = nn.Linear(dim_input, self.dim_output)
        else:
            self.pre = nn.Sequential()

        layers = []
        for _ in range(config['layers']):
            layers.append(ResLayer(config))

        self.layers = nn.ModuleList(layers)

    def forward(self, inputs, seqlens, indices=None):
        outputs = self.pre(inputs)
        for layer in self.layers:
            outputs = layer(outputs, seqlens, indices)
        return outputs


class MultiOutRNN(nn.Module):

    def __init__(self, dim_input, config):
        super(MultiOutRNN, self).__init__()
        layers = []
        dim_last = dim_input
        for i in range(config['layers']):
            layers.append(nn.GRU(dim_last, config['dim'], bidirectional=True, num_layers=1, batch_first=True))
            dim_last = config['dim'] * 2
        self.rnns = nn.ModuleList(layers)
        self.weights = nn.Linear(config['layers'], config['outputs'], bias=False)
        self.dim = config['dim'] * 2

    def forward(self, inputs, seqlens, indices=None):
        outputs = []
        for rnn in self.rnns:
            packed_inputs = rnn_utils.pack_padded_sequence(inputs, seqlens, batch_first=True)
            packed_outputs, _ = rnn(packed_inputs)
            inputs, _ = rnn_utils.pad_packed_sequence(packed_outputs, batch_first=True)
            outputs.append(inputs)
        outputs = torch.stack(outputs, -1)
        outputs = self.weights(outputs)
        return outputs


class ConcatLayersRNN(nn.Module):

    def __init__(self, dim_input, config):
        super(ConcatLayersRNN, self).__init__()
        layers = []
        dim_last = dim_input
        self.concat_input = config['concat-input']
        for i in range(config['layers']):
            layers.append(nn.GRU(dim_last, config['dim'], bidirectional=True, num_layers=1, batch_first=True))
            dim_last = config['dim'] * 2
        self.rnns = nn.ModuleList(layers)
        self.dim_output = (dim_input if config['concat-input'] else 0) + config['dim'] * 2 * config['layers']

    def forward(self, inputs, seqlens, indices=None):
        outputs = []
        if self.concat_input:
            outputs.append(inputs)
        for rnn in self.rnns:
            packed_inputs = rnn_utils.pack_padded_sequence(inputs, seqlens, batch_first=True)
            packed_outputs, _ = rnn(packed_inputs)
            inputs, _ = rnn_utils.pad_packed_sequence(packed_outputs, batch_first=True)
            outputs.append(inputs)
        outputs = torch.cat(outputs, -1)
        return outputs


class SplitRNN(nn.Module):

    def __init__(self, dim_input, config):
        super(SplitRNN, self).__init__()
        self.num_outputs = config['outputs']
        self.rnn = nn.GRU(dim_input, config['dim'] * self.num_outputs, bidirectional=True, num_layers=config['layers'],
                          batch_first=True)
        self.dim_output = config['dim'] * 2
        self.dim_hidden = config['dim']
        print("SplitRNN:", self.dim_output, self.dim_hidden)

    def forward(self, inputs, seqlens, indices=None):
        packed_inputs = rnn_utils.pack_padded_sequence(inputs, seqlens, batch_first=True)
        packed_outputs, _ = self.rnn(packed_inputs)
        outputs, _ = rnn_utils.pad_packed_sequence(packed_outputs, batch_first=True)

        splits = torch.split(outputs, self.dim_hidden, dim=-1)

        outputs = []
        for i in range(self.num_outputs):
            outputs.append(torch.cat((splits[i], splits[self.num_outputs + i]), -1))  # concat fw and bw
        return tuple(outputs)


def seq(rnn, inputs, seqlens):
    packed_inputs = rnn_utils.pack_padded_sequence(inputs, seqlens, batch_first=True)
    packed_outputs, _ = rnn(packed_inputs)
    outputs, _ = rnn_utils.pad_packed_sequence(packed_outputs, batch_first=True)
    return outputs


class RNNMultitask1(nn.Module):

    def __init__(self, dim_input, config):
        super(RNNMultitask1, self).__init__()
        self.dim_output = config['dim'] * 2
        self.rnn_ner = nn.GRU(dim_input, config['dim'], bidirectional=True, num_layers=1, batch_first=True)
        self.rnn_coref = nn.GRU(dim_input, config['dim'], bidirectional=True, num_layers=1, batch_first=True)
        self.rnn_rel = nn.GRU(dim_input, config['dim'], bidirectional=True, num_layers=1, batch_first=True)
        self.gate_nc = nn.Linear(self.dim_output, self.dim_output)
        self.gate_nr = nn.Linear(self.dim_output, self.dim_output)
        self.gate_cn = nn.Linear(self.dim_output, self.dim_output)
        self.gate_cr = nn.Linear(self.dim_output, self.dim_output)
        self.gate_rn = nn.Linear(self.dim_output, self.dim_output)
        self.gate_rc = nn.Linear(self.dim_output, self.dim_output)

    def forward(self, inputs, seqlens, indices=None):
        if isinstance(inputs, tuple):
            x_ner, x_coref, x_rel = inputs
        else:
            x_ner, x_coref, x_rel = inputs, inputs, inputs

        h_ner = seq(self.rnn_ner, x_ner, seqlens)
        h_coref = seq(self.rnn_coref, x_coref, seqlens)
        h_rel = seq(self.rnn_rel, x_rel, seqlens)

        out_ner = h_ner + torch.sigmoid(self.gate_nc(h_ner)) * h_coref + torch.sigmoid(self.gate_nr(h_ner)) * h_rel
        out_coref = h_coref + torch.sigmoid(self.gate_cn(h_coref)) * h_ner + torch.sigmoid(
            self.gate_cr(h_coref)) * h_rel
        out_rel = h_rel + torch.sigmoid(self.gate_rn(h_rel)) * h_ner + torch.sigmoid(self.gate_rc(h_rel)) * h_coref

        return out_ner, out_coref, out_rel


class RNNResidual(nn.Module):

    def __init__(self, dim_input, config):
        super(RNNResidual, self).__init__()
        self.layer = seq2seq_create(dim_input, config['layer'])
        self.dim_output = dim_input

    def forward(self, inputs, seqlens, indices=None):
        hiddens = self.layer(inputs, seqlens)
        if isinstance(hiddens, tuple):
            if isinstance(inputs, tuple):
                tmp = [x + y for x, y in zip(inputs, hiddens)]
            else:
                tmp = [inputs + y for y in hiddens]
            return tuple(tmp)
        else:
            return inputs + hiddens


class LayerSplit(nn.Module):

    def __init__(self, dim_input, split):
        super(LayerSplit, self).__init__()
        if isinstance(split, list):
            self.dim_hidden = []
            self.dim_hidden.extend([int(x / 2) for x in split])
            self.dim_hidden.extend([int(x / 2) for x in split])
            self.dim_outputs = split
        else:
            dim_hidden = int(dim_input / (2 * split))
            self.dim_hidden = [dim_hidden for _ in range(split)]
            self.dim_outputs = [2 * dim_hidden for _ in range(split)]
        self.num_outputs = len(self.dim_outputs)
        print('hidden:', self.dim_hidden)
        print('output:', self.num_outputs)

    def forward(self, inputs):
        splits = torch.split(inputs, self.dim_hidden, dim=-1)

        outputs = []
        for i in range(self.num_outputs):
            outputs.append(torch.cat((splits[i], splits[self.num_outputs + i]), -1))  # concat fw and bw

        return tuple(outputs)


class RNNLayers(nn.Module):

    def __init__(self, dim_input, config):
        super(RNNLayers, self).__init__()

        if 'linear' in config:
            self.linear = nn.Linear(dim_input, config['linear'])
            dim_last = config['linear']
        else:
            self.linear = nn.Sequential()
            dim_last = dim_input

        layers = []
        for _ in range(config['layers']):
            layer = seq2seq_create(dim_last, config['layer'])
            layers.append(layer)
            dim_last = layer.dim_output
        self.layers = nn.ModuleList(layers)
        self.dim_output = dim_last

        if 'split' in config:
            print("WARNING: split enabled")
            self.split = LayerSplit(self.dim_output, config['split'])
            self.dim_output = self.split.dim_outputs
        else:
            self.split = nn.Sequential()

    def forward(self, inputs, seqlens, indices=None):
        outputs = self.linear(inputs)
        for layer in self.layers:
            outputs = layer(outputs, seqlens)
        return self.split(outputs)

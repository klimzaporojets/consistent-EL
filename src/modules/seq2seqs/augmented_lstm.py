import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
# from allennlp.modules.stacked_bidirectional_lstm import StackedBidirectionalLstm


class AugmentedLSTM(nn.Module):

    def __init__(self, dim_input, config):
        super(AugmentedLSTM, self).__init__()
        self.dim_output = config['dim'] * 2
        # self.rnn = StackedBidirectionalLstm(dim_input, config['dim'], config['layers'])
        self.rnn = None

    def forward(self, inputs, seqlens, indices=None):
        packed_inputs = rnn_utils.pack_padded_sequence(inputs, seqlens, batch_first=True)
        packed_outputs, _ = self.rnn(packed_inputs)
        outputs, _ = rnn_utils.pad_packed_sequence(packed_outputs, batch_first=True)
        return outputs

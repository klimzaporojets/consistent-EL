import torch
import torch.nn as nn
from torch.autograd import Variable

from util.sequence import get_mask_from_sequence_lengths


class MaskedQRNNLayer(nn.Module):
    r"""Applies a single layer Quasi-Recurrent Neural Network (QRNN) to an input sequence.

    Args:
        input_size: The number of expected features in the input x.
        hidden_size: The number of features in the hidden state h. If not specified, the input size is used.
        save_prev_x: Whether to store previous inputs for use in future convolutional windows (i.e. for a continuing sequence such as in language modeling). If true, you must call reset to remove cached previous values of x. Default: False.
        window: Defines the size of the convolutional window (how many previous tokens to look when computing the QRNN values). Supports 1 and 2. Default: 1.
        zoneout: Whether to apply zoneout (i.e. failing to update elements in the hidden state) to the hidden state updates. Default: 0.
        output_gate: If True, performs QRNN-fo (applying an output gate to the output). If False, performs QRNN-f. Default: True.
        use_cuda: If True, uses fast custom CUDA kernel. If False, uses naive for loop. Default: True.

    Inputs: X, hidden
        - X (seq_len, batch, input_size): tensor containing the features of the input sequence.
        - hidden (batch, hidden_size): tensor containing the initial hidden state for the QRNN.

    Outputs: output, h_n
        - output (seq_len, batch, hidden_size): tensor containing the output of the QRNN for each timestep.
        - h_n (batch, hidden_size): tensor containing the hidden state for t=seq_len
    """

    def __init__(self, input_size, hidden_size=None, save_prev_x=False, zoneout=0, window=1, output_gate=True,
                 use_cuda=True):
        super(MaskedQRNNLayer, self).__init__()

        assert window in [1,
                          2], "This QRNN implementation currently only handles convolutional window of size 1 or size 2"
        self.window = window
        self.input_size = input_size
        self.hidden_size = hidden_size if hidden_size else input_size
        self.zoneout = zoneout
        self.save_prev_x = save_prev_x
        self.prevX = None
        self.output_gate = output_gate
        self.use_cuda = use_cuda

        # One large matmul with concat is faster than N small matmuls and no concat
        self.linear = nn.Linear(self.window * self.input_size,
                                3 * self.hidden_size if self.output_gate else 2 * self.hidden_size)

    def reset(self):
        # If you are saving the previous value of x, you should call this when starting with a new state
        self.prevX = None

    def forward(self, X, seqmask, hidden=None):
        seq_len, batch_size, _ = X.size()

        source = None
        if self.window == 1:
            source = X
        elif self.window == 2:
            # Construct the x_{t-1} tensor with optional x_{-1}, otherwise a zeroed out value for x_{-1}
            Xm1 = []
            Xm1.append(self.prevX if self.prevX is not None else X[:1, :, :] * 0)
            # Note: in case of len(X) == 1, X[:-1, :, :] results in slicing of empty tensor == bad
            if len(X) > 1:
                Xm1.append(X[:-1, :, :])
            Xm1 = torch.cat(Xm1, 0)
            # Convert two (seq_len, batch_size, hidden) tensors to (seq_len, batch_size, 2 * hidden)
            source = torch.cat([X, Xm1], 2)

        # Matrix multiplication for the three outputs: Z, F, O
        Y = self.linear(source)
        # Convert the tensor back to (batch, seq_len, len([Z, F, O]) * hidden_size)
        if self.output_gate:
            Y = Y.view(seq_len, batch_size, 3 * self.hidden_size)
            Z, F, O = Y.chunk(3, dim=2)
        else:
            Y = Y.view(seq_len, batch_size, 2 * self.hidden_size)
            Z, F = Y.chunk(2, dim=2)
        ###
        Z = torch.nn.functional.tanh(Z)
        F = torch.nn.functional.sigmoid(F)

        seqmask = seqmask.unsqueeze(2).expand_as(F)
        F = F * seqmask

        # If zoneout is specified, we perform dropout on the forget gates in F
        # If an element of F is zero, that means the corresponding neuron keeps the old value
        if self.zoneout:
            if self.training:
                mask = Variable(F.data.new(*F.size()).bernoulli_(1 - self.zoneout), requires_grad=False)
                F = F * mask
            else:
                F *= 1 - self.zoneout

        # Ensure the memory is laid out as expected for the CUDA kernel
        # This is a null op if the tensor is already contiguous
        Z = Z.contiguous()
        F = F.contiguous()
        # The O gate doesn't need to be contiguous as it isn't used in the CUDA kernel

        # Forget Mult
        # For testing QRNN without ForgetMult CUDA kernel, C = Z * F may be useful
        from torchqrnn.forget_mult import ForgetMult
        C = ForgetMult()(F, Z, hidden, use_cuda=self.use_cuda)

        # Apply (potentially optional) output gate
        if self.output_gate:
            H = torch.nn.functional.sigmoid(O) * C
        else:
            H = C
        H = H * seqmask

        # In an optimal world we may want to backprop to x_{t-1} but ...
        if self.window > 1 and self.save_prev_x:
            self.prevX = Variable(X[-1:, :, :].data, requires_grad=False)

        return H, C[-1:, :, :]


class MaskedBiQRNNLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MaskedBiQRNNLayer, self).__init__()
        self.qrnn_fw = MaskedQRNNLayer(input_size, hidden_size)
        self.qrnn_bw = MaskedQRNNLayer(input_size, hidden_size)

    def forward(self, input):
        X, mask = input
        Yfw, _ = self.qrnn_fw(X, mask)
        Ybw, _ = self.qrnn_bw(X.flip(0), mask.flip(0))
        Y = torch.cat((Yfw, Ybw.flip(0)), 2)
        return Y, mask


class MyDropout(nn.Module):

    def __init__(self, dropout):
        super(MyDropout, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        X, mask = input
        Y = self.dropout(X)
        return Y, mask


class MaskedBiQRNN(nn.Module):
    def __init__(self, input_size, config):
        super(MaskedBiQRNN, self).__init__()
        print("BiQRNN: batch_first only and no sequence_length")
        num_layers = config['layers']
        hidden_size = config['dim']
        dropout = config['dropout']

        layers = []
        last_size = input_size
        for i in range(num_layers):
            layers.append(MaskedBiQRNNLayer(last_size, hidden_size))
            if i < num_layers - 1:
                layers.append(MyDropout(dropout))
            last_size = hidden_size * 2

        self.dim = hidden_size * 2
        self.layers = nn.Sequential(*layers)

    def forward(self, X, seqlens):
        mask = get_mask_from_sequence_lengths(seqlens, X.size(1)).permute(1, 0).float()
        X = X.permute(1, 0, 2)
        Y, _ = self.layers((X, mask))
        return Y.permute(1, 0, 2)

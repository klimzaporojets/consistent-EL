import torch
import torch.nn as nn


class BiQRNNLayer(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0):
        super(BiQRNNLayer, self).__init__()
        from torchqrnn import QRNN
        self.qrnn_fw = QRNN(input_size, hidden_size, 1, dropout=dropout)
        self.qrnn_bw = QRNN(input_size, hidden_size, 1, dropout=dropout)

    def forward(self, X):
        Yfw, _ = self.qrnn_fw(X)
        Ybw, _ = self.qrnn_bw(X.flip(0))
        Y = torch.cat((Yfw, Ybw.flip(0)), 2)
        return Y


class BiQRNN(nn.Module):
    def __init__(self, input_size, config):
        super(BiQRNN, self).__init__()
        print("BiQRNN: batch_first only and no sequence_length")
        num_layers = config['layers']
        hidden_size = config['dim']
        dropout = config['dropout']

        layers = []
        last_size = input_size
        for i in range(num_layers):
            layers.append(BiQRNNLayer(last_size, hidden_size, dropout=dropout))
            last_size = hidden_size * 2

        self.dim = hidden_size * 2
        self.layers = nn.Sequential(*layers)

    def forward(self, X, seqlens):
        X = X.permute(1, 0, 2)
        return self.layers(X).permute(1, 0, 2)

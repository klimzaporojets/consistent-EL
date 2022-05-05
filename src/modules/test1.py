import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import math
import torchvision

from modules.test2 import MyLayerNorm
from util.sequence import get_mask_from_sequence_lengths


class MySelfAttention(nn.Module):

    def __init__(self, dim_input, config):
        super(MySelfAttention, self).__init__()
        self.num_heads = config['num_heads']
        self.dim_heads = config['dim_heads']
        self.dim = self.num_heads * self.dim_heads

        self.query = nn.Linear(dim_input, self.dim)
        self.key = nn.Linear(dim_input, self.dim)
        self.value = nn.Linear(dim_input, self.dim)
        self.dropout = nn.Dropout(config["att_dropout"])

        self.counter = 0

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.dim_heads)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, inputs, attention_mask):
        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)

        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)

        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.dim_heads)
        attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        if self.counter % 100 == 0:
            print("WRITING ATTENTION", attention_probs.size())
            for h in range(self.num_heads):
                torchvision.utils.save_image(attention_probs[0, h, :, :], 'head-{}.png'.format(h))
        self.counter += 1

        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, v)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.dim,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

class GlobalAttention(nn.Module):

    def __init__(self, dim_input, config):
        super(GlobalAttention, self).__init__()
        self.dim_attention = config['dim_attention']
        self.dim = dim_input

        self.query = nn.Linear(dim_input, self.dim_attention)
        self.key = nn.Linear(dim_input, self.dim_attention)
        self.dropout = nn.Dropout(config["dp_attention"])

        self.counter = 0

    def forward(self, inputs, attention_mask):
        q = self.query(inputs)
        k = self.key(inputs)
        v = inputs

        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        # attention_scores = attention_scores / math.sqrt(self.dim_heads)
        attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        if self.counter % 100 == 0:
            print("WRITING ATTENTION", attention_probs.size())
            torchvision.utils.save_image(attention_probs[0, :, :], 'attention.png')
        self.counter += 1

        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, v)
        return context_layer


class MySelfOutput(nn.Module):
    def __init__(self, config):
        super(MySelfOutput, self).__init__()
        self.dense = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.LayerNorm = MyLayerNorm(config['hidden_size'], eps=1e-12)
        self.dropout = nn.Dropout(config['hidden_dropout_prob'])

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class MyAttention(nn.Module):
    def __init__(self, dim_input, config):
        super(MyAttention, self).__init__()
        self.self = MySelfAttention(dim_input, config)
        self.output = MySelfOutput(config)
        self.dim = config['hidden_size']

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class MyLayer(nn.Module):
    def __init__(self, dim_input, config):
        super(MyLayer, self).__init__()
        self.attention = MyAttention(dim_input, config)
        self.fc1 = nn.Linear(config['hidden_size'], config['intermediate_size'])
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(config['intermediate_size'], config['hidden_size'])
        self.dropout = nn.Dropout(config['hidden_dropout_prob'])
        self.LayerNorm = MyLayerNorm(config['hidden_size'], eps=1e-12)
        self.dim = config['hidden_size']

    def forward(self, input_tensor, attention_mask):
        attention_output = self.attention(input_tensor, attention_mask)
        intermediate_output = self.act(self.fc1(attention_output))
        layer_output = self.dropout(self.fc2(intermediate_output))
        layer_output = self.LayerNorm(layer_output + attention_output)
        return layer_output


class Test1(nn.Module):

    def __init__(self, dim_input, config):
        super(Test1, self).__init__()
        print("Test1")
        self.rnn1 = nn.LSTM(dim_input, config['enc_dim'], bidirectional=True, num_layers=config['enc_layers'], dropout=config['dropout'])
        # self.att = MySelfAttention(config['enc_dim']*2, config)
        self.att = GlobalAttention(config['enc_dim']*2, config)
        #self.att = MyAttention(config['dim']*2, config)
        #self.att = MyLayer(config['dim'] * 2, config)
        #
        self.rnn2 = nn.LSTM(dim_input+self.att.dim+config['enc_dim'] * 2, config['dec_dim'], bidirectional=True, num_layers=config['dec_layers'], dropout=config['dropout'])
        self.dim = config['dec_dim']*2
        self.dropout = nn.Dropout(config["dropout"])

    def forward(self, inputs, seqlens):
        batchsize, maxlen, _ = inputs.size()
        mask = get_mask_from_sequence_lengths(seqlens, maxlen)
        # mask = mask.unsqueeze(1).unsqueeze(2)
        mask = mask.unsqueeze(1)
        mask = mask.float()
        mask = (1.0 - mask) * -10000.0

        packed_inputs = rnn_utils.pack_padded_sequence(inputs, seqlens, batch_first=True)
        packed_outputs, _ = self.rnn1(packed_inputs)
        outputs1, _ = rnn_utils.pad_packed_sequence(packed_outputs, batch_first=True)
        outputs1 = self.dropout(outputs1)
        # print('lstm1_out:', outputs1.size())

        outputs2 = self.att(outputs1, mask)
        # print('att_out:', outputs2.size())

        outputs = torch.cat((inputs,outputs1,outputs2),-1)
        # outputs = outputs2

        # print('outputs:', outputs.size())

        packed_inputs = rnn_utils.pack_padded_sequence(outputs, seqlens, batch_first=True)
        packed_outputs, _ = self.rnn2(packed_inputs)
        outputs, _ = rnn_utils.pad_packed_sequence(packed_outputs, batch_first=True)

        # print('lstm2_out:', outputs.size())

        return outputs

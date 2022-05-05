import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):

    def __init__(self, dim_input, config):
        super(SelfAttention, self).__init__()
        self.num_heads = config['num_heads']
        self.dim_heads = config['dim_heads']
        self.dim = self.num_heads * self.dim_heads

        self.query = nn.Linear(dim_input, self.dim)
        self.key = nn.Linear(dim_input, self.dim)
        self.value = nn.Linear(dim_input, self.dim)
        self.dropout = nn.Dropout(config["att_dropout"])

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.dim_heads)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, inputs):
        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)

        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)

        print('selfq:', q.size())
        print('selfq:', k.size())
        print('selfq:', v.size())

        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.dim_heads)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)

        print('probs:', attention_probs.size())

        context_layer = torch.matmul(attention_probs, v)
        print('conext:', context_layer.size())
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        print('conext:', context_layer.size())
        new_context_layer_shape = context_layer.size()[:-2] + (self.dim,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class MyAttention(nn.Module):
    def __init__(self, dim_input, config):
        super(MyAttention, self).__init__()
        self.w1 = 7
        self.w2 = 2 * self.w1 + 1
        self.dim_input = dim_input
        self.num_heads = 4
        self.dim_heads = 64
        self.dim_output = self.num_heads * self.dim_heads
        self.query = nn.Linear(self.dim_input, 4 * 64)
        self.key = nn.Linear(self.dim_input, 4 * 64)
        self.value = nn.Linear(self.dim_input, 4 * 64)
        # self.test = SelfAttention(114, config)
        self.position_embeddings = nn.Embedding(self.w2, self.dim_input)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.dim_heads)
        x = x.view(*new_x_shape)
        return x.permute(0, 1, 3, 2, 4)

    def forward(self, inputs):
        # print('before:', inputs.size())

        # self.test(inputs)
        window = F.pad(inputs, (0, 0, self.w1, self.w1), "constant", 0)
        window = window.unfold(1, self.w2, 1).permute(0, 1, 3, 2)

        position_ids = torch.arange(self.w2, dtype=torch.long, device=inputs.device)
        position_embeddings = self.position_embeddings(position_ids)
        window = window + position_embeddings

        q = self.query(inputs).unsqueeze(2)
        k = self.query(window)
        v = self.value(window)

        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)

        # print('q:', q.size())
        # print('k:', k.size())
        # print('v:', v.size())

        scores = torch.matmul(q, k.transpose(-1, -2))
        probs = scores
        context = torch.matmul(probs, v)
        new_shape = context.size()[0:-3] + (self.dim_output,)
        context = context.view(new_shape)

        return context


class MyLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(MyLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class MyLayer(nn.Module):
    def __init__(self, dim_input, config):
        super(MyLayer, self).__init__()
        self.attention = MyAttention(dim_input, config)
        self.dim_output = self.attention.dim_output
        self.fc1 = nn.Linear(self.dim_output, self.dim_output)
        self.nonlin = nn.Tanh()
        self.fc2 = nn.Linear(self.dim_output, self.dim_output)
        self.dropout = nn.Dropout(config['dropout'])
        self.norm = MyLayerNorm(self.dim_output)

    def forward(self, inputs):
        attention_outputs = self.attention(inputs)
        h = self.fc1(attention_outputs)
        h = self.nonlin(h)
        h = self.fc2(h)
        h = self.dropout(h)
        return self.norm(attention_outputs + h)


class Test2(nn.Module):
    def __init__(self, dim_input, config):
        super(Test2, self).__init__()
        self.layer1 = MyLayer(dim_input, config)
        self.layer2 = MyLayer(self.layer1.dim_output, config)
        self.layer3 = MyLayer(self.layer2.dim_output, config)
        self.layer4 = MyLayer(self.layer3.dim_output, config)
        self.dim = self.layer2.dim_output

    def forward(self, inputs, seqlens):
        outputs = self.layer1(inputs)
        outputs = self.layer2(outputs)
        outputs = self.layer3(outputs)
        outputs = self.layer4(outputs)
        return outputs

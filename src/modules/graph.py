# from modules.graphnet.generic import ModuleFF, ModuleDot, ModuleSimplified
# from modules.graphnet.nodes import ModuleNodes

# kzaporoj - commented because some parts might not be needed
graph_modules = {}


def graph_register(name, factory):
    print("register", name, factory)
    graph_modules[name] = factory


def create_graph(dim_input, dim_output, config):
    # if config['type'] == 'simplified':
    #     return ModuleSimplified(dim_input, dim_output, config)
    # elif config['type'] == 'revff':
    #     return ModuleRevFF(dim_input, dim_output, config)
    # elif config['type'] == 'gdot':
    #     return ModuleGDot(dim_input, dim_output, config)
    # elif config['type'] == 'biaffine':
    #     return ModuleBiAffine(dim_input, dim_output, config)
    # elif config['type'] == 'iter2':
    #     return ModuleIter2(dim_input, dim_output, config)
    # elif config['type'] == 'edges':
    #     return ModuleEdges(dim_input, dim_output, config)
    # elif config['type'] in graph_modules:
    #     return graph_modules[config['type']](dim_input, dim_output, config)
    # else:
    #     raise BaseException("no such module:", config['type'])
    raise BaseException("no such module:", config['type'])
#
#
# class ModuleGDot(nn.Module):
#     def __init__(self, dim_input, num_relations, config):
#         super(ModuleGDot, self).__init__()
#         self.dim_relation = config['dim']
#         self.num_relations = num_relations
#         self.left_gate = nn.Linear(dim_input, self.num_relations * self.dim_relation)
#         self.right_gate = nn.Linear(dim_input, self.num_relations * self.dim_relation)
#         self.left = nn.Linear(dim_input, self.num_relations * self.dim_relation, bias=False)
#         self.right = nn.Linear(dim_input, self.num_relations * self.dim_relation, bias=False)
#         self.bias = nn.Parameter(torch.Tensor(self.num_relations))
#         self.bias.data.zero_()
#         print("BIAS:", self.bias)
#
#     def forward(self, inputs):
#         shape = inputs.size()[:-1] + (self.num_relations, self.dim_relation)
#         lg = self.left_gate(inputs).view(shape).permute(0, 2, 1, 3)
#         rg = self.right_gate(inputs).view(shape).permute(0, 2, 3, 1)
#         l = self.left(inputs).view(shape).permute(0, 2, 1, 3)
#         r = self.right(inputs).view(shape).permute(0, 2, 3, 1)
#         inspect("lg:", lg)
#         inspect("rg:", rg)
#         inspect("left:", l)
#         inspect("right:", r)
#
#         scores = torch.matmul(torch.sigmoid(lg), r) + torch.matmul(l, torch.sigmoid(rg))
#         b = self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(2)
#         return scores + b
#
#
# class ModuleBiAffine(nn.Module):
#
#     def __init__(self, dim_input, dim_output, config):
#         super(ModuleBiAffine, self).__init__()
#         self.left = FeedForward(dim_input, config['ff'])
#         self.right = FeedForward(dim_input, config['ff'])
#         self.out_left = nn.Linear(self.left.dim_output, dim_output, bias=False)
#         self.out_bilinear = MyBilinear(self.left.dim_output, self.right.dim_output, dim_output)
#         self.out_right = nn.Linear(self.left.dim_output, dim_output, bias=False)
#         self.dim_output = dim_output
#
#     def forward(self, inputs, mask=None):
#         dims = (inputs.size()[0], inputs.size()[1], inputs.size()[1], self.dim_output)
#         l = self.left(inputs)
#         r = self.right(inputs)
#         out = self.out_bilinear(l, r)
#         out = self.out_left(l).unsqueeze(-3) + out + self.out_right(r).unsqueeze(-2)
#         return out
#
#
# class ModuleLinear(nn.Module):
#     def __init__(self, dim_input, dim_output, config=None):
#         super(ModuleLinear, self).__init__()
#         self.left = nn.Linear(dim_input, dim_output)
#         self.right = nn.Linear(dim_input, dim_output)
#
#     def forward(self, inputs, mask=None):
#         l = self.left(inputs).unsqueeze(-2)
#         r = self.right(inputs).unsqueeze(-3)
#         return l + r
#
#
# def inspect(name, x):
#     print("{}:\t{} - {}  avg:{} std:{}".format(name, x.min().item(), x.max().item(), x.mean().item(), x.std().item()))
#
#
# # most basic form
# class IterLayer1a(nn.Module):
#     def __init__(self, dim_input, dim_output, config):
#         super(IterLayer1a, self).__init__()
#         self.scorer = ModuleFF(dim_input, dim_output, config)
#
#     def forward(self, inputs, scores, mask):
#         return inputs, scores + self.scorer(inputs, mask)
#
#
# class IterLayer1(nn.Module):
#     def __init__(self, dim_input, dim_output, config):
#         super(IterLayer1, self).__init__()
#         self.bidirectional = config['bidirectional']
#         print("bidirectional:", self.bidirectional)
#         f = 3 if self.bidirectional else 2
#
#         self.weight = nn.Linear(dim_output, dim_input)
#         self.scorer = ModuleFF(f * dim_input, dim_output, config)
#         if 'alpha' in config:
#             self.alpha = config['alpha']
#             self.beta = 1 - self.alpha
#         else:
#             self.alpha = 1.0
#             self.beta = 1.0
#
#     def forward(self, inputs, scores, mask):
#         probs = torch.sigmoid(scores + (1 - mask.unsqueeze(-1)) * -100000)
#         # inspect("probs", probs)
#         g = self.weight(probs)
#
#         if self.bidirectional:
#             x = inputs.unsqueeze(1).expand_as(g)
#             y = x * g
#             u = y.sum(-2)
#             v = y.sum(-3)  # THIS HAS TO BE WRONG !!!
#             x = torch.cat((inputs, u, v), -1)
#         else:
#             x = inputs.unsqueeze(1).expand_as(g)
#             u = (x * g).sum(-2)
#             x = torch.cat((inputs, u), -1)
#
#         # return scores + self.scorer(x, mask)
#         return inputs, self.alpha * scores + self.beta * self.scorer(x, mask)
#
#
# class IterLayer1Fix(nn.Module):
#     def __init__(self, dim_input, dim_output, config):
#         super(IterLayer1Fix, self).__init__()
#         self.weight1 = nn.Linear(dim_output, dim_input)
#         self.weight2 = nn.Linear(dim_output, dim_input)
#         self.scorer = ModuleFF(3 * dim_input, dim_output, config)
#         print("WRONG")
#
#     def forward(self, inputs, scores, mask):
#         probs = torch.sigmoid(scores + (1 - mask.unsqueeze(-1)) * -100000)
#         # inspect("probs", probs)
#         g1 = self.weight1(probs)
#         g2 = self.weight2(probs)
#
#         # print('inputs1:', inputs.unsqueeze(1).size(), g1.size())
#         # print('inputs2:', inputs.unsqueeze(2).size(), g2.size())
#
#         y1 = inputs.unsqueeze(1).expand_as(g1) * g1  # [batch, 1, length, dim] -> [batch, length*, length, dim]
#         y2 = inputs.unsqueeze(2).expand_as(g2) * g2  # [batch, length, 1, dim] -> [batch, length, length*, dim]
#         # u = y1.sum(-2)
#         # v = y2.sum(-3)
#         u = y1.sum(-3)
#         v = y2.sum(-2)
#         x = torch.cat((inputs, u, v), -1)
#
#         return inputs, scores + self.scorer(x, mask)
#
#
# class IterLayer1Wrong(nn.Module):
#
#     def __init__(self, dim_input, dim_output, config):
#         super(IterLayer1Wrong, self).__init__()
#         self.weight1 = nn.Linear(dim_output, dim_input)
#         self.weight2 = nn.Linear(dim_output, dim_input)
#         self.scorer = ModuleFF(3 * dim_input, dim_output, config)
#         self.simplified = config['simplified']
#         print("SIMPLIFIED:", self.simplified)
#
#     def forward(self, inputs, scores, mask):
#         probs = torch.sigmoid(scores + (1 - mask.unsqueeze(-1)) * -100000)
#         # inspect("probs", probs)
#         g1 = self.weight1(probs)
#         g2 = self.weight2(probs)
#
#         if self.simplified:
#             u = g1.sum(-3)
#             v = g2.sum(-2)
#         else:
#             u = inputs * g1.sum(-3)
#             v = inputs * g2.sum(-2)
#
#         x = torch.cat((inputs, u, v), -1)
#
#         return inputs, scores + self.scorer(x, mask)
#
#
# # masking & normalization !!!
# class IterLayer1Final(nn.Module):
#
#     def __init__(self, dim_input, dim_output, config):
#         super(IterLayer1Final, self).__init__()
#         self.weight1 = nn.Linear(dim_output, dim_input)
#         self.weight2 = nn.Linear(dim_output, dim_input)
#         self.scorer = ModuleFF(3 * dim_input, dim_output, config)
#         self.normalize = config['normalize']
#         self.ln = config['ln']
#         if self.ln:
#             self.ln1 = LayerNorm(dim_input)
#             self.ln2 = LayerNorm(dim_input)
#
#         print("normalize:", self.normalize)
#         print("lnx:", self.ln)
#
#     def forward(self, inputs, scores, mask):
#         probs = torch.sigmoid(scores + (1 - mask.unsqueeze(-1)) * -100000)
#         # inspect("probs", probs)
#         g1 = self.weight1(probs) * mask.unsqueeze(-1)
#         g2 = self.weight2(probs) * mask.unsqueeze(-1)
#
#         if self.ln:
#             u = inputs * self.ln1(g1.sum(-3))
#             v = inputs * self.ln2(g2.sum(-2))
#         else:
#             u = inputs * g1.sum(-3)
#             v = inputs * g2.sum(-2)
#
#         if self.normalize:
#             normalization = (mask.sum(-1)[:, 0] + 1e-9).unsqueeze(1).unsqueeze(2)
#             u = u / normalization
#             v = v / normalization
#
#         x = torch.cat((inputs, u, v), -1)
#
#         return inputs, scores + self.scorer(x, mask)
#
#
# class IterLayerTest(nn.Module):
#
#     def __init__(self, dim_input, dim_output, config):
#         super(IterLayerTest, self).__init__()
#         self.weight1 = nn.Linear(dim_output, dim_input)
#         self.weight2 = nn.Linear(dim_output, dim_input)
#         self.scorer = ModuleFF(3 * dim_input, dim_output, config)
#         # print("tanh enabled, no element-wise mult")
#         print("try again, with normalization")
#
#     def forward(self, inputs, scores, mask):
#         probs = torch.sigmoid(scores + (1 - mask.unsqueeze(-1)) * -100000)
#         g1 = self.weight1(probs)
#         g2 = self.weight2(probs)
#
#         # u = inputs * g1.sum(-3)
#         # v = inputs * g2.sum(-2)
#         # x = torch.cat((u, v), -1)
#
#         # u = inputs * torch.tanh(g1).sum(-3)
#         # v = inputs * torch.tanh(g2).sum(-2)
#         # x = torch.cat((inputs, u, v), -1)
#
#         # u = torch.tanh(g1).sum(-3)
#         # v = torch.tanh(g2).sum(-2)
#         # x = torch.cat((inputs, u, v), -1)
#
#         u = (inputs.unsqueeze(-2) * (g1 * mask.unsqueeze(-1))).sum(-3)  # should be correct
#         v = (inputs.unsqueeze(-3) * (g2 * mask.unsqueeze(-1))).sum(-2)
#         normalization = (mask.sum(-1)[:, 0] + 1e-9).unsqueeze(1).unsqueeze(2)
#         x = torch.cat((inputs, u / normalization, v / normalization), -1)
#
#         return inputs, scores + self.scorer(x, mask)
#
#
# def softmax_sum(S, H):
#     P = F.softmax(S, -2)
#     S = P.transpose(2, 3)
#     s = S.view((-1,) + S.size()[-2:])
#     h = H.view((-1,) + H.size()[-2:])
#     y = torch.bmm(s, h)
#     return y.view(S.size()[0:2] + h.size()[-1:])
#
#
# class LayerExperiment1(nn.Module):
#     def __init__(self, dim_input, dim_output, config=None):
#         super(LayerExperiment1, self).__init__()
#         self.dim_hidden = config['dim']
#         self.ln = LayerNorm(dim_input)
#         self.left = nn.Linear(dim_input, 3 * self.dim_hidden)
#         self.right = nn.Linear(dim_input, 3 * self.dim_hidden)
#         self.att = nn.Linear(self.dim_hidden, 1)
#         self.xxx = nn.Linear(3 * self.dim_hidden, self.dim_hidden)
#         self.out = nn.Linear(self.dim_hidden, dim_output)
#
#     def forward(self, inputs, scores, mask):
#         n_b, n_l, n_i = inputs.size()
#         inputs = self.ln(inputs)
#
#         l = self.left(inputs).unsqueeze(-2)
#         r = self.right(inputs).unsqueeze(-3)
#         h = torch.relu(l + r)
#         h1, h2, h = torch.split(h, self.dim_hidden, -1)
#
#         a1 = self.att(h1) - (1 - mask.unsqueeze(-1)) * 1e20
#         a2 = self.att(h2) - (1 - mask.unsqueeze(-1)) * 1e20
#         c1 = softmax_sum(a1, h)
#         c2 = softmax_sum(a2.transpose(1, 2), h)
#
#         c1 = c1.unsqueeze(-2).expand(h.size())
#         c2 = c2.unsqueeze(-3).expand(h.size())
#         outputs = torch.cat((h, c1, c2), -1)
#         outputs = torch.relu(self.xxx(outputs))
#         return inputs, self.out(outputs)
#
#
# class LayerExperiment2(nn.Module):
#     def __init__(self, dim_input, dim_output, config=None):
#         super(LayerExperiment2, self).__init__()
#         self.dim_hidden = config['dim']
#         self.dim_hidden2 = config['dim2']
#         self.ln = LayerNorm(dim_input)
#         self.dp = nn.Dropout(config['dp'])
#         self.left = nn.Linear(dim_input, self.dim_hidden)
#         self.right = nn.Linear(dim_input, self.dim_hidden)
#         self.att = nn.Linear(self.dim_hidden, 1)
#         self.xxx = nn.Linear(2 * dim_input, self.dim_hidden2)
#         self.out = nn.Linear(self.dim_hidden2, dim_output)
#         print("OK2, no attention")
#
#     def forward(self, inputs, scores, mask):
#         n_b, n_l, n_i = inputs.size()
#         inputs = self.ln(inputs)
#         inputs = self.dp(inputs)
#
#         l = self.left(inputs).unsqueeze(-2)
#         r = self.right(inputs).unsqueeze(-3)
#         h = torch.relu(l + r)
#
#         # a = self.att(h) - (1 - mask.unsqueeze(-1)) * 1e20
#         # a = a.squeeze(-1)
#         # c1 = weighted_sum(inputs, F.softmax(a, -1))
#         # c2 = weighted_sum(inputs, F.softmax(a.transpose(1, 2), -1))
#
#         dims = (n_b, n_l, n_l, n_i)
#         i1 = inputs.unsqueeze(-2).expand(dims)
#         i2 = inputs.unsqueeze(-3).expand(dims)
#         # c1 = c1.unsqueeze(-2).expand(dims)
#         # c2 = c2.unsqueeze(-3).expand(dims)
#         # outputs = torch.cat((i1, i2, c1, c2), -1)
#         outputs = torch.cat((i1, i2), -1)
#         outputs = torch.relu(self.xxx(outputs))
#         return inputs, self.out(outputs)

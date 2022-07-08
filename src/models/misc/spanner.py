import logging
from collections import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from misc import settings
from models.misc.misc import batched_index_select

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger()


def create_all_spans(batch_size, length, width):
    """

    :param batch_size: example: {int} 1
    :param length: example: {int} 69
    :param width: example: {int} 5
    :return:
    """
    b = torch.arange(length, dtype=torch.long)
    w = torch.arange(width, dtype=torch.long)
    e = b.unsqueeze(-1) + w.unsqueeze(0)
    b = b.unsqueeze(-1).expand_as(e)

    b = b.unsqueeze(0).expand((batch_size,) + b.size())
    e = e.unsqueeze(0).expand((batch_size,) + e.size())
    return b, e


class SpanEndpointSpanBert(nn.Module):
    """
    The original idea is that, unlike SpanEndpoint, it accepts directly the masked spans.
    Also some extra stuff from https://github.com/lxucs/coref-hoi
    """

    def make_linear(self, in_features, out_features, bias=True, std=0.02):
        linear = nn.Linear(in_features, out_features, bias)
        init.normal_(linear.weight, std=std)
        if bias:
            init.zeros_(linear.bias)
        return linear

    def make_ffnn(self, feat_size, hidden_size, output_size):
        if hidden_size is None or hidden_size == 0 or hidden_size == [] or hidden_size == [0]:
            return self.make_linear(feat_size, output_size)

        if not isinstance(hidden_size, Iterable):
            hidden_size = [hidden_size]
        ffnn = [self.make_linear(feat_size, hidden_size[0]), nn.ReLU(), self.dropout]
        for i in range(1, len(hidden_size)):
            ffnn += [self.make_linear(hidden_size[i - 1], hidden_size[i]), nn.ReLU(), self.dropout]
        ffnn.append(self.make_linear(hidden_size[-1], output_size))
        return nn.Sequential(*ffnn)

    def __init__(self, dim_input, max_span_length, config):
        super(SpanEndpointSpanBert, self).__init__()
        self.span_embed = 'span_embed' in config
        self.dim_output = 2 * dim_input
        self.dim_input = dim_input
        self.span_average = config['average']
        self.dropout = nn.Dropout(p=config['dropout'])

        if self.span_embed:
            self.embed = nn.Embedding(max_span_length, config['span_embed'])
            # adapted from hoi, initialization with std 0.02
            init.normal_(self.embed.weight, std=0.02)
            self.dim_output += config['span_embed']

        if self.span_average:
            self.dim_output += dim_input

        if 'ff_dim' in config:
            self.ff = self.make_ffnn(self.dim_output, 0, output_size=config['ff_dim'])
            self.dim_output = config['ff_dim']
        else:
            self.ff = nn.Sequential()

        if 'attention_heads' in config and config['attention_heads']:
            self.attention_heads = True
            self.mention_token_attn = self.make_ffnn(self.dim_input, 0, output_size=1)
            self.dim_output += dim_input
        else:
            self.attention_heads = False

    def forward(self, inputs, b, e, max_width):
        # inputs.shape --> [1, 96, 768]; b.shape --> [1, 315]; e.shape --> [1, 315]; max_width --> 15
        b_vec = batched_index_select(inputs, b)
        # b_vec.shape --> [1, 315, 768]

        # e_vec = batched_index_select(inputs, torch.clamp(e, max=inputs.size(1) - 1))
        e_vec = batched_index_select(inputs, torch.clamp(e, max=inputs.size(1) - 1))
        # e_vec.shape --> [1, 315, 768]

        vecs = [b_vec, e_vec]

        if self.span_embed:
            candidate_width_idx = e - b
            candidate_width_emb = self.embed(e - b)
            candidate_width_emb = self.dropout(candidate_width_emb)
            vecs.append(candidate_width_emb)

        if self.attention_heads:
            # TODO: only will work for batch size of 1!
            curr_batch = 0
            num_candidates = b[curr_batch].shape[0]

            token_attn = torch.squeeze(self.mention_token_attn(inputs[curr_batch]), 1)

            num_subtokens = inputs[curr_batch].shape[0]  # num_words
            candidate_tokens = torch.unsqueeze(torch.arange(0, num_subtokens,
                                                            device=settings.device), 0).repeat(num_candidates, 1)
            candidate_tokens_mask = (candidate_tokens >= torch.unsqueeze(b[curr_batch], 1)) & \
                                    (candidate_tokens <= torch.unsqueeze(e[curr_batch], 1))

            candidate_tokens_attn_raw = torch.log(candidate_tokens_mask.float()) + \
                                        torch.unsqueeze(token_attn, 0)

            candidate_tokens_attn = nn.functional.softmax(candidate_tokens_attn_raw, dim=1)

            head_attn_emb = torch.matmul(candidate_tokens_attn, inputs[curr_batch])
            head_attn_emb.unsqueeze_(0)
            vecs.append(head_attn_emb)

        if self.span_average:
            vecs.append(span_average(inputs, b, e, max_width))

        vec = torch.cat(vecs, -1)
        return self.ff(vec), candidate_width_idx


def span_average(inputs, b, e, max_width):
    w = torch.arange(max_width).to(b.device)
    indices = b.unsqueeze(-1) + w.unsqueeze(0).unsqueeze(0)
    vectors = batched_index_select(inputs, torch.clamp(indices, max=inputs.size(1) - 1))

    mask = (indices <= e.unsqueeze(-1)).float()
    lengths = mask.sum(-1)
    probs = mask / lengths.unsqueeze(-1)
    output = torch.matmul(probs.unsqueeze(-2), vectors).squeeze(-2)
    return output

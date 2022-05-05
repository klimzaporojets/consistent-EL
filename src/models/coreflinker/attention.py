import torch
import torch.nn as nn
import torch.nn.functional as F
from models.coreflinker.scorers import create_pair_scorer


# plain attention before ffnn
class ModulePlainAttention(nn.Module):

    def __init__(self, dim_span, dim_output, span_pair_generator, config, squeeze=False):
        super(ModulePlainAttention, self).__init__()
        self.hidden_dim = config['hidden_dim']      # 150
        self.hidden_dp = config['hidden_dropout']   # 0.4
        self.squeeze = squeeze

        print("ModulePlainAttention")
        self.attention = create_pair_scorer(dim_span, 1, config, span_pair_generator)
        self.scorer = create_pair_scorer(dim_span*2, dim_output, config, span_pair_generator)

    def forward(self, all_spans, filtered_spans, sequence_lengths):
        update = filtered_spans['span_vecs']
        filtered_span_begin = filtered_spans['span_begin']
        filtered_span_end = filtered_spans['span_end']
        square_mask = filtered_spans['square_mask']

        att = self.attention(update, filtered_span_begin, filtered_span_end).squeeze(-1)
        probs = F.softmax(att - (1.0 - square_mask) * 1e23, dim=-1)
        ctxt = torch.matmul(probs, update)

        update = torch.cat((update,ctxt), -1)

        scores = self.scorer(update, filtered_span_begin, filtered_span_end)
        scores = scores.squeeze(-1) if self.squeeze else scores

        return all_spans, filtered_spans, scores
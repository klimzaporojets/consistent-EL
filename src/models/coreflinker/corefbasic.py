import torch.nn as nn

from modules.utils.misc import coref_add_scores
from models.coreflinker.scorers import OptFFpairs


class ModuleCorefBasic(nn.Module):

    def __init__(self, dim_span, coref_pruner, span_pair_generator, config):
        super(ModuleCorefBasic, self).__init__()

        print("ModuleCorefBasic")
        self.scorer = OptFFpairs(dim_span, 1, config, span_pair_generator)

    def forward(self, all_spans, filtered_spans, sequence_lengths):
        update = filtered_spans['span_vecs']
        filtered_span_begin = filtered_spans['span_begin']
        filtered_span_end = filtered_spans['span_end']

        coref_scores = self.scorer(update, filtered_span_begin, filtered_span_end).squeeze(-1)
        coref_scores = coref_add_scores(coref_scores, filtered_spans['span_scores'])

        return all_spans, filtered_spans, coref_scores

import torch.nn as nn

from modules.utils.misc import relation_add_scores
from models.coreflinker.scorers import OptFFpairs


class ModuleRelBasic(nn.Module):

    def __init__(self, dim_span, span_pair_generator, labels, config):
        super(ModuleRelBasic, self).__init__()

        print("ModuleRelBasic()")
        self.scorer = OptFFpairs(dim_span, len(labels), config, span_pair_generator)

    def forward(self, all_spans, filtered_spans, sequence_lengths):
        update = filtered_spans['span_vecs']
        filtered_span_begin = filtered_spans['span_begin']
        filtered_span_end = filtered_spans['span_end']
        square_mask = filtered_spans['square_mask']
        span_lengths = filtered_spans['span_lengths']

        relation_scores = self.scorer(update, filtered_span_begin, filtered_span_end)
        relation_scores = relation_add_scores(relation_scores, filtered_spans['span_scores'])

        return all_spans, filtered_spans, relation_scores

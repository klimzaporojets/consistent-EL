import torch.nn as nn
from modules.graph import create_graph


class ModuleRelSimple(nn.Module):

    def __init__(self, dim_span, labels, config):
        super(ModuleRelSimple, self).__init__()
        print('ModuleRelSimple:', config)
        self.scorer = create_graph(dim_span, len(labels), config['scorer'])

    def forward(self, all_spans, filtered_spans, sequence_lengths):
        update = filtered_spans['span_vecs']
        filtered_span_begin = filtered_spans['span_begin']
        filtered_span_end = filtered_spans['span_end']
        square_mask = filtered_spans['square_mask']
        span_lengths = filtered_spans['span_lengths']

        _, relation_scores = self.scorer(update, square_mask)

        return all_spans, filtered_spans, relation_scores

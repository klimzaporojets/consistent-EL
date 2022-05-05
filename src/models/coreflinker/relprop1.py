import torch
import torch.nn as nn

from modules.utils.misc import MyGate, relation_add_scores, overwrite_spans
from models.coreflinker.scorers import OptFFpairs


# relprop with a pruner
class ModuleRelProp1(nn.Module):

    def __init__(self, dim_span, shared_pruner, span_pair_generator, labels, config):
        super(ModuleRelProp1, self).__init__()
        self.rel_prop = config['rel_prop']

        print("ModuleRelProp1: creating relation pruner")
        self.pruner = shared_pruner.create_new()

        self.scorer = OptFFpairs(dim_span, len(labels), config, span_pair_generator)
        self.A = nn.Linear(len(labels), dim_span, bias=False)
        self.gate = MyGate(dim_span)

    def forward(self, all_spans, filtered_spans, sequence_lengths):
        all_spans, filtered_spans = self.pruner(all_spans, sequence_lengths)

        update = filtered_spans['span_vecs']
        filtered_span_begin = filtered_spans['span_begin']
        filtered_span_end = filtered_spans['span_end']
        square_mask = filtered_spans['square_mask']
        span_lengths = filtered_spans['span_lengths']

        update_all = all_spans.copy()
        update_filtered = filtered_spans.copy()

        relation_scores = self.scorer(update, filtered_span_begin, filtered_span_end)
        relation_scores = relation_add_scores(relation_scores, filtered_spans['span_scores'])

        if self.rel_prop > 0:
            for _ in range(self.rel_prop):
                probs = torch.relu(relation_scores) * square_mask.unsqueeze(-1)
                ctxt = (self.A(probs) * update.unsqueeze(-2)).sum(-3) / span_lengths.unsqueeze(-1).unsqueeze(-1)
                update = self.gate(update, ctxt)

                relation_scores = self.scorer(update, filtered_span_begin, filtered_span_end)
                relation_scores = relation_add_scores(relation_scores, self.pruner.scorer(update))

            update_filtered['span_vecs'] = update
            update_all['span_vecs'] = overwrite_spans(update_all['span_vecs'], filtered_spans['prune_indices'],
                                                      span_lengths, update)

        return update_all, update_filtered, relation_scores

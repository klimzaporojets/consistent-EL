import torch
import torch.nn as nn

from modules.utils.misc import MyGate, overwrite_spans, relation_add_scores
from models.coreflinker.scorers import OptFFpairs


# RelProp without a separate pruner
class ModuleRelProp(nn.Module):

    def __init__(self, dim_span, span_pair_generator, labels, config):
        super(ModuleRelProp, self).__init__()
        self.rel_prop = config['rel_prop']
        self.add_pruner_scores = config['add_pruner_scores']

        print("ModuleRelProp(rp={})".format(self.rel_prop, self.add_pruner_scores))

        self.scorer = OptFFpairs(dim_span, len(labels), config, span_pair_generator)
        self.A = nn.Linear(len(labels), dim_span, bias=False)
        self.gate = MyGate(dim_span)

    def forward(self, all_spans, filtered_spans, sequence_lengths):
        update = filtered_spans['span_vecs']
        filtered_span_begin = filtered_spans['span_begin']
        filtered_span_end = filtered_spans['span_end']
        square_mask = filtered_spans['square_mask']
        span_lengths = filtered_spans['span_lengths']

        update_all = all_spans.copy()
        update_filtered = filtered_spans.copy()

        relation_scores = self.scorer(update, filtered_span_begin, filtered_span_end)
        if self.add_pruner_scores:
            relation_scores = relation_add_scores(relation_scores, filtered_spans['span_scores'])

        if self.rel_prop > 0:
            # print('before:', update.norm().item())

            for _ in range(self.rel_prop):
                probs = torch.relu(relation_scores) * square_mask.unsqueeze(-1)
                ctxt = (self.A(probs) * update.unsqueeze(-2)).sum(-3) / span_lengths.unsqueeze(-1).unsqueeze(-1)
                update = self.gate(update, ctxt)

                relation_scores = self.scorer(update, filtered_span_begin, filtered_span_end)
                if self.add_pruner_scores:
                    relation_scores = relation_add_scores(relation_scores, filtered_spans['span_scores'])

            # print('after:', update.norm().item())

            update_filtered['span_vecs'] = update
            update_all['span_vecs'] = overwrite_spans(update_all['span_vecs'], filtered_spans['prune_indices'],
                                                      span_lengths, update)

        return update_all, update_filtered, relation_scores


# bidirectional version (incoming and outgoing relations), +sigmoid
class ModuleRelPropX(nn.Module):

    def __init__(self, dim_span, span_pair_generator, labels, config):
        super(ModuleRelPropX, self).__init__()
        self.rel_prop = config['rel_prop']

        print("ModuleRelPropX(rp={})".format(self.rel_prop))

        self.scorer = OptFFpairs(dim_span, len(labels), config, span_pair_generator)
        self.A = nn.Linear(len(labels), dim_span, bias=False)
        self.B = nn.Linear(len(labels), dim_span, bias=False)
        self.gate = MyGate(dim_span)

    def forward(self, all_spans, filtered_spans, sequence_lengths):
        update = filtered_spans['span_vecs']
        filtered_span_begin = filtered_spans['span_begin']
        filtered_span_end = filtered_spans['span_end']
        square_mask = filtered_spans['square_mask']
        span_lengths = filtered_spans['span_lengths']

        update_all = all_spans.copy()
        update_filtered = filtered_spans.copy()

        relation_scores = self.scorer(update, filtered_span_begin, filtered_span_end)

        if self.rel_prop > 0:
            # print('before:', update.norm().item())

            for _ in range(self.rel_prop):
                probs = torch.sigmoid(relation_scores) * square_mask.unsqueeze(-1)
                # A(probs):             [batch, length, length, dim_span]
                # update.unsqueeze(-2): [batch, length, 1, dim_spans]
                # TODO: try different form of normalization?
                ctxt1 = (self.A(probs) * update.unsqueeze(-2)).sum(-3) / span_lengths.unsqueeze(-1).unsqueeze(-1)
                ctxt2 = (self.B(probs) * update.unsqueeze(-3)).sum(-2) / span_lengths.unsqueeze(-1).unsqueeze(-1)
                update = self.gate(update, ctxt1 + ctxt2)

                relation_scores = self.scorer(update, filtered_span_begin, filtered_span_end)

            # print('after:', update.norm().item())

            update_filtered['span_vecs'] = update
            update_all['span_vecs'] = overwrite_spans(update_all['span_vecs'], filtered_spans['prune_indices'],
                                                      span_lengths, update)

        return update_all, update_filtered, relation_scores

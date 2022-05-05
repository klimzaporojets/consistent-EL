import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.utils.misc import MyGate, coref_add_scores, overwrite_spans
from models.coreflinker.scorers import OptFFpairs


# bidirectional version
class ModuleCorefProp2(nn.Module):

    def __init__(self, dim_span, coref_pruner, span_pair_generator, config):
        super(ModuleCorefProp2, self).__init__()
        self.coref_prop = config['coref_prop']
        self.update_coref_scores = config['update_coref_scores']

        print("ModuleCorefProp2(cp={})".format(self.coref_prop))

        self.coref_pruner = coref_pruner
        self.coref = OptFFpairs(dim_span, 1, config, span_pair_generator)
        self.gate1 = MyGate(dim_span)
        self.gate2 = MyGate(dim_span)

    def forward(self, all_spans, filtered_spans, sequence_lengths):
        update = filtered_spans['span_vecs']
        filtered_span_begin = filtered_spans['span_begin']
        filtered_span_end = filtered_spans['span_end']
        triangular_mask1 = filtered_spans['triangular_mask']
        triangular_mask2 = filtered_spans['triangular_mask'].permute(0, 2, 1)

        coref_scores = self.coref(update, filtered_span_begin, filtered_span_end).squeeze(-1)
        coref_scores = coref_add_scores(coref_scores, filtered_spans['span_scores'])

        update_all = all_spans.copy()
        update_filtered = filtered_spans.copy()

        if self.coref_prop > 0:

            for _ in range(self.coref_prop):
                probs1 = F.softmax(coref_scores - (1.0 - triangular_mask1) * 1e23, dim=-1)
                probs2 = F.softmax(coref_scores - (1.0 - triangular_mask2) * 1e23, dim=-1)

                ctxt1 = torch.matmul(probs1, update)
                ctxt2 = torch.matmul(probs2, update)

                update = self.gate1(update, ctxt1) + self.gate2(update, ctxt2)

                if self.update_coref_scores:
                    coref_scores = self.coref(update, filtered_span_begin, filtered_span_end).squeeze(-1)
                    coref_scores = coref_add_scores(coref_scores, self.coref_pruner(update))

            update_filtered['span_vecs'] = update

            update_all['span_vecs'] = overwrite_spans(update_all['span_vecs'], filtered_spans['prune_indices'],
                                                      filtered_spans['span_lengths'], update)

        return update_all, update_filtered, coref_scores

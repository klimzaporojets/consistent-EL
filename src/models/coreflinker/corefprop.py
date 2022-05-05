import torch
import torch.nn as nn
import torch.nn.functional as F

from models.coreflinker.scorers import OptFFpairs
from modules.utils.misc import MyGate, coref_add_scores, overwrite_spans


class ModuleCorefProp(nn.Module):

    def __init__(self, dim_span, coref_pruner, span_pair_generator, config):
        super(ModuleCorefProp, self).__init__()
        self.coref_prop = config['corefprop']['coref_prop']
        self.update_coref_scores = config['corefprop']['update_coref_scores']
        # self.shared_pruner = config['shared_pruner']

        print("ModuleCorefProp(cp={})".format(self.coref_prop))

        self.coref_pruner = coref_pruner
        self.coref = OptFFpairs(dim_span, 1, config['corefprop'], span_pair_generator)
        self.gate = MyGate(dim_span)
        # self.end_to_end = config['end_to_end']

    def forward(self, all_spans, filtered_spans, gold_spans, max_span_length=0, gold_spans_lengths=0):
        # filtered_spans_rest = None

        # if self.end_to_end:
        update = filtered_spans['span_vecs']
        # update.shape --> [1, 21, 2324]

        if update is None:
            return all_spans, filtered_spans, None

        filtered_span_begin = filtered_spans['span_begin']
        # filtered_span_begin.shape --> [1,21]
        filtered_span_end = filtered_spans['span_end']
        # filtered_span_end.shape --> [1,21]
        triangular_mask = filtered_spans['triangular_mask']
        # triangular_mask.shape --> [1,21,21]
        filtered_spans_rest = filtered_spans

        # filtered_spans_rest.keys() --> dict_keys(['prune_indices_hoi', 'span_vecs', 'span_scores', 'span_begin',
        # 'span_end', 'span_lengths', 'square_mask', 'triangular_mask', 'pruned_spans', 'gold_spans', 'enabled_spans'])
        #

        coref_scores = self.coref(update, filtered_span_begin, filtered_span_end).squeeze(-1)

        coref_scores = coref_add_scores(coref_scores, filtered_spans['span_scores'])

        update_all = all_spans.copy()
        update_filtered = filtered_spans_rest.copy()

        if self.coref_prop > 0:
            for _ in range(self.coref_prop):
                probs = F.softmax(coref_scores - (1.0 - triangular_mask) * 1e23, dim=-1)
                # coref_scores.shape --> [1,21,21] ; triangular_mask.shape --> [1,21,21];
                # triangular_mask.sum --> 231.0 (float); probs.shape --> [1,21,21] ; probs.sum() --> 21.0
                ctxt = torch.matmul(probs, update)
                # update.shape --> torch.Size([1, 21, 2324])
                # ctxt.shape --> [1, 21, 2324]
                update = self.gate(update, ctxt)
                # update.shape --> [1, 21, 2324]
                #
                if self.update_coref_scores:
                    coref_scores = self.coref(update, filtered_span_begin, filtered_span_end).squeeze(-1)
                    # filtered_span_begin.shape --> [1, 21, 1]; tensor([[[ 4], [ 6], [19], [23], [25], [28], ... ]])
                    # filtered_span_end.shape --> [1, 21, 1];   tensor([[[ 8], [ 6], [23], [27], [30], [28], ... ]])
                    if self.coref_pruner is not None:  # can be None if end_to_end is in false in model
                        coref_scores = coref_add_scores(coref_scores, self.coref_pruner(update))
                        # self.coref_pruner -->
                        # Sequential(
                        #   (0): Linear(in_features=2324, out_features=3000, bias=True)
                        #   (1): ReLU()
                        #   (2): Dropout(p=0.3, inplace=False)
                        #   (3): Linear(in_features=3000, out_features=3000, bias=True)
                        #   (4): ReLU()
                        #   (5): Dropout(p=0.3, inplace=False)
                        #   (6): Linear(in_features=3000, out_features=1, bias=True)
                        # )
                        # coref_scores.shape --> [1, 21, 21]

            update_filtered['span_vecs'] = update
            # update_filtered['span_vecs'].shape before update --> [1, 21, 2324]
            # update_filtered['span_vecs'].shape after update --> [1, 21, 2324]
            update_all['span_vecs'] = overwrite_spans(update_all['span_vecs'], filtered_spans_rest['prune_indices'],
                                                      filtered_spans_rest['span_lengths'], update)
            # update_all['span_vecs'].shape --> [1, 96, 15, 2324]
            # filtered_spans_rest['prune_indices'].shape --> [1, 21] --> tensor([[  64,   90,  289,  349,  380,  420,
            #   570,  617,  810,  872,  900,  907, 935,  938, 1022, 1098, 1113, 1188, 1194, 1215, 1216]])
            # filtered_spans_rest['span_lengths'] --> tensor([21])
            # update.shape --> [1, 21, 2324]

        return update_all, update_filtered, coref_scores

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.coreflinker.scorers import OptFFpairsHoi, OptFFpairs
from modules.utils.misc import MyGate, coref_add_scores, overwrite_spans_hoi


class ModuleCorefPropHoi(nn.Module):

    def __init__(self, dim_span, coref_pruner, config, span_pair_generator):
        super(ModuleCorefPropHoi, self).__init__()
        self.enabled = config['coref']['enabled']
        if self.enabled:
            self.coref_prop = config['coref']['corefprop']['coref_prop']
            print("ModuleCorefPropHoi(cp={})".format(self.coref_prop))
            self.update_coref_scores = config['coref']['corefprop']['update_coref_scores']
            self.init_weights_std = config['coref']['corefprop']['init_weights_std']
            self.coref_pruner = coref_pruner
            self.corefprop_type = config['coref']['corefprop']['type']
            if self.corefprop_type == 'ff_pairs':
                self.span_pair_generator = span_pair_generator
                self.coref = OptFFpairs(dim_span, 1, config['coref']['corefprop'], self.span_pair_generator)
            elif self.corefprop_type == 'ff_pairs_hoi':
                span_embed = config['span-extractor']['span_embed']
                self.coref = OptFFpairsHoi(dim_span, 1, config['coref']['corefprop'], span_embed)
            else:
                raise RuntimeError('not known corefprop.type from ModuleCorefPropHoi: ' +
                                   str(config['coref']['corefprop']['type']))

            self.gate = MyGate(dim_span, init_weights_std=self.init_weights_std)
        # self.end_to_end = config['end_to_end']

    def forward(self, all_spans, filtered_spans, gold_spans, max_span_length=0, gold_spans_lengths=0):
        # filtered_spans_rest = None

        # if self.end_to_end:
        update = filtered_spans['span_vecs']
        # update.shape --> [1, 21, 2324]

        if update is None or not self.enabled:
            return all_spans, filtered_spans, None

        filtered_span_begin = filtered_spans['span_begin']
        # filtered_span_begin.shape --> [1, 21]
        filtered_span_end = filtered_spans['span_end']
        # filtered_span_end.shape --> [1, 21]
        triangular_mask = filtered_spans['triangular_mask']
        # triangular_mask.shape --> [1, 21, 21]
        filtered_spans_rest = filtered_spans

        # filtered_spans_rest.keys() --> dict_keys(['prune_indices_hoi', 'span_vecs', 'span_scores', 'span_begin',
        # 'span_end', 'span_lengths', 'square_mask', 'triangular_mask', 'pruned_spans', 'gold_spans', 'enabled_spans'])
        #

        if self.corefprop_type == 'ff_pairs':
            coref_scores = self.coref(update, filtered_span_begin, filtered_span_end).squeeze(-1)
            coref_scores = coref_add_scores(coref_scores, filtered_spans['span_scores'])
        elif self.corefprop_type == 'ff_pairs_hoi':
            # TODO (09/04/2021) - this is wip!! DO NOT RUN IN PROD!
            coref_scores = self.coref(update, filtered_span_begin, filtered_span_end, triangular_mask,
                                      filtered_spans['span_scores']).squeeze(-1)
        else:
            raise RuntimeError('corefprop_type not recognized inside corefprop_hoi.py: ', self.corefprop_type)

        update_all = all_spans.copy()
        update_filtered = filtered_spans_rest.copy()

        if self.coref_prop > 0:
            for _ in range(self.coref_prop):
                probs = F.softmax(coref_scores - (1.0 - triangular_mask) * 1e23, dim=-1)
                # coref_scores.shape --> [1, 21, 21]; triangular_mask.shape --> [1, 21, 21];
                # triangular_mask.sum() --> 231.0 (float)
                ctxt = torch.matmul(probs, update)
                # ctxt.shape --> [1, 21, 2324]
                # update.shape --> [1, 21, 2324]
                update = self.gate(update, ctxt)
                # update.shape --> [1, 21, 2324]
                if self.update_coref_scores:
                    coref_scores = self.coref(update, filtered_span_begin, filtered_span_end).squeeze(-1)
                    # update.shape --> [1, 21, 2324]
                    # filtered_span_begin.shape --> [1, 21]
                    # filtered_spans_end.shape --> tensor([[ 3,  3,  5,  7,  9, 13, 23, 23, 24, 25, 27, 33, 42, 42, 48,
                    # 52, 52, 59, 78, 92, 93]])
                    if self.coref_pruner is not None:  # can be None if end_to_end is in false in model
                        coref_scores = coref_add_scores(coref_scores, self.coref_pruner(update))
                        # coref_scores.shape --> [1, 21, 21]
                        # update.shape --> [1, 21, 2324]
            # TODO - adapt here!
            update_filtered['span_vecs'] = update
            # update.shape --> [1, 21, 2324]

            update_all['cand_span_vecs'] = overwrite_spans_hoi(update_all['cand_span_vecs'],
                                                               filtered_spans_rest['prune_indices_hoi'],
                                                               filtered_spans_rest['span_lengths'], update)
            # update_all['span_vecs'] --> key error, relplaced by cand_span_vecs
            #   update_all['cand_span_vecs'].shape --> [1, 315, 2324]
            # filtered_spans_rest['prune_indices'] --> key error, replaced by prune_indices_hoi
            #   filtered_spans_rest['prune_indices_hoi'].shape --> [1, 21]
            # filtered_spans_rest['span_lengths'] --> tensor([21])
            # update.shape --> torch.Size([1, 21, 2324])

        return update_all, update_filtered, coref_scores

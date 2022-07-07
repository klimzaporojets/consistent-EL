import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.models.scorers import create_pair_scorer
from models.utils.misc import MyGate, overwrite_spans, overwrite_spans_hoi

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger()

class ModuleAttentionProp(nn.Module):

    def __init__(self, dim_span,
                 span_pair_generator, config):
        super(ModuleAttentionProp, self).__init__()
        self.hidden_dim = config['hidden_dim']  # 150
        self.hidden_dp = config['hidden_dropout']  # 0.4
        self.att_prop = config['att_prop']

        logger.info('ModuleAttentionProp(ap={})'.format(self.att_prop))
        if self.att_prop > 0:
            self.attention = create_pair_scorer(dim_span, 1, config, span_pair_generator)
            self.gate = MyGate(dim_span)

    def forward(self, all_spans, filtered_spans, sequence_lengths):
        update = filtered_spans['span_vecs']
        # update.shape --> [1, 21, 2324]

        filtered_span_begin = filtered_spans['span_begin']
        # filtered_span_begin.shape --> torch.Size([1, 21, 1])

        filtered_span_end = filtered_spans['span_end']
        # filtered_span_end.shape --> [1, 21, 1]

        if update is None or self.att_prop <= 0:
            return all_spans, filtered_spans

        if self.att_prop > 0:
            square_mask = filtered_spans['square_mask']
            # square_mask.shape --> [1, 21, 21]
            #
            update_all = all_spans.copy()
            # all_spans.shape -->
            #
            update_filtered = filtered_spans.copy()
            #
            #
            for _ in range(self.att_prop):
                scores = self.attention(update, filtered_span_begin, filtered_span_end).squeeze(-1)
                # scores.shape --> [1, 21, 21]
                #
                probs = F.softmax(scores - (1.0 - square_mask) * 1e23, dim=-1)
                # probs.shape --> [1, 21, 21]
                #
                ctxt = torch.matmul(probs, update)
                # ctxt.shape --> [1, 21, 2324]
                #
                update = self.gate(update, ctxt)
                # update.shape --> [1, 21, 2324]
                #

            update_filtered['span_vecs'] = update
            # update.shape --> [1, 21, 2324]
            #
            update_all['span_vecs'] = overwrite_spans(update_all['span_vecs'], filtered_spans['prune_indices'],
                                                      filtered_spans['span_lengths'], update)
            # update_all['span_vecs'].shape --> [1, 96, 15, 2324]
            # filtered_spans['prune_indices'].shape --> [1, 21]
            # filtered_spans['span_lengths'] --> tensor([21])
            # update.shape --> [1, 21, 2324]
            return update_all, update_filtered


class ModuleAttentionPropHoi(nn.Module):

    def __init__(self, dim_span,
                 span_pair_generator, config):
        super(ModuleAttentionPropHoi, self).__init__()
        self.hidden_dim = config['hidden_dim']  # 150
        self.hidden_dp = config['hidden_dropout']  # 0.4
        self.att_prop = config['att_prop']
        self.init_weights_std = config['init_weights_std']

        logger.info('ModuleAttentionProp(ap={})'.format(self.att_prop))
        if self.att_prop > 0:
            self.attention = create_pair_scorer(dim_span, 1, config, span_pair_generator)
            self.gate = MyGate(dim_span, self.init_weights_std)

    def forward(self, all_spans, filtered_spans, sequence_lengths):
        update = filtered_spans['span_vecs']
        # update.shape --> [1, 21, 2324]
        filtered_span_begin = filtered_spans['span_begin']
        # filtered_span_begin.shape --> [1, 21]
        filtered_span_end = filtered_spans['span_end']
        # filtered_span_end.shape --> [1, 21]

        if update is None or self.att_prop <= 0:
            return all_spans, filtered_spans

        if self.att_prop > 0:
            square_mask = filtered_spans['square_mask']

            update_all = all_spans
            update_filtered = filtered_spans

            for _ in range(self.att_prop):
                scores = self.attention(update, filtered_span_begin, filtered_span_end).squeeze(-1)
                probs = F.softmax(scores - (1.0 - square_mask) * 1e23, dim=-1)
                ctxt = torch.matmul(probs, update)
                update = self.gate(update, ctxt)
                # scores.shape --> [1, 21, 21]
                # probs.shape --> [1, 21, 21]
                # ctxt.shape --> [1, 21, 2324]
                # update.shape --> [1, 21, 2324]

            update_filtered['span_vecs'] = update
            # update.shape --> [1, 21, 2324]

            update_all['cand_span_vecs'] = overwrite_spans_hoi(update_all['cand_span_vecs'],
                                                               filtered_spans['prune_indices_hoi'],
                                                               filtered_spans['span_lengths'], update)

            # update_all['span_vecs'] --> key error, relplaced by cand_span_vecs
            #   update_all['cand_span_vecs'].shape --> [1, 315, 2324]
            # filtered_spans_rest['prune_indices'] --> key error, replaced by prune_indices_hoi
            #   filtered_spans_rest['prune_indices_hoi'].shape --> [1, 21]
            # filtered_spans_rest['span_lengths'] --> tensor([21])
            # update.shape --> torch.Size([1, 21, 2324])

            return update_all, update_filtered

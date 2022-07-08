import logging
from collections import Iterable
from typing import List

import torch
import torch.nn as nn
from torch.nn import init

from misc import settings
from models.utils.misc import spans_to_indices, sort_after_pruning, indices_to_spans, create_masks, filter_spans, \
    prune_spans, _extract_top_spans

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger()


def span_intersection(pred, gold):
    numer = 0
    for p, g in zip(pred, gold):
        numer += len(set(p) & set(g))
    return numer


def create_spans_targets(scores, gold_spans):
    targets = torch.zeros_like(scores)
    max_span_length = scores.size(2)
    for i, spans in enumerate(gold_spans):
        for begin, end in spans:
            if begin is not None and end is not None and end - begin < max_span_length:
                targets[i, begin, end - begin, 0] = 1.0
    return targets


def decode_accepted_spans(scores):
    num_batch = scores.size(0)
    max_span_length = scores.size(2)
    output = [list() for _ in range(num_batch)]
    for batch_idx, span_idx in torch.nonzero((scores.view(num_batch, -1) > 0).float()).tolist():
        begin = span_idx // max_span_length
        length = span_idx % max_span_length
        output[batch_idx].append((begin, begin + length))
    return output


class MentionPrunerSuper(nn.Module):
    def __init__(self):
        super(MentionPrunerSuper, self).__init__()

    def forward(self, all_spans, gold_spans, sequence_lengths, gold_spans_lengths=None, gold_span_tensors=None,
                doc_id=None, api_call=False):
        pass


class MentionPrunerGold(MentionPrunerSuper):
    def __init__(self, max_span_length, config):
        super(MentionPrunerGold, self).__init__()
        self.max_span_length = max_span_length
        self.scorer = None
        self.sort_after_pruning = config['sort_after_pruning']
        self.span_generated = 0
        self.span_recall_numer = 0
        self.span_recall_denom = 0
        self.span_loss = 0.0

    def forward(self, all_spans, gold_spans, sequence_lengths, gold_spans_lengths=None, gold_span_tensors=None,
                doc_id=None, api_call=False):
        span_vecs = all_spans['span_vecs']
        span_mask = all_spans['span_mask']
        span_begin = all_spans['span_begin']
        span_end = all_spans['span_end']

        gold_span_indices = spans_to_indices(gold_span_tensors, self.max_span_length)
        prune_scores = torch.zeros(span_mask.size()).unsqueeze(-1).to(settings.device)

        reindex = None
        if self.sort_after_pruning:
            pr_scores = prune_scores.view(prune_scores.size(0), -1)
            gold_span_indices, reindex = sort_after_pruning(gold_span_indices, gold_spans_lengths, pr_scores)

        gold_spans = indices_to_spans(gold_span_indices, gold_spans_lengths, self.max_span_length)

        square_mask, triangular_mask = create_masks(gold_spans_lengths, gold_span_indices.size(1))
        if gold_span_indices.size(-1) == 0:
            return 0, all_spans, {
                'prune_indices': gold_span_indices,
                'reindex_wrt_gold': reindex,
                'span_vecs': None,
                'span_scores': None,
                'span_begin': None,
                'span_end': None,
                'span_lengths': None,
                'square_mask': None,
                'triangular_mask': None,
                'spans': gold_spans,
                'enabled_spans': None
            }

        obj_pruner = 0
        enabled_spans = gold_spans

        self.span_generated += sum([len(x) for x in gold_spans])
        self.span_recall_numer += span_intersection(gold_spans, gold_spans)
        self.span_recall_denom += sum([len(x) for x in gold_spans])

        return obj_pruner, all_spans, {
            'prune_indices': gold_span_indices,
            'reindex_wrt_gold': reindex,
            'span_vecs': filter_spans(span_vecs, gold_span_indices),
            'span_scores': filter_spans(prune_scores, gold_span_indices),
            'span_begin': filter_spans(span_begin.view(prune_scores.size()), gold_span_indices),
            'span_end': filter_spans(span_end.view(prune_scores.size()), gold_span_indices),
            'span_lengths': gold_spans_lengths,
            'square_mask': square_mask,
            'triangular_mask': triangular_mask,
            'spans': gold_spans,
            'enabled_spans': enabled_spans
        }

    def end_epoch(self, dataset_name):
        logger.info('{}-span-generator: {} / {} = {}'.format(dataset_name, self.span_generated, self.span_recall_denom,
                                                             self.span_generated / (self.span_recall_denom + 1e-7)))
        logger.info('{}-span-recall: {} / {} = {}'.format(dataset_name, self.span_recall_numer, self.span_recall_denom,
                                                          self.span_recall_numer / (self.span_recall_denom + 1e-7)))
        logger.info('{}-span-loss: {}'.format(dataset_name, self.span_loss))
        self.span_generated = 0
        self.span_recall_numer = 0
        self.span_recall_denom = 0
        self.span_loss = 0.0


class MentionPruner(MentionPrunerSuper):

    def __init__(self, dim_span, max_span_length, config):
        super(MentionPruner, self).__init__()
        self.config = config
        self.dim_span = dim_span
        self.hidden_dim = config['hidden_dim']  # 150
        self.hidden_dp = config['hidden_dropout']  # 0.4
        self.max_span_length = max_span_length
        self.sort_after_pruning = config['sort_after_pruning']
        self.prune_ratio = config['prune_ratio']
        self.add_pruner_loss = config['add_pruner_loss']
        self.weight = config['weight'] if self.add_pruner_loss else None
        self.loss = nn.BCEWithLogitsLoss(reduction='none')

        self.scorer = nn.Sequential(
            nn.Linear(dim_span, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.hidden_dp),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.hidden_dp),
            nn.Linear(self.hidden_dim, 1)
        )

        logger.info('MentionPruner: %s %s %s %s ' %
                    (self.max_span_length, self.prune_ratio, self.sort_after_pruning, self.add_pruner_loss))
        self.span_generated = 0
        self.span_recall_numer = 0
        self.span_recall_numer_enabled = 0
        self.span_all_enabled = 0
        self.span_recall_denom = 0
        self.span_loss = 0.0

    def create_new(self):
        return MentionPruner(self.dim_span, self.max_span_length, self.config)

    def forward(self, all_spans, gold_spans, sequence_lengths, gold_spans_lengths=None, gold_span_tensors=None,
                doc_id=None, api_call=False):
        span_vecs = all_spans['span_vecs']
        span_mask = all_spans['span_mask']
        span_begin = all_spans['span_begin']
        span_end = all_spans['span_end']

        prune_scores = self.scorer(span_vecs) - (1.0 - span_mask.unsqueeze(-1)) * 1e4
        span_pruned_indices, span_lengths = prune_spans(prune_scores, sequence_lengths, self.sort_after_pruning,
                                                        prune_ratio=self.prune_ratio)
        pred_spans = indices_to_spans(span_pruned_indices, span_lengths, self.max_span_length)
        square_mask, triangular_mask = create_masks(span_lengths, span_pruned_indices.size(1))
        all_spans['span_scores'] = prune_scores

        self.span_generated += sum([len(x) for x in pred_spans])
        if not api_call:
            self.span_recall_numer += span_intersection(pred_spans, gold_spans)
            self.span_recall_denom += sum([len(x) for x in gold_spans])

        obj_pruner = 0
        enabled_spans = None
        if self.add_pruner_loss:
            enabled_spans = decode_accepted_spans(prune_scores)
            if not api_call:
                prune_targets = create_spans_targets(prune_scores, gold_spans)
                mask = (span_end < sequence_lengths.unsqueeze(-1).unsqueeze(-1)).float().unsqueeze(-1)

                obj_pruner = (self.loss(prune_scores, prune_targets) * mask).sum() * self.weight
                self.span_loss += obj_pruner.item()
                self.span_recall_numer_enabled += span_intersection(enabled_spans, gold_spans)
                self.span_all_enabled += sum([len(es) for es in enabled_spans])

        return obj_pruner, all_spans, {
            'prune_indices': span_pruned_indices,
            'span_vecs': filter_spans(span_vecs, span_pruned_indices),
            'span_scores': filter_spans(prune_scores, span_pruned_indices),
            'span_begin': filter_spans(span_begin.view(prune_scores.size()), span_pruned_indices),
            'span_end': filter_spans(span_end.view(prune_scores.size()), span_pruned_indices),
            'span_lengths': span_lengths,
            'square_mask': square_mask,
            'triangular_mask': triangular_mask,
            'spans': pred_spans,
            'enabled_spans': enabled_spans
        }

    def end_epoch(self, dataset_name):
        logger.info('{}-span-generator: {} / {} = {}'.format(dataset_name, self.span_generated, self.span_recall_denom,
                                                             self.span_generated / (self.span_recall_denom + 1e-7)))
        logger.info('{}-span-recall: {} / {} = {}'.format(dataset_name, self.span_recall_numer, self.span_recall_denom,
                                                          self.span_recall_numer / (self.span_recall_denom + 1e-7)))
        logger.info('{}-span-loss: {}'.format(dataset_name, self.span_loss))
        logger.info('{}-span-recall-enabled: {} / {} = {}'.format(dataset_name, self.span_recall_numer_enabled,
                                                                  self.span_recall_denom,
                                                                  self.span_recall_numer_enabled / (
                                                                          self.span_recall_denom + 1e-7)))
        logger.info('{}-span-precision-enabled: {} / {} = {}'.format(dataset_name, self.span_recall_numer_enabled,
                                                                     self.span_all_enabled,
                                                                     self.span_recall_numer_enabled / (
                                                                             self.span_all_enabled + 1e-7)))
        self.span_generated = 0
        self.span_recall_numer = 0
        self.span_recall_numer_enabled = 0
        self.span_recall_denom = 0
        self.span_loss = 0.0
        self.span_all_enabled = 0


class MentionPrunerSpanBert(MentionPrunerSuper):
    """
    Unlike MentionPruner does not have to be initialized with max_span_length, since for the case where we receive all
    the spans as input, it can be variable.
    """

    def __init__(self, dim_span, config):
        super(MentionPrunerSpanBert, self).__init__()
        self.config = config
        self.dim_span = dim_span
        self.hidden_dim = config['hidden_dim']  # 150
        self.hidden_dp = config['hidden_dropout']  # 0.4
        self.sort_after_pruning = config['sort_after_pruning']
        self.prune_ratio = config['prune_ratio']
        self.add_pruner_loss = config['add_pruner_loss']
        self.no_cross_overlap = config['no_cross_overlap']
        self.weight = config['weight'] if self.add_pruner_loss else None
        self.loss = nn.BCEWithLogitsLoss(reduction='none')

        self.scorer = nn.Sequential(
            nn.Linear(dim_span, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.hidden_dp),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.hidden_dp),
            nn.Linear(self.hidden_dim, 1)
        )

        logger.info('MentionPrunerSpanBert: %s %s %s ' %
                    (self.prune_ratio, self.sort_after_pruning, self.add_pruner_loss))
        self.span_generated = 0
        self.span_recall_numer = 0
        self.span_recall_numer_enabled = 0
        self.span_all_enabled = 0
        self.span_recall_denom = 0
        self.span_loss = 0.0

    def create_new(self):
        return MentionPrunerSpanBert(self.dim_span, self.config)

    def forward(self, all_spans, gold_spans, sequence_lengths, gold_spans_lengths=None, gold_span_tensors=None,
                doc_id=None, api_call=False, max_span_length=None):
        span_vecs = all_spans['span_vecs']
        span_mask = all_spans['span_mask']
        span_begin = all_spans['span_begin']
        span_end = all_spans['span_end']

        prune_scores = self.scorer(span_vecs) - (1.0 - span_mask.unsqueeze(-1)) * 1e4
        span_pruned_indices, span_lengths = prune_spans(prune_scores, sequence_lengths, self.sort_after_pruning,
                                                        prune_ratio=self.prune_ratio,
                                                        no_cross_overlap=self.no_cross_overlap,
                                                        span_mask=span_mask)
        # span_pruned_indices.shape --> [1,21] --> tensor([[  75,  316,  318,  334,  360,  362,  570,  637,  735,  737,  738,  814,
        #           815,  902, 1098, 1099, 1155, 1160, 1188, 1230, 1236]])
        # span_lengths --> tensor([21])
        # max_span_length --> 15
        pred_spans = indices_to_spans(span_pruned_indices, span_lengths, max_span_length)
        # pred_spans - [[(5, 5), (21, 22), (21, 24), (22, 26), (24, 24), (24, 26), (38, 38), (42, 49), (49, 49),
        # (49, 51), (49, 52), (54, 58), (54, 59), (60, 62), (73, 76), (73, 77), (77, 77), (77, 82), (79, 82),
        # (82, 82), (82, 88)]]
        square_mask, triangular_mask = create_masks(span_lengths, span_pruned_indices.size(1))
        # square_mask.shape --> [1,21,21]
        # triangular_mask.shape --> [1,21,21]

        all_spans['span_scores'] = prune_scores
        # prune_scores.shape --> [1,96,15,1]

        self.span_generated += sum([len(x) for x in pred_spans])
        if not api_call:
            self.span_recall_numer += span_intersection(pred_spans, gold_spans)
            self.span_recall_denom += sum([len(x) for x in gold_spans])

        obj_pruner = 0
        enabled_spans = None
        if self.add_pruner_loss:
            enabled_spans = decode_accepted_spans(prune_scores)
            # len(enabled_spans[0]) --> 124; <class 'list'>: [[(1, 2), (1, 6), (4, 5), (4, 8), ...]]
            if not api_call:
                prune_targets = create_spans_targets(prune_scores, gold_spans)
                # prune_targets.shape --> [1, 96, 15, 1]
                mask = span_mask.unsqueeze(-1)
                # mask.shape --> [1, 96, 15, 1]
                obj_pruner = (self.loss(prune_scores, prune_targets) * mask).sum() * self.weight
                #
                self.span_loss += obj_pruner.item()
                #
                self.span_recall_numer_enabled += span_intersection(enabled_spans, gold_spans)
                # len(gold_spans[0]) --> 9; gold_spans --> [[(5, 6), (7, 12), (51, 53), (55, 59), (61, 61), ...]]
                self.span_all_enabled += sum([len(es) for es in enabled_spans])
                # self.span_all_enabled --> 124

        return obj_pruner, all_spans, {
            'prune_indices': span_pruned_indices,
            # 'prune_indices'.shape --> [1,21] --> tensor([[  75,  316,  318,  334,  360,  362,  570,  637,  ...]])
            'span_vecs': filter_spans(span_vecs, span_pruned_indices),
            # 'span_vecs'.shape --> [1, 21, 2324]
            'span_scores': filter_spans(prune_scores, span_pruned_indices),
            # 'span_scores'.shape --> torch.Size([1, 21, 1])
            'span_begin': filter_spans(span_begin.view(prune_scores.size()), span_pruned_indices),
            # 'span_begin'.shape --> [1, 21, 1] --> tensor([[[ 5], [21], [21], [22], [24], [24], [38], [42], [49],....
            'span_end': filter_spans(span_end.view(prune_scores.size()), span_pruned_indices),
            # 'span_end'.shape --> [1, 21, 1] -->  tensor([[[ 5], [22], [24], [26], [24], [26], [38], [49], [49],...]]]
            'span_lengths': span_lengths,
            # span_lengths --> tensor([21])
            'square_mask': square_mask,
            #
            'triangular_mask': triangular_mask,
            #
            'spans': pred_spans,
            # <class 'list'>: [[(5, 5), (21, 22), (21, 24), (22, 26), (24, 24), (24, 26), (38, 38), (42, 49), ....]]
            'enabled_spans': enabled_spans
            # <class 'list'>: [[(1, 2), (1, 6), (4, 5), (4, 8), (5, 5), (6, 6), (6, 13), (11, 13), (13, 17), ...]]
        }

    def end_epoch(self, dataset_name):
        logger.info('{}-span-generator: {} / {} = {}'.format(dataset_name, self.span_generated, self.span_recall_denom,
                                                             self.span_generated / (self.span_recall_denom + 1e-7)))
        logger.info('{}-span-recall: {} / {} = {}'.format(dataset_name, self.span_recall_numer, self.span_recall_denom,
                                                          self.span_recall_numer / (self.span_recall_denom + 1e-7)))
        logger.info('{}-span-loss: {}'.format(dataset_name, self.span_loss))
        logger.info('{}-span-recall-enabled: {} / {} = {}'.format(dataset_name, self.span_recall_numer_enabled,
                                                                  self.span_recall_denom,
                                                                  self.span_recall_numer_enabled / (
                                                                          self.span_recall_denom + 1e-7)))
        logger.info('{}-span-precision-enabled: {} / {} = {}'.format(dataset_name, self.span_recall_numer_enabled,
                                                                     self.span_all_enabled,
                                                                     self.span_recall_numer_enabled / (
                                                                             self.span_all_enabled + 1e-7)))
        self.span_generated = 0
        self.span_recall_numer = 0
        self.span_recall_numer_enabled = 0
        self.span_recall_denom = 0
        self.span_loss = 0.0
        self.span_all_enabled = 0


class MentionPrunerSpanBertHoi(MentionPrunerSuper):
    """
    Includes some extra adaptations based on the code in https://github.com/lxucs/coref-hoi.
    Should be used with coreflinker_spanbert_hoi.py.
    """

    def make_linear(self, in_features, out_features, bias=True, std=0.02):
        linear = nn.Linear(in_features, out_features, bias)
        init.normal_(linear.weight, std=std)
        if bias:
            init.zeros_(linear.bias)
        return linear

    def make_ffnn(self, feat_size, hidden_size, output_size):
        if hidden_size is None or hidden_size == 0 or hidden_size == [] or hidden_size == [0]:
            return self.make_linear(feat_size, output_size)

        if not isinstance(hidden_size, Iterable):
            hidden_size = [hidden_size]
        ffnn = [self.make_linear(feat_size, hidden_size[0]), nn.ReLU(), self.dropout]
        for i in range(1, len(hidden_size)):
            ffnn += [self.make_linear(hidden_size[i - 1], hidden_size[i]), nn.ReLU(), self.dropout]
        ffnn.append(self.make_linear(hidden_size[-1], output_size))
        return nn.Sequential(*ffnn)

    def __init__(self, dim_span, config, feature_emb_size):
        super(MentionPrunerSpanBertHoi, self).__init__()
        self.config = config
        self.dim_span = dim_span
        self.hidden_dim = config['hidden_dim']  # 3000
        self.ffnn_depth = config['ffnn_depth']  # 1
        self.hidden_dp = config['hidden_dropout']  # 0.3
        self.dropout = nn.Dropout(p=self.hidden_dp)
        self.use_width_prior = config['use_width_prior']
        self.sort_after_pruning = config['sort_after_pruning']
        self.max_num_extracted_spans = config['max_num_extracted_spans']
        self.prune_ratio = config['prune_ratio']
        self.add_pruner_loss = config['add_pruner_loss']
        self.no_cross_overlap = config['no_cross_overlap']
        self.weight = config['weight'] if self.add_pruner_loss else None
        self.debug_stats = config['debug_stats']

        if self.debug_stats:
            self.pruner_losses = list()
            self.scores_norm = list()
            self.scores_mean = list()
            self.scores_std = list()
            self.scores_min = list()
            self.scores_max = list()

        self.loss = nn.BCEWithLogitsLoss(reduction='none')

        self.scorer = self.make_ffnn(dim_span, [self.hidden_dim] * self.ffnn_depth, output_size=1)

        if self.use_width_prior:
            self.span_width_score_ffnn = self.make_ffnn(feature_emb_size,
                                                        [self.hidden_dim] * self.ffnn_depth, output_size=1)

        logger.info('MentionPrunerSpanBertHoi: %s %s %s ' %
                    (self.prune_ratio, self.sort_after_pruning, self.add_pruner_loss))
        self.span_generated = 0
        self.span_recall_numer = 0
        self.span_recall_numer_enabled = 0
        self.span_all_enabled = 0
        self.span_recall_denom = 0
        self.span_loss = 0.0

    def create_new(self):
        return MentionPrunerSpanBert(self.dim_span, self.config)

    def get_mean(self, list_values: List):
        if len(list_values) == 0:
            return 0.0
        else:
            return (sum(list_values) / len(list_values))

    def forward(self, all_spans, sequence_lengths, gold_spans_lengths=None, gold_span_tensors=None,
                doc_id=None, api_call=False, max_span_length=None, emb_span_width_prior=None, predict=False):
        cand_span_vecs = all_spans['cand_span_vecs']
        span_begin = all_spans['cand_span_begin']
        span_end = all_spans['cand_span_end']
        candidate_width_idx = all_spans['cand_width_idx']
        # for now only single batch
        assert cand_span_vecs.shape[0] == 1

        prune_scores = torch.squeeze(self.scorer(cand_span_vecs[0]), 1)  # - (1.0 - span_mask.unsqueeze(-1)) * 1e4

        if self.use_width_prior:
            width_score = torch.squeeze(self.span_width_score_ffnn(emb_span_width_prior.weight), 1)
            candidate_width_score = width_score[candidate_width_idx[0]]
            prune_scores += candidate_width_score

        candidate_idx_sorted_by_score = torch.argsort(prune_scores, descending=True).tolist()
        candidate_starts_cpu, candidate_ends_cpu = span_begin[0].tolist(), span_end[0].tolist()

        span_length = (sequence_lengths.float() * self.prune_ratio + 1).int()
        span_length = int(min(self.max_num_extracted_spans, span_length))

        selected_idx_cpu = _extract_top_spans(candidate_idx_sorted_by_score, candidate_starts_cpu,
                                              candidate_ends_cpu, span_length, no_cross_overlap=self.no_cross_overlap)
        assert len(selected_idx_cpu) == span_length
        selected_idx = torch.tensor(selected_idx_cpu, device=settings.device)
        top_span_starts, top_span_ends = span_begin[0][selected_idx], span_end[0][selected_idx]

        all_spans['cand_span_scores'] = prune_scores.unsqueeze(0)
        pred_spans = None
        gold_spans = None
        if predict or self.debug_stats:
            se_tuples = torch.cat([top_span_starts.unsqueeze(0), top_span_ends.unsqueeze(0)], dim=0)
            pred_spans = [[tuple(l2) for l2 in l] for l in se_tuples.T.unsqueeze(0).tolist()]
            gold_spans = [[tuple(l2) for l2 in l] for l in gold_span_tensors.tolist()]

            self.span_generated += sum([len(x) for x in pred_spans])
            if not api_call:
                # TODO: try to make this tensorizable
                self.span_recall_numer += span_intersection(pred_spans, gold_spans)
                self.span_recall_denom += sum([len(x) for x in gold_spans])

        obj_pruner = 0
        enabled_spans = None
        if self.add_pruner_loss:
            if not api_call:
                gold_starts = gold_span_tensors[0][:, 0]
                gold_ends = gold_span_tensors[0][:, 1]
                same_start = (torch.unsqueeze(gold_starts, 1) == torch.unsqueeze(span_begin[0], 0))
                # same_start.shape --> torch.Size([186, 20768]); same_start.sum() --> 3639
                same_end = (torch.unsqueeze(gold_ends, 1) == torch.unsqueeze(span_end[0], 0))
                # same_end.shape --> torch.Size([186, 20768]); same_end.sum() --> 3460  # SOMEWHERE AROUND HERE SMALL MEMORY JUMP: 3,083 TO 3,103
                prune_targets = (same_start & same_end).float()

                # prune_targets = create_spans_targets(prune_scores, gold_spans)
                prune_targets: torch.Tensor = prune_targets.sum(dim=0)

                assert prune_targets.max() <= 1.0

                obj_pruner = self.loss(prune_scores, prune_targets).sum() * self.weight
                self.span_loss += obj_pruner.item()

                if torch.isnan(obj_pruner):
                    logger.warning('WARNING, torch.isnan(obj_pruner)')
                    obj_pruner = 0
                if self.debug_stats:
                    if isinstance(obj_pruner, torch.Tensor):
                        self.pruner_losses.append(obj_pruner.item())
                    else:
                        self.pruner_losses.append(obj_pruner)
                    self.scores_norm.append(prune_scores.norm().item())
                    self.scores_mean.append(prune_scores.mean().item())
                    self.scores_std.append(prune_scores.std().item())
                    self.scores_min.append(prune_scores.min().item())
                    self.scores_max.append(prune_scores.max().item())

                if predict or self.debug_stats:
                    assert span_begin.shape[0] == 1
                    assert span_end.shape[0] == 1
                    enabled_spans = torch.cat([span_begin, span_end]).T[prune_scores > 0]
                    enabled_spans = enabled_spans.unsqueeze(0)  # adding batch dimension
                    enabled_spans = [[tuple(l) for l in l2] for l2 in enabled_spans.tolist()]

                    self.span_recall_numer_enabled += span_intersection(enabled_spans, gold_spans)
                    self.span_all_enabled += sum([len(es) for es in enabled_spans])

        span_lengths_tensor = torch.tensor([span_length], device=settings.device)
        square_mask, triangular_mask = create_masks(span_lengths_tensor, span_length)
        # square_mask.shape --> [1,21,21]
        # triangular_mask.shape --> [1,21,21]

        cand_span_vecs = cand_span_vecs[0][selected_idx].unsqueeze(0)
        return obj_pruner, all_spans, {
            'prune_indices_hoi': selected_idx.unsqueeze(0),
            'span_vecs': cand_span_vecs,
            'span_scores': prune_scores[selected_idx].unsqueeze(0),
            'span_begin': top_span_starts.unsqueeze(0),
            'span_end': top_span_ends.unsqueeze(0),
            'span_lengths': span_lengths_tensor,
            'square_mask': square_mask,
            'triangular_mask': triangular_mask,
            'pruned_spans': pred_spans,
            'gold_spans': gold_spans,
            'enabled_spans': enabled_spans
        }

    def log_stats(self, dataset_name, predict, tb_logger, step_nr):
        tb_logger.log_value('{}-pruner-loss'.format(dataset_name), self.get_mean(self.pruner_losses), step_nr)

        # avg norm
        tb_logger.log_value('{}-pruner-norm'.format(dataset_name), self.get_mean(self.scores_norm), step_nr)
        tb_logger.log_value('{}-pruner-mean'.format(dataset_name), self.get_mean(self.scores_mean), step_nr)
        tb_logger.log_value('{}-pruner-std'.format(dataset_name), self.get_mean(self.scores_std), step_nr)
        tb_logger.log_value('{}-pruner-min'.format(dataset_name), self.get_mean(self.scores_min), step_nr)
        tb_logger.log_value('{}-pruner-max'.format(dataset_name), self.get_mean(self.scores_max), step_nr)

        self.end_epoch(dataset_name, predict)
        if self.debug_stats:
            self.pruner_losses = list()
            self.scores_norm = list()
            self.scores_mean = list()
            self.scores_std = list()
            self.scores_min = list()
            self.scores_max = list()

    def end_epoch(self, dataset_name, predict):
        if self.debug_stats or predict:
            logger.info('{}-span-generator: {} / {} = {}'.format(dataset_name, self.span_generated,
                                                                 self.span_recall_denom,
                                                                 self.span_generated / (self.span_recall_denom + 1e-7)))
            logger.info('{}-span-recall: {} / {} = {}'.format(dataset_name, self.span_recall_numer,
                                                              self.span_recall_denom,
                                                              self.span_recall_numer / (self.span_recall_denom + 1e-7)))
            logger.info('{}-span-loss: {}'.format(dataset_name, self.span_loss))
            logger.info('{}-span-recall-enabled: {} / {} = {}'.format(dataset_name, self.span_recall_numer_enabled,
                                                                      self.span_recall_denom,
                                                                      self.span_recall_numer_enabled / (
                                                                              self.span_recall_denom + 1e-7)))
            logger.info('{}-span-precision-enabled: {} / {} = {}'.format(dataset_name, self.span_recall_numer_enabled,
                                                                         self.span_all_enabled,
                                                                         self.span_recall_numer_enabled / (
                                                                                 self.span_all_enabled + 1e-7)))
        self.span_generated = 0
        self.span_recall_numer = 0
        self.span_recall_numer_enabled = 0
        self.span_recall_denom = 0
        self.span_loss = 0.0
        self.span_all_enabled = 0

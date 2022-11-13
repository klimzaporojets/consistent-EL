import json
import logging
import os
from typing import List

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from tensorboard_logger import Logger

from data_processing.dictionary import Dictionary
from metrics.coref import MetricCoref, MetricCorefAverage
from metrics.corefx import MetricCorefExternal
from metrics.linker import MetricLinkerImproved, MetricLinkAccuracy, MetricLinkAccuracyNoCandidates
from metrics.misc import MetricObjective
from misc import settings
from models.misc.misc import batched_index_select
from models.models.coreflinker_loss import create_candidate_mask, remove_disabled_spans, remove_disabled_spans_linking, \
    remove_disabled_scores_coref, convert_coref, m2i_to_clusters_linkercoref, \
    remove_disabled_scores_linking, remove_singletons_without_link
from models.utils.edmonds import mst_only_tree
from models.utils.misc import predict_scores_mtt

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger()


def predict_scores_coref_mtt(scores, pred_spans):
    """

    :param scores: upper triangular matrix
    :param linker:
    :param labels:
    :param no_nills:
    :return: scores for all the previous spans such as in this short example:
            [[{(1,1):[{'span':(1,1), 'score': 1.35}],
                (2,2):[{'span':(1,1), 'score': 1.35},
                        {'span':(2,2), 'score': 1.32}]}]]
    """
    to_ret_scores = []
    for batch_idx, scores_batch in enumerate(scores):
        batch_decoded = dict()
        spans_batch = pred_spans[batch_idx]
        for idx_span_base, curr_span_base in enumerate(spans_batch):
            for idx_span_coref, curr_span_coref in enumerate(spans_batch):
                if idx_span_coref <= idx_span_base:
                    if curr_span_base not in batch_decoded:
                        batch_decoded[curr_span_base] = []
                    batch_decoded[curr_span_base].append({'span': curr_span_coref,
                                                          'score': scores_batch[idx_span_base, idx_span_coref].item()})
        to_ret_scores.append(batch_decoded)

    return to_ret_scores


def create_coreflinker_mtt_z_mask_indexed(pred_spans, gold_spans, gold_clusters, linker_targets,
                                          candidates,
                                          candidate_lengths,
                                          unique_links, unique_links_lengths, unknown_id):
    """
    :param pred_spans:
    :param gold_spans:
    :param gold_clusters:
    :param linker_targets:
    :return:
        Binary (1 or 0) mask matrix of dimensions (not counting the first batch):
        [(1(root) + distinct links from pred spans + pred spans) x
        (1(root) + distinct links from pred spans + pred spans)]
        Indexed version that should be faster than create_coreflinker_mtt_z_mask
    """
    num_batch = len(pred_spans)  # 1
    max_spans = max([len(x) for x in pred_spans])  # 21

    coreflinker_mtt_z_mask_indexed = torch.zeros(num_batch, 1 + unique_links.shape[-1] + max_spans,
                                                 1 + unique_links.shape[-1] + max_spans, device=settings.device,
                                                 dtype=torch.float32)  # .shape --> torch.Size([1, 89, 89])
    z_mask_lengths = list()
    for batch, (pred, gold, clusters) in enumerate(zip(pred_spans, gold_spans, gold_clusters)):
        span_offset = unique_links_lengths[batch] + 1
        curr_z_mask_length = span_offset.item() + len(pred)  # root + links + spans
        z_mask_lengths.append(curr_z_mask_length)

        len_pred = len(pred)
        mask_span_to_span = torch.ones(len_pred, len_pred, device=settings.device)
        ind = np.diag_indices(mask_span_to_span.shape[0])
        mask_span_to_span[ind[0], ind[1]] = torch.zeros(mask_span_to_span.shape[0], device=settings.device)

        coreflinker_mtt_z_mask_indexed[batch, span_offset:, span_offset:] = mask_span_to_span[:]

        mix_cross: torch.Tensor = candidates[batch].unsqueeze(-1) == unique_links[batch]
        mask_link_to_span = mix_cross.sum(-2)
        mask_link_to_span = mask_link_to_span.T

        coreflinker_mtt_z_mask_indexed[batch, 1:span_offset, span_offset:] = mask_link_to_span[:]
        coreflinker_mtt_z_mask_indexed[batch, 0, 1:] = 1.0

    return coreflinker_mtt_z_mask_indexed, torch.tensor(z_mask_lengths, dtype=torch.int32, device=settings.device)


def m2i_to_clusters_linkercoref_mtt(m2i, coref_col_to_link_id=None, links_dictionary: Dictionary = None,
                                    nr_candidates=0):
    """

    :param m2i: <class 'list'>: [36, 4, 36, 14, 36, 4, 4, 4, 32]
    :return:
    """
    clusters = {}
    m2c = {}
    for m, c in enumerate(m2i):
        if c == -1:
            # not a valid cluster (i.e. should happen if filter_singletons_with_matrix is in true and the span points to
            # the matrix column that indicates it is a not valid entity mention).
            continue

        if c not in clusters:
            clusters[c] = []

        link_token = ''
        if c < nr_candidates:
            link_id = coref_col_to_link_id[c + 1]
            link_token = links_dictionary.get(link_id)

        # if points to 'NILL', just makes it point to itself, this is because we can not cluster entities based on 'NILL'
        # since different entities can point to 'NILL'. Same with NONE.
        if link_token == 'NILL' or link_token == 'NONE':
            if m not in clusters:
                clusters[m] = []
            clusters[m].append(m)
            m2c[m] = clusters[m]
        else:
            clusters[c].append(m)
            m2c[m] = clusters[c]

    # clusters: <class 'dict'>: {0: [0], 1: [1, 6, 7, 10], 2: [2], 3: [3, 11], 4: [4], 5: [5], 8: [8, 12], 9: [9], 13: [13]}
    # clusters.values(): <class 'list'>: [[0], [1, 6, 7, 10], [2], [3, 11], [4], [5], [8, 12], [9], [13]]
    # m2c: <class 'dict'>: {0: [0], 1: [1, 6, 7, 10], 2: [2], 3: [3, 11], 4: [4], 5: [5], 6: [1, 6, 7, 10],
    #   7: [1, 6, 7, 10], 8: [8, 12], 9: [9], 10: [1, 6, 7, 10], 11: [3, 11], 12: [8, 12], 13: [13]}
    return list(clusters.values()), m2c


# def dfs_edges(b_pred_matrix_mst, from_node=0, traversed_edges = list()):
def dfs_edges(b_pred_matrix_mst, from_node=0):
    """

    :param b_pred_matrix_mst:
    :return: gets edges of depth first search traverse over the adjacency matrix passed as parameter
    """
    to_ret = []
    for to_node in range(b_pred_matrix_mst.shape[1]):
        if b_pred_matrix_mst[from_node, to_node] == 1.0:
            to_ret += [(from_node, to_node)] + dfs_edges(b_pred_matrix_mst, to_node)
    return to_ret


def convert_coref_mtt(clusters, spans, number_candidates=None, links_dictionary: Dictionary = None,
                      coref_col_to_link_id: dict = None):
    """

    :param clusters: it is a tuple, see below for components (clusters and m2c)
    :param spans: <class 'list'>: [(3, 4), (5, 7), (41, 43), (45, 47), (49, 49), (50, 50), (51, 52), (53, 54), (57, 57)]
    :return:
    """
    (clusters, m2c) = clusters

    out_coref_clusters = [[spans[m - number_candidates] for m in cluster if m >= number_candidates]
                          for cluster in clusters]  # IndexError: list index out of range
    out_coref_clusters = [span_cluster for span_cluster in out_coref_clusters if len(span_cluster) > 0]

    out_mention_to_link_id = dict()

    for i in range(number_candidates):
        span_ids = [span_id for span_id in m2c[i] if span_id >= number_candidates]
        for curr_span_id in span_ids:
            if curr_span_id not in out_mention_to_link_id:
                out_mention_to_link_id[curr_span_id] = [i]
            else:
                out_mention_to_link_id[curr_span_id].append(i)
                logger.warning('WARNING ON CANDIDATE LINKING!!: %s has been resolved to multiple '
                               'candidates: %s' % (curr_span_id, out_mention_to_link_id[curr_span_id]))

    # +1 because we do not count the root taken into account in coref_col_to_link_id
    out_mention_to_link_id2 = [(spans[k - number_candidates] + (links_dictionary.get(coref_col_to_link_id[v[0] + 1]),))
                               for k, v in out_mention_to_link_id.items()]

    spans_with_link = set([(span_start, span_end) for span_start, span_end, _ in out_mention_to_link_id2])

    return out_coref_clusters, out_mention_to_link_id2, spans_with_link


def get_smart_arsinh_exp(tensor_to_simplify: torch.Tensor):
    return tensor_to_simplify + torch.sqrt(torch.pow(tensor_to_simplify, 2) + 1.0)


class LossCorefLinkerMTTHoi(nn.Module):

    def __init__(self, link_task, coref_task, entity_dictionary, config, end_to_end):
        super(LossCorefLinkerMTTHoi, self).__init__()

        self.enabled = config['enabled']
        self.coref_task = coref_task
        self.link_task = link_task
        self.entity_dictionary = entity_dictionary

        if self.enabled:
            self.labels = self.entity_dictionary.tolist()
            self.unknown_dict = entity_dictionary.lookup('###UNKNOWN###')

        self.weight = config.get('weight', 1.0)
        self.filter_singletons_with_pruner = config['filter_singletons_with_pruner']
        self.filter_only_singletons = config['filter_only_singletons']
        self.filter_singletons_with_ner = config['filter_singletons_with_ner']
        self.singletons = self.filter_singletons_with_pruner or self.filter_singletons_with_ner
        self.end_to_end = end_to_end
        self.float_precision = config['float_precision']
        self.multihead_nil = config['multihead_nil']
        self.log_inf_mask = config['log_inf_mask']
        self.exp_trick = config['exp_trick']

        if self.float_precision == 'float64':
            self.torch_float_precision = torch.float64
        else:
            self.torch_float_precision = torch.float32

        self.print_debugging = config['print_debugging']
        self.print_debugging_matrices = config['print_debugging_matrices']

        self.nonlinear_function = config['nonlinear_function']
        self.smart_arsinh = config['smart_arsinh']
        self.enforce_scores = config['enforce_scores']
        self.min_score_max = config['min_score_max']
        self.min_score_min = config['min_score_min']
        self.max_score_max = config['max_score_max']
        self.max_score_min = config['max_score_min']

        self.zeros_to_clusters = config['zeros_to_clusters']
        self.zeros_to_links = config['zeros_to_links']
        self.root_link_max_spans_to_link = config['root_link_max_spans_to_link']
        self.root_link_min_zero = config['root_link_min_zero']
        # nil_partition_implementation: implementation based on "3.1. Partition functions via matrix determinants" of
        # https://www.aclweb.org/anthology/D07-1015/
        self.nil_partition_implementation = config['nil_partition_implementation']

        if self.print_debugging:
            self.mtt_hoi_loss = list()
            self.stat_link_span_norm = list()
            self.stat_link_span_mean = list()
            self.stat_link_span_std = list()
            self.stat_link_span_min = list()
            self.stat_link_span_max = list()
            self.stat_span_span_norm = list()
            self.stat_span_span_mean = list()
            self.stat_span_span_std = list()
            self.stat_span_span_min = list()
            self.stat_span_span_max = list()

    def create_scores_mtt_pred(self, scores, unique_links, candidates, candidate_lengths, torch_float_precision):
        """

        :param scores:
        :return:

        """
        # resources to help: https://discuss.pytorch.org/t/find-indices-of-one-tensor-in-another/84889

        # scores.shape: [1,14,31]
        # candidates.shape: [1,14,16]
        # unique_links.shape: [1,30]

        # here broadcasts candidate ids to unique ids in unique_links
        # mix_cross.shape: [batch, spans, # candidates, # unique_candidates] --> [1, 14, 16, 30]
        # mix_cross is a matrix that maps each of the candidates (dim 2) for each of the spans (dim 1) to the respective
        # position in unique_links (dim 3). dim 0 is the batch.
        scrs_mtt_root_to_spans = scores[:, :, :1].transpose(-2, -1)  # scores.shape --> torch.Size([1, 21, 38])
        # scrs_mtt_root_to_spans.shape --> torch.Size([1, 1, 21])
        cand_max_length = candidate_lengths.max().item()
        # 16
        candidates = candidates[:, :, :cand_max_length]
        # candidates.shape --> torch.Size([1, 21, 16])
        scrs_mtt_spans = scores[:, :, cand_max_length + 1:].transpose(-2, -1)
        # scrs_mtt_spans.shape --> torch.Size([1, 21, 21])

        if unique_links.shape[-1] > 0:
            mix_cross: torch.Tensor = (candidates.unsqueeze(-1) == unique_links)
            # mix_cross.shape --> torch.Size([1, 21, 16, 67])
            scrs_mtt_expd_links_to_spans = torch.zeros_like(mix_cross, device=settings.device,
                                                            dtype=torch_float_precision)
            # scrs_mtt_expd_links_to_spans.shape --> torch.Size([1, 21, 16, 67])
            scrs_links_to_spans = scores[:, :, 1:candidates.shape[-1] + 1]
            # scrs_links_to_spans.shape --> torch.Size([1, 21, 16])
            scrs_expd_links_to_spans = scrs_links_to_spans.unsqueeze(-1).expand(mix_cross.size())
            # scrs_expd_links_to_spans.shape --> torch.Size([1, 21, 16, 67])
            scrs_mtt_expd_links_to_spans[mix_cross] = scrs_expd_links_to_spans[mix_cross]
            # scrs_mtt_expd_links_to_spans.shape --> torch.Size([1, 21, 16, 67])

            ones_multiplier = torch.ones(1, mix_cross.shape[2], dtype=torch_float_precision, device=settings.device)
            # ones_multiplier.shape --> torch.Size([1, 16])
            scrs_mtt_links_to_spans = torch.matmul(ones_multiplier, scrs_mtt_expd_links_to_spans).squeeze(-2)
            # scrs_mtt_links_to_spans.shape --> torch.Size([1, 21, 67])
            scrs_mtt_links_to_spans = scrs_mtt_links_to_spans.transpose(-2, -1)
            # scrs_mtt_links_to_spans.shape --> torch.Size([1, 67, 21])

            # scores for predicted block (the rightmost)
            scrs_mtt_pred_bloc = torch.cat([scrs_mtt_root_to_spans, scrs_mtt_links_to_spans, scrs_mtt_spans], dim=-2)
            # scrs_mtt_pred_bloc.shape --> torch.Size([1, 89, 21])
        else:
            # if no candidate links, then don't add it completely
            scrs_mtt_pred_bloc = torch.cat([scrs_mtt_root_to_spans, scrs_mtt_spans], dim=-2)

        # the leftmost fixed bloc
        scrs_mtt_fixed_bloc = torch.zeros(scrs_mtt_pred_bloc.shape[0], scrs_mtt_pred_bloc.shape[1],
                                          1 + unique_links.shape[-1], device=settings.device,
                                          dtype=torch_float_precision)

        # repeat just the maximum score from the root to candidate links
        if self.root_link_max_spans_to_link:
            # the score root->link is the maximum score link to span predicted for that link
            # as opposed to overall maximum is root_link_max_spans_to_link is not activated.
            max_to_assign = scrs_mtt_pred_bloc[0, 1:unique_links.shape[-1] + 1, :].max(dim=-1).values
            if self.root_link_min_zero:
                max_to_assign[max_to_assign < 0.0] = 0.0

            scrs_mtt_fixed_bloc[0, 0, 1:unique_links.shape[-1] + 1] = max_to_assign
        elif not (self.zeros_to_clusters or self.zeros_to_links):
            # if it is lower than 0 then, assign 0 depending on hyperparameter
            scrs_mtt_fixed_bloc[0, 0, 1:unique_links.shape[-1] + 1] = torch.max(scrs_mtt_pred_bloc)
        #
        scrs_mtt_complete_matrix = torch.cat([scrs_mtt_fixed_bloc, scrs_mtt_pred_bloc], dim=-1)
        # scrs_mtt_complete_matrix.shape --> torch.Size([1, 89, 89])
        return scrs_mtt_complete_matrix

    def create_coreflinker_mtt_target_mask_multihead(self, pred_spans, gold_spans, gold_clusters, linker_targets,
                                                     candidates, unique_links, unique_links_lengths):
        """
        Unlike create_coreflinker_mtt_target_mask that produces only a single entry (1.0 mask activation) from root
        to a span (the first span in a cluster) in a particular NIL cluster, this one returns multiple matrices with
        activations for each of the spans ("heads") in a NIL cluster.
        For details: see https://docs.google.com/presentation/d/12vVEcWkg-BygOM_ui1l0jaE_JJ0RDYwW7wFBqPGgkvY/edit#slide=id.g18c18415074_0_0
        For details of what create_coreflinker_mtt_target_mask is doing see slide https://docs.google.com/presentation/d/12vVEcWkg-BygOM_ui1l0jaE_JJ0RDYwW7wFBqPGgkvY/edit#slide=id.g18c18415074_0_143

        :param pred_spans:
        :param gold_spans:
        :param gold_clusters:
        :param linker_targets:
        :return:
            Binary (1 or 0) mask matrix of dimensions (not counting the first batch):
            [(1(root) + distinct links from pred spans + pred spans) x
            (1(root) + distinct links from pred spans + pred spans)]
        """

        num_batch = len(pred_spans)  # 1
        max_spans = max([len(x) for x in pred_spans])  # 9

        coreflinker_mtt_targets = torch.zeros(num_batch, 1 + unique_links.shape[-1] + max_spans,
                                              1 + unique_links.shape[-1] + max_spans,
                                              dtype=torch.float32,
                                              device=settings.device)
        # coreflinker_mtt_targets.shape --> torch.Size([1, 89, 89])
        nr_correct_candidates_per_mention = linker_targets.sum(-1)
        target_mask_lengths = list()
        batched_lst_multiheads = list()
        # targets for span-span coref # TODO!! - also check that linker_targets come as batch!!!
        for batch, (pred, gold, clusters) in enumerate(zip(pred_spans, gold_spans, gold_clusters)):
            # root to links always in 1
            coreflinker_mtt_targets[batch, 0, 1:unique_links_lengths[batch] + 1] = 1.0
            span_offset = 1 + unique_links_lengths[batch].item()  # span_offset --> 68
            curr_target_length = span_offset + len(pred)  # root + links + spans # 89
            target_mask_lengths.append(curr_target_length)
            gold2cluster = dict()
            for idx, span in enumerate(gold):
                gold2cluster[span] = clusters[idx].item()
            # gold2cluster --> <class 'dict'>: {(5, 6): 4, (7, 12): 5, (51, 53): 1, (55, 59): 2, (61, 61): 6, (62, 62): 7, (63, 67): 8, (68, 72): 9, (76, 76): 6}
            # len(gold2cluster) --> 9
            pred2cluster_struct = dict()
            if clusters.shape[-1] > 0:
                max_gold_cl_id = clusters.max().item()
            else:
                max_gold_cl_id = 0
            #
            to_assign_cl_id = max_gold_cl_id + 1
            #
            for idx_span1, span1 in enumerate(pred):
                if span1 not in gold2cluster:
                    new_cl_id = to_assign_cl_id
                    to_assign_cl_id += 1
                    pred2cluster_struct[new_cl_id] = {'cluster_id': to_assign_cl_id,
                                                      'spans': [(idx_span1, span1)],
                                                      # if it is not in gold2cluster, then there is no way to know
                                                      # if the link is valid or not
                                                      'is_valid_link': False}
                    #
                else:
                    cluster_id = gold2cluster[span1]
                    nr_correct_candidates_in_span = nr_correct_candidates_per_mention[batch, idx_span1].item()
                    is_valid_link = False
                    # in theory nr_correct_candidates_in_span can be at most 1 (only a single correct candidate)
                    if nr_correct_candidates_in_span > 0:
                        is_valid_link = True
                        correct_link_id = candidates[batch, idx_span1, linker_targets[batch, idx_span1] > 0.5].item()
                        offset_correct_link = (unique_links[batch] == correct_link_id).nonzero().item()
                        coreflinker_mtt_targets[batch, offset_correct_link + 1, idx_span1 + span_offset] = 1.0

                    for idx_span2, span2 in enumerate(pred):
                        if idx_span2 != idx_span1 and span2 in gold2cluster and gold2cluster[span1] == gold2cluster[
                            span2]:
                            coreflinker_mtt_targets[batch, idx_span1 + span_offset, idx_span2 + span_offset] = 1.0

                    if cluster_id not in pred2cluster_struct:
                        pred2cluster_struct[cluster_id] = {'cluster_id': cluster_id,
                                                           'spans': [(idx_span1, span1)],
                                                           'is_valid_link': False}
                    else:
                        pred2cluster_struct[cluster_id]['spans'].append((idx_span1, span1))

                    if is_valid_link:
                        pred2cluster_struct[cluster_id]['is_valid_link'] = True

            lst_multiheads = list()  # probably put this on top
            for cluster_struct in pred2cluster_struct.values():
                is_valid_link = cluster_struct['is_valid_link']
                if not is_valid_link:
                    # puts the first one directly into coref_mtt_targets
                    spans = cluster_struct['spans']
                    idx_1st_span = spans[0][0]
                    coreflinker_mtt_targets[batch, 0, idx_1st_span + span_offset] = 1.0
                    if self.multihead_nil == 'multihead_old':
                        len_spans = len(spans)
                        indices = [0]  # the first index always 0 (root)
                        curr_indices = indices + [curr_ind[0] + span_offset for curr_ind in spans]
                        # :1 because the first head is already taken care in coreflinker_mtt_targets
                        for idx, curr_head in enumerate(spans[1:]):
                            mtt_targets = torch.zeros(len_spans + 1, len_spans + 1, dtype=self.torch_float_precision,
                                                      device=settings.device)

                            # +2 because of root and the first head that is already taken care in coreflinker_mtt_targets
                            mtt_targets[0, 2 + idx] = 1.0
                            # the connections between spans (inter_spans) all in 1 except the main diagonal
                            inter_spans = torch.ones(mtt_targets.shape[0] - 1, mtt_targets.shape[1] - 1,
                                                     dtype=self.torch_float_precision, device=settings.device)

                            ind = np.diag_indices(inter_spans.shape[0])
                            inter_spans[ind[0], ind[1]] = torch.zeros(inter_spans.shape[0],
                                                                      dtype=self.torch_float_precision,
                                                                      device=settings.device)
                            mtt_targets[1:, 1:] = inter_spans[:]
                            lst_multiheads.append({'mtt_targets': mtt_targets, 'indices': curr_indices})
                    elif self.multihead_nil == 'none':
                        for curr_head in spans[1:]:
                            coreflinker_mtt_targets[batch, 0, curr_head[0] + span_offset] = 1.0

            batched_lst_multiheads.append(lst_multiheads)
        return coreflinker_mtt_targets, torch.tensor(target_mask_lengths, dtype=torch.int32, device=settings.device), \
               batched_lst_multiheads

    def create_coreflinker_mtt_target_mask_prod(self, pred_spans, gold_spans, gold_clusters, linker_targets,
                                                candidates, unique_links, unique_links_lengths):
        """
        This version is the extension of create_coreflinker_mtt_target_mask_multihead with some formulations in the
        comments of the following slide:
        https://docs.google.com/presentation/d/12vVEcWkg-BygOM_ui1l0jaE_JJ0RDYwW7wFBqPGgkvY/edit#slide=id.g18c18415074_0_78
        The idea is to use the output of this function to calculate the product of tree weights of all nil clusters:
        target score = ( MTT(mask_linked*score) * (mask_nil_cluster1_head1*score + mask_nil_cluster1_head2*score...) *
            * (mask_nil_cluster2_head1*score + mask_nil_cluster2_head2*score....) * ....)
        For this we need to return:
            1- the mask of linked entities ("mask_linked" in the comment above).
            2- A list of masks, each for a specific nil cluster (mask_nil_cluster1 for example). Where the first row
            is the root entry row to the cluster. So the mask for cluster with 2 mentios will be:
                 R  m1 m2
            R  [[0, 0, 0], --> 1s for each mx (head) will have to be added to get "mask_nil_cluster_headx"
            m1  [0, 0, 1],
            m2  [0, 1, 0]]  --> 0s in the main diagonal and all combinations possible between m1 and m2.
            The caller to this
            function will have to change the first row (root row) putting 1s in different heads to obtain
            the "mask_nil_cluster_headx" masks in the comment above.

        :param pred_spans:
        :param gold_spans:
        :param gold_clusters:
        :param linker_targets:
        :return:
            The original idea is to return two type of target masks (with respective indices):
                1- The mask for linked entities.
                2- A list of masks, each for a specific nil cluster
        """

        num_batch = len(pred_spans)  # 1
        max_spans = max([len(x) for x in pred_spans])  # 9

        coreflinker_mtt_targets = torch.zeros(num_batch, 1 + unique_links.shape[-1] + max_spans,
                                              1 + unique_links.shape[-1] + max_spans,
                                              dtype=torch.float32,
                                              device=settings.device)
        # coreflinker_mtt_targets.shape --> torch.Size([1, 89, 89])
        nr_correct_candidates_per_mention = linker_targets.sum(-1)
        target_mask_lengths = list()
        batched_lst_multiheads = list()
        # targets for span-span coref
        target_not_nil_indices = list()
        target_nil_clusters_indices = list()
        for batch, (pred, gold, clusters) in enumerate(zip(pred_spans, gold_spans, gold_clusters)):
            target_not_nil_indices_b = [0]
            # also adds the connections to unique links that we know are always present from our architecture
            #   first only will work in case batch size is of 1
            assert unique_links.shape[0] == 1
            target_not_nil_indices_b.extend(list(range(1, unique_links.shape[-1] + 1)))
            # this one not initialized with root because it will be a list of lists
            target_nil_clusters_indices_b = []
            # root to links always in 1
            coreflinker_mtt_targets[batch, 0, 1:unique_links_lengths[batch] + 1] = 1.0
            span_offset = 1 + unique_links_lengths[batch].item()  # span_offset --> 68
            curr_target_length = span_offset + len(pred)  # root + links + spans # 89
            target_mask_lengths.append(curr_target_length)
            gold2cluster = dict()
            for idx, span in enumerate(gold):
                gold2cluster[span] = clusters[idx].item()
            # gold2cluster --> <class 'dict'>: {(5, 6): 4, (7, 12): 5, (51, 53): 1, (55, 59): 2, (61, 61): 6, (62, 62): 7, (63, 67): 8, (68, 72): 9, (76, 76): 6}
            pred2cluster_struct = dict()
            if clusters.shape[-1] > 0:
                max_gold_cl_id = clusters.max().item()
            else:
                max_gold_cl_id = 0
            #
            to_assign_cl_id = max_gold_cl_id + 1
            #
            for idx_span1, span1 in enumerate(pred):
                if span1 not in gold2cluster:
                    new_cl_id = to_assign_cl_id
                    to_assign_cl_id += 1
                    pred2cluster_struct[new_cl_id] = {'cluster_id': to_assign_cl_id,
                                                      'spans': [(idx_span1, span1)],
                                                      # if it is not in gold2cluster, then there is no way to know
                                                      # if the link is valid or not
                                                      'is_valid_link': False}
                    #
                else:
                    cluster_id = gold2cluster[span1]
                    nr_correct_candidates_in_span = nr_correct_candidates_per_mention[batch, idx_span1].item()
                    is_valid_link = False
                    # target mask from link to spans: in theory nr_correct_candidates_in_span can be at most 1
                    # (only a single correct candidate)
                    if nr_correct_candidates_in_span > 0:
                        is_valid_link = True
                        correct_link_id = candidates[batch, idx_span1, linker_targets[batch, idx_span1] > 0.5].item()
                        offset_correct_link = (unique_links[batch] == correct_link_id).nonzero().item()
                        coreflinker_mtt_targets[batch, offset_correct_link + 1, idx_span1 + span_offset] = 1.0

                    # target mask from span to span
                    for idx_span2, span2 in enumerate(pred):
                        if idx_span2 != idx_span1 and span2 in gold2cluster and \
                                gold2cluster[span1] == gold2cluster[span2]:
                            # THIS NOT ALWAYS HAS TO BE ADDED, BECAUSE NIL CLUSTERS CAN BE HERE AS WELL!
                            coreflinker_mtt_targets[batch, idx_span1 + span_offset, idx_span2 + span_offset] = 1.0

                    if cluster_id not in pred2cluster_struct:
                        pred2cluster_struct[cluster_id] = {'cluster_id': cluster_id,
                                                           'spans': [(idx_span1, span1)],
                                                           'is_valid_link': False}
                    else:
                        pred2cluster_struct[cluster_id]['spans'].append((idx_span1, span1))

                    if is_valid_link:
                        pred2cluster_struct[cluster_id]['is_valid_link'] = True

            lst_multiheads = list()  # probably put this on top
            for cluster_struct in pred2cluster_struct.values():
                is_valid_link = cluster_struct['is_valid_link']
                if not is_valid_link:
                    curr_targets_nil_cluster = [0]
                    # puts the first one directly into coref_mtt_targets
                    spans = cluster_struct['spans']
                    for curr_head in spans:
                        curr_targets_nil_cluster.append(curr_head[0] + span_offset)
                        # if the partition is activated, then 1s have to be set to all the rooted nodes
                        if self.nil_partition_implementation:
                            coreflinker_mtt_targets[batch, 0, curr_head[0] + span_offset] = 1.0

                    curr_targets_nil_cluster = sorted(curr_targets_nil_cluster)
                    target_nil_clusters_indices_b.append(torch.tensor(curr_targets_nil_cluster,
                                                                      dtype=torch.long, device=settings.device))
                else:
                    for curr_span in cluster_struct['spans']:
                        target_not_nil_indices_b.append(span_offset + curr_span[0])

            target_not_nil_indices_b = sorted(target_not_nil_indices_b)
            target_not_nil_indices.append(torch.tensor(target_not_nil_indices_b, dtype=torch.long,
                                                       device=settings.device))
            target_nil_clusters_indices.append(target_nil_clusters_indices_b)

            batched_lst_multiheads.append(lst_multiheads)

        return coreflinker_mtt_targets, target_not_nil_indices, target_nil_clusters_indices, \
               torch.tensor(target_mask_lengths, dtype=torch.int32, device=settings.device)

    def decode_m2i_coreflinker_mtt(self, pred_masked_scores, pred_tree_mst,
                                   link_id_to_coref_col,
                                   dic, unique_links,
                                   unique_links_lengths, pred_spans):
        """

        :return: This function should return the "root" each of the mentions point to. The "root" can be either an "entity",
        in which case this would be a link. Or can be another mention; in which case it would be a mention that can not be linked
        or that does not have a link I guess.
        """
        span_to_pointers_detail_info = list()

        decoded_m2i_coref_linker = list()

        for batch, (b_unique_links, b_unique_links_lengths) in enumerate(zip(unique_links, unique_links_lengths)):
            b_link_id_to_coref_col = link_id_to_coref_col[batch]

            b_decoded_m2i_coref_linker = list()
            b_span_to_pointers_detail_info = dict()
            # -1 because we do not count the root node
            b_decoded_m2i_coref_linker.extend(
                [b_link_id_to_coref_col[link_id.item()] - 1 for link_id in b_unique_links])

            b_pred_tree_mst = pred_tree_mst[batch]
            b_pred_spans = pred_spans[batch]
            b_pred_masked_scores = pred_masked_scores[batch]
            my_edges = list(nx.dfs_edges(b_pred_tree_mst))

            nr_nodes = b_pred_masked_scores.shape[0]

            nr_links = b_unique_links_lengths.item()

            b_decoded_m2i_coref_linker.extend(list(range(nr_links, nr_nodes - 1)))
            assert len(b_pred_spans) == nr_nodes - nr_links - 1
            # -1 because we do not add the root node
            # assert len(b_decoded_m2i_coref_linker) == b_pred_matrix_mst.shape[0] - 1

            initial_node = 0
            pred_span_to_how_pointed = dict()
            for curr_edge in my_edges:
                if curr_edge[0] == 0:
                    initial_node = 0  # if the outcoming edge is from root, the initial_node is root (0)
                else:
                    if initial_node == 0:
                        initial_node = curr_edge[0]
                curr_input_node = curr_edge[1]
                if initial_node != 0:
                    # points to the first non-root node; -1 because the root node is not in b_decoded_m2i_coref_linker list
                    b_decoded_m2i_coref_linker[curr_input_node - 1] = initial_node - 1
                    # span_to_pointers_detail_info

                # fills in the information on how the span is connected
                if curr_input_node > nr_links:  # it is a span node
                    curr_span_id = curr_input_node - nr_links - 1
                    in_span = b_pred_spans[curr_span_id]
                    coref_connection_score = b_pred_masked_scores[curr_edge[0], curr_edge[1]].item()
                    if curr_edge[0] == 0:
                        pred_span_to_how_pointed[in_span] = {'coref_connection_type': 'root',
                                                             'coref_connection_pointer': in_span,
                                                             'coref_connection_score': coref_connection_score}
                    elif 0 < curr_edge[0] < nr_links + 1:
                        link_id = b_unique_links[curr_edge[0] - 1]
                        link_name = dic.get(link_id.item())
                        pred_span_to_how_pointed[in_span] = {'coref_connection_type': 'link',
                                                             'coref_connection_pointer': link_name,
                                                             'coref_connection_score': coref_connection_score}
                    elif curr_edge[0] > nr_links:
                        other_mention_span = b_pred_spans[curr_edge[0] - nr_links - 1]
                        pred_span_to_how_pointed[in_span] = {'coref_connection_type': 'mention_other',
                                                             'coref_connection_pointer': other_mention_span,
                                                             'coref_connection_score': coref_connection_score}
                    else:
                        raise Exception('This should not happen, something wrong with coreflinker_mtt 1!')
            for curr_span in b_pred_spans:
                if curr_span in pred_span_to_how_pointed:
                    b_span_to_pointers_detail_info[curr_span] = pred_span_to_how_pointed[curr_span]
                else:
                    b_span_to_pointers_detail_info[curr_span] = {'coref_connection_type': 'unknown',
                                                                 'coref_connection_pointer': None,
                                                                 'coref_connection_score': None}

            decoded_m2i_coref_linker.append(b_decoded_m2i_coref_linker)
            span_to_pointers_detail_info.append(b_span_to_pointers_detail_info)

        # example of span_to_pointer_detail_info:
        # <class 'dict'>:
        # {(1, 4): {'coref_connection_type': 'root', 'coref_connection_pointer': (1, 4), 'coref_connection_score': -5.024434566497803},
        # (6, 6): {'coref_connection_type': 'root', 'coref_connection_pointer': (6, 6), 'coref_connection_score': -4.879373073577881},
        # (9, 9): {'coref_connection_type': 'link', 'coref_connection_pointer': 'Cologne', 'coref_connection_score': 7.0420918464660645},
        # (10, 10): {'coref_connection_type': 'mention_other', 'coref_connection_pointer': (6, 6), 'coref_connection_score': -4.88646936416626},
        # (11, 11): {'coref_connection_type': 'root', 'coref_connection_pointer': (11, 11), 'coref_connection_score': 4.993085861206055},
        # (13, 13): {'coref_connection_type': 'link', 'coref_connection_pointer': 'Germany', 'coref_connection_score': 7.003166675567627},
        # (13, 16): {'coref_connection_type': 'root', 'coref_connection_pointer': (13, 16), 'coref_connection_score': 6.339137554168701},
        # (27, 30): {'coref_connection_type': 'root', 'coref_connection_pointer': (27, 30), 'coref_connection_score': 6.166959762573242},
        # (32, 32): {'coref_connection_type': 'link', 'coref_connection_pointer': 'Cologne', 'coref_connection_score': 6.726534366607666},
        # (35, 35): {'coref_connection_type': 'root', 'coref_connection_pointer': (35, 35), 'coref_connection_score': 5.208428859710693},
        # (36, 39): {'coref_connection_type': 'root', 'coref_connection_pointer': (36, 39), 'coref_connection_score': -5.176065444946289},
        # (42, 42): {'coref_connection_type': 'root', 'coref_connection_pointer': (42, 42), 'coref_connection_score': 6.069662570953369},
        # (45, 48): {'coref_connection_type': 'root', 'coref_connection_pointer': (45, 48), 'coref_connection_score': 6.158422470092773},
        # (58, 58): {'coref_connection_type': 'root', 'coref_connection_pointer': (58, 58), 'coref_connection_score': 6.718040466308594},
        # (59, 59): {'coref_connection_type': 'root', 'coref_connection_pointer': (59, 59), 'coref_connection_score': 5.985378265380859},
        # (60, 60): {'coref_connection_type': 'root', 'coref_connection_pointer': (60, 60), 'coref_connection_score': 6.3143792152404785}}
        # example of decoded_m2i_coref_linker:  [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 160, 68, 160, 1, 165, 166, 68, 168, 169, 160, 171, 172, 173, 174, 175]]
        return decoded_m2i_coref_linker, span_to_pointers_detail_info

    def get_mtt_cost_partition(self, targets_mask, pred_scores_mtt_exp_space, type='logdet'):
        targets_scores_exp = targets_mask * pred_scores_mtt_exp_space

        filtered_scores_exp = targets_scores_exp[1:, :][:, 1:]

        laplacian_tgt_scores = torch.eye(filtered_scores_exp.shape[-2], filtered_scores_exp.shape[-1],
                                         dtype=self.torch_float_precision, device=settings.device)
        laplacian_tgt_scores = laplacian_tgt_scores * filtered_scores_exp.sum(dim=-2)  # main diagonal
        laplacian_tgt_scores = laplacian_tgt_scores + (filtered_scores_exp * -1.0)
        laplacian_tgt_scores[0, :] = targets_scores_exp[0][1:]

        if type == 'logdet':
            mtt_cost = torch.logdet(laplacian_tgt_scores)
        else:
            raise RuntimeError('not implemented type in get_mtt_cost' + type)
        return mtt_cost

    def get_mtt_cost(self, targets_mask, pred_scores_mtt_exp_space, type='logdet'):
        targets_scores_exp = targets_mask * pred_scores_mtt_exp_space

        laplacian_tgt_scores = torch.eye(targets_scores_exp.shape[-2], targets_scores_exp.shape[-1],
                                         dtype=self.torch_float_precision, device=settings.device)
        laplacian_tgt_scores = laplacian_tgt_scores * targets_scores_exp.sum(dim=-2)  # main diagonal
        laplacian_tgt_scores = laplacian_tgt_scores + (targets_scores_exp * -1.0)

        if type == 'logdet':
            mtt_cost = torch.logdet(laplacian_tgt_scores[1:, 1:])
        elif type == 'det':
            mtt_cost = torch.det(laplacian_tgt_scores[1:, :][:, 1:])
        else:
            raise RuntimeError('not implemented type in get_mtt_cost' + type)
        return mtt_cost

    def get_mtt_loss(self, targets_mask, pred_scores_mtt, z_mask, torch_float_precision):

        if self.smart_arsinh:
            if self.enforce_scores:
                pred_scores_mtt[pred_scores_mtt > self.max_score_max] = self.max_score_max
                pred_scores_mtt[pred_scores_mtt < self.min_score_min] = self.min_score_min
                targets_scores = targets_mask * pred_scores_mtt
            else:
                targets_scores = targets_mask * pred_scores_mtt
        else:
            targets_scores = targets_mask * pred_scores_mtt

        targets_scores_exp = targets_scores
        assert targets_scores_exp[0].diagonal().sum() == 0.0
        # TODO: instead of unsqueeze(0), implement it in a batched way!!
        laplacian_tgt_scores = torch.eye(targets_scores_exp.shape[-2], targets_scores_exp.shape[-1],
                                         dtype=torch_float_precision, device=settings.device).unsqueeze(0)
        laplacian_tgt_scores = laplacian_tgt_scores * targets_scores_exp.sum(dim=-2)  # main diagonal
        laplacian_tgt_scores += (targets_scores_exp * -1.0)

        if self.smart_arsinh:
            if self.enforce_scores:
                z_score_enforcer = torch.zeros_like(pred_scores_mtt)
                z_score_enforcer[z_mask.bool()] = pred_scores_mtt[0].min().item()
                z_score_enforcer[targets_mask.bool()] = pred_scores_mtt.max().item()
                z_scores = z_mask * z_score_enforcer
            else:
                z_scores = z_mask * pred_scores_mtt
        else:
            z_scores = z_mask * pred_scores_mtt

        # TODO: instead of unsqueeze(0), implement it in a batched way!!
        z_scores_exp = z_scores
        assert z_scores_exp[0].diagonal().sum() == 0.0
        laplacian_z_scores = torch.eye(z_scores_exp.shape[-2], z_scores_exp.shape[-1],
                                       device=settings.device, dtype=torch_float_precision).unsqueeze(0)
        laplacian_z_scores = laplacian_z_scores * z_scores_exp.sum(dim=-2)  # main diagonal
        laplacian_z_scores += (z_scores_exp * -1.0)

        # get the mtts
        laplacian_tgt_scores_exp = laplacian_tgt_scores
        mtt_det_tgt_scores = torch.slogdet(laplacian_tgt_scores_exp[0, 1:, 1:])
        laplacian_z_scores_exp = laplacian_z_scores
        mtt_det_z_scores = torch.slogdet(laplacian_z_scores_exp[0, 1:, 1:])

        curr_loss = mtt_det_z_scores[1] - mtt_det_tgt_scores[1]

        if torch.isinf(curr_loss):
            logger.warning('!!!!WARNING, CURR LOSS IN INF, setting to 0!!! mtt_det_z_scores[1] %s '
                           'mtt_det_tgt_scores[1] %s' % (mtt_det_z_scores[1], mtt_det_tgt_scores[1]))
            curr_loss = None
        elif torch.isnan(curr_loss):
            logger.warning('!!!!WARNING, CURR LOSS IN NAN, setting to 0!!! mtt_det_z_scores[1] %s  '
                           'mtt_det_tgt_scores[1] %s', (mtt_det_z_scores[1], mtt_det_tgt_scores[1]))
            curr_loss = None

        return curr_loss

    def forward(self, scores, gold_m2i, filtered_spans, gold_spans, linker,
                predict=False, pruner_spans=None, ner_spans=None, api_call=False, only_loss=False):
        output = {}
        output_coref = {}
        output_linking = {}

        if self.enabled and scores is not None:
            linker_candidates = linker['candidates']
            candidate_lengths = linker['candidate_lengths']
            targets = linker.get('targets')
            prune_indices_hoi = filtered_spans['prune_indices_hoi']
            # linker_candidates.shape --> torch.Size([1, 335, 16])
            # prune_indices_hoi.shape --> torch.Size([1, 21])
            linker_candidates = batched_index_select(linker_candidates, prune_indices_hoi)
            # linker_candidates.shape --> torch.Size([1, 21, 16])
            # targets.shape --> torch.Size([1, 335, 16])
            targets = batched_index_select(targets, prune_indices_hoi)
            # targets.shape --> torch.Size([1, 21, 16])
            # candidate_lengths.shape --> torch.Size([1, 335])
            candidate_lengths = batched_index_select(candidate_lengths.unsqueeze(-1), prune_indices_hoi).squeeze(-1)
            # candidate_lengths.shape --> torch.Size([1, 21])
            if not self.end_to_end:
                raise RuntimeError('LossCorefLinkerMTTHoi has not end-to-end')

            candidates = linker_candidates.to(settings.device)  # torch.Size([1, 9, 17])
            # candidates.shape --> torch.Size([1, 21, 16])

            linker_mask = create_candidate_mask(candidates.size(-1), candidate_lengths).float().to(settings.device)
            # linker_mask.shape --> torch.Size([1, 21, 16])

            unique_links_batches = list()
            unique_links_lengths = list()
            nill_id = self.entity_dictionary.lookup('NILL')
            for content_batch in candidates.view(candidates.shape[0], -1):
                unique_batch_content = torch.unique(content_batch)
                unique_batch_content = unique_batch_content[unique_batch_content != self.unknown_dict]
                unique_batch_content = unique_batch_content[unique_batch_content != nill_id]
                unique_links_lengths.append(unique_batch_content.shape[0])
                unique_links_batches.append(unique_batch_content)
            unique_links = torch.nn.utils.rnn.pad_sequence(unique_links_batches, batch_first=True)
            unique_links_lengths = torch.tensor(unique_links_lengths, dtype=torch.int32, device=settings.device)

            linker_target = targets * linker_mask

            pruned_spans = filtered_spans['pruned_spans']
            # len(pruned_spans[0]) --> 21
            # targets_mask.shape --> torch.Size([1, 89, 89])
            #
            #
            #
            z_mask, z_mask_lengths = create_coreflinker_mtt_z_mask_indexed(pruned_spans, gold_spans, gold_m2i,
                                                                           linker_target,
                                                                           candidates,
                                                                           candidate_lengths=candidate_lengths,
                                                                           unique_links=unique_links,
                                                                           unique_links_lengths=unique_links_lengths,
                                                                           unknown_id=self.unknown_dict)
            # z_mask.shape --> torch.Size([1, 89, 89])

            if self.float_precision == 'float64':
                scores = scores.double()

            # scores to mtt matrix,
            pred_scores_mtt = self.create_scores_mtt_pred(scores, unique_links, candidates, candidate_lengths,
                                                          torch_float_precision=self.torch_float_precision)

            # first passes to the exponential space
            if self.nonlinear_function == 'arsinh' and self.smart_arsinh:
                pred_scores_mtt_exp_space = get_smart_arsinh_exp(pred_scores_mtt)
            elif self.nonlinear_function == 'arsinh' and not self.smart_arsinh:
                pred_scores_mtt_exp_space = torch.exp(torch.arcsinh(pred_scores_mtt))
            else:
                pred_scores_mtt_exp_space = torch.exp(pred_scores_mtt)

            if self.print_debugging:

                if self.nonlinear_function == 'arsinh':
                    to_analyze = torch.arcsinh(pred_scores_mtt) * z_mask
                else:
                    to_analyze = pred_scores_mtt * z_mask
                if unique_links.shape[1] > 0:
                    to_analyze_link_span = to_analyze[0, 1:1 + unique_links.shape[1], :][:,
                                           1 + unique_links.shape[1]:]
                    to_analyze_link_span_mask = z_mask[0, 1:1 + unique_links.shape[1], :][:,
                                                1 + unique_links.shape[1]:]
                    to_analyze_link_span = to_analyze_link_span[to_analyze_link_span_mask.bool()]
                    norm_link_span = torch.norm(to_analyze_link_span).item()
                    mean_link_span = torch.mean(to_analyze_link_span).item()
                    std_link_span = torch.std(to_analyze_link_span).item()
                    min_link_span = torch.min(to_analyze_link_span).item()
                    max_link_span = torch.max(to_analyze_link_span).item()
                    self.stat_link_span_norm.append(norm_link_span)
                    self.stat_link_span_mean.append(mean_link_span)
                    self.stat_link_span_std.append(std_link_span)
                    self.stat_link_span_min.append(min_link_span)
                    self.stat_link_span_max.append(max_link_span)

                to_analyze_span_span = to_analyze[0, 1 + unique_links.shape[1]:, :][:,
                                       1 + unique_links.shape[1]:]

                to_analyze_span_span_mask = z_mask[0, 1 + unique_links.shape[1]:, :][:,
                                            1 + unique_links.shape[1]:]
                to_analyze_span_span = to_analyze_span_span[to_analyze_span_span_mask.bool()]
                norm_span_span = torch.norm(to_analyze_span_span).item()
                mean_span_span = torch.mean(to_analyze_span_span).item()
                std_span_span = torch.std(to_analyze_span_span).item()
                min_span_span = torch.min(to_analyze_span_span).item()
                max_span_span = torch.max(to_analyze_span_span).item()
                self.stat_span_span_norm.append(norm_span_span)
                self.stat_span_span_mean.append(mean_span_span)
                self.stat_span_span_std.append(std_span_span)
                self.stat_span_span_min.append(min_span_span)
                self.stat_span_span_max.append(max_span_span)

            if self.float_precision == 'float64':
                z_mask = z_mask.double()

            tot_loss = None
            if self.multihead_nil == 'multihead_old':
                targets_mask, target_mask_lengths, batched_multiheads = \
                    self.create_coreflinker_mtt_target_mask_multihead(pruned_spans, gold_spans, gold_m2i, linker_target,
                                                                      candidates, unique_links, unique_links_lengths)
                #
                # want to be sure that the dimensions match to the ones expected, if no match, then print the details of the problem
                expected_dim = candidates.shape[-2] + 1 + unique_links.shape[-1]
                if expected_dim != pred_scores_mtt_exp_space.shape[-1] or expected_dim != \
                        pred_scores_mtt_exp_space.shape[-2] or \
                        targets_mask.shape[-1] != pred_scores_mtt_exp_space.shape[-1] or \
                        targets_mask.shape[-2] != pred_scores_mtt_exp_space.shape[-2]:
                    logger.error(
                        '!!!ERROR IN DIMENSIONS!!! SOMETHING GOT WRONG, printing the details of hyperparameters')
                    logger.error('the expected dim is: %s' % expected_dim)
                    logger.error('the shape in pred_scores_mtt_exp_space is: %s' % pred_scores_mtt_exp_space.shape)
                    logger.error('target mask.shape: %s' % targets_mask.shape)
                    logger.error('scores.shape: %s' % scores.shape)
                    logger.error('unique_links.shape: %s' % unique_links.shape)
                    logger.error('unique_links content: %s' % list(unique_links[0]))
                    logger.error('candidates.shape: %s' % candidates.shape)
                    logger.error('candidates content: %s' % list(candidates[0]))
                    logger.error('candidate_lengths.shape: %s' % candidate_lengths.shape)
                    logger.error('candidate_lengths content: %s' % list(candidate_lengths))

                    # here serializes the model and other objects parameters to the function
                    torch.save({'scores': scores,
                                'gold_m2i': gold_m2i,
                                'filtered_spans': filtered_spans,
                                'gold_spans': gold_spans,
                                'linker': linker,
                                'predict': predict,
                                'pruner_spans': pruner_spans,
                                'ner_spans': ner_spans,
                                'api_call': api_call,
                                'model_state': self.state_dict()}, 'failed_model_scores.bin')

                tot_loss = self.get_mtt_loss(targets_mask=targets_mask, pred_scores_mtt=pred_scores_mtt_exp_space,
                                             z_mask=z_mask, torch_float_precision=self.torch_float_precision)

                for curr_multihead in batched_multiheads[0]:
                    curr_targets_mask = curr_multihead['mtt_targets']
                    indices = curr_multihead['indices']
                    curr_pred_scores_mtt = pred_scores_mtt_exp_space[0, indices, :][:, indices]
                    curr_z_mask = z_mask[0, indices, :][:, indices]
                    curr_loss = self.get_mtt_loss(targets_mask=curr_targets_mask, pred_scores_mtt=curr_pred_scores_mtt,
                                                  z_mask=curr_z_mask, torch_float_precision=self.torch_float_precision)
                    if curr_loss is not None:
                        if tot_loss is not None:
                            tot_loss = tot_loss + curr_loss
                        else:
                            tot_loss = curr_loss
            elif self.multihead_nil == 'multihead_prod':
                if predict and not only_loss:
                    if self.print_debugging_matrices:
                        pass

                targets_mask, target_not_nil_indices, target_nil_clusters_indices, target_mask_lengths = \
                    self.create_coreflinker_mtt_target_mask_prod(pruned_spans, gold_spans, gold_m2i, linker_target,
                                                                 candidates, unique_links, unique_links_lengths)
                # for now batch size 1
                assert len(target_not_nil_indices) == 1
                assert len(target_nil_clusters_indices) == 1
                target_not_nil_indices = target_not_nil_indices[0]
                target_nil_clusters_indices = target_nil_clusters_indices[0]
                z_log_cost = self.get_mtt_cost(z_mask[0], pred_scores_mtt_exp_space[0], type='logdet')
                tot_loss = z_log_cost

                if self.print_debugging_matrices and (predict and not only_loss):
                    debug_nil_clusters_weights = list()
                    debug_nil_cluster_idxs_in_matrix = list()
                    debug_not_nil_idxs_in_matrix = list()

                if self.print_debugging_matrices and (predict and not only_loss):
                    not_nil_log_cost = torch.Tensor([0.0])
                # if there are actually not nil clusters predicted:
                if target_not_nil_indices.shape[-1] > 1:
                    if self.print_debugging_matrices and (predict and not only_loss):
                        debug_not_nil_idxs_in_matrix = target_not_nil_indices.tolist()

                    targets_mask_not_nil = targets_mask[0, target_not_nil_indices, :][:, target_not_nil_indices]
                    pred_scores_not_nil = pred_scores_mtt_exp_space[0, target_not_nil_indices, :][:,
                                          target_not_nil_indices]
                    not_nil_log_cost = self.get_mtt_cost(targets_mask_not_nil, pred_scores_not_nil, type='logdet')
                    tot_loss = tot_loss - not_nil_log_cost

                tot_log_cost_nil_clusters = None

                if self.nil_partition_implementation:
                    # implementation based on "3.1. Partition functions via matrix determinants" of
                    # https://www.aclweb.org/anthology/D07-1015/
                    # print('nil_partition_implementation to implement')
                    for curr_nil_cluster_indices in target_nil_clusters_indices:
                        curr_targets_mask = targets_mask[0, curr_nil_cluster_indices, :][:, curr_nil_cluster_indices]
                        curr_pred_scores = pred_scores_mtt_exp_space[0, curr_nil_cluster_indices, :][:,
                                           curr_nil_cluster_indices]
                        curr_nil_cost_logdet = self.get_mtt_cost_partition(curr_targets_mask, curr_pred_scores,
                                                                           type='logdet')
                        if tot_log_cost_nil_clusters is None:
                            tot_log_cost_nil_clusters = curr_nil_cost_logdet
                        else:
                            tot_log_cost_nil_clusters = tot_log_cost_nil_clusters + curr_nil_cost_logdet

                        if self.print_debugging_matrices and (predict and not only_loss):
                            debug_nil_clusters_weights.append(curr_nil_cost_logdet.item())
                            debug_nil_cluster_idxs_in_matrix.append(curr_nil_cluster_indices.tolist())
                else:
                    for curr_nil_cluster_indices in target_nil_clusters_indices:
                        curr_targets_mask = targets_mask[0, curr_nil_cluster_indices, :][:, curr_nil_cluster_indices]
                        # curr_targets_mask = curr_targets_mask
                        curr_pred_scores = pred_scores_mtt_exp_space[0, curr_nil_cluster_indices, :][:,
                                           curr_nil_cluster_indices]
                        nr_spans = curr_nil_cluster_indices.shape[-1] - 1

                        tot_cost_curr_cluster = None
                        # to be used for exp_trick, different var to first check if this trick actually gives equal loss
                        tot_cost_curr_cluster_logdet = []
                        max_log_det = None
                        # from 1 and to nr_spans + 1 because of the root node
                        for curr_span_idx in range(1, nr_spans + 1):

                            # detach() doesn't seem to work, so using clone()
                            curr_targets_mask_loop = curr_targets_mask.clone()
                            curr_targets_mask_loop[0, :] = 0  # 0 from root to the rest of the spans
                            curr_targets_mask_loop[0, curr_span_idx] = 1
                            # NOT log cost, but just determinant based cost of a particular connection!
                            if self.exp_trick:
                                curr_nil_cost_logdet = self.get_mtt_cost(curr_targets_mask_loop, curr_pred_scores,
                                                                         type='logdet')
                                if torch.isinf(curr_nil_cost_logdet):
                                    logger.warning('WARNING, torch.isinf(curr_nil_cost_logdet)')
                                    continue
                                if torch.isnan(curr_nil_cost_logdet):
                                    logger.warning('WARNING, torch.isnan(curr_nil_cost_logdet)')
                                    continue

                                tot_cost_curr_cluster_logdet.append(curr_nil_cost_logdet)
                                if max_log_det is None:
                                    max_log_det = curr_nil_cost_logdet
                                elif max_log_det < curr_nil_cost_logdet:
                                    max_log_det = curr_nil_cost_logdet
                            else:
                                curr_nil_cost = self.get_mtt_cost(curr_targets_mask_loop, curr_pred_scores, type='det')
                                if tot_cost_curr_cluster is None:
                                    tot_cost_curr_cluster = curr_nil_cost
                                else:
                                    tot_cost_curr_cluster = tot_cost_curr_cluster + curr_nil_cost

                        if self.exp_trick:
                            if max_log_det is None or torch.isnan(max_log_det):
                                logger.warning('WARNING, torch.isnan(max_log_det) or None: %s' % max_log_det)
                                continue

                            tot_log_cost_nil_clusters_exp_trick = max_log_det
                            sum_exp = None
                            for curr_term in tot_cost_curr_cluster_logdet:
                                if torch.isnan(torch.exp(curr_term - max_log_det)):
                                    logger.warning('WARNING, torch.isnan(torch.exp(curr_term - max_log_det))')
                                    continue

                                if sum_exp is None:
                                    sum_exp = torch.exp(curr_term - max_log_det)
                                else:
                                    sum_exp = sum_exp + torch.exp(curr_term - max_log_det)
                            log_sum_exp = torch.log(sum_exp)

                            if torch.isnan(log_sum_exp):
                                logger.warning('WARNING, torch.isnan(log_sum_exp)')

                            tot_log_cost_nil_clusters_exp_trick = tot_log_cost_nil_clusters_exp_trick + log_sum_exp

                            if torch.isnan(tot_log_cost_nil_clusters_exp_trick):
                                logger.warning('WARNING, torch.isnan(tot_log_cost_nil_clusters_exp_trick)')

                            if tot_log_cost_nil_clusters is None:
                                tot_log_cost_nil_clusters = tot_log_cost_nil_clusters_exp_trick
                            else:
                                # not product, but sum because in log space
                                tot_log_cost_nil_clusters = tot_log_cost_nil_clusters + tot_log_cost_nil_clusters_exp_trick
                            if self.print_debugging_matrices and (predict and not only_loss):
                                debug_nil_clusters_weights.append(tot_log_cost_nil_clusters_exp_trick.item())
                        else:
                            assert tot_cost_curr_cluster is not None

                            if tot_log_cost_nil_clusters is None:
                                tot_log_cost_nil_clusters = torch.log(tot_cost_curr_cluster)
                            else:
                                # not product, but sum because in log space
                                tot_log_cost_nil_clusters = tot_log_cost_nil_clusters + torch.log(tot_cost_curr_cluster)
                if tot_log_cost_nil_clusters is not None:
                    if torch.isnan(tot_log_cost_nil_clusters):
                        logger.warning('WARNING, torch.isnan(tot_log_cost_nil_clusters)')

                    tot_loss = tot_loss - tot_log_cost_nil_clusters

                # """
                # {
                # "scores":[[0,-inf,+10],..],
                # "clusters":[[1,2],[3],[4,5,6]],
                # "items":["ROOT", "Brussels_(city)", "brussel@42:44", ""],
                # "obj":-3.4545,
                # "numer":4.0,
                # "denom":5.0
                # }"""
                else:
                    logger.warning('WARNING, tot_log_cost_nil_clusters in None')

            else:
                raise RuntimeError('multihead_nil type not recognized in forward of coreflinker_mtt_hoi: ' +
                                   self.multihead_nil)

            if tot_loss is None:
                tot_loss = 0

            if self.print_debugging:
                if isinstance(tot_loss, torch.Tensor):
                    self.mtt_hoi_loss.append(tot_loss.item())
                else:
                    self.mtt_hoi_loss.append(tot_loss)

            output['loss'] = tot_loss
            output_coref['loss'] = tot_loss
            output_linking['loss'] = tot_loss

            if predict and not only_loss:

                nill_id = self.entity_dictionary.lookup('NILL')
                if self.nonlinear_function == 'arsinh':
                    pred_masked_scores = torch.arcsinh(pred_scores_mtt)
                else:
                    pred_masked_scores = pred_scores_mtt

                # the inverse of mask receives a very small score (less than the current minimum)
                if self.log_inf_mask:
                    # just another way of putting very low number (-inf) to masked scores, trying it out
                    pred_masked_scores = pred_masked_scores + torch.log(z_mask)
                else:
                    z_mask_inverse = 1.0 - z_mask
                    min_pred_mask_scores = min(pred_masked_scores.min().item(), 0.0)
                    pred_masked_scores = pred_masked_scores + (z_mask_inverse) * (min_pred_mask_scores - 999999)

                pred_tree_mst = mst_only_tree(pred_masked_scores, target_mask_lengths, z_mask)

                candidate_ids = []
                for curr_cand_batch, _ in enumerate(candidates):
                    unique_curr_batch = candidates[curr_cand_batch].unique(sorted=True)
                    unique_curr_batch = unique_curr_batch[unique_curr_batch != nill_id]
                    # (16/10/2020) - 0 is used for padding, so remove it
                    unique_curr_batch = unique_curr_batch[unique_curr_batch != self.unknown_dict]
                    candidate_ids.append(unique_curr_batch)

                link_id_to_coref_col = list()
                for batch_id, candidate_ids_batch in enumerate(candidate_ids):
                    link_id_to_coref_col.append(dict())
                    for matrix_idx_link, link_dict_id in enumerate(candidate_ids_batch):
                        # + 1 because the first one is root
                        link_id_to_coref_col[batch_id][link_dict_id.item()] = matrix_idx_link + 1

                coref_col_to_link_id = list()
                for batch_id, link_id_to_coref_col_batch in enumerate(link_id_to_coref_col):
                    coref_col_to_link_id.append({v: k for k, v in link_id_to_coref_col_batch.items()})

                decoded_m2i_coref_linker, span_to_pointer_detail_info = \
                    self.decode_m2i_coreflinker_mtt(pred_masked_scores,
                                                    pred_tree_mst,
                                                    link_id_to_coref_col=link_id_to_coref_col,
                                                    dic=self.entity_dictionary,
                                                    unique_links=unique_links,
                                                    unique_links_lengths=unique_links_lengths,
                                                    pred_spans=pruned_spans)
                # print('THIS PRINT HAS TO BE DELETED!')
                # logits.shape --> torch.Size([1, 14, 30])
                # lengths_coref --> tensor([14])
                # lengths_linker --> tensor([16])
                # linker_candidates.shape --> torch.Size([1, 14, 16])
                # candidate_ids[0].shape --> torch.Size([49])
                # link_id_to_coref_col --> <class 'list'>: [{2237: 0, 2552: 1, 10719: 2, 10720: 3, 11729: 4, 11734: 5, 11735: 6, 11736: 7, 11737: 8, 11738: 9, 11739: 10, 11740: 11, 11741: 12, 11742: 13, 11743: 14, 11744: 15, 11745: 16, 11746: 17, 11747: 18, 11748: 19, 14221: 20, 25253: 21, 34142: 22, 34210: 23, 34211: 24, 34213: 25, 34214: 26, 34215: 27, 34216: 28, 34217: 29, 34218: 30, 34219: 31, 34220: 32, 118110: 33, 118129: 34, 118130: 35, 118131: 36, 118132: 37, 118133: 38, 118134: 39, 118135: 40, 118136: 41, 118137: 42, 118138: 43, 118139: 44, 118140: 45, 118141: 46, 118142: 47, 118143: 48}]
                # coref_col_to_link_id --> <class 'list'>: [{0: 2237, 1: 2552, 2: 10719, 3: 10720, 4: 11729, 5: 11734, 6: 11735, 7: 11736, 8: 11737, 9: 11738, 10: 11739, 11: 11740, 12: 11741, 13: 11742, 14: 11743, 15: 11744, 16: 11745, 17: 11746, 18: 11747, 19: 11748, 20: 14221, 21: 25253, 22: 34142, 23: 34210, 24: 34211, 25: 34213, 26: 34214, 27: 34215, 28: 34216, 29: 34217, 30: 34218, 31: 34219, 32: 34220, 33: 118110, 34: 118129, 35: 118130, 36: 118131, 37: 118132, 38: 118133, 39: 118134, 40: 118135, 41: 118136, 42: 118137, 43: 118138, 44: 118139, 45: 118140, 46: 118141, 47: 118142, 48: 118143}]
                # pruned_spans --> <class 'list'>: [[(3, 4), (3, 7), (5, 6), (5, 7), (41, 43), (45, 47), (49, 49), (49, 50), (50, 50), (50, 52), (51, 52), (51, 54), (53, 54), (57, 57)]]
                # scores.shape --> torch.Size([1, 14, 30])

                # example of decoded_m2i_coref_linker:
                #   <class 'list'>: [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                #   23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                #   48, 49, 50, 51, 52, 53, 54, 5, 56, 57, 58, 59, 60, 61, 5]]
                #
                #
                # example of span_to_pointer_detail_info:
                #   <class 'list'>:
                #   [{(3, 4): {'coref_connection_type': 'mention_self', 'coref_connection_pointer': (3, 4), 'coref_connection_score': -0.0},
                #   (3, 7): {'coref_connection_type': 'mention_self', 'coref_connection_pointer': (3, 7), 'coref_connection_score': -0.0},
                #   (5, 6): {'coref_connection_type': 'mention_self', 'coref_connection_pointer': (5, 6), 'coref_connection_score': -0.0},
                #   (5, 7): {'coref_connection_type': 'mention_self', 'coref_connection_pointer': (5, 7), 'coref_connection_score': -0.0},
                #   (41, 43): {'coref_connection_type': 'mention_self', 'coref_connection_pointer': (41, 43), 'coref_connection_score': -0.0},
                #   (45, 47): {'coref_connection_type': 'mention_self', 'coref_connection_pointer': (45, 47), 'coref_connection_score': -0.0},
                #   (49, 49): {'coref_connection_type': 'link', 'coref_connection_pointer': 'Berlin'},
                #   (49, 50): {'coref_connection_type': 'mention_self', 'coref_connection_pointer': (49, 50), 'coref_connection_score': -0.0},
                #   (50, 50): {'coref_connection_type': 'mention_self', 'coref_connection_pointer': (50, 50), 'coref_connection_score': -0.0},
                #   (50, 52): {'coref_connection_type': 'mention_self', 'coref_connection_pointer': (50, 52), 'coref_connection_score': -0.0},
                #   (51, 52): {'coref_connection_type': 'mention_self', 'coref_connection_pointer': (51, 52), 'coref_connection_score': -0.0},
                #   (51, 54): {'coref_connection_type': 'mention_self', 'coref_connection_pointer': (51, 54), 'coref_connection_score': -0.0},
                #   (53, 54): {'coref_connection_type': 'mention_self', 'coref_connection_pointer': (53, 54), 'coref_connection_score': -0.0},
                #   (57, 57): {'coref_connection_type': 'link', 'coref_connection_pointer': 'Berlin'}}]

                # here gets the coref cluster spans only
                output_pred = list()
                if scores is not None:
                    for decoded_m2i_b, pruned_spans_b, candidate_ids_b, coref_col_to_link_id_b in \
                            zip(decoded_m2i_coref_linker, pruned_spans, candidate_ids, coref_col_to_link_id):
                        # len(decoded_m2i_b) --> 176
                        # len(pruned_spans_b) --> 16
                        # candidate_ids_b.shape --> 160
                        # len(coref_col_to_link_id_b) --> 160
                        var1 = m2i_to_clusters_linkercoref_mtt(decoded_m2i_b, coref_col_to_link_id_b,
                                                               self.entity_dictionary,
                                                               candidate_ids_b.size(-1))
                        # var1 example:
                        # var1[0] --> clusters --> <class 'list'>: [[0, 148], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16], [17], [18], [19], [20], [21], [22], [23], [24], [25], [26], [27], [28], [29], [30], [31], [32], [33], [34], [35], [36], [37], [38], [39], [40], [41], [42], [43], [44], [45], [46], [47], [48], [49], [50], [51], [52, 146, 152], [53], [54], [55], [56], [57], [58], [59], [60], [61], [62], [63], [64], [65], [66], [67], [68], [69], [70], [71], [72], [73], [74], [75], [76], [77], [78], [79], [80], [81], [82], [83], [84], [85], [86], [87], [88], [89], [90], [91], [92], [93], [94], [95], [96], [97], [98], [99], [100], [101], [102], [103], [104], [105], [106], [107], [108], [109], [110], [111], [112], [113], [114], [115], [116], [117], [118], [119], [120], [121], [122], [123], [124], [125], [126], [127], [128], [129], [130], [131], [132], [133], [134], [135], [136], [137], [138], [139], [140], [141], [142], [143], [144], [145, 147], [149], [150], [151], [153], [154], [155], [156], [157], [158], [159]]
                        # var1[1] --> mentions to clusters --> <class 'dict'>: {0: [0, 148], 1: [1], 2: [2], 3: [3], 4: [4], 5: [5], 6: [6], 7: [7], 8: [8], 9: [9], 10: [10], 11: [11], 12: [12], 13: [13], 14: [14], 15: [15], 16: [16], 17: [17], 18: [18], 19: [19], 20: [20], 21: [21], 22: [22], 23: [23], 24: [24], 25: [25], 26: [26], 27: [27], 28: [28], 29: [29], 30: [30], 31: [31], 32: [32], 33: [33], 34: [34], 35: [35], 36: [36], 37: [37], 38: [38], 39: [39], 40: [40], 41: [41], 42: [42], 43: [43], 44: [44], 45: [45], 46: [46], 47: [47], 48: [48], 49: [49], 50: [50], 51: [51], 52: [52, 146, 152], 53: [53], 54: [54], 55: [55], 56: [56], 57: [57], 58: [58], 59: [59], 60: [60], 61: [61], 62: [62], 63: [63], 64: [64], 65: [65], 66: [66], 67: [67], 68: [68], 69: [69], 70: [70], 71: [71], 72: [72], 73: [73], 74: [74], 75: [75], 76: [76], 77: [77], 78: [78], 79: [79], 80: [80], 81: [81], 82: [82], 83: [83], 84: [84], 85: [85], 86: [86], 87: [87], 88: [88], 89: [89], 90: [90], 91: [91], 92: [92], 93: [93], 94: [94], 95: [95], 96: [96], 97: [97], 98: [98], 99: [99], 100: [100], 101: [101], 102: [102], 103: [103], 104: [104], 105: [105], 106: [106], 107: [107], 108: [108], 109: [109], 110: [110], 111: [111], 112: [112], 113: [113], 114: [114], 115: [115], 116: [116], 117: [117], 118: [118], 119: [119], 120: [120], 121: [121], 122: [122], 123: [123], 124: [124], 125: [125], 126: [126], 127: [127], 128: [128], 129: [129], 130: [130], 131: [131], 132: [132], 133: [133], 134: [134], 135: [135], 136: [136], 137: [137], 138: [138], 139: [139], 140: [140], 141: [141], 142: [142], 143: [143], 144: [144], 145: [145, 147], 146: [52, 146, 152], 147: [145, 147], 148: [0, 148], 149: [149], 150: [150], 151: [151], 152: [52, 146, 152], 153: [153], 154: [154], 155: [155], 156: [156], 157: [157], 158: [158], 159: [159]}
                        var2 = convert_coref_mtt(var1, pruned_spans_b, number_candidates=candidate_ids_b.size(-1),
                                                 links_dictionary=self.entity_dictionary,
                                                 coref_col_to_link_id=coref_col_to_link_id_b)
                        # var2 example:
                        output_pred.append(var2)
                else:
                    output_pred = [[] for _ in pruned_spans]

                output_coref['pred'] = [out[0] for out in output_pred]
                output_coref['pred_pointers'] = span_to_pointer_detail_info
                output_linking['pred'] = [out[1] for out in output_pred]
                output_linking['spans_with_link'] = [out[2] for out in output_pred]

                # this gives correct answer; this is why it is used instead of linker_candidates.size(-1)
                max_nr_candidates = candidate_lengths.max().item()

                linker_spans = filtered_spans['pruned_spans']

                cols_to_ignore = 0  # in case we want to use the matrix itself to filter incorrect mentions

                # + 1 because also the link score to root is counted
                s = predict_scores_mtt(scores[:, :, cols_to_ignore:max_nr_candidates + 1], linker_spans,
                                       linker_candidates, candidate_lengths, self.entity_dictionary)

                output_coref['scores'] = predict_scores_coref_mtt(scores[:, :, max_nr_candidates + cols_to_ignore + 1:],
                                                                  pred_spans=pruned_spans)

                output_linking['scores'] = s

                if not api_call:
                    output_coref['gold'] = [convert_coref(m2i_to_clusters_linkercoref(x.tolist()), y,
                                                          number_candidates=0,
                                                          links_dictionary=self.entity_dictionary)[0] for x, y in
                                            zip(gold_m2i, gold_spans)]

                    output_linking['gold'] = linker['gold']
                else:
                    output_coref['gold'] = [None for _ in gold_spans]
                    output_linking['gold'] = [None for _ in gold_spans]

                if self.filter_singletons_with_pruner:
                    if self.print_debugging_matrices:
                        coref_pred_all = output_coref['pred'].copy()
                        link_pred_all = output_linking['pred'].copy()
                    # this assumes that pruner is able to predict spans
                    if not self.filter_only_singletons:
                        # does the old stuff
                        output_coref['pred'] = remove_disabled_spans(output_coref['pred'], pruner_spans)
                        coref_flat = [{item for sublist in batch for item in sublist} for batch in output_coref['pred']]
                        output_linking['pred'] = remove_disabled_spans_linking(output_linking['pred'], coref_flat)
                        output_coref['scores'] = remove_disabled_scores_coref(output_coref['scores'], coref_flat)
                        output_linking['scores'] = remove_disabled_scores_linking(output_linking['scores'], coref_flat)
                    else:
                        # does the new stuff: puts focus on removing only singletons
                        # (focus on predicted singleton spans WITHOUT LINK)
                        output_coref['pred'] = remove_singletons_without_link(output_coref['pred'],
                                                                              output_linking['pred'],
                                                                              pruner_spans)
                        coref_flat = [{item for sublist in batch for item in sublist} for batch in output_coref['pred']]
                        output_linking['pred'] = remove_disabled_spans_linking(output_linking['pred'], coref_flat)
                        output_coref['scores'] = remove_disabled_scores_coref(output_coref['scores'], coref_flat)
                        output_linking['scores'] = remove_disabled_scores_linking(output_linking['scores'], coref_flat)

                if self.filter_singletons_with_ner:
                    output_coref['pred'] = remove_disabled_spans(output_coref['pred'], ner_spans)
                    output_linking['pred'] = remove_disabled_spans_linking(output_linking['pred'], ner_spans)
                    output_coref['scores'] = None  # TODO
                    output_linking['scores'] = None  # TODO
                    raise NotImplementedError  # TODO!! first resolve the two previous TODOs!!! ,

                ###################################### BEGIN: debugging json dump
                if self.print_debugging_matrices:
                    to_print = dict()

                    to_log_scores = pred_scores_mtt + torch.log(z_mask)
                    to_print['scores'] = to_log_scores[0].tolist()
                    to_print['z_mask'] = z_mask.tolist()
                    to_print['target_mask'] = targets_mask.tolist()

                    link_id_to_cluster = dict()
                    # first cluster links and spans
                    for curr_idx_link in range(1, unique_links.shape[-1] + 1):
                        for curr_idx_span in range(unique_links.shape[-1] + 1, unique_links.shape[-1] + len(
                                filtered_spans['pruned_spans'][0]) + 1):
                            if targets_mask[0][curr_idx_link][curr_idx_span].item() == 1.0:
                                if curr_idx_link not in link_id_to_cluster:
                                    link_id_to_cluster[curr_idx_link] = [curr_idx_link]
                                link_id_to_cluster[curr_idx_link].append(curr_idx_span)

                    so_far_clusters = list(link_id_to_cluster.values())
                    idx_to_cluster = dict()
                    for curr_cluster in so_far_clusters:
                        for curr_cl_el in curr_cluster:
                            idx_to_cluster[curr_cl_el] = curr_cluster

                    span_id_to_span_id = dict()
                    # now cluster spans with spans
                    for curr_idx_span1 in range(unique_links.shape[-1] + 1, unique_links.shape[-1] + len(
                            filtered_spans['pruned_spans'][0]) + 1):
                        for curr_idx_span2 in range(unique_links.shape[-1] + 1, unique_links.shape[-1] + len(
                                filtered_spans['pruned_spans'][0]) + 1):
                            if curr_idx_span2 < curr_idx_span1:
                                continue
                            if targets_mask[0][curr_idx_span1][curr_idx_span2] == 1.0:
                                if curr_idx_span1 not in span_id_to_span_id:
                                    span_id_to_span_id[curr_idx_span1] = []
                                span_id_to_span_id[curr_idx_span1].append(curr_idx_span2)

                    traversed_spans_links = set([item for sublist in so_far_clusters for item in sublist])
                    traversed_spans = set()
                    for curr_idx_span in range(unique_links.shape[-1] + 1, unique_links.shape[-1] + len(
                            filtered_spans['pruned_spans'][0]) + 1):
                        existing_cluster = None
                        if curr_idx_span in traversed_spans:
                            continue
                        if curr_idx_span in idx_to_cluster:
                            existing_cluster = idx_to_cluster[curr_idx_span]
                        cluster = [curr_idx_span]
                        traversed_spans.add(curr_idx_span)
                        if curr_idx_span in span_id_to_span_id:
                            next_span = span_id_to_span_id[curr_idx_span]
                            while len(next_span) > 0:
                                next_next_span = []
                                for curr_next_span in next_span:
                                    if curr_next_span in idx_to_cluster:
                                        existing_cluster = idx_to_cluster[curr_next_span]
                                    cluster.append(curr_next_span)
                                    traversed_spans.add(curr_next_span)
                                    if curr_next_span in span_id_to_span_id:
                                        next_next_span.extend(span_id_to_span_id[curr_next_span])
                                next_next_span = list(set(next_next_span))
                                next_span = next_next_span
                        cluster = list(set(cluster))
                        if existing_cluster is not None:
                            for cidx in cluster:
                                if cidx not in traversed_spans_links:
                                    existing_cluster.append(cidx)
                                    idx_to_cluster[cidx] = existing_cluster
                        else:
                            so_far_clusters.append(cluster)
                    to_print_clusters = so_far_clusters
                    flattened_to_print_clusters = set([item for sublist in to_print_clusters for item in sublist])
                    all_possible_items = set(range(1, z_mask.shape[-1]))

                    to_fill_singletons = all_possible_items.difference(flattened_to_print_clusters)
                    # adds singletons (ex: links) of all the not used clusters/elements
                    for tofill in to_fill_singletons:
                        to_print_clusters.append([tofill])
                    to_print['clusters_gold'] = to_print_clusters

                    items = ['ROOT']
                    link_to_idx_in_mtt_matrix = dict()
                    span_to_idx_in_mtt_matrix = dict()
                    span_to_link = dict()

                    for curr_link_idx, curr_link in enumerate(unique_links[0]):
                        link_decoded = self.entity_dictionary.get(curr_link.item())
                        items.append(link_decoded)
                        # + 1 because of root node
                        link_to_idx_in_mtt_matrix[link_decoded] = curr_link_idx + 1

                    for curr_span_idx, curr_pruned_span in enumerate(filtered_spans['pruned_spans'][0]):
                        # +1 because of root node
                        span_idx_offset = unique_links.shape[-1] + 1
                        span_to_idx_in_mtt_matrix[curr_pruned_span] = curr_span_idx + span_idx_offset

                        tok_begin_id = filtered_spans['subtoken_map'][0][curr_pruned_span[0]]
                        tok_end_id = filtered_spans['subtoken_map'][0][curr_pruned_span[1]]
                        pos_text_begin = filtered_spans['begin_token'][0][tok_begin_id]
                        pos_text_end = filtered_spans['end_token'][0][tok_end_id]
                        span_text = filtered_spans['content'][0][pos_text_begin:pos_text_end]
                        items.append('{}@{}:{}'.format(span_text, pos_text_begin, pos_text_end))

                    for curr_link_pred in link_pred_all[0]:
                        span_to_link[(curr_link_pred[0], curr_link_pred[1])] = curr_link_pred[2]

                    # TODO: get the content maybe using tokens
                    to_print['target_nil_cluster_indices'] = [tnci.tolist() for tnci in target_nil_clusters_indices]
                    to_print['target_not_nil_indices'] = [tnni.tolist() for tnni in target_not_nil_indices]
                    to_print['items'] = items
                    to_print['target_nil_cluster_indices_dec'] = \
                        [[items[curr_ind] for curr_ind in curr_cl] for curr_cl in
                         to_print['target_nil_cluster_indices']]
                    to_print['target_not_nil_indices_dec'] = \
                        [items[curr_ind] for curr_ind in to_print['target_not_nil_indices']]
                    to_print['clusters_gold_dec'] = \
                        [[items[curr_ind] for curr_ind in curr_cl] for curr_cl in to_print['clusters_gold']]
                    to_print['denom'] = z_log_cost.item()
                    to_print['numer'] = z_log_cost.item() - tot_loss.item()
                    to_print['numer_explained'] = {
                        'not_nil_idxs_in_matrix': debug_not_nil_idxs_in_matrix,
                        'not_nil_weight': not_nil_log_cost.item(),
                        'nil_clusters_weights': debug_nil_clusters_weights,
                        'nil_clusters_idxs_in_matrix: ': debug_nil_cluster_idxs_in_matrix
                    }
                    span_idx_to_candidates = dict()
                    assert candidates[0].shape[0] == len(filtered_spans['pruned_spans'][0])
                    for span_idx, (curr_span_candidates, curr_span_cand_length) in \
                            enumerate(zip(candidates[0], candidate_lengths[0])):
                        ########################
                        curr_cand_length = curr_span_cand_length.item()
                        curr_candidates = curr_span_candidates.tolist()[:curr_cand_length]
                        curr_candidates = [cnd for cnd in curr_candidates if
                                           cnd != self.unknown_dict and cnd != nill_id]
                        curr_candidates = [self.entity_dictionary.get(cnd) for cnd in curr_candidates]
                        span_idx = unique_links.shape[-1] + 1 + span_idx
                        span_idx_to_candidates[span_idx] = curr_candidates

                    clusters_pred = []
                    to_print['span_to_candidates'] = span_idx_to_candidates

                    for curr_o_coref in output_coref['pred'][0]:
                        curr_cluster_links = set()
                        to_add_cluster = curr_o_coref.copy()
                        cluster_link = []
                        for curr_cluster_span in to_add_cluster:
                            if curr_cluster_span in span_to_link:
                                curr_span_link = span_to_link[curr_cluster_span]
                                if curr_span_link not in curr_cluster_links:
                                    curr_cluster_links.add(curr_span_link)
                                    cluster_link.append(curr_span_link)
                        # only one link per cluster
                        assert len(curr_cluster_links) <= 1
                        to_add_cluster = cluster_link + to_add_cluster
                        # now pass everything to index
                        cluster_link_indexes_to_mtt_matrix = []
                        for curr_element in to_add_cluster:
                            if curr_element in span_to_idx_in_mtt_matrix:
                                cluster_link_indexes_to_mtt_matrix.append(span_to_idx_in_mtt_matrix[curr_element])
                            elif curr_element in link_to_idx_in_mtt_matrix:
                                cluster_link_indexes_to_mtt_matrix.append(link_to_idx_in_mtt_matrix[curr_element])

                        clusters_pred.append(cluster_link_indexes_to_mtt_matrix)

                    clusters_all = []
                    for curr_o_coref in coref_pred_all[0]:
                        curr_cluster_links = set()
                        to_add_cluster = curr_o_coref.copy()
                        cluster_link = []
                        for curr_cluster_span in to_add_cluster:
                            if curr_cluster_span in span_to_link:
                                curr_span_link = span_to_link[curr_cluster_span]
                                if curr_span_link not in curr_cluster_links:
                                    curr_cluster_links.add(curr_span_link)
                                    cluster_link.append(curr_span_link)
                        # only one link per cluster
                        assert len(curr_cluster_links) <= 1
                        to_add_cluster = cluster_link + to_add_cluster
                        # now pass everything to index
                        cluster_link_indexes_to_mtt_matrix = []
                        for curr_element in to_add_cluster:
                            if curr_element in span_to_idx_in_mtt_matrix:
                                cluster_link_indexes_to_mtt_matrix.append(span_to_idx_in_mtt_matrix[curr_element])
                            elif curr_element in link_to_idx_in_mtt_matrix:
                                cluster_link_indexes_to_mtt_matrix.append(link_to_idx_in_mtt_matrix[curr_element])

                        clusters_all.append(cluster_link_indexes_to_mtt_matrix)

                    to_print['clusters_pred_enabled'] = clusters_pred
                    to_print['clusters_pred_all'] = clusters_all

                    debugging_file = 'ep_{:04d}_{}_debugging_mtt.json'.format(settings.epoch,
                                                                              filtered_spans['doc_id'][0])
                    debugging_path = settings.debugging_path
                    os.makedirs(settings.debugging_path, exist_ok=True)

                    with open(os.path.join(debugging_path, debugging_file), 'w') as fp:
                        json.dump(to_print, fp)

                ###################################### END: debugging json dump
        else:
            output['loss'] = 0.0
            output_coref['loss'] = 0.0
            output_coref['pred'] = [None for x in gold_spans]
            output_coref['pred_pointers'] = [None for x in gold_spans]
            output_coref['gold'] = [None for x in gold_spans]
            output_coref['scores'] = [None for x in gold_spans]

            # TODO: see well what have to add here for pred_linking and gold_linking
            output_linking['pred'] = [None for x in gold_spans]
            output_linking['gold'] = [None for x in gold_spans]
            output_linking['loss'] = 0.0
            output_linking['scores'] = [None for x in gold_spans]

        # kzaporoj - None for the link part , not yet
        return output['loss'], output_linking, output_coref

    def get_mean(self, list_values: List):
        if len(list_values) == 0:
            return 0.0
        else:
            return (sum(list_values) / len(list_values))

    def log_stats(self, dataset_name, tb_logger: Logger, step_nr):
        if self.print_debugging:
            logger.info('{}-coreflinker_mtt_hoi-loss: {}'.format(dataset_name, self.mtt_hoi_loss))
            tb_logger.log_value('{}-mtt-loss'.format(dataset_name), self.get_mean(self.mtt_hoi_loss), step_nr)

            # avg norm
            tb_logger.log_value('{}-mtt-norm-link_span'.format(dataset_name),
                                self.get_mean(self.stat_link_span_norm), step_nr)
            tb_logger.log_value('{}-mtt-mean-link_span'.format(dataset_name),
                                self.get_mean(self.stat_link_span_mean), step_nr)
            tb_logger.log_value('{}-mtt-std-link_span'.format(dataset_name),
                                self.get_mean(self.stat_link_span_std), step_nr)
            tb_logger.log_value('{}-mtt-min-link_span'.format(dataset_name),
                                self.get_mean(self.stat_link_span_min), step_nr)
            tb_logger.log_value('{}-mtt-max-link_span'.format(dataset_name),
                                self.get_mean(self.stat_link_span_max), step_nr)
            ###
            tb_logger.log_value('{}-mtt-norm-span_span'.format(dataset_name),
                                self.get_mean(self.stat_span_span_norm), step_nr)
            tb_logger.log_value('{}-mtt-mean-span_span'.format(dataset_name),
                                self.get_mean(self.stat_span_span_mean), step_nr)
            tb_logger.log_value('{}-mtt-std-span_span'.format(dataset_name),
                                self.get_mean(self.stat_span_span_std), step_nr)
            tb_logger.log_value('{}-mtt-min-span_span'.format(dataset_name),
                                self.get_mean(self.stat_span_span_min), step_nr)
            tb_logger.log_value('{}-mtt-max-span_span'.format(dataset_name),
                                self.get_mean(self.stat_span_span_max), step_nr)

            self.mtt_hoi_loss = list()
            self.stat_link_span_norm = list()
            self.stat_link_span_mean = list()
            self.stat_link_span_std = list()
            self.stat_link_span_min = list()
            self.stat_link_span_max = list()
            self.stat_span_span_norm = list()
            self.stat_span_span_mean = list()
            self.stat_span_span_std = list()
            self.stat_span_span_min = list()
            self.stat_span_span_max = list()

    def create_metrics(self):
        out = []
        if self.enabled:
            metrics = [
                MetricCoref(self.coref_task, 'muc', MetricCoref.muc),
                MetricCoref(self.coref_task, 'bcubed', MetricCoref.b_cubed, verbose=False),
                MetricCoref(self.coref_task, 'ceafe', MetricCoref.ceafe, verbose=False),
            ]

            out.extend(metrics)
            out.append(MetricCorefAverage(self.coref_task, 'avg', metrics))
            out.append(MetricObjective(self.coref_task))
            out.append(MetricCorefExternal(self.coref_task))

            out.extend([MetricLinkerImproved(self.link_task),
                        MetricLinkerImproved(self.link_task, 'links'),
                        MetricLinkerImproved(self.link_task, 'nills'),
                        MetricLinkAccuracy(self.link_task),
                        MetricLinkAccuracyNoCandidates(self.link_task),
                        MetricObjective(self.link_task)])

        return out

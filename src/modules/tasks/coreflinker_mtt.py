import json
import time

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
# from allennlp.nn.util import batched_index_select

import settings
from datass.dictionary import Dictionary
from metrics.coref import MetricCoref, MetricCorefAverage
from metrics.corefx import MetricCorefExternal
from metrics.linker import MetricLinkerImproved, MetricLinkAccuracy, MetricLinkAccuracyNoCandidates
from metrics.misc import MetricObjective
from modules.graph import create_graph
# from modules.tasks.linker import predict_scores
from modules.misc.misc import batched_index_select
from modules.tasks.coreflinker import create_candidate_mask, remove_disabled_spans, remove_disabled_spans_linking, \
    remove_disabled_scores_coref, convert_coref, m2i_to_clusters_linkercoref, \
    remove_disabled_scores_linking
from modules.utils.misc import predict_scores_mtt
from util.edmonds import mst_only_tree


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
        # print('batch_idx is of ', batch_idx)
        # print('scores_batch is of ', scores_batch.shape)
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


def create_scores_mtt_pred(scores, unique_links, candidates, candidate_lengths, torch_float_precision):
    """

    :param scores:
    :return:

    """
    # resources to help: https://discuss.pytorch.org/t/find-indices-of-one-tensor-in-another/84889
    # >>> a = torch.arange(10)
    # >>> b = torch.arange(2, 7)[torch.randperm(5)]
    # print((b.unsqueeze(1) == a).nonzero())

    # print('the passed scores to create_scores_mtt_pred is: ', scores)
    # scores.shape: [1,14,31]
    # candidates.shape: [1,14,16]
    # unique_links.shape: [1,30]

    # here broadcasts candidate ids to unique ids in unique_links
    # mix_cross.shape: [batch, spans, # candidates, # unique_candidates] --> [1, 14, 16, 30]
    # mix_cross is a matrix that maps each of the candidates (dim 2) for each of the spans (dim 1) to the respective
    # position in unique_links (dim 3). dim 0 is the batch.
    scrs_mtt_root_to_spans = scores[:, :, :1].transpose(-2, -1)
    cand_max_length = candidate_lengths.max().item()
    candidates = candidates[:, :, :cand_max_length]
    scrs_mtt_spans = scores[:, :, cand_max_length + 1:].transpose(-2, -1)

    if unique_links.shape[-1] > 0:
        mix_cross: torch.Tensor = (candidates.unsqueeze(-1) == unique_links)

        scrs_mtt_expd_links_to_spans = torch.zeros_like(mix_cross, device=settings.device, dtype=torch_float_precision)
        scrs_links_to_spans = scores[:, :, 1:candidates.shape[-1] + 1]
        scrs_expd_links_to_spans = scrs_links_to_spans.unsqueeze(-1).expand(mix_cross.size())
        scrs_mtt_expd_links_to_spans[mix_cross] = scrs_expd_links_to_spans[mix_cross]

        ones_multiplier = torch.ones(1, mix_cross.shape[2], dtype=torch_float_precision, device=settings.device)
        scrs_mtt_links_to_spans = torch.matmul(ones_multiplier, scrs_mtt_expd_links_to_spans).squeeze(-2)
        scrs_mtt_links_to_spans = scrs_mtt_links_to_spans.transpose(-2, -1)

        # scrs_mtt_links_to_spans --> block of scores from entity links to spans
        # TODO: not sure if this transpose is necessary since it is square
        # commented because in case of 0 candidates, it fails
        # scrs_mtt_spans = scores[:, :, candidates.shape[-1] + 1:].transpose(-2, -1)
        # scrs_mtt_spans --> bloc of scores between spans

        # scrs_mtt_root_to_spans --> bloc of scores from root to spans

        # scores for predicted block (the rightmost)
        scrs_mtt_pred_bloc = torch.cat([scrs_mtt_root_to_spans, scrs_mtt_links_to_spans, scrs_mtt_spans], dim=-2)
    else:
        # if no candidate links, then don't add it completely
        scrs_mtt_pred_bloc = torch.cat([scrs_mtt_root_to_spans, scrs_mtt_spans], dim=-2)

    # the leftmost fixed bloc
    # TODO: when using batch size > 1, change this to take it into account
    scrs_mtt_fixed_bloc = torch.zeros(scrs_mtt_pred_bloc.shape[0], scrs_mtt_pred_bloc.shape[1],
                                      1 + unique_links.shape[-1], device=settings.device,
                                      dtype=torch_float_precision)
    # dtype=torch.float32)

    # repeat just the maximum score from the root to candidate links
    # TODO: make this batched
    scrs_mtt_fixed_bloc[0, 0, 1:unique_links.shape[-1] + 1] = torch.max(scrs_mtt_pred_bloc)

    # print('pred_scores done?')
    scrs_mtt_complete_matrix = torch.cat([scrs_mtt_fixed_bloc, scrs_mtt_pred_bloc], dim=-1)

    return scrs_mtt_complete_matrix


def create_coreflinker_mtt_z_mask(pred_spans, gold_spans, gold_clusters, linker_targets,
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
    """
    num_batch = len(pred_spans)  # 1
    max_spans = max([len(x) for x in pred_spans])  # 9

    coreflinker_mtt_z_mask = torch.zeros(num_batch, 1 + unique_links.shape[-1] + max_spans,
                                         1 + unique_links.shape[-1] + max_spans, device=settings.device,
                                         dtype=torch.float32)
    z_mask_lengths = list()
    for batch, (pred, gold, clusters) in enumerate(zip(pred_spans, gold_spans, gold_clusters)):
        span_offset = unique_links_lengths[batch] + 1
        curr_z_mask_length = span_offset.item() + len(pred)  # root + links + spans
        z_mask_lengths.append(curr_z_mask_length)

        link_id_to_idx = dict()
        for idx_link, link_id in enumerate(unique_links[batch]):
            link_id_to_idx[link_id.item()] = idx_link
        gold2cluster = {}
        for idx, span in enumerate(gold):
            gold2cluster[span] = clusters[idx].item()
        # processed_clusters = set()
        for idx1, span1 in enumerate(pred):
            # between spans
            for idx2, span2 in enumerate(pred):
                if idx2 != idx1:
                    coreflinker_mtt_z_mask[batch, idx1 + span_offset, idx2 + span_offset] = 1.0

            # from links to spans
            if candidate_lengths[batch, idx1] > 0:
                for cand_id in candidates[batch, idx1]:
                    cand_id = cand_id.item()
                    if cand_id != unknown_id:
                        # + 1 because of the root which is in the first row
                        coreflinker_mtt_z_mask[batch, link_id_to_idx[cand_id] + 1, idx1 + span_offset] = 1.0

            # from root to span
            coreflinker_mtt_z_mask[batch, 0, idx1 + span_offset] = 1.0

        # from root to all the links of the spans
        for curr_link_idx in range(unique_links_lengths[batch]):
            # + 1 because of the root which is in the first column
            coreflinker_mtt_z_mask[batch, 0, curr_link_idx + 1] = 1.0

    return coreflinker_mtt_z_mask, torch.tensor(z_mask_lengths, dtype=torch.int32, device=settings.device)


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
    max_spans = max([len(x) for x in pred_spans])  # 9

    coreflinker_mtt_z_mask_indexed = torch.zeros(num_batch, 1 + unique_links.shape[-1] + max_spans,
                                                 1 + unique_links.shape[-1] + max_spans, device=settings.device,
                                                 dtype=torch.float32)
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


def create_coreflinker_mtt_target_mask(pred_spans, gold_spans, gold_clusters, linker_targets, candidates,
                                       unique_links, unique_links_lengths):
    """

    :param pred_spans:
    :param gold_spans:
    :param gold_clusters:
    :param linker_targets ; :
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
    nr_candidate_links_per_mention = linker_targets.sum(-1)
    target_mask_lengths = list()
    # targets for span-span coref # TODO!! - also check that linker_targets come as batch!!!
    for batch, (pred, gold, clusters) in enumerate(zip(pred_spans, gold_spans, gold_clusters)):
        # root to links always in 1
        coreflinker_mtt_targets[batch, 0, 1:unique_links_lengths[batch] + 1] = 1.0
        span_offset = 1 + unique_links_lengths[batch].item()
        curr_target_length = span_offset + len(pred)  # root + links + spans
        target_mask_lengths.append(curr_target_length)
        gold2cluster = {}
        for idx, span in enumerate(gold):
            gold2cluster[span] = clusters[idx].item()
        processed_clusters = set()
        for idx1, span1 in enumerate(pred):
            num_coref_spans_found = 0
            # num_span_links_found = 0
            if span1 in gold2cluster:
                for idx2, span2 in enumerate(pred):
                    # if idx2 < idx1 and span2 in gold2cluster and gold2cluster[span1] == gold2cluster[span2]:
                    if idx2 != idx1 and span2 in gold2cluster and gold2cluster[span1] == gold2cluster[span2]:
                        # coref_targets[batch, idx1, idx2] = 1.0
                        coreflinker_mtt_targets[batch, idx1 + span_offset, idx2 + span_offset] = 1.0
                        num_coref_spans_found += 1

            # (kzaporoj) - here also checks whether a particular mention has entity linking candidates, and if so,
            # also sets the num_found to num_candidates
            # print('here todo with num_found of linking candidates')
            num_span_links_found = nr_candidate_links_per_mention[batch, idx1].item()
            if num_span_links_found == 0:
                # TODO!!! - potentially buggy code, in case the correct link is not in the candidates of the first span!
                if (span1 not in gold2cluster) or \
                        (span1 in gold2cluster and gold2cluster[span1] not in processed_clusters):
                    coreflinker_mtt_targets[batch, 0, idx1 + span_offset] = 1.0
            else:
                # connects to the correct link
                correct_link_id = candidates[batch, idx1, linker_targets[batch, idx1] > 0.5].item()
                # correct_link_id = candidates[batch, idx1, linker_targets[batch, idx1] == 1.0].item()
                offset_correct_link = (unique_links[batch] == correct_link_id).nonzero().item()
                coreflinker_mtt_targets[batch, offset_correct_link + 1, idx1 + span_offset] = 1.0
                # raise Exception('STILL HAVE TO DEBUG THIS PART, TO SEE IF correct_link_id is correctly calculated')

            if span1 in gold2cluster:
                processed_clusters.add(gold2cluster[span1])

    return coreflinker_mtt_targets, torch.tensor(target_mask_lengths, dtype=torch.int32, device=settings.device)


def create_coreflinker_mtt_target_mask_multihead(pred_spans, gold_spans, gold_clusters, linker_targets, candidates,
                                                 unique_links, unique_links_lengths,
                                                 use_multihead=True, torch_float_precision=torch.float32):
    """
    Unlike create_coreflinker_mtt_target_mask, this that produces only a single entry (1.0 mask activation) from root
    to a span (the first span in a cluster) in a particular NIL cluster, this one returns multiple matrices with
    activations for each of the spans ("heads") in a NIL cluster.
    For details: see slide https://docs.google.com/presentation/d/1Za2gCNq55gp1MCTlg4p0CN-JGyKjj4WtzxpYJKDZz3E/edit#slide=id.gbea2b70095_0_61
    For details of what create_coreflinker_mtt_target_mask is doing see slide https://docs.google.com/presentation/d/1Za2gCNq55gp1MCTlg4p0CN-JGyKjj4WtzxpYJKDZz3E/edit#slide=id.gbea2b70095_0_0

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
    nr_candidate_links_per_mention = linker_targets.sum(-1)
    target_mask_lengths = list()
    batched_lst_multiheads = list()
    # targets for span-span coref # TODO!! - also check that linker_targets come as batch!!!
    for batch, (pred, gold, clusters) in enumerate(zip(pred_spans, gold_spans, gold_clusters)):
        # root to links always in 1
        coreflinker_mtt_targets[batch, 0, 1:unique_links_lengths[batch] + 1] = 1.0
        span_offset = 1 + unique_links_lengths[batch].item()
        curr_target_length = span_offset + len(pred)  # root + links + spans
        target_mask_lengths.append(curr_target_length)
        gold2cluster = dict()
        for idx, span in enumerate(gold):
            gold2cluster[span] = clusters[idx].item()

        pred2cluster_struct = dict()
        if clusters.shape[-1] > 0:
            max_gold_cl_id = clusters.max().item()
        else:
            max_gold_cl_id = 0
        to_assign_cl_id = max_gold_cl_id + 1
        for idx_span1, span1 in enumerate(pred):
            if span1 not in gold2cluster:
                new_cl_id = to_assign_cl_id
                to_assign_cl_id += 1
                pred2cluster_struct[new_cl_id] = {'cluster_id': to_assign_cl_id,
                                                  'spans': [(idx_span1, span1)],
                                                  # if it is not in gold2cluster, then there is no way to know
                                                  # if the link is valid or not
                                                  'is_valid_link': False}
            else:
                cluster_id = gold2cluster[span1]
                num_span_links_found = nr_candidate_links_per_mention[batch, idx_span1].item()
                is_valid_link = False
                if num_span_links_found > 0:
                    is_valid_link = True
                    correct_link_id = candidates[batch, idx_span1, linker_targets[batch, idx_span1] > 0.5].item()
                    offset_correct_link = (unique_links[batch] == correct_link_id).nonzero().item()
                    coreflinker_mtt_targets[batch, offset_correct_link + 1, idx_span1 + span_offset] = 1.0

                for idx_span2, span2 in enumerate(pred):
                    if idx_span2 != idx_span1 and span2 in gold2cluster and gold2cluster[span1] == gold2cluster[span2]:
                        coreflinker_mtt_targets[batch, idx_span1 + span_offset, idx_span2 + span_offset] = 1.0
                        # num_coref_spans_found += 1

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
                if use_multihead:
                    len_spans = len(spans)
                    indices = [0]  # the first index always 0 (root)
                    curr_indices = indices + [curr_ind[0] + span_offset for curr_ind in spans]
                    # :1 because the first head is already taken care in coreflinker_mtt_targets
                    for idx, curr_head in enumerate(spans[1:]):
                        mtt_targets = torch.zeros(len_spans + 1, len_spans + 1, dtype=torch_float_precision,
                                                  device=settings.device)

                        # +2 because of root and the first head that is already taken care in coreflinker_mtt_targets
                        mtt_targets[0, 2 + idx] = 1.0
                        # the connections between spans (inter_spans) all in 1 except the main diagonal
                        inter_spans = torch.ones(mtt_targets.shape[0] - 1, mtt_targets.shape[1] - 1,
                                                 dtype=torch_float_precision,
                                                 device=settings.device)
                        ind = np.diag_indices(inter_spans.shape[0])
                        inter_spans[ind[0], ind[1]] = torch.zeros(inter_spans.shape[0],
                                                                  dtype=torch_float_precision,
                                                                  device=settings.device)
                        mtt_targets[1:, 1:] = inter_spans[:]
                        lst_multiheads.append({'mtt_targets': mtt_targets, 'indices': curr_indices})
                        # mask_span_to_span[ind[0], ind[1]] = torch.zeros(mask_span_to_span.shape[0], device=settings.device)

        # TODO: now a loop over all elements in pred2cluster_struct and
        #  1- if is_valid_link in True, just add it into a big matrix (coref_mtt_targets)
        #  2- if is_valid_link in False, then for each one create a separate M matrices where M is the number of
        #  spans in 'spans', each matrix will have a different head, also the corresponding offset of the matrix
        #  will have to be added. IS THIS OFFSET REALLY NEEDED or even possible (non-contiguous spans)?????
        #  TO GET ACCESS TO THE REAL SCORES?? SO MAYBE just pass the 'scores' matrix here and instead of mask
        #  assign the real scores there
        #    - !!maybe the first head in M add it to the coref_mtt_targets matrix; this way it will be possible to
        #      calculate laplacian and determinant, as well as configurable on whether multihead is needed!!
        #  3- IS IT POSSIBLE TO GET TO KNOW IF A PARTICULAR CLUSTER IS NIL HERE to make NIL configurable and differentiate
        #  it from the case where the spans just don't have a valid link??
        batched_lst_multiheads.append(lst_multiheads)
    return coreflinker_mtt_targets, torch.tensor(target_mask_lengths, dtype=torch.int32, device=settings.device), \
           batched_lst_multiheads


def m2i_to_clusters_linkercoref_mtt(m2i, coref_col_to_link_id=None,
                                    links_dictionary: Dictionary = None, nr_candidates=0):
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
            # if link_token == 'NILL':
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


def decode_m2i_coreflinker_mtt(pred_masked_scores, pred_tree_mst,
                               # lengths_coref, lengths_linker,
                               candidate_ids,
                               link_id_to_coref_col,
                               # coref_col_to_link_id,
                               dic, unique_links,
                               unique_links_lengths, pred_spans):
    """

    :return: This function should return the "root" each of the mentions point to. The "root" can be either an "entity",
    in which case this would be a link. Or can be another mention; in which case it would be a mention that can not be linked
    or that does not have a link I guess.
    """
    # to_ret = []
    span_to_pointers_detail_info = list()

    decoded_m2i_coref_linker = list()

    for batch, (b_unique_links, b_unique_links_lengths) in enumerate(zip(unique_links, unique_links_lengths)):
        # print('decode_m2i_coreflinker_mtt processing batch ', batch)
        b_link_id_to_coref_col = link_id_to_coref_col[batch]
        # b_decoded_m2i_coref_linker = [0]  # 0 for root
        # NO, now I think that it should not include the root, only the nodes representing links or spans
        b_decoded_m2i_coref_linker = list()
        b_span_to_pointers_detail_info = dict()
        # -1 because we do not count the root node
        b_decoded_m2i_coref_linker.extend([b_link_id_to_coref_col[link_id.item()] - 1 for link_id in b_unique_links])
        # b_pred_matrix_mst = pred_matrix_mst[batch]
        b_pred_tree_mst = pred_tree_mst[batch]
        b_pred_spans = pred_spans[batch]
        b_pred_masked_scores = pred_masked_scores[batch]
        my_edges = list(nx.dfs_edges(b_pred_tree_mst))
        # print('dfs edges: ', list(edges))
        # my_edges = dfs_edges(b_pred_matrix_mst, from_node=0)
        # print('my edges: ', my_edges)
        # print('nicely printed matrix: ')

        # assert torch.diagonal(b_pred_matrix_mst).sum() == 0.0  # no self referring spans
        # assert b_pred_matrix_mst.shape[0] == b_pred_matrix_mst.shape[1]  # square adjacency matrix

        # for row in range(b_pred_matrix_mst.shape[0]):
        #     curr_col = ''
        #     for col in range(b_pred_matrix_mst.shape[1]):
        #         if curr_col != '':
        #             curr_col += '\t'
        #         curr_col += str(int(b_pred_matrix_mst[row, col].item()))
        #     print(curr_col)

        # nr_nodes = b_pred_matrix_mst.shape[0]
        nr_nodes = b_pred_masked_scores.shape[0]

        nr_links = b_unique_links_lengths.item()

        b_decoded_m2i_coref_linker.extend(list(range(nr_links, nr_nodes - 1)))
        assert len(b_pred_spans) == nr_nodes - nr_links - 1
        # -1 because we do not add the root node
        # assert len(b_decoded_m2i_coref_linker) == b_pred_matrix_mst.shape[0] - 1

        initial_node = 0
        pred_span_to_how_pointed = dict()
        for curr_edge in my_edges:
            # print('curr edge is as follows: ', curr_edge)
            if curr_edge[0] == 0:
                initial_node = 0
            else:
                if initial_node == 0:
                    initial_node = curr_edge[0]
            curr_input_node = curr_edge[1]
            input_node_span = curr_input_node - 1
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
                    link_name = dic.get(link_id)
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

    return decoded_m2i_coref_linker, span_to_pointers_detail_info


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
                print('WARNING ON CANDIDATE LINKING!!: ', curr_span_id, ' has been resolved to multiple '
                                                                        'candidates: ',
                      out_mention_to_link_id[curr_span_id])

    # +1 because we do not count the root taken into account in coref_col_to_link_id
    out_mention_to_link_id2 = [(spans[k - number_candidates] + (links_dictionary.get(coref_col_to_link_id[v[0] + 1]),))
                               for k, v in out_mention_to_link_id.items()]

    spans_with_link = set([(span_start, span_end) for span_start, span_end, _ in out_mention_to_link_id2])

    return out_coref_clusters, out_mention_to_link_id2, spans_with_link


def get_mtt_loss(targets_mask, pred_scores_mtt, z_mask, torch_float_precision, print_debugging=False):
    # TODO: we are here!!! WIP!!
    # tot_loss = None
    # get the laplacian of target, first applies the mask
    # TODO: add for the batch calculation
    if print_debugging:
        to_print_debugging = dict()
        to_print_debugging['target_mask'] = targets_mask[0].tolist()
        to_print_debugging['z_mask'] = z_mask[0].tolist()
        to_print_debugging['scores'] = pred_scores_mtt[0].tolist()

    targets_scores = targets_mask * torch.exp(pred_scores_mtt)

    targets_scores_exp = targets_scores
    # TODO: instead of unsqueeze(0), implement it in a batched way!!
    laplacian_tgt_scores = torch.eye(targets_scores_exp.shape[-2], targets_scores_exp.shape[-1],
                                     dtype=torch_float_precision, device=settings.device).unsqueeze(0)
    laplacian_tgt_scores = laplacian_tgt_scores * targets_scores_exp.sum(dim=-2)  # main diagonal
    laplacian_tgt_scores += (targets_scores_exp * -1.0)

    z_scores = z_mask * torch.exp(pred_scores_mtt)

    # TODO: instead of unsqueeze(0), implement it in a batched way!!
    z_scores_exp = z_scores
    laplacian_z_scores = torch.eye(z_scores_exp.shape[-2], z_scores_exp.shape[-1],
                                   device=settings.device, dtype=torch_float_precision).unsqueeze(0)
    laplacian_z_scores = laplacian_z_scores * z_scores_exp.sum(dim=-2)  # main diagonal
    laplacian_z_scores += (z_scores_exp * -1.0)

    # get the mtts
    laplacian_tgt_scores_exp = laplacian_tgt_scores
    mtt_det_tgt_scores = torch.slogdet(laplacian_tgt_scores_exp[0, 1:, 1:])
    # mtt_det_tgt_scores = torch.det(laplacian_tgt_scores_exp[0, 1:, 1:])
    laplacian_z_scores_exp = laplacian_z_scores
    mtt_det_z_scores = torch.slogdet(laplacian_z_scores_exp[0, 1:, 1:])
    # mtt_det_z_scores = torch.det(laplacian_z_scores_exp[0, 1:, 1:])

    curr_loss = mtt_det_z_scores[1] - mtt_det_tgt_scores[1]

    if print_debugging:
        to_print_debugging['z_slogdet'] = mtt_det_z_scores[1].item()
        to_print_debugging['target_slogdet'] = mtt_det_tgt_scores[1].item()
        to_print_debugging['z_laplacian'] = laplacian_z_scores[0].tolist()
        to_print_debugging['target_laplacian'] = laplacian_tgt_scores[0].tolist()
        to_print_debugging['loss'] = curr_loss.item()
        with open('debugging_mtt.json', 'w') as fp:
            json.dump(to_print_debugging, fp)

    if torch.isinf(curr_loss):
        print('!!!!WARNING, CURR LOSS IN INF, setting to 0!!! mtt_det_z_scores[1]', mtt_det_z_scores[1],
              'mtt_det_tgt_scores[1]', mtt_det_tgt_scores[1])
        # curr_loss = 0
        curr_loss = None
    elif torch.isnan(curr_loss):
        print('!!!!WARNING, CURR LOSS IN NAN, setting to 0!!! mtt_det_z_scores[1]', mtt_det_z_scores[1],
              'mtt_det_tgt_scores[1]', mtt_det_tgt_scores[1])
        # curr_loss = 0
        curr_loss = None

    return curr_loss
    # if curr_loss is not None:
    #     if tot_loss is not None:
    #         tot_loss = tot_loss + curr_loss
    #     else:
    #         tot_loss = curr_loss


class LossCorefLinkerMTT(nn.Module):

    def __init__(self, link_task, coref_task, entity_dictionary, config, end_to_end):
        super(LossCorefLinkerMTT, self).__init__()

        self.enabled = config['enabled']
        self.coref_task = coref_task
        self.link_task = link_task
        self.entity_dictionary = entity_dictionary

        if self.enabled:
            self.labels = self.entity_dictionary.tolist()
            self.unknown_dict = entity_dictionary.lookup('###UNKNOWN###')

        self.weight = config.get('weight', 1.0)
        self.filter_singletons_with_pruner = config['filter_singletons_with_pruner']
        self.filter_singletons_with_ner = config['filter_singletons_with_ner']
        self.singletons = self.filter_singletons_with_pruner or self.filter_singletons_with_ner
        self.end_to_end = end_to_end
        self.float_precision = config['float_precision']
        self.multihead_nil = config['multihead_nil']

        if self.float_precision == 'float64':
            self.torch_float_precision = torch.float64
        else:
            self.torch_float_precision = torch.float32

        self.print_debugging = False
        if 'print_debugging' in config:
            self.print_debugging = config['print_debugging']

    def forward(self, scores, gold_m2i, filtered_spans, gold_spans, linker,
                predict=False, pruner_spans=None, ner_spans=None, api_call=False):
        output = {}
        output_coref = {}
        output_linking = {}

        if self.enabled and scores is not None:
            # pred_spans = filtered_spans['spans']

            linker_candidates = linker['candidates']
            candidate_lengths = linker['candidate_lengths']
            targets = linker.get('targets')
            linker_candidates = batched_index_select(linker_candidates, filtered_spans['prune_indices'])
            targets = batched_index_select(targets, filtered_spans['prune_indices'])
            candidate_lengths = batched_index_select(candidate_lengths.unsqueeze(-1),
                                                     filtered_spans['prune_indices']).squeeze(-1)
            if self.end_to_end:
                # if it is end-to-end, we only select the candidates pruned by pruner in order to avoid
                #   using too much memory
                pred_spans_idx = filtered_spans['prune_indices']
            else:
                pred_spans_idx = filtered_spans['reindex_wrt_gold']
                linker_candidates = batched_index_select(linker_candidates, pred_spans_idx)
                targets = batched_index_select(targets, pred_spans_idx)
                candidate_lengths = batched_index_select(candidate_lengths.unsqueeze(-1), pred_spans_idx).squeeze(-1)

            candidates = linker_candidates.to(scores.device)  # torch.Size([1, 9, 17])
            #
            # nill_id = self.entity_dictionary.lookup('NILL')
            #
            linker_mask = create_candidate_mask(candidates.size(-1), candidate_lengths).float().to(scores.device)

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

            # targets = batched_index_select(targets, pred_spans_idx)
            linker_target = targets * linker_mask

            pred_spans = filtered_spans['spans']
            targets_mask, target_mask_lengths, batched_multiheads = \
                create_coreflinker_mtt_target_mask_multihead(pred_spans, gold_spans, gold_m2i, linker_target,
                                                             candidates, unique_links, unique_links_lengths,
                                                             use_multihead=self.multihead_nil,
                                                             torch_float_precision=self.torch_float_precision)

            z_mask, z_mask_lengths = create_coreflinker_mtt_z_mask_indexed(pred_spans, gold_spans, gold_m2i,
                                                                           linker_target,
                                                                           candidates,
                                                                           candidate_lengths=candidate_lengths,
                                                                           unique_links=unique_links,
                                                                           unique_links_lengths=unique_links_lengths,
                                                                           unknown_id=self.unknown_dict)
            if self.float_precision == 'float64':
                scores = scores.double()

            # scores to mtt matrix,
            # TODO: implement this in batched way, returning also the mask_lengths
            #  as in create_coreflinker_mtt_target_mask and create_coreflinker_mtt_z_mask
            pred_scores_mtt = create_scores_mtt_pred(scores, unique_links, candidates, candidate_lengths,
                                                     torch_float_precision=self.torch_float_precision)

            # want to be sure that the dimensions match to the ones expected, if no match, then print the details of the problem
            expected_dim = candidates.shape[-2] + 1 + unique_links.shape[-1]
            if expected_dim != pred_scores_mtt.shape[-1] or expected_dim != pred_scores_mtt.shape[-2] or \
                    targets_mask.shape[-1] != pred_scores_mtt.shape[-1] or \
                    targets_mask.shape[-2] != pred_scores_mtt.shape[-2]:
                print('!!!ERROR IN DIMENSIONS!!! SOMETHING GOT WRONG, printing the details of hyperparameters')
                print('the expected dim is: ', expected_dim)
                print('the shape in pred_scores_mtt is: ', pred_scores_mtt.shape)
                print('target mask.shape: ', targets_mask.shape)
                print('scores.shape: ', scores.shape)
                print('unique_links.shape: ', unique_links.shape)
                print('unique_links content: ', list(unique_links[0]))
                print('candidates.shape: ', candidates.shape)
                print('candidates content: ', list(candidates[0]))
                print('candidate_lengths.shape: ', candidate_lengths.shape)
                print('candidate_lengths content: ', list(candidate_lengths))

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

            if self.float_precision == 'float64':
                # targets_mask = targets_mask.double()
                pred_scores_mtt = pred_scores_mtt.double()
                z_mask = z_mask.double()

            tot_loss = get_mtt_loss(targets_mask=targets_mask, pred_scores_mtt=pred_scores_mtt,
                                    z_mask=z_mask, torch_float_precision=self.torch_float_precision,
                                    print_debugging=self.print_debugging)

            if self.multihead_nil:
                # TODO: instead of batched_multiheads[0], implement a batched version
                for curr_multihead in batched_multiheads[0]:
                    curr_targets_mask = curr_multihead['mtt_targets']
                    indices = curr_multihead['indices']
                    curr_pred_scores_mtt = pred_scores_mtt[0, indices, :][:, indices]
                    curr_z_mask = z_mask[0, indices, :][:, indices]
                    curr_loss = get_mtt_loss(targets_mask=curr_targets_mask, pred_scores_mtt=curr_pred_scores_mtt,
                                             z_mask=curr_z_mask, torch_float_precision=self.torch_float_precision,
                                             # for now do not print multihead matrices anyway
                                             print_debugging=False)
                    if curr_loss is not None:
                        if tot_loss is not None:
                            tot_loss = tot_loss + curr_loss
                        else:
                            tot_loss = curr_loss
                    # print('WIP: working on this one')
            if tot_loss is None:
                tot_loss = 0

            output['loss'] = tot_loss
            output_coref['loss'] = tot_loss
            output_linking['loss'] = tot_loss

            if predict:
                nill_id = self.entity_dictionary.lookup('NILL')
                pred_masked_scores = z_mask * pred_scores_mtt

                # the inverse of mask receives a very small score (less than the current minimum)
                z_mask_inverse = 1.0 - z_mask
                min_pred_mask_scores = min(pred_masked_scores.min().item(), 0.0)
                pred_masked_scores = pred_masked_scores + (z_mask_inverse) * (min_pred_mask_scores - 999999)

                # print('executing edmonds with ', int(z_mask.sum().item()), ' edges and ', z_mask.shape[-1], ' nodes: ',
                #       unique_links.shape[-1], ' unique links and ', z_mask.shape[-1] - unique_links.shape[-1] - 1, ' spans')
                start = time.time()

                pred_tree_mst = mst_only_tree(pred_masked_scores, target_mask_lengths, z_mask)

                end = time.time()
                # print('done with edmonds in ', (end - start), ' seconds')
                # TODO (24/02/2021)!!! no need for candidate_ids, it can just be replaced with unique_links, right????
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
                    decode_m2i_coreflinker_mtt(pred_masked_scores,
                                               pred_tree_mst,
                                               # lengths_coref, lengths_linker, # these two are probably not needed
                                               candidate_ids,
                                               link_id_to_coref_col=link_id_to_coref_col,
                                               # coref_col_to_link_id=coref_col_to_link_id,
                                               dic=self.entity_dictionary,
                                               unique_links=unique_links,
                                               unique_links_lengths=unique_links_lengths,
                                               pred_spans=pred_spans)
                # print('THIS PRINT HAS TO BE DELETED!')
                # logits.shape --> torch.Size([1, 14, 30])
                # lengths_coref --> tensor([14])
                # lengths_linker --> tensor([16])
                # linker_candidates.shape --> torch.Size([1, 14, 16])
                # candidate_ids[0].shape --> torch.Size([49])
                # link_id_to_coref_col --> <class 'list'>: [{2237: 0, 2552: 1, 10719: 2, 10720: 3, 11729: 4, 11734: 5, 11735: 6, 11736: 7, 11737: 8, 11738: 9, 11739: 10, 11740: 11, 11741: 12, 11742: 13, 11743: 14, 11744: 15, 11745: 16, 11746: 17, 11747: 18, 11748: 19, 14221: 20, 25253: 21, 34142: 22, 34210: 23, 34211: 24, 34213: 25, 34214: 26, 34215: 27, 34216: 28, 34217: 29, 34218: 30, 34219: 31, 34220: 32, 118110: 33, 118129: 34, 118130: 35, 118131: 36, 118132: 37, 118133: 38, 118134: 39, 118135: 40, 118136: 41, 118137: 42, 118138: 43, 118139: 44, 118140: 45, 118141: 46, 118142: 47, 118143: 48}]
                # coref_col_to_link_id --> <class 'list'>: [{0: 2237, 1: 2552, 2: 10719, 3: 10720, 4: 11729, 5: 11734, 6: 11735, 7: 11736, 8: 11737, 9: 11738, 10: 11739, 11: 11740, 12: 11741, 13: 11742, 14: 11743, 15: 11744, 16: 11745, 17: 11746, 18: 11747, 19: 11748, 20: 14221, 21: 25253, 22: 34142, 23: 34210, 24: 34211, 25: 34213, 26: 34214, 27: 34215, 28: 34216, 29: 34217, 30: 34218, 31: 34219, 32: 34220, 33: 118110, 34: 118129, 35: 118130, 36: 118131, 37: 118132, 38: 118133, 39: 118134, 40: 118135, 41: 118136, 42: 118137, 43: 118138, 44: 118139, 45: 118140, 46: 118141, 47: 118142, 48: 118143}]
                # pred_spans --> <class 'list'>: [[(3, 4), (3, 7), (5, 6), (5, 7), (41, 43), (45, 47), (49, 49), (49, 50), (50, 50), (50, 52), (51, 52), (51, 54), (53, 54), (57, 57)]]
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
                output_pred = [convert_coref_mtt(m2i_to_clusters_linkercoref_mtt(x, coref_col_to_link_id_b,
                                                                                 self.entity_dictionary,
                                                                                 candidate_ids_b.size(-1)),
                                                 y, number_candidates=candidate_ids_b.size(-1),
                                                 links_dictionary=self.entity_dictionary,
                                                 coref_col_to_link_id=coref_col_to_link_id_b)
                               for x, y, candidate_ids_b, coref_col_to_link_id_b in
                               zip(decoded_m2i_coref_linker, pred_spans, candidate_ids,
                                   coref_col_to_link_id)] if scores is not None else [[] for _ in pred_spans]

                output_coref['pred'] = [out[0] for out in output_pred]
                output_coref['pred_pointers'] = span_to_pointer_detail_info
                output_linking['pred'] = [out[1] for out in output_pred]
                output_linking['spans_with_link'] = [out[2] for out in output_pred]

                # this gives 1 if candidate list is empty, this why it is commented
                # max_nr_candidates = linker_candidates.size(-1)

                # this gives correct answer; this is why it is used instead of linker_candidates.size(-1)
                max_nr_candidates = candidate_lengths.max().item()

                linker_spans = filtered_spans['spans']

                cols_to_ignore = 0  # in case we want to use the matrix itself to filter incorrect mentions

                # + 1 because also the link score to root is counted
                s = predict_scores_mtt(scores[:, :, cols_to_ignore:max_nr_candidates + 1], linker_spans,
                                       linker_candidates, candidate_lengths, self.entity_dictionary)

                # TODO!!! - WHERE THESE 'scores' ARE USED??? ARE THEY REALLY NEEDED???
                output_coref['scores'] = predict_scores_coref_mtt(scores[:, :, max_nr_candidates + cols_to_ignore + 1:],
                                                                  pred_spans=pred_spans)

                output_linking['scores'] = s

                if not api_call:
                    output_coref['gold'] = [convert_coref(m2i_to_clusters_linkercoref(x.tolist()), y,
                                                          number_candidates=0,
                                                          links_dictionary=self.entity_dictionary)[0] for x, y in
                                            zip(gold_m2i, gold_spans)]

                    # TODO - number_candidates!!!
                    output_linking['gold'] = linker['gold']
                else:
                    output_coref['gold'] = [None for _ in gold_spans]
                    output_linking['gold'] = [None for _ in gold_spans]

                if self.filter_singletons_with_pruner:
                    # this assumes that pruner is able to predict spans
                    output_coref['pred'] = remove_disabled_spans(output_coref['pred'], pruner_spans)
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


def create_target_matrix(clusters):
    cluster2mentions = {}
    for mention, cluster in enumerate(clusters.tolist()):
        if cluster not in cluster2mentions:
            cluster2mentions[cluster] = []
        cluster2mentions[cluster].append(mention)

    number_of_mentions = clusters.size()[0]
    target = torch.zeros(number_of_mentions, number_of_mentions, device=settings.device)
    for cluster, mentions in cluster2mentions.items():
        for m1 in mentions:
            for m2 in mentions:
                target[m1, m2] = 1
    return target


def logZ(scores):
    dim = scores.size()[0] + 1

    S = torch.zeros(dim, dim)
    S[1:, 1:] = scores
    A = torch.exp(S + torch.eye(dim) * -10000)
    D = torch.diag(A.sum(0))
    L = D - A

    L[0, 1:] = 1 / scores.size()[0]
    return L.logdet()


class TaskCorefMTT(nn.Module):

    def __init__(self, dim_input, config):
        super(TaskCorefMTT, self).__init__()
        self.module = create_graph(dim_input, 1, config['scorer'])

    def forward(self, mentions, clusters):
        scores = self.module(mentions).squeeze(-1).squeeze(0)
        targets = create_target_matrix(clusters[0]).to(scores.device)

        scores = scores - scores.max()

        g = logZ(scores + (1 - targets) * -10000)
        z = logZ(scores)

        return z - g

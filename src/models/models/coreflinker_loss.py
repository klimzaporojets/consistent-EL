import logging

import torch
import torch.nn as nn

from data_processing.dictionary import Dictionary
from metrics.coref import MetricCoref, MetricCorefAverage
from metrics.corefx import MetricCorefExternal
from metrics.linker import MetricLinkerImproved, MetricLinkAccuracy, MetricLinkAccuracyNoCandidates
from metrics.misc import MetricObjective
from misc import settings
from models.misc.misc import batched_index_select
from models.utils.math import logsumexp
from models.utils.misc import predict_scores, get_mask_from_sequence_lengths

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger()


def create_coreflinker_target_forward(pred_spans, gold_spans, gold_clusters, linker_targets,
                                      filter_singletons_with_matrix):
    num_batch = len(pred_spans)  # 1
    max_spans = max([len(x) for x in pred_spans])  # 9

    coref_targets = torch.zeros(num_batch, max_spans, max_spans, device=settings.device)  # torch.Size([1, 9, 9])
    if filter_singletons_with_matrix:
        no_mention_targets = torch.zeros(num_batch, max_spans, 1, device=settings.device)
    else:
        no_mention_targets = None

    nr_candidate_links_per_mention = linker_targets.sum(-1)
    # targets for span-span coref
    for batch, (pred, gold, clusters) in enumerate(zip(pred_spans, gold_spans, gold_clusters)):
        gold2cluster = {}
        for idx, span in enumerate(gold):
            gold2cluster[span] = clusters[idx].item()

        for idx1, span1 in enumerate(pred):
            num_found = 0
            if span1 in gold2cluster:
                for idx2, span2 in enumerate(pred):
                    if idx2 < idx1 and span2 in gold2cluster and gold2cluster[span1] == gold2cluster[span2]:
                        coref_targets[batch, idx1, idx2] = 1.0
                        num_found += 1

            # (kzaporoj) - here also checks whether a particular mention has entity linking candidates, and if so,
            # also sets the num_found to num_candidates
            num_found += nr_candidate_links_per_mention[batch, idx1].item()
            if num_found == 0:
                if filter_singletons_with_matrix:
                    if span1 not in gold2cluster:
                        # if it is not a valid mention, put into no_mention column
                        no_mention_targets[batch, idx1, 0] = 1.0
                    else:
                        # if it is a singleton, still main diagonal
                        coref_targets[batch, idx1, idx1] = 1.0
                else:
                    coref_targets[batch, idx1, idx1] = 1.0

    # targets for span-entity coref
    if filter_singletons_with_matrix:
        linkercoref_targets = torch.cat([no_mention_targets, linker_targets, coref_targets],
                                        dim=-1)  # torch.Size([1, 9, 27])
    else:
        linkercoref_targets = torch.cat([linker_targets, coref_targets], dim=-1)  # torch.Size([1, 9, 26])

    return linkercoref_targets


def create_candidate_mask(max_cand_length, candidate_lengths):
    tmp = torch.arange(max_cand_length, device=settings.device)
    tmp = tmp.unsqueeze(0).unsqueeze(0)
    candidate_lengths = candidate_lengths.unsqueeze(-1)
    mask = tmp < candidate_lengths
    return mask


def convert_coref(clusters, spans, number_candidates=None, links_dictionary: Dictionary = None,
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
                logger.warning('WARNING ON CANDIDATE LINKING!!: %s has been resolved to multiple candidates: %s'
                               % (curr_span_id, out_mention_to_link_id[curr_span_id]))

    # to start with, leaves only the first one (first predicted candidate link)
    out_mention_to_link_id = [(spans[k - number_candidates] + (links_dictionary.get(coref_col_to_link_id[v[0]]),))
                              for k, v in out_mention_to_link_id.items()]

    spans_with_link = set([(span_start, span_end) for span_start, span_end, _ in out_mention_to_link_id])

    return out_coref_clusters, out_mention_to_link_id, spans_with_link


# remove singletons containing only a disabled span
def remove_disabled_spans(clusters, enabled_spans):
    out = []
    for cs, spans in zip(clusters, enabled_spans):
        enabled = set(spans)
        out.append([cluster for cluster in cs if len(cluster) > 1 or cluster[0] in enabled])
    return out


def remove_singletons_without_link(clusters, pred_links, enabled_spans):
    out = []
    for cs, curr_pred_links, spans in zip(clusters, pred_links, enabled_spans):
        span_to_link = dict()
        for curr_link_pred in curr_pred_links:
            span_to_link[(curr_link_pred[0], curr_link_pred[1])] = curr_link_pred[2]
        enabled = set(spans)
        out.append([cluster for cluster in cs if len(cluster) > 1 or cluster[0] in enabled
                    or cluster[0] in span_to_link])
    return out


def remove_disabled_spans_linking(mentions_to_links, enabled_spans):
    """

    :param mentions_to_links: <class 'list'>: [[(66, 67, 'Haplogroup_DE'), (29, 33, 'Haplogroup_DE'),
       (12, 12, 'Specialty_(medicine)'), ...]]
    :param enabled_spans: <class 'list'>: [{(36, 42), (46, 49), (56, 62), (41, 46), (25, 31), (12, 12), (61, 63), ...}]
    :return:
    """
    out = []
    for cs, spans in zip(mentions_to_links, enabled_spans):
        enabled = set(spans)
        out.append([linked_span for linked_span in cs if (linked_span[0], linked_span[1]) in enabled])
    return out


def remove_disabled_scores_linking(linking_scores, enabled_spans):
    """

    :param linking_scores: <class 'list'>: [[((0, 0), ['WorldCom_scandal', 'MCI_Inc.'], [-20.77992820739746, -23.525156021118164]),
                ((3, 3), ['Temperate_climate', 'Håkan_Mild', 'Concussion', ...)]]
    :param enabled_spans:
    :return:
    """
    out = []
    for cs, enabled_spans in zip(linking_scores, enabled_spans):
        out.append([linked_score for linked_score in cs if (linked_score[0][0], linked_score[0][1]) in enabled_spans])
    return out


def remove_disabled_scores_coref(coref_scores, enabled_spans):
    """

    :param coref_scores: <class 'list'>: [{(0, 0): [{'span': (0, 0), 'score': -0.0}],
        (3, 3): [{'span': (0, 0), 'score': -12.838151931762695}, {'span': (3, 3), 'score': -0.0}]...}]
    :param enabled_spans:
    :return:
    """
    out = []
    for cs, enabled_spans in zip(coref_scores, enabled_spans):
        out.append({k: v for k, v in cs.items() if (k[0], k[1]) in enabled_spans})
    return out


def m2i_to_clusters_linkercoref(m2i, coref_col_to_link_id=None,
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
            link_id = coref_col_to_link_id[c]
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


def decode_m2i_coreflinker(scores, lengths_coref, lengths_linker, linker_candidates, candidate_ids=None,
                           link_id_to_coref_col=None, dic: Dictionary = None,
                           coref_col_to_link_id=None, pred_spans=None, real_scores=None,
                           filter_singletons_with_matrix=False, ignore_no_mention_chains=False):
    """

    :param scores:
    :param lengths:
    :return: This function should return the "root" each of the mentions point to. The "root" can be either an "entity",
    in which case this would be a link. Or can be another mention; in which case it would be a mention that can not be linked
    or that does not have a link I guess.
    """
    output = []
    span_to_pointers_detail_info = []

    for b, (length_coref, length_linker) in enumerate(zip(lengths_coref.tolist(), lengths_linker.tolist())):
        sparse_link_matrix_length = candidate_ids[b].size(-1)

        span_to_pointer_detail_info = dict()

        # this is the old version of m2i_coref with only the idxs and no information on targeted entry
        m2i_coref = list(range(sparse_link_matrix_length + length_coref))

        for curr_idx in list(range(sparse_link_matrix_length + length_coref)):
            if curr_idx >= sparse_link_matrix_length:  # initialization of pointer details
                span = pred_spans[b][curr_idx - sparse_link_matrix_length]
                span_to_pointer_detail_info[span] = {'coref_connection_type': 'mention_self',
                                                     'coref_connection_pointer': span,
                                                     # TODO: uncomment this if needed to predict
                                                     'coref_connection_score': 0.0
                                                     }

        len_candidates_in_dense_matrix = length_linker
        len_ignore = 0
        if filter_singletons_with_matrix:
            len_ignore = 1  # ignores mentio if points to the first column
        if length_coref > 0:
            _, indices = torch.max(scores[b, 0:length_coref, :], -1)
            # the start=sparse_link_matrix_length is needed to take into account the actual length of link candidates
            # in the document
            for src, dst_dense in enumerate(indices.tolist(), start=sparse_link_matrix_length + len_ignore):
                if src < sparse_link_matrix_length + length_coref + len_ignore and \
                        dst_dense < length_coref + length_linker + len_ignore:

                    if dst_dense < len_candidates_in_dense_matrix + len_ignore:  # points to the link (or ignore)
                        if dst_dense < len_ignore:
                            # we just ignore if it points to a column in matrix that indicates that span is not a
                            # mention
                            src_span = pred_spans[b][src - sparse_link_matrix_length - len_ignore]
                            span_to_pointer_detail_info[src_span] = \
                                {'coref_connection_type': 'no_mention',
                                 'coref_connection_pointer': src_span}
                            m2i_coref[src - len_ignore] = -1
                            continue

                        # we shift back, thus only taking into account now coref and linking matrix elements
                        dst_dense -= len_ignore
                        src -= len_ignore

                        link_dict_id = linker_candidates[b, src - sparse_link_matrix_length, dst_dense].item()
                        # this if is needed because some point to NILL/candidates that do not exist
                        if link_dict_id in link_id_to_coref_col[b]:
                            link_matrix_column = link_id_to_coref_col[b][link_dict_id]
                            m2i_coref[src] = link_matrix_column
                            src_span = pred_spans[b][src - sparse_link_matrix_length]
                            span_to_pointer_detail_info[src_span] = \
                                {'coref_connection_type': 'link',
                                 'coref_connection_pointer': dic.get(link_dict_id)}
                    else:
                        # we shift back, thus only taking into account now coref and linking matrix elements
                        dst_dense -= len_ignore
                        src -= len_ignore
                        # this delta is needed to adjust to all document's candidate's length
                        delta_length_all_doc_candidates = sparse_link_matrix_length - len_candidates_in_dense_matrix
                        to_point_to = m2i_coref[dst_dense + delta_length_all_doc_candidates]

                        if to_point_to == -1:
                            if ignore_no_mention_chains:
                                m2i_coref[src] = -1
                                continue
                            else:
                                # puts both in the same valid cluster
                                to_point_to = dst_dense + delta_length_all_doc_candidates
                                m2i_coref[dst_dense + delta_length_all_doc_candidates] = to_point_to

                                m2i_coref[src] = to_point_to

                        coref_conn_type = 'mention_self' if dst_dense + delta_length_all_doc_candidates == src \
                            else 'mention_other'

                        src_span = pred_spans[b][src - sparse_link_matrix_length]
                        pointed_span = pred_spans[b][dst_dense - len_candidates_in_dense_matrix]

                        span_to_pointer_detail_info[src_span] = {'coref_connection_type': coref_conn_type,
                                                                 'coref_connection_pointer': pointed_span,
                                                                 'coref_connection_score':
                                                                     real_scores[b,
                                                                                 src - sparse_link_matrix_length,
                                                                                 dst_dense + len_ignore].item()}
                        if to_point_to < sparse_link_matrix_length:
                            link_dict_id = coref_col_to_link_id[b][to_point_to]
                            link_name = dic.get(link_dict_id)
                            if link_name == 'NILL' or link_name == 'NONE':
                                # we can not point to NILL since NILL is not discriminatory, so just point to the
                                # mention the coreference is connecting to instead of the link pointed by that mention
                                m2i_coref[src] = dst_dense + delta_length_all_doc_candidates
                            else:
                                m2i_coref[src] = to_point_to
                        else:
                            m2i_coref[src] = to_point_to
                else:
                    # sanity check: this should never ever happen !!!
                    logger.error('ERROR: invalid index')
                    logger.error('length_coref: %s' % length_coref)
                    logger.error('scores: %s' % scores[b, 0:length_coref, :])
                    logger.error('scores: %s' % scores.min().item(), scores.max().item())
                    logger.error('indices: %s' % indices)
                    logger.error('LENGTHS COREF: %s' % lengths_coref)
                    logger.error('LENGTHS LINKER: %s' % lengths_linker)
                    torch.save(scores, 'scores.pt')
                    torch.save(lengths_coref, 'lengths_coref.pt')
                    torch.save(lengths_linker, 'lengths_linker.pt')
                    exit(1)
        output.append(m2i_coref)
        span_to_pointers_detail_info.append(span_to_pointer_detail_info)

    return output, span_to_pointers_detail_info


def predict_scores_coref(scores, pred_spans):
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


class CorefLinkerLossDisabled(nn.Module):
    def __init__(self):
        super(CorefLinkerLossDisabled, self).__init__()
        self.enabled = False

    def forward(self, scores, gold_m2i, filtered_spans, gold_spans, linker,
                predict=False, pruner_spans=None, ner_spans=None, api_call=False, only_loss=False):
        output = {}
        output_coref = {}
        output_linking = {}

        output['loss'] = 0.0
        output_coref['loss'] = 0.0
        output_coref['pred'] = [[] for x in gold_spans]
        output_coref['pred_pointers'] = [[] for x in gold_spans]
        output_coref['gold'] = [[] for x in gold_spans]
        output_coref['scores'] = [[] for x in gold_spans]

        output_linking['pred'] = [[] for x in gold_spans]
        output_linking['gold'] = [[] for x in gold_spans]
        output_linking['loss'] = 0.0
        output_linking['scores'] = [[] for x in gold_spans]

        # kzaporoj - None for the link part , not yet
        return output['loss'], output_linking, output_coref

    def create_metrics(self):
        return []

    def log_stats(self, dataset_name, predict, tb_logger, step_nr):
        pass


class CorefLinkerLoss(nn.Module):

    def __init__(self, link_task, coref_task, entity_dictionary, config, end_to_end):
        super(CorefLinkerLoss, self).__init__()

        self.enabled = config['enabled']

        self.entity_dictionary = entity_dictionary
        if self.enabled:
            self.labels = self.entity_dictionary.tolist()
            self.unknown_dict = entity_dictionary.lookup('###UNKNOWN###')

        self.link_task = link_task
        self.coref_task = coref_task
        self.weight = config.get('weight', 1.0)
        self.filter_singletons_with_pruner = config['filter_singletons_with_pruner']
        self.filter_singletons_with_matrix = config['filter_singletons_with_matrix']
        self.ignore_no_mention_chains = config['ignore_no_mention_chains']
        self.filter_singletons_with_ner = config['filter_singletons_with_ner']
        self.no_nil_in_targets = config['no_nil_in_targets']
        self.doc_level_candidates = config['doc_level_candidates']
        self.singletons = self.filter_singletons_with_pruner or self.filter_singletons_with_ner  # write out singletons to json
        self.end_to_end = end_to_end

    def forward(self, scores, gold_m2i, filtered_spans, gold_spans, linker,
                predict=False, pruner_spans=None, ner_spans=None, api_call=False):
        output = {}
        output_coref = {}
        output_linking = {}

        if self.enabled and scores is not None:
            pred_spans = filtered_spans['spans']

            linker_candidates = linker['candidates']
            candidate_lengths = linker['candidate_lengths']
            targets = linker.get('targets')

            if self.end_to_end:
                # if it is end-to-end, we only select the candidates pruned by pruner in order to avoid
                #   using too much memory
                pred_spans_idx = filtered_spans['prune_indices']
            else:
                pred_spans_idx = filtered_spans['reindex_wrt_gold']

            linker_candidates = batched_index_select(linker_candidates, pred_spans_idx)
            candidate_lengths = batched_index_select(candidate_lengths.unsqueeze(-1), pred_spans_idx).squeeze(-1)

            candidates = linker_candidates.to(scores.device)  # torch.Size([1, 9, 17])

            nill_id = self.entity_dictionary.lookup('NILL')

            linker_mask = create_candidate_mask(candidates.size(-1), candidate_lengths).float().to(scores.device)

            targets_matrix = None
            tot_cand_lengths_in_gold_mentions = None
            if not api_call:
                tot_cand_lengths_in_gold_mentions = linker.get('total_cand_lengths_in_gold_mentions', None)
                if tot_cand_lengths_in_gold_mentions is not None:
                    tot_cand_lengths_in_gold_mentions = \
                        batched_index_select(tot_cand_lengths_in_gold_mentions.unsqueeze(-1),
                                             pred_spans_idx).squeeze(-1)

                targets = batched_index_select(targets, pred_spans_idx)
                linker_target = targets * linker_mask

                targets_matrix = create_coreflinker_target_forward(pred_spans, gold_spans, gold_m2i, linker_target,
                                                                   self.filter_singletons_with_matrix)

            if scores is not None:
                lengths_coref = torch.LongTensor([len(x) for x in pred_spans]).to(settings.device)

                lengths_linker = torch.LongTensor([linker_mask.shape[-1]]).repeat(linker_mask.shape[0]).to(
                    settings.device)

                triangular_mask = torch.ones(linker_mask.size()[-2], linker_mask.size()[-2], device=settings.device
                                             ).tril(0).unsqueeze(0)
                triangular_mask = torch.cat(linker_mask.shape[0] * [triangular_mask])

                # concatenate entity mask to the span-span coreference mask
                triangular_mask = torch.cat([linker_mask, triangular_mask], dim=-1)
                if self.filter_singletons_with_matrix:
                    no_mentions_mask = torch.ones(linker_mask.size()[-2], 1, device=settings.device).unsqueeze(0)
                    triangular_mask = torch.cat([no_mentions_mask, triangular_mask], dim=-1)

                constant = scores.max().item() + 100000
                additive_mask = (1 - triangular_mask) * -constant
                logits = torch.nn.functional.log_softmax(scores + additive_mask.to(scores.device), dim=-1)

            if scores is not None and targets_matrix is not None:
                loss = - logsumexp(logits + (1 - targets_matrix) * -100000)
                mask = get_mask_from_sequence_lengths(lengths_coref, lengths_coref.max().item()).float()
                output['loss'] = self.weight * (mask * loss).sum()
            elif not api_call:  # when api call is performed (e.g. with only text), we do not get the gold annotation
                raise BaseException('HUH')
            else:
                output['loss'] = 0.0

            output_coref['loss'] = output['loss']
            output_linking['loss'] = output['loss']

            if predict:
                candidate_ids = []
                for curr_cand_batch, _ in enumerate(candidates):
                    unique_curr_batch = candidates[curr_cand_batch].unique(sorted=True)
                    if self.no_nil_in_targets:
                        unique_curr_batch = unique_curr_batch[unique_curr_batch != nill_id]
                    # (16/10/2020) - 0 is used for padding, so remove it
                    unique_curr_batch = unique_curr_batch[unique_curr_batch != self.unknown_dict]

                    candidate_ids.append(unique_curr_batch)
                link_id_to_coref_col = list()
                for batch_id, candidate_ids_batch in enumerate(candidate_ids):
                    link_id_to_coref_col.append(dict())
                    for matrix_idx_link, link_dict_id in enumerate(candidate_ids_batch):
                        link_id_to_coref_col[batch_id][link_dict_id.item()] = matrix_idx_link

                coref_col_to_link_id = list()
                for batch_id, link_id_to_coref_col_batch in enumerate(link_id_to_coref_col):
                    coref_col_to_link_id.append({v: k for k, v in link_id_to_coref_col_batch.items()})

                decoded_m2i_coref_linker, span_to_pointer_detail_info = \
                    decode_m2i_coreflinker(logits, lengths_coref, lengths_linker,
                                           linker_candidates, candidate_ids,
                                           link_id_to_coref_col=link_id_to_coref_col,
                                           dic=self.entity_dictionary,
                                           coref_col_to_link_id=coref_col_to_link_id,
                                           pred_spans=pred_spans, real_scores=scores,
                                           filter_singletons_with_matrix=self.filter_singletons_with_matrix,
                                           ignore_no_mention_chains=self.ignore_no_mention_chains)

                # decoded_m2i_coref_linker - <class 'list'>: [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 5, 55, 56, 57, 58, 59, 60, 5, 62]]
                # pred_spans - <class 'list'>: [[(3, 4), (5, 6), (5, 7), (41, 43), (45, 47), (49, 49), (49, 50), (50, 50), (50, 52), (51, 52), (51, 54), (53, 54), (57, 57), (57, 59)]]
                # here gets the coref cluster spans only
                output_pred = [
                    convert_coref(m2i_to_clusters_linkercoref(x, coref_col_to_link_id_b,
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
                # THIS ONLY WILL WORK IF NOT END-TO-END!!!
                for batch_id in range(len(output_linking['pred'])):
                    curr_pred_spans = pred_spans[batch_id]
                    curr_spans_with_link = output_linking['spans_with_link'][batch_id]
                    curr_predicted_links = output_linking['pred'][batch_id]
                    for idx_span, curr_span in enumerate(curr_pred_spans):
                        if curr_span not in curr_spans_with_link:
                            if self.end_to_end:
                                pass
                            elif tot_cand_lengths_in_gold_mentions[batch_id][idx_span] > 0:
                                curr_predicted_links.append(curr_span + ('NILL',))

                max_nr_candidates = linker_candidates.size(-1)

                if self.end_to_end:
                    linker_spans = filtered_spans['spans']
                else:
                    linker_spans = filtered_spans['spans']

                cols_to_ignore = 0
                if self.filter_singletons_with_matrix:
                    cols_to_ignore = 1

                s = predict_scores(scores[:, :, cols_to_ignore:max_nr_candidates + cols_to_ignore], linker_spans,
                                   linker_candidates,
                                   candidate_lengths, self.labels)

                output_coref['scores'] = predict_scores_coref(scores[:, :, max_nr_candidates + cols_to_ignore:],
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

                # for loss just copies the loss from output for both linker and coref ; this 'loss' is needed when
                # later evaluating in metrics.misc.MetricObjective#update2 for example (see below when adding metrics
                # in create_metrics(self) method)

                if self.filter_singletons_with_pruner:
                    # this assumes that pruner is able to predict spans
                    output_coref['pred'] = remove_disabled_spans(output_coref['pred'], pruner_spans)
                    coref_flat = [{item for sublist in batch for item in sublist} for batch in output_coref['pred']]
                    output_linking['pred'] = remove_disabled_spans_linking(output_linking['pred'], coref_flat)
                    output_coref['scores'] = remove_disabled_scores_coref(output_coref['scores'], coref_flat)
                    output_linking['scores'] = remove_disabled_scores_linking(output_linking['scores'], coref_flat)

                if self.filter_singletons_with_matrix:
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
                    #  if not the builder will just output all mentions that have associated 'scores'

        else:
            output['loss'] = 0.0
            output_coref['loss'] = 0.0
            output_coref['pred'] = [None for x in gold_spans]
            output_coref['pred_pointers'] = [None for x in gold_spans]
            output_coref['gold'] = [None for x in gold_spans]
            output_coref['scores'] = [None for x in gold_spans]

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

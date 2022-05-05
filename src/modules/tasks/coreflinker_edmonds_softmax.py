"""Using softmax score (as in coreflinker.py) but in both directions and also using edmonds algorithm to decode. """
import time

import torch
import torch.nn as nn
# from allennlp.nn.util import batched_index_select

import settings
from metrics.coref import MetricCoref, MetricCorefAverage
from metrics.corefx import MetricCorefExternal
from metrics.linker import MetricLinkerImproved, MetricLinkAccuracy, MetricLinkAccuracyNoCandidates
from metrics.misc import MetricObjective
from modules.misc.misc import batched_index_select
from modules.tasks.coreflinker import create_candidate_mask, remove_disabled_spans, remove_disabled_spans_linking, \
    remove_disabled_scores_coref, remove_disabled_scores_linking, convert_coref, m2i_to_clusters_linkercoref
from modules.tasks.coreflinker_mtt import create_coreflinker_mtt_z_mask_indexed, \
    decode_m2i_coreflinker_mtt, m2i_to_clusters_linkercoref_mtt, convert_coref_mtt, predict_scores_coref_mtt
from modules.utils.misc import predict_scores_mtt
from util.edmonds import mst_only_tree
from util.math import logsumexp
from util.sequence import get_mask_from_sequence_lengths


# def create_linkercoref_target_forward(pred_spans, gold_spans, gold_clusters, linker_targets, linker_mask):
def create_coreflinker_esm_target_forward(pred_spans, gold_spans, gold_clusters, linker_targets,
                                          filter_singletons_with_matrix):
    num_batch = len(pred_spans)  # 1
    max_spans = max([len(x) for x in pred_spans])  # 9

    coref_targets = torch.zeros(num_batch, max_spans, max_spans, device=settings.device)  # torch.Size([1, 9, 9])
    if filter_singletons_with_matrix:
        no_mention_targets = torch.zeros(num_batch, max_spans, 1, device=settings.device)
    else:
        no_mention_targets = None

    # obj = self.loss(scores, targets)  # torch.Size([1, 9, 17]) -> targets shape
    # obj = (scores_mask * obj).sum() * self.weight

    # nr_candidate_links_per_mention = (linker_targets.int() & linker_mask.int()).sum(-1)
    nr_candidate_links_per_mention = linker_targets.sum(-1)
    # targets for span-span coref # TODO!! - also check that linker_targets come as batch!!!
    for batch, (pred, gold, clusters) in enumerate(zip(pred_spans, gold_spans, gold_clusters)):
        gold2cluster = {}
        for idx, span in enumerate(gold):
            gold2cluster[span] = clusters[idx].item()

        for idx1, span1 in enumerate(pred):
            num_found = 0
            if span1 in gold2cluster:
                for idx2, span2 in enumerate(pred):
                    if idx2 != idx1 and span2 in gold2cluster and gold2cluster[span1] == gold2cluster[span2]:
                        coref_targets[batch, idx1, idx2] = 1.0
                        num_found += 1

            # (kzaporoj) - here also checks whether a particular mention has entity linking candidates, and if so,
            # also sets the num_found to num_candidates
            # print('here todo with num_found of linking candidates')
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


def create_scores_esm_pred(scores, unique_links, candidates, candidate_lengths, torch_float_precision,
                           scrs_mtt_spans):
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

    # TODO!!! SEE IF THIS WILL WORK FOR edmonds_softmax, SINCE I THINK EVEN THOUGH THERE IS NO CANDIDATES,
    #  THE LENGTH IS STILL 1 in candidates matrix! (there is still one column), right???, check this!!
    # cand_max_length = candidate_lengths.max().item()

    # candidates = candidates[:, :, :cand_max_length]
    # scrs_mtt_spans = scores[:, :, cand_max_length:].transpose(-2, -1)
    # assert scrs_mtt_spans.shape[-1] == scrs_mtt_spans.shape[-2]  # it has to be square: # spans x # spans

    scrs_mtt_root_to_spans = torch.diagonal(scrs_mtt_spans, dim1=-2, dim2=-1).unsqueeze(-2)
    # scrs_mtt_root_to_spans = scores[:, :, :1].transpose(-2, -1)

    if unique_links.shape[-1] > 0:
        mix_cross: torch.Tensor = (candidates.unsqueeze(-1) == unique_links)

        scrs_mtt_expd_links_to_spans = torch.zeros_like(mix_cross, device=settings.device, dtype=torch_float_precision)
        scrs_links_to_spans = scores[:, :, 1:candidates.shape[-1]+1]
        # scrs_links_to_spans = scores[:, :, :candidates.shape[-1]]
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


class LossCorefLinkerESM(nn.Module):

    def __init__(self, link_task, coref_task, entity_dictionary, config, end_to_end):
        super(LossCorefLinkerESM, self).__init__()

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

        # TODO: filter_singletons_with_matrix is still not implemented
        self.filter_singletons_with_matrix = config['filter_singletons_with_matrix']
        self.singletons = self.filter_singletons_with_pruner or self.filter_singletons_with_ner
        self.end_to_end = end_to_end
        # self.float_precision = config['float_precision']
        # self.multihead_nil = config['multihead_nil']

        # if self.float_precision == 'float64':
        #     self.torch_float_precision = torch.float64
        # else:
        self.torch_float_precision = torch.float32

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
            linker_candidates = batched_index_select(linker_candidates, filtered_spans['prune_indices'])
            targets = batched_index_select(targets, filtered_spans['prune_indices'])
            candidate_lengths = batched_index_select(candidate_lengths.unsqueeze(-1),
                                                     filtered_spans['prune_indices']).squeeze(-1)
            pred_spans_idx = filtered_spans['prune_indices']
            if not self.end_to_end:
                pred_spans_idx = filtered_spans['reindex_wrt_gold']
                linker_candidates = batched_index_select(linker_candidates, pred_spans_idx)
                targets = batched_index_select(targets, pred_spans_idx)
                candidate_lengths = batched_index_select(candidate_lengths.unsqueeze(-1), pred_spans_idx).squeeze(-1)

            candidates = linker_candidates.to(scores.device)  # torch.Size([1, 9, 17])
            #
            # nill_id = self.entity_dictionary.lookup('NILL')
            #

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

            ## taken from coreflinker.py
            tot_cand_lengths_in_gold_mentions = linker['total_cand_lengths_in_gold_mentions']
            if tot_cand_lengths_in_gold_mentions is not None:
                tot_cand_lengths_in_gold_mentions = \
                    batched_index_select(tot_cand_lengths_in_gold_mentions.unsqueeze(-1),
                                         pred_spans_idx).squeeze(-1)

            linker_mask = create_candidate_mask(candidates.size(-1), candidate_lengths).float().to(scores.device)
            linker_target = targets * linker_mask

            targets_matrix = create_coreflinker_esm_target_forward(pred_spans, gold_spans, gold_m2i, linker_target,
                                                                   self.filter_singletons_with_matrix)

            lengths_coref = torch.LongTensor([len(x) for x in pred_spans]).to(settings.device)
            # lengths_linker = torch.LongTensor([linker_mask.shape[-1]]).repeat(linker_mask.shape[0]).to(settings.device)

            # triangular_mask = torch.ones(linker_mask.size()[-2], linker_mask.size()[-2], device=settings.device
            #                              ).tril(0).unsqueeze(0)
            triangular_mask = torch.ones(linker_mask.size()[-2], linker_mask.size()[-2], device=settings.device
                                         ).unsqueeze(0)

            triangular_mask = torch.cat(linker_mask.shape[0] * [triangular_mask])
            # concatenate entity mask to the span-span coreference mask
            triangular_mask = torch.cat([linker_mask, triangular_mask], dim=-1)
            if self.filter_singletons_with_matrix:
                no_mentions_mask = torch.ones(linker_mask.size()[-2], 1, device=settings.device).unsqueeze(0)
                triangular_mask = torch.cat([no_mentions_mask, triangular_mask], dim=-1)

            constant = scores.max().item() + 100000
            additive_mask = (1 - triangular_mask) * -constant
            logits = torch.nn.functional.log_softmax(scores + additive_mask.to(scores.device), dim=-1)

            loss = - logsumexp(logits + (1 - targets_matrix) * -100000)
            mask = get_mask_from_sequence_lengths(lengths_coref, lengths_coref.max().item()).float()
            output['loss'] = self.weight * (mask * loss).sum()

            output_coref['loss'] = output['loss']
            output_linking['loss'] = output['loss']

            if predict:
                cand_max_length = candidate_lengths.max().item()

                candidates = candidates[:, :, :cand_max_length]
                scrs_mtt_spans = scores[:, :, cand_max_length:].transpose(-2, -1)
                assert scrs_mtt_spans.shape[-1] == scrs_mtt_spans.shape[-2]  # it has to be square: # spans x # spans

                scrs_mtt_root_to_spans = torch.diagonal(scrs_mtt_spans, dim1=-2, dim2=-1).unsqueeze(-2)
                scrs_mtt_root_to_spans = scrs_mtt_root_to_spans.transpose(-2,-1)
                scores = torch.cat([scrs_mtt_root_to_spans, scores], dim=-1)

                pred_scores_mtt = create_scores_esm_pred(scores, unique_links, candidates, candidate_lengths,
                                                         torch_float_precision=self.torch_float_precision,
                                                         scrs_mtt_spans=scrs_mtt_spans)

                z_mask, z_mask_lengths = create_coreflinker_mtt_z_mask_indexed(pred_spans, gold_spans, gold_m2i,
                                                                               linker_target,
                                                                               candidates,
                                                                               candidate_lengths=candidate_lengths,
                                                                               unique_links=unique_links,
                                                                               unique_links_lengths=unique_links_lengths,
                                                                               unknown_id=self.unknown_dict)

                pred_masked_scores = z_mask * pred_scores_mtt

                # the inverse of mask receives a very small score (less than the current minimum)
                z_mask_inverse = 1.0 - z_mask
                min_pred_mask_scores = min(pred_masked_scores.min().item(), 0.0)
                pred_masked_scores = pred_masked_scores + (z_mask_inverse) * (min_pred_mask_scores - 999999)

                print('executing edmonds with ', int(z_mask.sum().item()), ' edges and ', z_mask.shape[-1], ' nodes: ',
                      unique_links.shape[-1], ' unique links and ',
                      z_mask.shape[-1] - unique_links.shape[-1] - 1, ' spans')
                start = time.time()

                pred_tree_mst = mst_only_tree(pred_masked_scores, z_mask_lengths, z_mask)

                end = time.time()
                print('done with edmonds in ', (end - start), ' seconds')

                # candidate_ids = []
                # for curr_cand_batch, _ in enumerate(candidates):
                #     unique_curr_batch = candidates[curr_cand_batch].unique(sorted=True)
                #     unique_curr_batch = unique_curr_batch[unique_curr_batch != nill_id]
                #     # (16/10/2020) - 0 is used for padding, so remove it
                #     unique_curr_batch = unique_curr_batch[unique_curr_batch != self.unknown_dict]
                #
                #     candidate_ids.append(unique_curr_batch)

                link_id_to_coref_col = list()
                for batch_id, candidate_ids_batch in enumerate(unique_links):
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
                                               # candidate_ids,
                                               unique_links,
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
                               # zip(decoded_m2i_coref_linker, pred_spans, candidate_ids,
                               zip(decoded_m2i_coref_linker, pred_spans, unique_links,
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

                # TODO (24/02/2021): we are here adapting: maybe from the beggining add the column of diagonal to scores instead
                #   of only doing this inside create_scores_esm_pred only:
                #   scrs_mtt_root_to_spans = torch.diagonal(scrs_mtt_spans, dim1=-2, dim2=-1).unsqueeze(-2)??
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

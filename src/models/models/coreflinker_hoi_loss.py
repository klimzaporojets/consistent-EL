import torch
import torch.nn as nn

from misc import settings
from metrics.coref import MetricCoref, MetricCorefAverage
from metrics.corefx import MetricCorefExternal
from metrics.linker import MetricLinkerImproved, MetricLinkAccuracy, MetricLinkAccuracyNoCandidates
from metrics.misc import MetricObjective
# from modules.tasks.linker import predict_scores
from models.misc.misc import batched_index_select
from models.models.coreflinker_loss import create_candidate_mask, create_coreflinker_target_forward, decode_m2i_coreflinker, \
    convert_coref, m2i_to_clusters_linkercoref, remove_disabled_spans, remove_disabled_spans_linking, \
    remove_disabled_scores_coref, remove_disabled_scores_linking, predict_scores_coref, remove_singletons_without_link
from models.utils.misc import predict_scores, get_mask_from_sequence_lengths
from models.utils.math import logsumexp


class CorefLinkerLossHoi(nn.Module):

    def __init__(self, link_task, coref_task, entity_dictionary, config, end_to_end):
        super(CorefLinkerLossHoi, self).__init__()

        self.enabled = config['enabled']

        self.entity_dictionary = entity_dictionary
        if self.enabled:
            self.labels = self.entity_dictionary.tolist()
            self.unknown_dict = entity_dictionary.lookup('###UNKNOWN###')

        self.link_task = link_task
        self.coref_task = coref_task
        self.weight = config.get('weight', 1.0)
        self.filter_singletons_with_pruner = config['filter_singletons_with_pruner']
        self.filter_only_singletons = config['filter_only_singletons']

        self.filter_singletons_with_matrix = config['filter_singletons_with_matrix']
        self.ignore_no_mention_chains = config['ignore_no_mention_chains']
        self.filter_singletons_with_ner = config['filter_singletons_with_ner']
        self.no_nil_in_targets = config['no_nil_in_targets']
        self.doc_level_candidates = config['doc_level_candidates']
        self.singletons = self.filter_singletons_with_pruner or self.filter_singletons_with_ner  # write out singletons to json
        self.end_to_end = end_to_end

    def forward(self, scores, gold_m2i, filtered_spans, gold_spans, linker,
                predict=False, pruner_spans=None, ner_spans=None, api_call=False, only_loss=False):
        output = {}
        output_coref = {}
        output_linking = {}

        if self.enabled and scores is not None:
            # pred_spans = filtered_spans['spans']
            pred_spans = filtered_spans['pruned_spans']

            linker_candidates = linker['candidates']
            candidate_lengths = linker['candidate_lengths']
            targets = linker.get('targets')

            if self.end_to_end:
                # if it is end-to-end, we only select the candidates pruned by pruner in order to avoid
                #   using too much memory
                # pred_spans_idx = filtered_spans['prune_indices']
                pred_spans_idx = filtered_spans['prune_indices_hoi']
            else:
                pred_spans_idx = filtered_spans['reindex_wrt_gold']

            linker_candidates = batched_index_select(linker_candidates, pred_spans_idx)
            candidate_lengths = batched_index_select(candidate_lengths.unsqueeze(-1), pred_spans_idx).squeeze(-1)

            candidates = linker_candidates.to(scores.device)  # torch.Size([1, 9, 17])

            nill_id = self.entity_dictionary.lookup('NILL')
            # none_id = self.entity_dictionary.lookup('NONE')

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
                raise BaseException("HUH")
            else:
                output['loss'] = 0.0

            output_coref['loss'] = output['loss']
            output_linking['loss'] = output['loss']

            if predict and not only_loss:
                candidate_ids = []
                for curr_cand_batch, _ in enumerate(candidates):
                    unique_curr_batch = candidates[curr_cand_batch].unique(sorted=True)
                    if self.no_nil_in_targets:
                        unique_curr_batch = unique_curr_batch[unique_curr_batch != nill_id]
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
                    linker_spans = filtered_spans['pruned_spans']
                else:
                    linker_spans = filtered_spans['pruned_spans']

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

                    output_linking['gold'] = linker['gold']
                else:
                    output_coref['gold'] = [None for _ in gold_spans]
                    output_linking['gold'] = [None for _ in gold_spans]

                # for loss just copies the loss from output for both linker and coref ; this 'loss' is needed when
                # later evaluating in metrics.misc.MetricObjective#update2 for example (see below when adding metrics
                # in create_metrics(self) method)

                if self.filter_singletons_with_pruner:
                    if not self.filter_only_singletons:
                        # this assumes that pruner is able to predict spans
                        output_coref['pred'] = remove_disabled_spans(output_coref['pred'], pruner_spans)
                        coref_flat = [{item for sublist in batch for item in sublist} for batch in output_coref['pred']]
                        output_linking['pred'] = remove_disabled_spans_linking(output_linking['pred'], coref_flat)
                        output_coref['scores'] = remove_disabled_scores_coref(output_coref['scores'], coref_flat)
                        output_linking['scores'] = remove_disabled_scores_linking(output_linking['scores'], coref_flat)
                    else:
                        # output_coref['pred'] = remove_disabled_spans(output_coref['pred'], pruner_spans)
                        output_coref['pred'] = remove_singletons_without_link(output_coref['pred'],
                                                                              output_linking['pred'],
                                                                              pruner_spans)
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

    def log_stats(self, dataset_name, predict, tb_logger, step_nr):
        pass  # for now nothing here

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

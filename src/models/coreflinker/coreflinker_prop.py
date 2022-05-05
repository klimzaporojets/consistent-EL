import torch
import torch.nn as nn
import torch.nn.functional as F
# from allennlp.nn.util import batched_index_select

from models.coreflinker.scorers import OptFFpairsCorefLinkerNaive, OptFFpairsCorefLinkerBase, OptFFpairs
from modules.misc.misc import batched_index_select
from modules.text_field import TextFieldEmbedderTokens
from modules.utils.misc import MyGate, overwrite_spans, coref_add_scores_coreflinker


# def create_candidate_mask(max_cand_length, candidate_lengths):
#     tmp = torch.arange(max_cand_length)
#     tmp = tmp.unsqueeze(0).unsqueeze(0)
#     candidate_lengths = candidate_lengths.unsqueeze(-1)
#     mask = tmp < candidate_lengths
#     return mask


class ModuleLinkerCorefProp(nn.Module):
    def __init__(self, dim_span, coref_pruner, span_pair_generator, config, dictionaries):
        # TODO: we are here: pass dim_entity here
        super(ModuleLinkerCorefProp, self).__init__()

        self.coref_prop = config['linkercoref_prop']['coref_prop']
        self.update_coref_scores = config['linkercoref_prop']['update_coref_scores']
        # if false, will imply to do union on the spans selected of pruner + linkable mentions
        # if true, will imply to take only the spans predicted by the pruner
        # self.end_to_end = end_to_end

        # self.append_from_pruner = config['append_from_pruner']
        # self.no_nil_in_targets = config['no_nil_in_targets']
        self.doc_level_candidates = config['doc_level_candidates']

        # self.only_mentions_with_valid_candidates = config['only_mentions_with_valid_candidates']

        # self.shared_pruner = config['shared_pruner']

        print("ModuleCorefProp(cp={})".format(self.coref_prop))

        self.coref_pruner = coref_pruner

        self.enabled = config['enabled']
        if self.enabled:
            self.coref = OptFFpairs(dim_span, 1, config['linkercoref_prop'], span_pair_generator)
            self.entity_embedder = TextFieldEmbedderTokens(dictionaries,
                                                           config['entity_embedder'])  # here is entity embedder 1

            if config['model_type'] == 'base':
                self.linker_coref = OptFFpairsCorefLinkerBase(dim_span, self.entity_embedder.dim,
                                                              1, config['linkercoref_prop'], span_pair_generator)
            elif config['model_type'] == 'super-naive':
                self.linker_coref = OptFFpairsCorefLinkerNaive(dim_span, self.entity_embedder.dim,
                                                               1, config['linkercoref_prop'], span_pair_generator)
        else:
            self.entity_embedder = None

        self.gate = MyGate(dim_span)

    def forward(self, all_spans, filtered_spans, sequence_lengths, linker):

        if not self.enabled or filtered_spans['span_vecs'] is None:
            return all_spans, filtered_spans, None

        all_vecs = all_spans['span_vecs']  # torch.Size([1, 69, 5, 1676])

        if self.doc_level_candidates:
            linker_candidates = linker['candidates_no_nill_doc']
        else:
            linker_candidates = linker['candidates']

        reindex = filtered_spans['reindex_wrt_gold']
        linker_candidates = batched_index_select(linker_candidates, reindex)

        triangular_mask = filtered_spans['triangular_mask']

        # Only process spans with linker information
        # (11/10/2020) - old code that is changed by filtered_spans that now have to come good thanks to
        #    "end_to_end_mentions" cofig parameter on model level

        # max_span_length = all_vecs.size(2)

        # linker_spans = linker['gold_spans_tensors']
        # linker_indices = spans_to_indices(linker_spans,
        #                                   max_span_length)  # tensor([[ 16,  27, 207, 227, 245, 250, 256, 266, 285]])
        #
        # linker_vecs = filter_spans(all_vecs, linker_indices.to(all_vecs.device))  # torch.Size([1, 9, 1676])
        #
        # linker_span_embeddings = linker_vecs  # torch.Size([1, 9, 1676])

        linker_span_embeddings = filtered_spans['span_vecs']

        # (11/10/2020) - end old code that is changed by filtered_spans that now have to come good thanks to
        #    "end_to_end_mentions" cofig parameter on model level

        # (11/10/2020) - new code that takes directly the filtered_spans
        # linker_span_embeddings = filtered_spans['span_vecs'] # TODO: this has to be changed, can not rely on the exact positoin of filtered['span_vecs']!!!
        # (11/10/2020) - end new code that takes directly the filtered_spans

        candidates = linker_candidates.to(all_vecs.device)  # torch.Size([1, 9, 17])

        # resorts the candidates according to the orders of the spans in the filtered_spans
        # candidates = candidates[filtered_spans['index_candidates']]

        candidate_vecs = self.entity_embedder(candidates)  # torch.Size([1, 9, 17, 200])

        # if not self.end_to_end:
        update_mentions = linker_span_embeddings  # torch.Size([1, 9, 1676])
        update_entities = candidate_vecs  # torch.Size([1, 9, 17, 200])

        # mentions_span_begin = linker['gold_spans_tensors'][..., 0].unsqueeze(-1)  # torch.Size([1, 9, 1])
        # mentions_span_end = linker['gold_spans_tensors'][..., 1].unsqueeze(-1)  # torch.Size([1, 9, 1])

        mentions_span_begin = filtered_spans['span_begin']  # torch.Size([1, 9, 1])
        mentions_span_end = filtered_spans['span_end']  # torch.Size([1, 9, 1])

        linker_coref_scores = self.linker_coref(update_mentions,
                                                update_entities,
                                                mentions_span_begin,
                                                mentions_span_end).squeeze(-1)

        # the "eye" is here , this is why I still add 0s and then it nullifies the main diagonal
        linker_coref_scores = coref_add_scores_coreflinker(linker_coref_scores, filtered_spans['span_scores'],
                                                           filter_singletons_with_matrix=False)

        update_all = all_spans.copy()
        update_filtered = filtered_spans.copy()

        if self.coref_prop > 0:
            # only get the coref part from linker_coref_scores:
            coref_scores = linker_coref_scores[..., -triangular_mask.size(-1):]
            for _ in range(self.coref_prop):
                probs = F.softmax(coref_scores - (1.0 - triangular_mask) * 1e23, dim=-1)
                ctxt = torch.matmul(probs, update_mentions)
                update_mentions = self.gate(update_mentions, ctxt)

                if self.update_coref_scores:
                    # coref_scores = self.coref(update_mentions, linker['spans_tensors'][:, :, 0].unsqueeze(-1),
                    #                           linker['spans_tensors'][:, :, 1].unsqueeze(-1)).squeeze(-1)
                    coref_scores = self.coref(update_mentions, mentions_span_begin, mentions_span_end).squeeze(-1)
                    # coref_scores = self.coref(update_mentions, filtered_span_begin, filtered_span_end).squeeze(-1)

                    # 14/10/2020 - added this part
                    eye = torch.eye(coref_scores.size(1)).unsqueeze(0).to(coref_scores)
                    coref_scores[:, :, -coref_scores.size(1):] = coref_scores[:, :, -coref_scores.size(1):] * \
                                                                 (1.0 - eye)

            linker_coref_scores[..., -triangular_mask.size(-1):] = coref_scores
            update_filtered['span_vecs'] = update_mentions
            update_all['span_vecs'] = overwrite_spans(update_all['span_vecs'], filtered_spans['prune_indices'],
                                                      filtered_spans['span_lengths'], update_mentions)

            # 19/10/2020 - (kzaporoj) - should not use filtered_spans in not E2E!!!
            # update_filtered['span_vecs'] = update_mentions
            # update_all['span_vecs'] = overwrite_spans(update_all['span_vecs'], filtered_spans['prune_indices'],
            #                                           filtered_spans['span_lengths'], update_mentions)

        return update_all, update_filtered, linker_coref_scores


class ModuleCorefLinkerDisabled(nn.Module):
    def __init__(self):
        super(ModuleCorefLinkerDisabled, self).__init__()
        self.enabled = False

    def forward(self, all_spans, filtered_spans, sequence_lengths, linker):
        return all_spans, filtered_spans, None

    def log_stats(self, dataset_name, predict, tb_logger, step_nr):
        pass


class ModuleCorefLinkerPropE2E(nn.Module):
    def __init__(self, dim_span, coref_pruner, span_pair_generator, config, dictionaries):
        # TODO: we are here: pass dim_entity here
        super(ModuleCorefLinkerPropE2E, self).__init__()

        self.coref_prop = config['coreflinker_prop']['coref_prop']
        self.update_coref_scores = config['coreflinker_prop']['update_coref_scores']
        # if false, will imply to do union on the spans selected of pruner + linkable mentions
        # if true, will imply to take only the spans predicted by the pruner
        # self.end_to_end = end_to_end

        # self.append_from_pruner = config['append_from_pruner']
        self.no_nil_in_targets = config['no_nil_in_targets']
        self.doc_level_candidates = config['doc_level_candidates']
        self.filter_singletons_with_matrix = config['filter_singletons_with_matrix']
        self.subtract_pruner_for_singletons = config['subtract_pruner_for_singletons']

        print("ModuleCorefProp(cp={})".format(self.coref_prop))

        self.coref_pruner = coref_pruner

        self.enabled = config['enabled']
        if self.enabled:
            if self.coref_prop > 0:
                self.coref = OptFFpairs(dim_span, 1, config['coreflinker_prop'], span_pair_generator)

            self.entity_embedder = TextFieldEmbedderTokens(dictionaries,
                                                           config['entity_embedder'])  # here is entity embedder 1

            if config['model_type'] == 'base':
                self.linker_coref = OptFFpairsCorefLinkerBase(dim_span, self.entity_embedder,
                                                              1, config['coreflinker_prop'], span_pair_generator,
                                                              filter_singletons_with_matrix=self.filter_singletons_with_matrix,
                                                              dictionaries=dictionaries)
            elif config['model_type'] == 'super-naive':
                self.linker_coref = OptFFpairsCorefLinkerNaive(dim_span, self.entity_embedder.dim,
                                                               1, config['coreflinker_prop'], span_pair_generator,
                                                               filter_singletons_with_matrix=self.filter_singletons_with_matrix)
        else:
            self.entity_embedder = None

        self.gate = MyGate(dim_span)

    def forward(self, all_spans, filtered_spans, sequence_lengths, linker):

        # if (not self.enabled) or (linker_spans.shape[1] == 0 and (not self.end_to_end)):
        #     return all_spans, filtered_spans, None
        if not self.enabled or filtered_spans['span_vecs'] is None:
            return all_spans, filtered_spans, None

        all_vecs = all_spans['span_vecs']  # torch.Size([1, 69, 5, 1676])

        # prune_indices
        # all_vecs should be of # torch.Size([1, 69, 5, 1676])
        # filtered_vecs = filtered_spans['span_vecs']  # torch.Size([1, 14, 1676])

        # TODO (13/10/2020) - changing to include end-to-end candidates, which would be the candidates not for gold
        #   BUT for the filtered_spans!

        # linker_candidates = batched_index_select(linker['cands_all_spans_no_nill'], filtered_spans['prune_indices'])
        linker_candidates = batched_index_select(linker['candidates'], filtered_spans['prune_indices'])
        # linker_candidates.shape --> [1, 21, 18]
        # linker['candidates'].shape --> [1, 1440, 18]
        # filtered_spans['prune_indices'].shape --> [1, 21]

        # TODO 13/10/2020 - WE ARE HERE: use batched_index_select from allennlp to get the candidates.
        #    batch index select of linker['cands_all_spans_no_nill'] based on indices in filtered_spans['prune_indices']

        # TODO (13/10/2020) - end changing to include end-to-end candidates, which would be the candidates not for gold
        #   BUT for the filtered_spans!

        filtered_span_begin = filtered_spans['span_begin']
        # filtered_span_begin.shape --> [1, 21, 1]
        #
        filtered_span_end = filtered_spans['span_end']
        # filtered_span_end.shape --> [1, 21, 1]
        #

        triangular_mask = filtered_spans['triangular_mask']
        # triangular_mask.shape --> [1, 21, 21]
        #

        # Only process spans with linker information
        # (11/10/2020) - old code that is changed by filtered_spans that now have to come good thanks to
        #    "end_to_end_mentions" cofig parameter on model level
        # max_span_length = all_vecs.size(2)
        # linker_spans = linker['spans_tensors']
        # linker_indices = spans_to_indices(linker_spans,
        #                                   max_span_length)  # tensor([[ 16,  27, 207, 227, 245, 250, 256, 266, 285]])
        #
        # linker_vecs = filter_spans(all_vecs, linker_indices.to(all_vecs.device))  # torch.Size([1, 9, 1676])
        #
        # linker_span_embeddings = linker_vecs  # torch.Size([1, 9, 1676])
        # (11/10/2020) - end old code that is changed by filtered_spans that now have to come good thanks to
        #    "end_to_end_mentions" cofig parameter on model level

        # (11/10/2020) - new code that takes directly the filtered_spans
        # TODO: this has to be changed, can not rely on the exact position of filtered['span_vecs']!!!
        linker_span_embeddings = filtered_spans['span_vecs']
        # linker_span_embeddings.shape --> [1, 21, 2324]
        # (11/10/2020) - end new code that takes directly the filtered_spans

        candidates = linker_candidates.to(all_vecs.device)  # torch.Size([1, 9, 17])
        # candidates.shape --> [1, 21, 18]
        candidate_vecs = self.entity_embedder(candidates)  # torch.Size([1, 9, 17, 200])
        # candidate_vecs.shape --> [1, 21, 18, 200]

        # if not self.end_to_end:
        update_mentions = linker_span_embeddings  # torch.Size([1, 9, 1676])
        # update_mentions.shape --> [1, 21, 2324]
        update_entities = candidate_vecs  # torch.Size([1, 9, 17, 200])
        # update_entities.shape --> [1, 21, 18, 200]

        # mentions_span_begin = linker['spans_tensors'][:, :, 0].unsqueeze(-1)  # torch.Size([1, 9, 1])
        # mentions_span_end = linker['spans_tensors'][:, :, 1].unsqueeze(-1)  # torch.Size([1, 9, 1])
        # mentions_span_begin = filtered_spans['span_begin']  # torch.Size([1, 9, 1])
        # mentions_span_end = filtered_spans['span_end']  # torch.Size([1, 9, 1])

        linker_coref_scores = self.linker_coref(update_mentions,
                                                update_entities,
                                                filtered_span_begin,
                                                filtered_span_end).squeeze(-1)
        # linker_coref_scores.shape --> [1, 21, 39]
        # filtered_spans['span_scores'].shape --> [1,21,1]
        linker_coref_scores = coref_add_scores_coreflinker(linker_coref_scores, filtered_spans['span_scores'],
                                                           self.filter_singletons_with_matrix,
                                                           subtract_pruner_for_singletons=self.subtract_pruner_for_singletons)
        # linker_coref_scores.shape --> [1, 21, 39]
        #
        update_all = all_spans.copy()
        #
        #
        update_filtered = filtered_spans.copy()
        #
        #

        if self.coref_prop > 0:
            # only get the coref part from linker_coref_scores:
            coref_scores = linker_coref_scores[..., -triangular_mask.size(-1):]  # [1, 21, 21]
            for _ in range(self.coref_prop):
                probs = F.softmax(coref_scores - (1.0 - triangular_mask) * 1e23, dim=-1)
                ctxt = torch.matmul(probs, update_mentions)
                update_mentions = self.gate(update_mentions, ctxt)
                # probs.shape --> [1, 21, 21]; ctxt.shape --> [1, 21, 2324]; update_mentions.shape --> [1, 21, 2324]
                if self.update_coref_scores:
                    # TODO kzaporoj (11/10/2020) - will it work with this self.coref(...)???
                    #  Or should we have a separate approach for joint coref+linker?
                    coref_scores = self.coref(update_mentions, filtered_span_begin, filtered_span_end).squeeze(-1)
                    # coref_scores.shape --> [1, 21, 21]
                    if self.coref_pruner is not None:
                        coref_scores = coref_add_scores_coreflinker(coref_scores, self.coref_pruner(update_mentions),
                                                                    self.filter_singletons_with_matrix,
                                                                    subtract_pruner_for_singletons=self.subtract_pruner_for_singletons)
                        # [1, 21, 21]
                        #
                    else:
                        # 14/10/2020 - added this part
                        eye = torch.eye(coref_scores.size(1)).unsqueeze(0).to(coref_scores)
                        #
                        #
                        coref_scores[:, :, -coref_scores.size(1):] = coref_scores[:, :, -coref_scores.size(1):] * \
                                                                     (1.0 - eye)
                        #
                        #

            linker_coref_scores[..., -triangular_mask.size(-1):] = coref_scores
            # [1, 21, 39]
            #
            update_filtered['span_vecs'] = update_mentions
            # update_mentions.shape --> [1, 21, 2324]
            #
            #
            update_all['span_vecs'] = overwrite_spans(update_all['span_vecs'], filtered_spans['prune_indices'],
                                                      filtered_spans['span_lengths'], update_mentions)
            # update_all['span_vecs'].shape --> [1, 96, 15, 2324]
            # filtered_spans['prune_indices'].shape --> [1,21]
            # filtered_spans['span_lengths'] --> tensor([21])
            # update_mentions.shape --> [1, 21, 2324]
        return update_all, update_filtered, linker_coref_scores

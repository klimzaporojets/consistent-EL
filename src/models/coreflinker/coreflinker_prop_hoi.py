import torch
import torch.nn as nn
import torch.nn.functional as F
# from allennlp.nn.util import batched_index_select

from models.coreflinker.scorers import OptFFpairsCorefLinkerNaive, OptFFpairsCorefLinkerBase, \
    OptFFpairsCorefLinkerBaseHoi
from modules.entity_embeddings import KolitsasEntityEmbeddings
from modules.misc.misc import batched_index_select
from modules.text_field import TextFieldEmbedderTokens
from modules.utils.misc import MyGate, coref_add_scores_coreflinker, filter_spans, overwrite_spans_hoi


class ModuleCorefLinkerPropE2EHoi(nn.Module):
    def __init__(self, dim_span, coref_pruner, span_pair_generator, config, dictionaries):
        # TODO: we are here: pass dim_entity here
        super(ModuleCorefLinkerPropE2EHoi, self).__init__()

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

        print("ModuleCorefPropE2EHoi(cp={})".format(self.coref_prop))

        self.coref_pruner = coref_pruner

        self.enabled = config['enabled']
        if self.enabled:
            # self.coref_prop_type = config['coreflinker_prop']['coref_prop_type']
            self.init_weights_std = config['coreflinker_prop']['init_weights_std']
            # if self.coref_prop > 0:
            #     self.coref = OptFFpairs(dim_span, 1, config['coreflinker_prop'], span_pair_generator)
            self.embeddings_type = config['entity_embedder']['type']

            if self.embeddings_type == 'yamada-johannes':
                self.entity_embedder = TextFieldEmbedderTokens(dictionaries, config['entity_embedder'])
            elif self.embeddings_type == 'kolitsas':
                self.entity_embedder = KolitsasEntityEmbeddings(dictionaries, config['entity_embedder'])
            else:
                raise RuntimeError('Unrecognized embeddings type in ModuleCorefLinkerPropE2EHoi: ' + self.embeddings_type)

            if config['model_type'] == 'base':
                self.linker_coref = OptFFpairsCorefLinkerBase(dim_span, self.entity_embedder,
                                                              1, config['coreflinker_prop'], span_pair_generator,
                                                              filter_singletons_with_matrix=self.filter_singletons_with_matrix,
                                                              dictionaries=dictionaries)
            elif config['model_type'] == 'base-hoi':
                self.linker_coref = OptFFpairsCorefLinkerBaseHoi(dim_span, self.entity_embedder,
                                                                 1, config['coreflinker_prop'], span_pair_generator,
                                                                 filter_singletons_with_matrix=self.filter_singletons_with_matrix,
                                                                 dictionaries=dictionaries)
            elif config['model_type'] == 'super-naive':
                self.linker_coref = OptFFpairsCorefLinkerNaive(dim_span, self.entity_embedder.dim,
                                                               1, config['coreflinker_prop'], span_pair_generator,
                                                               filter_singletons_with_matrix=self.filter_singletons_with_matrix)
            self.gate = MyGate(dim_span, self.init_weights_std)
        else:
            self.entity_embedder = None

    def forward(self, all_spans, filtered_spans, sequence_lengths, linker):

        # if (not self.enabled) or (linker_spans.shape[1] == 0 and (not self.end_to_end)):
        #     return all_spans, filtered_spans, None
        if not self.enabled or filtered_spans['span_vecs'] is None:
            return all_spans, filtered_spans, None

        cand_all_vecs = all_spans['cand_span_vecs']
        # cand_all_vecs.shape --> [1, 335, 2324]

        # prune_indices
        # all_vecs should be of # torch.Size([1, 69, 5, 1676])
        # filtered_vecs = filtered_spans['span_vecs']  # torch.Size([1, 14, 1676])

        # linker_candidates = batched_index_select(linker['candidates'], filtered_spans['prune_indices'])
        linker_indices_hoi = filtered_spans['prune_indices_hoi']
        # linker_indices_hoi.shape --> [1, 21]
        linker_candidates = batched_index_select(linker['candidates'], linker_indices_hoi)
        # linker_candidates.shape --> [1, 21, 18]

        filtered_span_begin = filtered_spans['span_begin']
        # filtered_span_begin.shape --> [1, 21]

        filtered_span_end = filtered_spans['span_end']
        # filtered_span_end.shape --> [1, 21]

        triangular_mask = filtered_spans['triangular_mask']
        # triangular_mask.shape --> [1, 21, 21]

        # linker_span_embeddings = filtered_spans['span_vecs']
        linker_span_embeddings = filter_spans(cand_all_vecs, linker_indices_hoi.to(cand_all_vecs.device))
        # linker_span_embeddings.shape --> [1, 21, 2324]
        #
        # (11/10/2020) - end new code that takes directly the filtered_spans

        candidates = linker_candidates.to(cand_all_vecs.device)  # [1, 21, 18]
        candidate_vecs = self.entity_embedder(candidates)  # [1, 21, 18, 200]

        # if not self.end_to_end:
        update_mentions = linker_span_embeddings  # [1, 21, 2324]
        update_entities = candidate_vecs  # [1, 21, 18, 200]

        # mentions_span_begin = linker['spans_tensors'][:, :, 0].unsqueeze(-1)  # torch.Size([1, 9, 1])
        # mentions_span_end = linker['spans_tensors'][:, :, 1].unsqueeze(-1)  # torch.Size([1, 9, 1])
        # mentions_span_begin = filtered_spans['span_begin']  # torch.Size([1, 9, 1])
        # mentions_span_end = filtered_spans['span_end']  # torch.Size([1, 9, 1])

        linker_coref_scores = self.linker_coref(update_mentions,
                                                update_entities,
                                                filtered_span_begin,
                                                filtered_span_end,
                                                do_only_coref=False).squeeze(-1)
        # linker_coref_scores.shape --> [1, 21, 39]
        # filtered_spans['span_scores'].shape --> torch.Size([1, 21])
        linker_coref_scores = coref_add_scores_coreflinker(linker_coref_scores,
                                                           # filtered_spans['span_scores'],
                                                           filtered_spans['span_scores'].unsqueeze(-1),
                                                           self.filter_singletons_with_matrix,
                                                           subtract_pruner_for_singletons=self.subtract_pruner_for_singletons)

        # TODO: do we need this copy()??, for now just disableing the copy()
        # update_all = all_spans.copy()
        # update_filtered = filtered_spans.copy()
        update_all = all_spans
        update_filtered = filtered_spans

        if self.coref_prop > 0:
            # coref propagation only on coreference (mentio) part, not the link part
            # only get the coref part from linker_coref_scores:
            # TODO: get coref+linker part graph propagation
            coref_scores = linker_coref_scores[..., -triangular_mask.size(-1):]
            for _ in range(self.coref_prop):
                # if self.coref_prop_type =='only_mentions':
                probs = F.softmax(coref_scores - (1.0 - triangular_mask) * 1e23, dim=-1)  # probs.shape --> [1, 21, 21]
                # probs = F.softmax(linker_coref_scores - (1.0 - triangular_mask) * 1e23, dim=-1)
                ctxt = torch.matmul(probs,
                                    update_mentions)  # update_mentions.shape -->  torch.Size([1, 21, 2324]) ; ctxt.shape --> torch.Size([1, 21, 2324])
                update_mentions = self.gate(update_mentions, ctxt)  # update_mentions.shape -->
                # TODO: update_mentions and update_entities have to be calculated separately!

                if self.update_coref_scores:
                    # coref_scores = self.coref(update_mentions, filtered_span_begin, filtered_span_end).squeeze(-1)
                    # DONE: here figure out what to do!!! can not use just another nnet such as self.coref
                    #  (like it was previously in the line above)
                    #  I think (now 18/04/2021) it HAS TO BE the same self.linker_coref nnet! BUT with the option of
                    #  only enable the coref part and not the linker part there! OR TODO: BOTH (coref+linker) parts.
                    #  HERE add a parameter like "do_only_coref".
                    coref_scores = self.linker_coref(update_mentions,
                                                     update_entities,
                                                     filtered_span_begin,
                                                     filtered_span_end, do_only_coref=True).squeeze(-1)
                    # coref_scores.shape --> torch.Size([1, 21, 21])
                    if self.coref_pruner is not None:
                        coref_scores = coref_add_scores_coreflinker(coref_scores, self.coref_pruner(update_mentions),
                                                                    self.filter_singletons_with_matrix,
                                                                    subtract_pruner_for_singletons=self.subtract_pruner_for_singletons)
                    else:
                        # 14/10/2020 - added this part
                        eye = torch.eye(coref_scores.size(1)).unsqueeze(0).to(coref_scores)
                        coref_scores[:, :, -coref_scores.size(1):] = coref_scores[:, :, -coref_scores.size(1):] * \
                                                                     (1.0 - eye)

            linker_coref_scores[..., -triangular_mask.size(-1):] = coref_scores
            update_filtered['span_vecs'] = update_mentions  # torch.Size([1, 21, 2324])
            update_all['cand_span_vecs'] = overwrite_spans_hoi(update_all['cand_span_vecs'],
                                                               filtered_spans['prune_indices_hoi'],
                                                               filtered_spans['span_lengths'], update_mentions)

        return update_all, update_filtered, linker_coref_scores

    def log_stats(self, dataset_name, predict, tb_logger, step_nr):
        pass

import torch
import torch.nn as nn
# from allennlp.nn.util import batched_index_select

import settings
from models.coreflinker.scorers import OptFFpairsCorefLinkerMTTBaseHoi
from modules.entity_embeddings import KolitsasEntityEmbeddings
from modules.misc.misc import batched_index_select
from modules.text_field import TextFieldEmbedderTokens
from modules.utils.misc import MyGate, filter_spans


class ModuleCorefLinkerMTTPropE2EHoi(nn.Module):
    def __init__(self, dim_span, coref_pruner, span_pair_generator, config, dictionaries):
        # TODO: we are here: pass dim_entity here
        super(ModuleCorefLinkerMTTPropE2EHoi, self).__init__()

        # no graph propagations in the first version
        self.coref_prop = None

        print("ModuleCorefLinkerMTTPropE2E(cp={})".format(self.coref_prop))

        self.coref_pruner = coref_pruner

        # self.coref = OptFFpairs(dim_span, 1, config['linkercoref_prop'], span_pair_generator)

        self.enabled = config['enabled']
        self.nonlinear_function = config['nonlinear_function']
        self.smart_arsinh = config['smart_arsinh']
        self.float_precision = config['float_precision']

        if self.enabled:
            self.zeros_to_clusters = config['zeros_to_clusters']
            self.zeros_to_links = config['zeros_to_links']
            self.init_weights_std = config['coreflinker_prop']['init_weights_std']
            self.add_pruner_to_root = config['add_pruner_to_root']
            self.root_link_max_spans_to_link = config['root_link_max_spans_to_link']
            self.embeddings_type = config['entity_embedder']['type']
            # self.entity_embedder = TextFieldEmbedderTokens(dictionaries, config['entity_embedder'])
            if self.embeddings_type == 'yamada-johannes':
                self.entity_embedder = TextFieldEmbedderTokens(dictionaries, config['entity_embedder'])
            elif self.embeddings_type == 'kolitsas':
                self.entity_embedder = KolitsasEntityEmbeddings(dictionaries, config['entity_embedder'])
            else:
                raise RuntimeError(
                    'Unrecognized embeddings type in ModuleCorefLinkerPropE2EHoi: ' + self.embeddings_type)

            # if config['model_type'] == 'base':
            self.linker_coref = OptFFpairsCorefLinkerMTTBaseHoi(dim_span, self.entity_embedder,
                                                                1, config['coreflinker_prop'], span_pair_generator,
                                                                # filter_singletons_with_matrix=False,
                                                                dictionaries=dictionaries,
                                                                zeros_to_clusters=self.zeros_to_clusters,
                                                                zeros_to_links=self.zeros_to_links)
            # elif config['model_type'] == 'super-naive':
            #     self.linker_coref = OptFFpairsLinkerCorefNaive(dim_span, self.entity_embedder.dim,
            #                                                    1, config['model_details'], span_pair_generator,
            #                                                    filter_singletons_with_matrix=False)
            self.gate = MyGate(dim_span, init_weights_std=self.init_weights_std)
        else:
            self.entity_embedder = None

    def coref_add_scores_coreflinker(self, coref_scores, filtered_prune_scores, filter_singletons_with_matrix,
                                     subtract_pruner_for_singletons=True):
        scores_left = filtered_prune_scores  # .shape --> torch.Size([1, 21, 1])

        scores_right = filtered_prune_scores.squeeze(-1).unsqueeze(-2)  # .shape --> torch.Size([1, 1, 21])
        if not (self.zeros_to_clusters or self.zeros_to_links):
            if self.add_pruner_to_root:
                coref_scores = coref_scores + scores_left
            else:
                coref_scores[:, :, 1:] = coref_scores[:, :, 1:] + scores_left
        else:
            # the connections to root are not added
            coref_scores[:, :, 1:] = coref_scores[:, :, 1:] + scores_left

        coref_scores[:, :, -scores_right.size(-1):] = coref_scores[:, :, -scores_right.size(-1):] + scores_right
        # coref_scores.shape --> torch.Size([1, 21, 37])
        # scores_right.shape --> torch.Size([1, 1, 21])
        # scores_right.size(-1) --> 21

        # zero-out self references (without this pruner doesn't work)
        eye = torch.eye(coref_scores.size(1), device=settings.device).unsqueeze(0).to(coref_scores)
        #
        #
        coref_scores[:, :, -coref_scores.size(1):] = coref_scores[:, :, -coref_scores.size(1):] * (1.0 - eye)
        # coref_scores.shape --> torch.Size([1, 21, 37])
        # eye.shape --> torch.Size([1, 21, 21]) ; coref_scores[:, :, -coref_scores.size(1):].shape --> torch.Size([1, 21, 21])
        # if filter_singletons_with_matrix:
        #     # also adds (TODO substracts???) the pruner scores to the mentions in the positions to be ignored
        #     # has to subtract twice because it was added using scores_left already (see above)
        #     if subtract_pruner_for_singletons:
        #         coref_scores[:, :, :1] = coref_scores[:, :, :1] - scores_left - scores_left
        #     # print('TODO: ADD OR SUBSTRACT PRUNER SCORES FOR THE NOT ENTITY MENTION SINGLETONS??? ')
        return coref_scores
        # coref_scores.shape --> torch.Size([1, 21, 37])

    def forward(self, all_spans, filtered_spans, sequence_lengths, linker):

        # if (not self.enabled) or (linker_spans.shape[1] == 0 and (not self.end_to_end)):
        #     return all_spans, filtered_spans, None
        if not self.enabled or filtered_spans['span_vecs'] is None:
            return all_spans, filtered_spans, None

        # all_vecs = all_spans['span_vecs']  # torch.Size([1, 69, 5, 1676])
        cand_all_vecs = all_spans['cand_span_vecs']
        # cand_all_vecs.shape --> torch.Size([1, 335, 2324])
        # prune_indices
        # all_vecs should be of # torch.Size([1, 69, 5, 1676])
        # filtered_vecs = filtered_spans['span_vecs']  # torch.Size([1, 14, 1676])

        # TODO (13/10/2020) - changing to include end-to-end candidates, which would be the candidates not for gold
        #   BUT for the filtered_spans!

        # linker_candidates = batched_index_select(linker['cands_all_spans_no_nill'], filtered_spans['prune_indices'])
        linker_indices_hoi = filtered_spans['prune_indices_hoi']
        # linker_indices_hoi.shape --> torch.Size([1, 21])

        # linker_candidates = batched_index_select(linker['candidates'], filtered_spans['prune_indices'])
        linker_candidates = batched_index_select(linker['candidates'], linker_indices_hoi)
        # linker['candidates'].shape --> torch.Size([1, 335, 16])
        # linker_candidates.shape --> torch.Size([1, 21, 16])

        # TODO 13/10/2020 - WE ARE HERE: use batched_index_select from allennlp to get the candidates.
        #    batch index select of linker['cands_all_spans_no_nill'] based on indices in filtered_spans['prune_indices']

        # TODO (13/10/2020) - end changing to include end-to-end candidates, which would be the candidates not for gold
        #   BUT for the filtered_spans!

        filtered_span_begin = filtered_spans['span_begin']  # --> torch.Size([1, 21])
        filtered_span_end = filtered_spans['span_end']  # --> torch.Size([1, 21])

        # triangular_mask = filtered_spans['triangular_mask']

        # (11/10/2020) - new code that takes directly the filtered_spans
        # TODO: this has to be changed, can not rely on the exact position of filtered['span_vecs']!!!
        # linker_span_embeddings = filtered_spans['span_vecs']
        linker_span_embeddings = filter_spans(cand_all_vecs, linker_indices_hoi.to(cand_all_vecs.device))
        # linker_span_embeddings.shape --> torch.Size([1, 21, 2324])
        # (11/10/2020) - end new code that takes directly the filtered_spans

        candidates = linker_candidates.to(settings.device)  # torch.Size([1, 9, 17])
        # candidate_lengths = batched_index_select(linker['candidate_lengths'].unsqueeze(-1),
        #                                          filtered_spans['prune_indices']).squeeze(-1)
        candidate_lengths = batched_index_select(linker['candidate_lengths'].unsqueeze(-1), linker_indices_hoi) \
            .squeeze(-1)
        # linker['candidate_lengths'].shape --> torch.Size([1, 335])
        # filtered_spans['prune_indices_hoi'].shape --> torch.Size([1, 21])
        # candidate_lengths.shape --> torch.Size([1, 21])

        max_cand_length = candidate_lengths.max().item()
        candidates = candidates[:, :, :max_cand_length]

        candidate_vecs = self.entity_embedder(candidates)  # torch.Size([1, 9, 17, 200])

        # if not self.end_to_end:
        update_mentions = linker_span_embeddings  # torch.Size([1, 21, 2324])
        update_entities = candidate_vecs  # torch.Size([1, 21, 16, 200]) (before was torch.Size([1, 9, 17, 200]), correct???)

        linker_coref_scores = self.linker_coref(update_mentions, update_entities, filtered_span_begin,
                                                filtered_span_end, candidate_lengths=candidate_lengths,
                                                max_cand_length=max_cand_length).squeeze(-1)
        # candidate_lengths=linker['candidate_lengths']).squeeze(-1)
        # linker_coref_scores.shape -->
        linker_coref_scores = self.coref_add_scores_coreflinker(linker_coref_scores,
                                                                filtered_spans['span_scores'].unsqueeze(-1),
                                                                # self.filter_singletons_with_matrix,
                                                                False, subtract_pruner_for_singletons=False)
        # linker_coref_scores.shape --> torch.Size([1, 21, 38])
        # if self.nonlinear_function is not None and self.nonlinear_function == 'arsinh':
        #     if not self.smart_arsinh:
        #         # if it is smart_arsinh, it does the simplification directly in coreflinker_mtt_hoi.py
        #         linker_coref_scores = torch.arcsinh(linker_coref_scores)
        # subtract_pruner_for_singletons=self.subtract_pruner_for_singletons)

        # TODO: is this.copy() necessary??? maybe when doing graph propagation, but not now
        # update_all = all_spans.copy()
        # update_filtered = filtered_spans.copy()
        update_all = all_spans
        update_filtered = filtered_spans

        # if self.float_precision == 'float64':
        #     linker_coref_scores = linker_coref_scores.double()

        return update_all, update_filtered, linker_coref_scores

    def log_stats(self, dataset_name, predict, tb_logger, step_nr):
        # TODO: think maybe some stats useful here
        pass

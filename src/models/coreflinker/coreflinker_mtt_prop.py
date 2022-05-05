import torch
import torch.nn as nn
# from allennlp.nn.util import batched_index_select

from models.coreflinker.scorers import OptFFpairsCorefLinkerMTTBase
from modules.misc.misc import batched_index_select
from modules.text_field import TextFieldEmbedderTokens
from modules.utils.misc import MyGate, coref_add_scores_coreflinker


class ModuleCorefLinkerMTTPropE2E(nn.Module):
    def __init__(self, dim_span, coref_pruner, span_pair_generator, config, dictionaries):
        # TODO: we are here: pass dim_entity here
        super(ModuleCorefLinkerMTTPropE2E, self).__init__()

        # no graph propagations in the first version
        self.coref_prop = None

        print("ModuleCorefLinkerMTTPropE2E(cp={})".format(self.coref_prop))

        self.coref_pruner = coref_pruner

        # self.coref = OptFFpairs(dim_span, 1, config['linkercoref_prop'], span_pair_generator)

        self.enabled = config['enabled']
        self.nonlinear_function = config['nonlinear_function']
        self.float_precision = config['float_precision']

        if self.enabled:
            self.entity_embedder = TextFieldEmbedderTokens(dictionaries, config['entity_embedder'])

            # if config['model_type'] == 'base':
            self.linker_coref = OptFFpairsCorefLinkerMTTBase(dim_span, self.entity_embedder,
                                                             1, config['coreflinker_prop'], span_pair_generator,
                                                             # filter_singletons_with_matrix=False,
                                                             dictionaries=dictionaries)
            # elif config['model_type'] == 'super-naive':
            #     self.linker_coref = OptFFpairsLinkerCorefNaive(dim_span, self.entity_embedder.dim,
            #                                                    1, config['model_details'], span_pair_generator,
            #                                                    filter_singletons_with_matrix=False)
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

        # TODO 13/10/2020 - WE ARE HERE: use batched_index_select from allennlp to get the candidates.
        #    batch index select of linker['cands_all_spans_no_nill'] based on indices in filtered_spans['prune_indices']

        # TODO (13/10/2020) - end changing to include end-to-end candidates, which would be the candidates not for gold
        #   BUT for the filtered_spans!

        filtered_span_begin = filtered_spans['span_begin']
        filtered_span_end = filtered_spans['span_end']

        # triangular_mask = filtered_spans['triangular_mask']

        # (11/10/2020) - new code that takes directly the filtered_spans
        # TODO: this has to be changed, can not rely on the exact position of filtered['span_vecs']!!!
        linker_span_embeddings = filtered_spans['span_vecs']
        # (11/10/2020) - end new code that takes directly the filtered_spans

        candidates = linker_candidates.to(all_vecs.device)  # torch.Size([1, 9, 17])
        candidate_lengths = batched_index_select(linker['candidate_lengths'].unsqueeze(-1),
                                                 filtered_spans['prune_indices']).squeeze(-1)

        max_cand_length = candidate_lengths.max().item()
        candidates = candidates[:, :, :max_cand_length]

        candidate_vecs = self.entity_embedder(candidates)  # torch.Size([1, 9, 17, 200])

        # if not self.end_to_end:
        update_mentions = linker_span_embeddings  # torch.Size([1, 9, 1676])
        update_entities = candidate_vecs  # torch.Size([1, 9, 17, 200])

        linker_coref_scores = self.linker_coref(update_mentions, update_entities, filtered_span_begin,
                                                filtered_span_end,
                                                candidate_lengths=candidate_lengths,
                                                max_cand_length=max_cand_length).squeeze(-1)
        # candidate_lengths=linker['candidate_lengths']).squeeze(-1)

        linker_coref_scores = coref_add_scores_coreflinker(linker_coref_scores, filtered_spans['span_scores'],
                                                           # self.filter_singletons_with_matrix,
                                                           False,
                                                           subtract_pruner_for_singletons=False)

        if self.nonlinear_function is not None and self.nonlinear_function == 'arsinh':
            linker_coref_scores = torch.arcsinh(linker_coref_scores)
        # subtract_pruner_for_singletons=self.subtract_pruner_for_singletons)

        update_all = all_spans.copy()
        update_filtered = filtered_spans.copy()

        # if self.float_precision == 'float64':
        #     linker_coref_scores = linker_coref_scores.double()

        return update_all, update_filtered, linker_coref_scores

import logging

import torch
import torch.nn as nn

from models.models.scorers import OptFFpairsCorefLinkerMTTBase
from models.misc.misc import batched_index_select
from models.utils.misc import MyGate, coref_add_scores_coreflinker
from models.misc.text_field import TextFieldEmbedderTokens

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger()

class ModuleCorefLinkerMTTPropE2E(nn.Module):
    def __init__(self, dim_span, coref_pruner, span_pair_generator, config, dictionaries):
        # TODO: we are here: pass dim_entity here
        super(ModuleCorefLinkerMTTPropE2E, self).__init__()

        # no graph propagations in the first version
        self.coref_prop = None

        logger.info('ModuleCorefLinkerMTTPropE2E(cp={})'.format(self.coref_prop))

        self.coref_pruner = coref_pruner

        self.enabled = config['enabled']
        self.nonlinear_function = config['nonlinear_function']
        self.float_precision = config['float_precision']

        if self.enabled:
            self.entity_embedder = TextFieldEmbedderTokens(dictionaries, config['entity_embedder'])

            self.linker_coref = OptFFpairsCorefLinkerMTTBase(dim_span, self.entity_embedder,
                                                             1, config['coreflinker_prop'], span_pair_generator,
                                                             dictionaries=dictionaries)
        else:
            self.entity_embedder = None

        self.gate = MyGate(dim_span)

    def forward(self, all_spans, filtered_spans, sequence_lengths, linker):
        if not self.enabled or filtered_spans['span_vecs'] is None:
            return all_spans, filtered_spans, None

        all_vecs = all_spans['span_vecs']  # torch.Size([1, 69, 5, 1676])

        linker_candidates = batched_index_select(linker['candidates'], filtered_spans['prune_indices'])

        filtered_span_begin = filtered_spans['span_begin']
        filtered_span_end = filtered_spans['span_end']

        # (11/10/2020) - new code that takes directly the filtered_spans
        linker_span_embeddings = filtered_spans['span_vecs']
        # (11/10/2020) - end new code that takes directly the filtered_spans

        candidates = linker_candidates.to(all_vecs.device)  # torch.Size([1, 9, 17])
        candidate_lengths = batched_index_select(linker['candidate_lengths'].unsqueeze(-1),
                                                 filtered_spans['prune_indices']).squeeze(-1)

        max_cand_length = candidate_lengths.max().item()
        candidates = candidates[:, :, :max_cand_length]

        candidate_vecs = self.entity_embedder(candidates)  # torch.Size([1, 9, 17, 200])

        update_mentions = linker_span_embeddings  # torch.Size([1, 9, 1676])
        update_entities = candidate_vecs  # torch.Size([1, 9, 17, 200])

        linker_coref_scores = self.linker_coref(update_mentions, update_entities, filtered_span_begin,
                                                filtered_span_end,
                                                candidate_lengths=candidate_lengths,
                                                max_cand_length=max_cand_length).squeeze(-1)

        linker_coref_scores = coref_add_scores_coreflinker(linker_coref_scores, filtered_spans['span_scores'],
                                                           False,
                                                           subtract_pruner_for_singletons=False)

        if self.nonlinear_function is not None and self.nonlinear_function == 'arsinh':
            linker_coref_scores = torch.arcsinh(linker_coref_scores)

        update_all = all_spans.copy()
        update_filtered = filtered_spans.copy()

        return update_all, update_filtered, linker_coref_scores

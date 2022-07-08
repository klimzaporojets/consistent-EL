import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.misc.entity_embeddings import KolitsasEntityEmbeddings
from models.misc.misc import batched_index_select
from models.misc.text_field import TextFieldEmbedderTokens
from models.models.scorers import OptFFpairsCorefLinkerNaive, OptFFpairsCorefLinkerBase, \
    OptFFpairsCorefLinkerBaseHoi
from models.utils.misc import MyGate, filter_spans, coref_add_scores_coreflinker, overwrite_spans_hoi

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger()


class ModuleCorefLinkerPropE2EHoi(nn.Module):
    def __init__(self, dim_span, coref_pruner, span_pair_generator, config, dictionaries):
        super(ModuleCorefLinkerPropE2EHoi, self).__init__()

        self.coref_prop = config['coreflinker_prop']['coref_prop']
        self.update_coref_scores = config['coreflinker_prop']['update_coref_scores']

        self.no_nil_in_targets = config['no_nil_in_targets']
        self.doc_level_candidates = config['doc_level_candidates']
        self.filter_singletons_with_matrix = config['filter_singletons_with_matrix']
        self.subtract_pruner_for_singletons = config['subtract_pruner_for_singletons']

        logger.info('ModuleCorefPropE2EHoi(cp={})'.format(self.coref_prop))

        self.coref_pruner = coref_pruner

        self.enabled = config['enabled']
        if self.enabled:
            self.init_weights_std = config['coreflinker_prop']['init_weights_std']

            self.embeddings_type = config['entity_embedder']['type']

            if self.embeddings_type == 'yamada-johannes':
                self.entity_embedder = TextFieldEmbedderTokens(dictionaries, config['entity_embedder'])
            elif self.embeddings_type == 'kolitsas':
                self.entity_embedder = KolitsasEntityEmbeddings(dictionaries, config['entity_embedder'])
            else:
                raise RuntimeError(
                    'Unrecognized embeddings type in ModuleCorefLinkerPropE2EHoi: ' + self.embeddings_type)

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

        if not self.enabled or filtered_spans['span_vecs'] is None:
            return all_spans, filtered_spans, None

        cand_all_vecs = all_spans['cand_span_vecs']
        # cand_all_vecs.shape --> [1, 335, 2324]

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

        update_all = all_spans
        update_filtered = filtered_spans

        if self.coref_prop > 0:
            # coref propagation only on coreference (mentio) part, not the link part
            coref_scores = linker_coref_scores[..., -triangular_mask.size(-1):]
            for _ in range(self.coref_prop):
                probs = F.softmax(coref_scores - (1.0 - triangular_mask) * 1e23, dim=-1)  # probs.shape --> [1, 21, 21]
                ctxt = torch.matmul(probs, update_mentions)
                # update_mentions.shape -->  torch.Size([1, 21, 2324]) ; ctxt.shape --> torch.Size([1, 21, 2324])
                update_mentions = self.gate(update_mentions, ctxt)  # update_mentions.shape -->
                # TODO: update_mentions and update_entities have to be calculated separately!

                if self.update_coref_scores:
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

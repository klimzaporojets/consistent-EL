import logging

import torch
import torch.nn as nn

from misc import settings
from models.misc.entity_embeddings import KolitsasEntityEmbeddings
from models.misc.misc import batched_index_select
from models.misc.text_field import TextFieldEmbedderTokens
from models.models.scorers import OptFFpairsCorefLinkerMTTBaseHoi
from models.utils.misc import MyGate, filter_spans

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger()


class ModuleCorefLinkerMTTPropE2EHoi(nn.Module):
    def __init__(self, dim_span, coref_pruner, span_pair_generator, config, dictionaries):
        super(ModuleCorefLinkerMTTPropE2EHoi, self).__init__()

        # no graph propagations in the first version
        self.coref_prop = None

        logger.info('ModuleCorefLinkerMTTPropE2E(cp={})'.format(self.coref_prop))

        self.coref_pruner = coref_pruner

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
            if self.embeddings_type == 'yamada-johannes':
                self.entity_embedder = TextFieldEmbedderTokens(dictionaries, config['entity_embedder'])
            elif self.embeddings_type == 'kolitsas':
                self.entity_embedder = KolitsasEntityEmbeddings(dictionaries, config['entity_embedder'])
            else:
                raise RuntimeError(
                    'Unrecognized embeddings type in ModuleCorefLinkerPropE2EHoi: %s' % self.embeddings_type)

            self.linker_coref = OptFFpairsCorefLinkerMTTBaseHoi(dim_span, self.entity_embedder,
                                                                1, config['coreflinker_prop'], span_pair_generator,
                                                                dictionaries=dictionaries,
                                                                zeros_to_clusters=self.zeros_to_clusters,
                                                                zeros_to_links=self.zeros_to_links)

            self.gate = MyGate(dim_span, init_weights_std=self.init_weights_std)
        else:
            self.entity_embedder = None

    def coref_add_scores_coreflinker(self, coref_scores, filtered_prune_scores):
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

        return coref_scores
        # coref_scores.shape --> torch.Size([1, 21, 37])

    def forward(self, all_spans, filtered_spans, sequence_lengths, linker):

        if not self.enabled or filtered_spans['span_vecs'] is None:
            return all_spans, filtered_spans, None

        cand_all_vecs = all_spans['cand_span_vecs']
        # cand_all_vecs.shape --> torch.Size([1, 335, 2324])
        # prune_indices
        # all_vecs should be of # torch.Size([1, 69, 5, 1676])

        linker_indices_hoi = filtered_spans['prune_indices_hoi']
        # linker_indices_hoi.shape --> torch.Size([1, 21])

        linker_candidates = batched_index_select(linker['candidates'], linker_indices_hoi)
        # linker['candidates'].shape --> torch.Size([1, 335, 16])
        # linker_candidates.shape --> torch.Size([1, 21, 16])

        filtered_span_begin = filtered_spans['span_begin']  # --> torch.Size([1, 21])
        filtered_span_end = filtered_spans['span_end']  # --> torch.Size([1, 21])

        # (11/10/2020) - new code that takes directly the filtered_spans
        linker_span_embeddings = filter_spans(cand_all_vecs, linker_indices_hoi.to(cand_all_vecs.device))
        # linker_span_embeddings.shape --> torch.Size([1, 21, 2324])
        # (11/10/2020) - end new code that takes directly the filtered_spans

        candidates = linker_candidates.to(settings.device)  # torch.Size([1, 9, 17])

        candidate_lengths = batched_index_select(linker['candidate_lengths'].unsqueeze(-1), linker_indices_hoi) \
            .squeeze(-1)
        # linker['candidate_lengths'].shape --> torch.Size([1, 335])
        # filtered_spans['prune_indices_hoi'].shape --> torch.Size([1, 21])
        # candidate_lengths.shape --> torch.Size([1, 21])

        max_cand_length = candidate_lengths.max().item()
        candidates = candidates[:, :, :max_cand_length]

        candidate_vecs = self.entity_embedder(candidates)  # torch.Size([1, 9, 17, 200])

        update_mentions = linker_span_embeddings  # torch.Size([1, 21, 2324])
        update_entities = candidate_vecs  # torch.Size([1, 21, 16, 200]) (before was torch.Size([1, 9, 17, 200]), correct???)

        linker_coref_scores = self.linker_coref(update_mentions, update_entities, filtered_span_begin,
                                                filtered_span_end, candidate_lengths=candidate_lengths,
                                                max_cand_length=max_cand_length).squeeze(-1)
        # linker_coref_scores.shape -->
        linker_coref_scores = self.coref_add_scores_coreflinker(linker_coref_scores,
                                                                filtered_spans['span_scores'].unsqueeze(-1))
        # linker_coref_scores.shape --> torch.Size([1, 21, 38])
        update_all = all_spans
        update_filtered = filtered_spans

        return update_all, update_filtered, linker_coref_scores

    def log_stats(self, dataset_name, predict, tb_logger, step_nr):
        # TODO: think maybe some stats useful here
        pass

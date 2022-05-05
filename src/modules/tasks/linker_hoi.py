from collections import Iterable

import torch
import torch.nn as nn
# from allennlp.nn.util import batched_index_select
from torch.nn import init

from metrics.linker import MetricLinkerImproved, MetricLinkAccuracy, MetricLinkAccuracyNoCandidates
from metrics.misc import MetricObjective
from modules.entity_embeddings import KolitsasEntityEmbeddings
from modules.misc.misc import batched_index_select
from modules.tasks.coreflinker import convert_coref, m2i_to_clusters_linkercoref
from modules.tasks.linker import create_candidate_mask, predict_links_e2e
from modules.text_field import TextFieldEmbedderTokens
from modules.utils.entity import EntityEmbbederKB
from modules.utils.misc import filter_spans, indices_to_spans


# TODO: define this function somewhere in utils, I think it is already implemented somewhere
def span_to_cluster(clusters):
    span_to_cluster = dict()
    for curr_cluster_spans in clusters:
        for curr_span in curr_cluster_spans:
            span_to_cluster[curr_span] = curr_cluster_spans

    return span_to_cluster


class LossLinkerE2EHoi(nn.Module):

    def make_linear(self, in_features, out_features, bias=True, std=0.02):
        linear = nn.Linear(in_features, out_features, bias)
        if std is not None:
            init.normal_(linear.weight, std=std)
            if bias:
                init.zeros_(linear.bias)
            return linear

    def make_ffnn(self, feat_size, hidden_size, output_size, init_weights_std):
        if hidden_size is None or hidden_size == 0 or hidden_size == [] or hidden_size == [0]:
            return self.make_linear(feat_size, output_size, std=init_weights_std)

        if not isinstance(hidden_size, Iterable):
            hidden_size = [hidden_size]
        ffnn = [self.make_linear(feat_size, hidden_size[0], std=init_weights_std), nn.ReLU(), self.dropout]
        for i in range(1, len(hidden_size)):
            ffnn += [self.make_linear(hidden_size[i - 1], hidden_size[i], std=init_weights_std), nn.ReLU(),
                     self.dropout]
        ffnn.append(self.make_linear(hidden_size[-1], output_size, std=init_weights_std))
        return nn.Sequential(*ffnn)

    def __init__(self, name, dim_span, dictionaries, config, max_span_length):
        super(LossLinkerE2EHoi, self).__init__()
        self.name = name
        self.enabled = True
        self.max_span_length = max_span_length
        self.init_weights_std = config['init_weights_std']
        self.scorers_ffnn_depth = config['scorers_ffnn_depth']

        self.embeddings_type = config['entity_embedder']['type']

        if self.embeddings_type == 'yamada-johannes':
            self.entity_embedder = TextFieldEmbedderTokens(dictionaries, config['entity_embedder'])
        elif self.embeddings_type == 'kolitsas':
            self.entity_embedder = KolitsasEntityEmbeddings(dictionaries, config['entity_embedder'])
        else:
            raise RuntimeError('Unrecognized embeddings type in LossLinkerE2EHoi: ' + self.embeddings_type)

        if 'kb_embedder' in config:
            self.kb_embedder = EntityEmbbederKB(dictionaries, config['kb_embedder'])
        else:
            self.kb_embedder = None

        hidden_dim = config['hidden_dim']
        hidden_dp = config['hidden_dropout']
        self.dropout = nn.Dropout(p=hidden_dp)

        # layers = []
        dim = dim_span + self.entity_embedder.dim
        if 'kb_embedder' in config:
            dim += self.kb_embedder.dim_output
        self.layers = self.make_ffnn(dim, [hidden_dim] * self.scorers_ffnn_depth, output_size=1,
                                     init_weights_std=self.init_weights_std)
        # for _ in range(config['layers']):
        #     layers.extend([
        #         nn.Linear(dim, hidden_dim),
        #         nn.ReLU(),
        #         nn.Dropout(hidden_dp),
        #     ])
        #     dim = hidden_dim
        # layers.append(nn.Linear(dim, 1))
        # self.layers = nn.Sequential(*layers)

        # self.loss = nn.CrossEntropyLoss(reduction='none')
        self.loss = nn.BCEWithLogitsLoss(reduction='none')
        self.labels = self.entity_embedder.dictionary.tolist()
        self.entity_dictionary = self.entity_embedder.dictionary
        self.weight = config['weight']
        self.source = config['source']  # either the 'pruned' or 'all' spans

    def forward(self, spans_all, linker, filtered_spans, gold_m2i=None, gold_spans=None, coref_pred=None,
                predict=False, only_loss=False):
        if self.source == 'all':
            nr_possible_spans = spans_all['span_mask'].sum((-1, -2), dtype=torch.int32)
            span_masked_scores = spans_all['span_mask'].view(spans_all['span_mask'].size(0), -1)
            all_linker_indices = []
            all_linker_spans = []
            # TODO (kzaporoj) - don't like this code, improve it
            for batch in range(spans_all['span_mask'].size(0)):
                linker_indices, _ = torch.sort(torch.topk(span_masked_scores[batch], nr_possible_spans[batch],
                                                          largest=True, sorted=True)[1], 0)
                all_linker_indices.append(linker_indices)
                linker_spans = indices_to_spans(linker_indices.unsqueeze(0),
                                                torch.tensor(
                                                    [span_masked_scores.size(-1) * span_masked_scores.size(-2)]),
                                                self.max_span_length)
                all_linker_spans.append(linker_spans[0])
            linker_spans = all_linker_spans
            linker_indices = torch.stack(all_linker_indices)
        else:
            # linker_indices_hoi = filtered_spans['prune_indices']
            linker_indices_hoi = filtered_spans['prune_indices_hoi']
            # linker_indices_hoi.shape --> [1, 21]
            linker_spans = filtered_spans['pruned_spans']
            # linker_spans --> <class 'list'>: [[(3, 6), (4, 6), (5, 5), (25, 25), (26, 26), (31, 34), ...]]
            # len(linker_spans) --> 21

        output_coref = dict()
        if linker_indices_hoi.size(1) > 0:
            # TODO 11/04/2021 - we are here, refactoring
            cand_all_vecs = spans_all['cand_span_vecs']  # torch.Size([1, 69, 5, 1676])
            # cand_all_vecs.shape --> torch.Size([1, 315, 2324])
            linker_candidates = batched_index_select(linker['candidates'], linker_indices_hoi)
            candidate_lengths = linker['candidate_lengths']
            candidate_lengths = batched_index_select(candidate_lengths.unsqueeze(-1), linker_indices_hoi).squeeze(-1)

            # filter_spans()
            # targets = linker['targets_all_spans']
            targets = linker['targets']
            targets = batched_index_select(targets, linker_indices_hoi)

            # linker_vecs = filter_spans(all_vecs, linker_indices_hoi.to(cand_all_vecs.device))  # torch.Size([1, 9, 1676])
            linker_vecs = filter_spans(cand_all_vecs,
                                       linker_indices_hoi.to(cand_all_vecs.device))  # torch.Size([1, 9, 1676])

            candidates = linker_candidates.to(cand_all_vecs.device)  # torch.Size([1, 9, 17])
            candidate_vecs = self.entity_embedder(candidates)  # torch.Size([1, 9, 17, 200])
            if self.kb_embedder is not None:
                kb_vecs = self.kb_embedder(candidates)
                candidate_vecs = torch.cat((candidate_vecs, kb_vecs), -1)

            dims = (linker_vecs.size(0), linker_vecs.size(1), candidate_vecs.size(2),
                    linker_vecs.size(2))  # <class 'tuple'>: (1, 9, 17, 1676)
            linker_vecs = linker_vecs.unsqueeze(-2).expand(dims)  # torch.Size([1, 9, 17, 1676])
            vecs = torch.cat((linker_vecs, candidate_vecs), -1)  # torch.Size([1, 9, 17, 1876])

            scores = self.layers(vecs).squeeze(-1)  # torch.Size([1, 9, 17])
            scores_mask = create_candidate_mask(scores, candidate_lengths).float().to(
                cand_all_vecs.device)  # torch.Size([1, 9, 17])

            obj = self.loss(scores, targets)  # torch.Size([1, 9, 17]) -> targets shape
            obj = (scores_mask * obj).sum() * self.weight

            if predict and not only_loss:
                _, predictions = (scores - (1.0 - scores_mask) * 1e23).max(dim=-1)

                if coref_pred is not None:
                    pred, s = predict_links_e2e(predictions, linker_spans=linker_spans,
                                                linker_candidates=linker_candidates,
                                                candidate_lengths=candidate_lengths,
                                                labels=self.labels, scores=scores, coref_pred=coref_pred['pred'])
                else:
                    pred, s = predict_links_e2e(predictions, linker_spans=linker_spans,
                                                linker_candidates=linker_candidates,
                                                candidate_lengths=candidate_lengths,
                                                labels=self.labels, scores=scores, coref_pred=None)

                # <class 'list'>: [[(5, 7, 'NILL'), (41, 43, 'NILL'), (45, 47, 'NILL'), (49, 49, 'Berlin'),
                # (51, 52, 'NILL'), (53, 54, 'NILL'), (57, 57, 'Berlin')]]
                pred_clusters = list()
                for curr_links_doc in pred:
                    curr_doc_link_to_span = dict()
                    for curr_link_doc in curr_links_doc:
                        link = curr_link_doc[2]
                        span_begin = curr_link_doc[0]
                        span_end = curr_link_doc[1]
                        if link != 'NILL':
                            if link not in curr_doc_link_to_span:
                                curr_doc_link_to_span[link] = list()
                            curr_doc_link_to_span[link].append((span_begin, span_end))
                    list_clusters = list(curr_doc_link_to_span.values())
                    list_clusters = [sorted(x, key=lambda xl: xl[0]) for x in list_clusters]
                    pred_clusters.append(list_clusters)
                # pred_clusters = []

                if coref_pred is not None:
                    # merges the clusters of LINK and the ones predicted by coref
                    #   rule 1: only the ones that are not already in cluster spans
                    #   rule 2: new cluster if no current clusters with the same link
                    #       - if current cluster with the same link, append it to the cluster with most links/total size of cluster
                    for idx_batch, curr_link_clusters in enumerate(pred_clusters):
                        # print('pred link clusters: ', curr_link_clusters)
                        pred_coref_clusters = coref_pred['pred'][idx_batch]
                        curr_doc_links = pred[idx_batch]
                        link_span_to_cluster = span_to_cluster(curr_link_clusters)
                        spans_in_coref = set([item for sublist in pred_coref_clusters for item in sublist])
                        spans_in_linker = set([item for sublist in curr_link_clusters for item in sublist])
                        spans_not_covered_in_coref = spans_in_linker.difference(spans_in_coref)
                        added_spans = set()
                        for curr_span_not_covered in spans_not_covered_in_coref:
                            if curr_span_not_covered in added_spans:
                                continue
                            linker_span_cluster = set(link_span_to_cluster[curr_span_not_covered])
                            max_intersection_cluster_size = 0
                            max_intersection_cluster_cluster = None
                            for curr_pred_coref_cluster in pred_coref_clusters:
                                intersection = linker_span_cluster.intersection(set(curr_pred_coref_cluster))

                                if len(intersection) > max_intersection_cluster_size:
                                    max_intersection_cluster_size = len(intersection)
                                    max_intersection_cluster_cluster = curr_pred_coref_cluster
                            if max_intersection_cluster_size > 0:
                                # appends to the cluster with highest intersection
                                max_intersection_cluster_cluster.append(curr_span_not_covered)
                                added_spans.add(curr_span_not_covered)
                                # TODO: also for scores matrix and other matrices! <class 'dict'>: {'loss': 0.0, 'pred': [[]], 'pred_pointers': [[]], 'gold': [[]], 'scores': [[]]}
                            else:
                                # creates a new cluster
                                pred_coref_clusters.append(link_span_to_cluster[curr_span_not_covered])
                                for added_span in link_span_to_cluster[curr_span_not_covered]:
                                    added_spans.add(added_span)
                                # TODO: also for scores matrix and other matrices! <class 'dict'>: {'loss': 0.0, 'pred': [[]], 'pred_pointers': [[]], 'gold': [[]], 'scores': [[]]}

                        # s0- span to cluster dict for curr_link_clusters
                        # s1- subtraction flattened curr_link_clusters from pred_coref_clusters
                        # s2- assignment of the linking span to the predicted cluster whose spans have highest intersection with the
                        #   spans of the linking cluster. If no intersection detected, or if the span is singleton, then new
                        #   cluster is formed and added.
                        # print('pred coref clusters: ', curr_link_clusters)
                        # pass
                # print('curr doc link to span: ', pred_clusters)
                if coref_pred is None:
                    output_coref['loss'] = obj
                    output_coref['pred'] = pred_clusters
                    output_coref['pred_pointers'] = [None for x in linker['gold']]
                    output_coref['gold'] = [convert_coref(m2i_to_clusters_linkercoref(x.tolist()), y,
                                                          number_candidates=0,
                                                          links_dictionary=self.entity_dictionary)[0] for x, y in
                                            zip(gold_m2i, gold_spans)]

                    output_coref['scores'] = [None for _ in linker['gold']]
                else:
                    # (28/04/2021) - !this is needed because seems like coref_pred is not passed by reference! -->
                    # ups, I think I am wrong here, they are passed by reference (double check!) --> if so no need for
                    # the assignments below
                    output_coref['loss'] = coref_pred['loss']
                    output_coref['pred'] = coref_pred['pred']
                    output_coref['pred_pointers'] = coref_pred['pred_pointers']
                    output_coref['gold'] = coref_pred['gold']

                    output_coref['scores'] = coref_pred['scores']

            else:
                pred = [[] for _ in linker['gold']]
                s = [None for _ in linker['gold']]
                output_coref['loss'] = obj
                output_coref['pred'] = [None for _ in linker['gold']]
                output_coref['pred_pointers'] = [None for _ in linker['gold']]
                output_coref['gold'] = [None for _ in linker['gold']]
                output_coref['scores'] = [None for _ in linker['gold']]

            # s = predict_scores(scores, linker_spans, candidates, candidate_lengths, self.labels)
        else:
            obj = 0
            pred = [[] for _ in linker['gold']]
            s = [None for _ in linker['gold']]
            #
            output_coref['loss'] = 0.0

            # TODO - check this, do we need 'pred_pointers'???!
            output_coref['pred'] = [None for _ in linker['gold']]
            output_coref['pred_pointers'] = [None for _ in linker['gold']]
            output_coref['gold'] = [None for _ in linker['gold']]
            output_coref['scores'] = [None for _ in linker['gold']]

        output = {
            'loss': obj,
            'pred': pred,
            'gold': linker['gold'],
            'scores': s
        }

        return obj, output, output_coref

    def create_metrics(self):
        return [MetricLinkerImproved(self.name), MetricLinkerImproved(self.name, 'links'),
                MetricLinkerImproved(self.name, 'nills'),
                MetricLinkAccuracy(self.name),
                MetricLinkAccuracyNoCandidates(self.name),
                MetricObjective(self.name)]

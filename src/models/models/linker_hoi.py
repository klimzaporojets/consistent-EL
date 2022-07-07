from collections import Iterable

import torch
import torch.nn as nn
from torch.nn import init

from misc import settings
from metrics.linker import MetricLinkerImproved, MetricLinkAccuracy, MetricLinkAccuracyNoCandidates
from metrics.misc import MetricObjective
from models.misc.misc import batched_index_select
from models.utils.misc import indices_to_spans, filter_spans
from models.misc.entity_embeddings import KolitsasEntityEmbeddings
from models.models.coreflinker_loss import convert_coref, m2i_to_clusters_linkercoref
from models.misc.text_field import TextFieldEmbedderTokens
from models.utils.entity import EntityEmbbederKB


def span_to_cluster(clusters):
    span_to_cluster = dict()
    for curr_cluster_spans in clusters:
        for curr_span in curr_cluster_spans:
            span_to_cluster[curr_span] = curr_cluster_spans

    return span_to_cluster


def create_candidate_mask(scores, candidate_lengths):
    tmp = torch.arange(scores.size(-1), device=settings.device)
    tmp = tmp.unsqueeze(0).unsqueeze(0)
    candidate_lengths = candidate_lengths.unsqueeze(-1)
    mask = tmp < candidate_lengths
    return mask


def predict_links_e2e(predictions, linker_spans, linker_candidates, candidate_lengths, labels, scores,
                      coref_pred=None):
    output = [list() for _ in range(predictions.size(0))]
    output_scores = [list() for _ in range(predictions.size(0))]

    predictions = predictions.cpu()
    for batch_idx, span_idx in torch.nonzero(candidate_lengths).tolist():
        begin, end = linker_spans[batch_idx][span_idx]

        candidate = linker_candidates[batch_idx, span_idx, predictions[batch_idx, span_idx].item()].item()

        if coref_pred is not None:
            coref_spans = set([item for sublist in coref_pred[batch_idx] for item in sublist])
            if labels[candidate] == 'NILL':
                if linker_spans[batch_idx][span_idx] not in coref_spans:
                    continue

        if labels[candidate] != 'NONE':
            output[batch_idx].append((begin, end, labels[candidate]))
            num_candidates = candidate_lengths[batch_idx, span_idx].item()
            span = linker_spans[batch_idx][span_idx]
            c_ = [labels[c] for c in linker_candidates[batch_idx, span_idx, :num_candidates].tolist()]
            s_ = scores[batch_idx, span_idx, :num_candidates].tolist()
            output_scores[batch_idx].append((span, c_, s_))
    return output, output_scores


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

        dim = dim_span + self.entity_embedder.dim
        if 'kb_embedder' in config:
            dim += self.kb_embedder.dim_output
        self.layers = self.make_ffnn(dim, [hidden_dim] * self.scorers_ffnn_depth, output_size=1,
                                     init_weights_std=self.init_weights_std)

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
        else:
            linker_indices_hoi = filtered_spans['prune_indices_hoi']
            # linker_indices_hoi.shape --> [1, 21]
            linker_spans = filtered_spans['pruned_spans']
            # linker_spans --> <class 'list'>: [[(3, 6), (4, 6), (5, 5), (25, 25), (26, 26), (31, 34), ...]]
            # len(linker_spans) --> 21

        output_coref = dict()
        if linker_indices_hoi.size(1) > 0:
            cand_all_vecs = spans_all['cand_span_vecs']  # torch.Size([1, 69, 5, 1676])
            # cand_all_vecs.shape --> torch.Size([1, 315, 2324])
            linker_candidates = batched_index_select(linker['candidates'], linker_indices_hoi)
            candidate_lengths = linker['candidate_lengths']
            candidate_lengths = batched_index_select(candidate_lengths.unsqueeze(-1), linker_indices_hoi).squeeze(-1)

            targets = linker['targets']
            targets = batched_index_select(targets, linker_indices_hoi)

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

                if coref_pred is not None:
                    # merges the clusters of LINK and the ones predicted by coref
                    #   rule 1: only the ones that are not already in cluster spans
                    #   rule 2: new cluster if no current clusters with the same link
                    #       - if current cluster with the same link, append it to the cluster with most links/total size of cluster
                    for idx_batch, curr_link_clusters in enumerate(pred_clusters):
                        pred_coref_clusters = coref_pred['pred'][idx_batch]
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
                            else:
                                # creates a new cluster
                                pred_coref_clusters.append(link_span_to_cluster[curr_span_not_covered])
                                for added_span in link_span_to_cluster[curr_span_not_covered]:
                                    added_spans.add(added_span)

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

        else:
            obj = 0
            pred = [[] for _ in linker['gold']]
            s = [None for _ in linker['gold']]
            output_coref['loss'] = 0.0

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

import torch
import torch.nn as nn
# from allennlp.nn.util import batched_index_select

import settings
from metrics.linker import MetricLinkerImproved, MetricLinkAccuracy, MetricLinkAccuracyNoCandidates
from metrics.misc import MetricObjective
from modules.misc.misc import batched_index_select
from modules.tasks.coreflinker import convert_coref, m2i_to_clusters_linkercoref
from modules.text_field import TextFieldEmbedderTokens
from modules.utils.entity import EntityEmbbederKB
from modules.utils.misc import filter_spans, spans_to_indices, indices_to_spans, predict_scores


def collate_spans(instances):
    max_spans = max([len(x) for x in instances])
    if settings.device == 'cuda':
        output = torch.cuda.LongTensor(len(instances), max_spans, 2)
    else:
        output = torch.LongTensor(len(instances), max_spans, 2)
    output[:, :, :] = 0
    for b, instance in enumerate(instances):
        num_spans = len(instance)
        if num_spans > 0:
            if settings.device == 'cuda':
                output[b, :num_spans, :] = torch.cuda.LongTensor(instance)
            else:
                output[b, :num_spans, :] = torch.LongTensor(instance)

    return output


def create_candidate_mask(scores, candidate_lengths):
    tmp = torch.arange(scores.size(-1), device=settings.device)
    tmp = tmp.unsqueeze(0).unsqueeze(0)
    candidate_lengths = candidate_lengths.unsqueeze(-1)
    mask = tmp < candidate_lengths
    return mask


def collate_candidates(instances):
    """

    :param instances: looks like this:
    <class 'list'>: [[[], [188646], [188646], [188646, 89029, 16360, 31792], [8254, 188646, 8253], [], [188646], [188646], [188646, 8254, 8253]]]
    :return:
    """
    max_spans = max([len(spans) for spans in instances])
    max_candidates = max(
        [max([len(x) for x in spans] + [1]) for spans in instances])  # if there are no candidates, return 1
    if settings.device == 'cuda':
        output = torch.cuda.LongTensor(len(instances), max_spans, max_candidates)
        lengths = torch.cuda.LongTensor(len(instances), max_spans)
    else:
        output = torch.LongTensor(len(instances), max_spans, max_candidates)  # dim: torch.Size([1, 9, 4])
        lengths = torch.LongTensor(len(instances), max_spans)  # dim: torch.Size([1, 9])

    output[:, :, :] = 0
    lengths[:, :] = 0
    for b, instance in enumerate(instances):
        num_spans = len(instance)
        for s, candidates in enumerate(instance):
            if settings.device == 'cuda':
                output[b, s, :len(candidates)] = torch.cuda.LongTensor(candidates)
            else:
                output[b, s, :len(candidates)] = torch.LongTensor(candidates)

        if settings.device == 'cuda':
            lengths[b, :num_spans] = torch.cuda.LongTensor([len(candidates) for candidates in instance])
        else:
            lengths[b, :num_spans] = torch.LongTensor([len(candidates) for candidates in instance])

    return output, lengths


def collate_candidates_in_pytorch(instances, unknown_id):
    """
    Same as  collate_candidates function, but when the candidates come in pytorch vectors already
    :param instances: looks like this:
        <class 'list'>: [[tensor([46390, 46388, 46386, 28356, 46389, 39176, 46385, 46387, 44380], dtype=torch.int32),
            tensor([], dtype=torch.int32), tensor([], dtype=torch.int32),
            tensor([], dtype=torch.int32), tensor([], dtype=torch.int32),
            tensor([], dtype=torch.int32), tensor([], dtype=torch.int32),
            tensor([], dtype=torch.int32), tensor([], dtype=torch.int32),
            tensor([], dtype=torch.int32),
            tensor([30704,  5673, 22142, 23304, 30706, 30705, 30702, 16532, 30701, 30703], dtype=torch.int32),
            tensor([30706, 30701, 16532, 23304,  5673, 30704, 30703, 22142, 30702, 30705], dtype=torch.int32)]....]
    :return:
    """
    max_spans = max([len(spans) for spans in instances])
    max_candidates = max(
        [max([x.shape[-1] for x in spans] + [1]) for spans in instances])  # if there are no candidates, return 1
    # [max([len(x) for x in spans] + [1]) for spans in instances])  # if there are no candidates, return 1
    if settings.device == 'cuda':
        output = torch.cuda.LongTensor(len(instances), max_spans, max_candidates)
        lengths = torch.cuda.LongTensor(len(instances), max_spans)
    else:
        output = torch.LongTensor(len(instances), max_spans, max_candidates)  # dim: torch.Size([1, 9, 4])
        lengths = torch.LongTensor(len(instances), max_spans)  # dim: torch.Size([1, 9])

    output[:, :, :] = unknown_id
    lengths[:, :] = 0
    for b, instance in enumerate(instances):
        num_spans = len(instance)
        for s, candidates in enumerate(instance):
            # if settings.device == 'cuda':
            output[b, s, :candidates.shape[-1]] = candidates.to(device=settings.device)
            # else:
            #     output[b, s, :candidates.shape[-1]] = candidates
            # output[b, s, :len(candidates)] = torch.LongTensor(candidates)

        if settings.device == 'cuda':
            lengths[b, :num_spans] = torch.cuda.LongTensor([candidates.shape[-1] for candidates in instance])
        else:
            lengths[b, :num_spans] = torch.LongTensor([candidates.shape[-1] for candidates in instance])

    return output, lengths


def collate_candidates_doc_level(instances, nr_instances):
    # max_spans = max([len(spans) for spans in instances])
    max_spans = max(nr_instances)
    max_candidates = max([len(candidates) for candidates in instances])  # if there are no candidates, return 1
    if settings.device == 'cuda':
        output = torch.cuda.LongTensor(len(instances), max_spans, max_candidates)
        lengths = torch.cuda.LongTensor(len(instances), max_spans)
    else:
        output = torch.LongTensor(len(instances), max_spans, max_candidates)
        lengths = torch.LongTensor(len(instances), max_spans)

    output[:, :, :] = 0
    lengths[:, :] = 0
    for b, (doc_candidates, nr_mentions) in enumerate(zip(instances, nr_instances)):
        for s in range(max_spans):
            if settings.device == 'cuda':
                output[b, s, :len(doc_candidates)] = torch.cuda.LongTensor(doc_candidates)
            else:
                output[b, s, :len(doc_candidates)] = torch.LongTensor(doc_candidates)

        if settings.device == 'cuda':
            lengths[b, :nr_mentions] = torch.cuda.LongTensor([len(doc_candidates) for _ in range(nr_mentions)])
        else:
            lengths[b, :nr_mentions] = torch.LongTensor([len(doc_candidates) for _ in range(nr_mentions)])

    return output, lengths


def collate_targets(instances, max_candidates):
    max_spans = max([len(x) for x in instances])

    output = torch.zeros(len(instances), max_spans, max_candidates, device=settings.device)
    for b, instance in enumerate(instances):
        for s, target in enumerate(instance):
            if target != -1:  # (kzaporoj) - in case targets not nill
                output[b, s, target] = 1.0
    return output


def collate_tot_cand_lengths(instances):
    max_mentions_len = max([x.size(-1) for x in instances])

    if max_mentions_len == 0:
        return None

    output = torch.zeros(len(instances), max_mentions_len, device=settings.device, dtype=torch.int32)

    for b, instance in enumerate(instances):
        output[b, :instance.size(-1)] = instance
        # for s, target in enumerate(instance):
        #     if target != -1:  # (kzaporoj) - in case targets not nill
        #         output[b, s, target] = 1.0
    return output


class LinkerNone(nn.Module):

    def __init__(self):
        super(LinkerNone, self).__init__()
        self.enabled = False

    def forward(self, spans_all, linker, predict=True):
        # num_batch = spans_all['span_vecs'].size(0)
        # only one batch
        num_batch = 1
        output = {
            'loss': 0.0,
            'pred': [None for _ in range(num_batch)],
            'gold': [None for _ in range(num_batch)],
            'scores': [None for _ in range(num_batch)]
        }

        return 0.0, output

    def create_metrics(self):
        return []


# def predict_scores(scores, linker_spans, linker_candidates, candidate_lengths, labels):
#     output = [list() for _ in linker_spans]
#     for batch_idx, span_idx in torch.nonzero(candidate_lengths).tolist():
#         num_candidates = candidate_lengths[batch_idx, span_idx].item()
#         span = linker_spans[batch_idx][span_idx]
#         c_ = [labels[c] for c in linker_candidates[batch_idx, span_idx, :num_candidates].tolist()]
#         s_ = scores[batch_idx, span_idx, :num_candidates].tolist()
#         output[batch_idx].append((span, c_, s_))
#     return output


def predict_links(predictions, linker_spans, linker_candidates, candidate_lengths, labels):
    # linker_spans = linker['spans']
    # linker_candidates = linker['candidates']
    # candidate_lengths = linker['candidate_lengths']

    output = [list() for _ in range(predictions.size(0))]

    predictions = predictions.cpu()
    for batch_idx, span_idx in torch.nonzero(candidate_lengths).tolist():
        # begin, end = linker_spans[batch_idx][span_idx]
        begin, end = linker_spans[batch_idx][span_idx]
        candidate = linker_candidates[batch_idx, span_idx, predictions[batch_idx, span_idx].item()].item()
        output[batch_idx].append((begin, end, labels[candidate]))

    return output


def predict_links_e2e(predictions, linker_spans, linker_candidates, candidate_lengths, labels, scores,
                      coref_pred=None):
    # output = [list() for _ in linker_spans]
    # for batch_idx, span_idx in torch.nonzero(candidate_lengths).tolist():
    #     num_candidates = candidate_lengths[batch_idx, span_idx].item()
    #     span = linker_spans[batch_idx][span_idx]
    #     c_ = [labels[c] for c in linker_candidates[batch_idx, span_idx, :num_candidates].tolist()]
    #     s_ = scores[batch_idx, span_idx, :num_candidates].tolist()
    #     output[batch_idx].append((span, c_, s_))
    # return output

    output = [list() for _ in range(predictions.size(0))]
    output_scores = [list() for _ in range(predictions.size(0))]

    predictions = predictions.cpu()
    for batch_idx, span_idx in torch.nonzero(candidate_lengths).tolist():
        begin, end = linker_spans[batch_idx][span_idx]

        candidate = linker_candidates[batch_idx, span_idx, predictions[batch_idx, span_idx].item()].item()
        # the NONE are not predicted
        # if labels[candidate] != 'NONE':
        # # TODO: if coreference was predicted, then we are guiding ourselves by coreference if NILL , so if it is
        # NILL and wasn't predicted by coreference, it is ignored.
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


# TODO: make softmax and binary version
class LossLinker(nn.Module):

    def __init__(self, name, dim_span, dictionaries, config):
        super(LossLinker, self).__init__()
        self.name = name
        self.enabled = True

        self.entity_embedder = TextFieldEmbedderTokens(dictionaries, config['entity_embedder'])

        if 'kb_embedder' in config:
            self.kb_embedder = EntityEmbbederKB(dictionaries, config['kb_embedder'])
        else:
            self.kb_embedder = None

        hidden_dim = config['hidden_dim']
        hidden_dp = config['hidden_dropout']

        layers = []
        dim = dim_span + self.entity_embedder.dim
        if 'kb_embedder' in config:
            dim += self.kb_embedder.dim_output
        for _ in range(config['layers']):
            layers.extend([
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(hidden_dp),
            ])
            dim = hidden_dim
        layers.append(
            nn.Linear(dim, 1)
        )
        self.layers = nn.Sequential(*layers)

        # self.loss = nn.CrossEntropyLoss(reduction='none')
        self.loss = nn.BCEWithLogitsLoss(reduction='none')
        self.labels = self.entity_embedder.dictionary.tolist()
        self.weight = config['weight']
        self.source = config['source']  # either the 'pruned' or 'all' spans

    def forward(self, spans_all, linker, filtered_spans):
        # linker_spans = linker['spans_tensors']
        linker_span_tensors = torch.cat([filtered_spans['span_begin'], filtered_spans['span_end']], dim=-1)

        if linker_span_tensors.size(1) > 0:
            all_vecs = spans_all['span_vecs']  # torch.Size([1, 69, 5, 1676])
            linker_candidates = linker['candidates']  # torch.Size([1, 9, 17])
            candidate_lengths = linker['candidate_lengths']  # torch.Size([1, 9])
            targets = linker['targets'].to(all_vecs.device)  # torch.Size([1, 9, 17])

            pred_spans_idx = filtered_spans['reindex_wrt_gold']
            linker_candidates = batched_index_select(linker_candidates, pred_spans_idx)
            targets = batched_index_select(targets, pred_spans_idx)
            candidate_lengths = batched_index_select(candidate_lengths.unsqueeze(-1), pred_spans_idx).squeeze(-1)

            max_span_length = all_vecs.size(2)  # 5

            # Only process spans with linker information
            linker_indices = spans_to_indices(linker_span_tensors,
                                              max_span_length)  # tensor([[ 16,  27, 207, 227, 245, 250, 256, 266, 285]])
            linker_vecs = filter_spans(all_vecs, linker_indices.to(all_vecs.device))  # torch.Size([1, 9, 1676])

            candidates = linker_candidates.to(all_vecs.device)  # torch.Size([1, 9, 17])
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
                all_vecs.device)  # torch.Size([1, 9, 17])
            # scores_mask content:
            # tensor([[[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            #          [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            #          [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            #          [1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            #          [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            #          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            #          [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            #          [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            #          [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]]])

            # print('candidates:', linker_candidates.size())
            # print('scores:', scores.size())
            # print('targets:', targets.size())

            obj = self.loss(scores, targets)  # torch.Size([1, 9, 17]) -> targets shape
            obj = (scores_mask * obj).sum() * self.weight

            # scores = targets
            _, predictions = (scores - (1.0 - scores_mask) * 1e23).max(dim=-1)
            linker_spans = filtered_spans['spans']
            pred = predict_links(predictions, linker_spans, linker_candidates, candidate_lengths, self.labels)
            s = predict_scores(scores, linker_spans, linker_candidates, candidate_lengths, self.labels)
            # pred = predict_links(predictions, linker_span_tensors, linker_candidates, candidate_lengths, self.labels)
            # s = predict_scores(scores, linker_span_tensors, linker_candidates, candidate_lengths, self.labels)
        else:
            obj = 0
            pred = [[] for _ in linker['gold']]
            s = [None for _ in linker['gold']]

        output = {
            'loss': obj,
            'pred': pred,
            'gold': linker['gold'],
            'scores': s
        }

        return obj, output

    def create_metrics(self):
        return [MetricLinkerImproved(self.name), MetricLinkerImproved(self.name, 'links'),
                MetricLinkerImproved(self.name, 'nills'),
                MetricLinkAccuracy(self.name),
                MetricLinkAccuracyNoCandidates(self.name),
                MetricObjective(self.name)]


class LossLinkerE2E(nn.Module):

    def __init__(self, name, dim_span, dictionaries, config, max_span_length):
        super(LossLinkerE2E, self).__init__()
        self.name = name
        self.enabled = True
        self.max_span_length = max_span_length

        self.entity_embedder = TextFieldEmbedderTokens(dictionaries, config['entity_embedder'])

        if 'kb_embedder' in config:
            self.kb_embedder = EntityEmbbederKB(dictionaries, config['kb_embedder'])
        else:
            self.kb_embedder = None

        hidden_dim = config['hidden_dim']
        hidden_dp = config['hidden_dropout']

        layers = []
        dim = dim_span + self.entity_embedder.dim
        if 'kb_embedder' in config:
            dim += self.kb_embedder.dim_output
        for _ in range(config['layers']):
            layers.extend([
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(hidden_dp),
            ])
            dim = hidden_dim
        layers.append(
            nn.Linear(dim, 1)
        )
        self.layers = nn.Sequential(*layers)

        # self.loss = nn.CrossEntropyLoss(reduction='none')
        self.loss = nn.BCEWithLogitsLoss(reduction='none')
        self.labels = self.entity_embedder.dictionary.tolist()
        self.entity_dictionary = self.entity_embedder.dictionary
        self.weight = config['weight']
        self.source = config['source']  # either the 'pruned' or 'all' spans

    def forward(self, spans_all, linker, filtered_spans, gold_m2i=None, gold_spans=None):
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
            linker_indices = filtered_spans['prune_indices']
            # linker_indices.shape --> [1, 21]
            # linker_indices --> tensor([[  60,   94,  199,  242,  334,  349,  362,  392,  527,  588,  721,  800,
            #           902,  907,  930,  938, 1013, 1066, 1113, 1144, 1240]])
            linker_spans = filtered_spans['spans']
            # <class 'list'>: [[(4, 4), (6, 10), (13, 17), (16, 18), (22, 26), (23, 27), (24, 26), (26, 28), ... ]]
            # len(linker_spans[0]) --> 21

        output_coref = dict()
        if linker_indices.size(1) > 0:
            all_vecs = spans_all['span_vecs']  # torch.Size([1, 69, 5, 1676])

            linker_candidates = batched_index_select(linker['candidates'], linker_indices)
            candidate_lengths = linker['candidate_lengths']
            candidate_lengths = batched_index_select(candidate_lengths.unsqueeze(-1), linker_indices).squeeze(-1)

            # filter_spans()
            # targets = linker['targets_all_spans']
            targets = linker['targets']
            targets = batched_index_select(targets, linker_indices)

            linker_vecs = filter_spans(all_vecs, linker_indices.to(all_vecs.device))  # torch.Size([1, 9, 1676])

            candidates = linker_candidates.to(all_vecs.device)  # torch.Size([1, 9, 17])
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
                all_vecs.device)  # torch.Size([1, 9, 17])

            obj = self.loss(scores, targets)  # torch.Size([1, 9, 17]) -> targets shape
            obj = (scores_mask * obj).sum() * self.weight

            _, predictions = (scores - (1.0 - scores_mask) * 1e23).max(dim=-1)
            pred, s = predict_links_e2e(predictions, linker_spans=linker_spans,
                                        linker_candidates=linker_candidates, candidate_lengths=candidate_lengths,
                                        labels=self.labels, scores=scores)

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

            # print('curr doc link to span: ', pred_clusters)
            output_coref['loss'] = obj
            output_coref['pred'] = pred_clusters
            output_coref['pred_pointers'] = [None for x in linker['gold']]
            output_coref['gold'] = [convert_coref(m2i_to_clusters_linkercoref(x.tolist()), y,
                                                  number_candidates=0,
                                                  links_dictionary=self.entity_dictionary)[0] for x, y in
                                    zip(gold_m2i, gold_spans)]

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

import torch
import torch.nn as nn

from metrics.coref import MetricCoref, MetricCorefAverage
from metrics.coref import decode_m2i, m2i_to_clusters
from metrics.coref import set_backup
from metrics.corefx import MetricCorefExternal
from metrics.misc import MetricObjective
from modules.graph import create_graph
from modules.tasks.coref_edmonds import LossCorefEdmonds
from modules.tasks.coreflinker import predict_scores_coref, remove_disabled_scores_coref
from util.math import logsumexp
from util.sequence import get_mask_from_sequence_lengths


def create_coref_target_forward(pred_spans, gold_spans, gold_clusters):
    num_batch = len(pred_spans)
    max_spans = max([len(x) for x in pred_spans])

    targets = torch.zeros(num_batch, max_spans, max_spans)

    for batch, (pred, gold, clusters) in enumerate(zip(pred_spans, gold_spans, gold_clusters)):
        gold2cluster = {}
        for idx, span in enumerate(gold):
            gold2cluster[span] = clusters[idx].item()

        for idx1, span1 in enumerate(pred):
            num_found = 0
            if span1 in gold2cluster:
                for idx2, span2 in enumerate(pred):
                    if idx2 < idx1 and span2 in gold2cluster and gold2cluster[span1] == gold2cluster[span2]:
                        targets[batch, idx1, idx2] = 1.0
                        num_found += 1

            if num_found == 0:
                targets[batch, idx1, idx1] = 1.0

    return targets


def create_coref_target_backward(pred_spans, gold_spans, gold_clusters):
    num_batch = len(pred_spans)
    max_spans = max([len(x) for x in pred_spans])

    targets = torch.zeros(num_batch, max_spans, max_spans)

    for batch, (pred, gold, clusters) in enumerate(zip(pred_spans, gold_spans, gold_clusters)):
        gold2cluster = {}
        for idx, span in enumerate(gold):
            gold2cluster[span] = clusters[idx].item()

        for idx1, span1 in enumerate(pred):
            num_found = 0
            if span1 in gold2cluster:
                for idx2, span2 in enumerate(pred):
                    if idx2 > idx1 and span2 in gold2cluster and gold2cluster[span1] == gold2cluster[span2]:
                        targets[batch, idx1, idx2] = 1.0
                        num_found += 1

            if num_found == 0:
                targets[batch, idx1, idx1] = 1.0

    return targets


def create_coref(config):
    if config['type'] == 'coref-a':
        return LossCoref('coref', config)
    elif config['type'] == 'coref-e':
        return LossCorefEdmonds('coref', config)


class TaskCoref(nn.Module):

    def __init__(self, task, dim_input, config):
        super(TaskCoref, self).__init__()
        self.task = task
        self.dim_hidden = config['dim']
        self.left = nn.Linear(dim_input, self.dim_hidden, bias=False)
        self.right = nn.Linear(dim_input, self.dim_hidden, bias=False)
        self.weight = config.get('weight', 1.0)
        self.enabled = config['enabled']

    def forward(self, inputs, targets, lengths):
        l = self.left(inputs)
        r = self.right(inputs).permute(0, 2, 1)
        scores = torch.matmul(l, r)

        set_backup(scores)

        # print('->', scores.size(), scores.min().item(), scores.max().item())

        triangular_mask = torch.ones(scores.size()[1:]).tril(0).unsqueeze(0)
        constant = scores.max().item() + 100000
        additive_mask = (1 - triangular_mask) * -constant
        logits = torch.nn.functional.log_softmax(scores + additive_mask.to(scores.device), dim=-1)

        if targets is not None:
            loss = - logsumexp(logits + (1 - targets) * -100000)
            mask = get_mask_from_sequence_lengths(lengths, lengths.max().item())
            m2is = None
            output = self.weight * (mask.float() * loss).sum()
        else:
            output = None
            m2is = decode_m2i(logits, lengths)

        return output, logits, m2is

    def create_metrics(self):
        return [MetricCoref(self.task, 'muc', MetricCoref.muc), MetricCoref(self.task, 'bcubed', MetricCoref.b_cubed),
                MetricObjective(self.task)] if self.enabled else []


class TaskCoref2(nn.Module):

    def __init__(self, task, dim_input, config):
        super(TaskCoref2, self).__init__()
        self.task = task
        self.module = create_graph(dim_input, 1, config['scorer'])
        self.weight = config.get('weight', 1.0)
        self.enabled = config['enabled']
        print("Task {}: enabled={} weight={}".format(self.task, self.enabled, self.weight))

    def set_weight(self, W0):
        # self.weight = W0
        self.weight = 0.5
        print("Task {} weight: {}".format(self.task, self.weight))

    def forward(self, inputs, targets, lengths):
        triangular_mask = torch.ones(inputs.size()[1], inputs.size()[1]).tril(0).unsqueeze(0).to(inputs.device)

        scores = self.module(inputs, triangular_mask, lengths=lengths).squeeze(-1)

        set_backup(scores)

        # print('->', scores.size(), scores.min().item(), scores.max().item())

        constant = scores.max().item() + 100000
        additive_mask = (1 - triangular_mask) * -constant
        logits = torch.nn.functional.log_softmax(scores + additive_mask.to(scores.device), dim=-1)

        if targets is not None:
            loss = - logsumexp(logits + (1 - targets) * -100000)
            mask = get_mask_from_sequence_lengths(lengths, lengths.max().item())
            m2is = None
            output = self.weight * (mask.float() * loss).sum()
        else:
            output = None
            m2is = decode_m2i(logits, lengths)

        return output, logits, m2is

    def create_metrics(self):
        return [MetricCoref(self.task, 'muc', MetricCoref.muc), MetricCoref(self.task, 'bcubed', MetricCoref.b_cubed),
                MetricObjective(self.task)] if self.enabled else []


def convert(clusters, spans):
    out = [[spans[m] for m in cluster] for cluster in clusters]
    return out


# remove singletons containing only a disabled span
def remove_disabled_spans(clusters, enabled_spans):
    out = []
    for cs, spans in zip(clusters, enabled_spans):
        enabled = set(spans)
        out.append([cluster for cluster in cs if len(cluster) > 1 or cluster[0] in enabled])
    return out


# simplified
class LossCoref(nn.Module):

    def __init__(self, task, config):
        super(LossCoref, self).__init__()
        self.task = task
        self.weight = config.get('weight', 1.0)
        self.enabled = config['enabled']
        self.filter_singletons_with_pruner = config['filter_singletons_with_pruner']
        self.filter_singletons_with_ner = config['filter_singletons_with_ner']
        self.singletons = self.filter_singletons_with_pruner or self.filter_singletons_with_ner  # write out singletons to json
        if 'singletons' in config:
            self.singletons = config['singletons']

        # self.end_to_end = config['end_to_end']

    def forward(self, scores, gold_m2i, pred_spans, gold_spans, predict=False, pruner_spans=None, span_lengths=None,
                ner_spans=None):
        output = {}

        if self.enabled and scores is not None:
            # TODO!!! tensorize this function!! take ideas from https://github.com/lxucs/coref-hoi
            targets = create_coref_target_forward(pred_spans, gold_spans, gold_m2i).to(scores.device)
            #
            if scores is not None:
                # span_lengths = torch.LongTensor([len(x) for x in pred_spans]).to(scores.device)

                triangular_mask = torch.ones(scores.size()[1:]).tril(0).unsqueeze(0)
                constant = scores.max().item() + 100000
                additive_mask = (1 - triangular_mask) * -constant
                logits = torch.nn.functional.log_softmax(scores + additive_mask.to(scores.device), dim=-1)

            if scores is not None and targets is not None:
                loss = - logsumexp(logits + (1 - targets) * -100000)
                mask = get_mask_from_sequence_lengths(span_lengths, span_lengths.max().item()).float()
                output['loss'] = self.weight * (mask * loss).sum()
            else:
                raise BaseException("HUH")
                # output['loss'] = torch.tensor(0.0).cuda()

            if predict:
                output['pred'] = [convert(m2i_to_clusters(x)[0], y) for x, y in
                                  zip(decode_m2i(logits, span_lengths), pred_spans)] \
                    if scores is not None else [[] for _ in pred_spans]
                output['scores'] = predict_scores_coref(scores, pred_spans=pred_spans)

                output['pred_pointers'] = [{} for x in gold_spans]
                output['gold'] = [convert(m2i_to_clusters(x.tolist())[0], y) for x, y in zip(gold_m2i, gold_spans)]

                if self.filter_singletons_with_pruner:
                    # this assumes that pruner is able to predict spans
                    output['pred'] = remove_disabled_spans(output['pred'], pruner_spans)
                    coref_flat = [{item for sublist in batch for item in sublist} for batch in output['pred']]
                    output['scores'] = remove_disabled_scores_coref(output['scores'], coref_flat)
                if self.filter_singletons_with_ner:
                    output['pred'] = remove_disabled_spans(output['pred'], ner_spans)

        else:
            output['loss'] = 0  # torch.tensor(0.0).cuda() (skip minibatch)
            # if gold_spans is None:
            #     gold_spans = []
            output['pred'] = [None for x in gold_spans]
            output['gold'] = [None for x in gold_spans]
            output['scores'] = [None for x in gold_spans]
            output['pred_pointers'] = [{} for x in gold_spans]

        return output['loss'], output

    def create_metrics(self):
        out = []
        if self.enabled:
            metrics = [
                MetricCoref(self.task, 'muc', MetricCoref.muc),
                MetricCoref(self.task, 'bcubed', MetricCoref.b_cubed, verbose=False),
                MetricCoref(self.task, 'ceafe', MetricCoref.ceafe, verbose=False),
            ]

            out.extend(metrics)
            out.append(MetricCorefAverage(self.task, 'avg', metrics))
            out.append(MetricObjective(self.task))
            out.append(MetricCorefExternal(self.task))
        return out

        # , MetricCoref2(self.task, 'muc2'), MetricCoref2(self.task, 'bcubed2')
        # , MetricObjective(self.task)


class LossBidirectionalCoref(nn.Module):

    def __init__(self, task, config):
        super(LossBidirectionalCoref, self).__init__()
        self.task = task
        self.weight = config.get('weight', 1.0)
        self.enabled = config['enabled']

    def forward(self, scores, gold_m2i, pred_spans, gold_spans, predict=False):
        output = {}

        if self.enabled:
            if scores is not None:
                lengths = torch.LongTensor([len(x) for x in pred_spans]).to(scores.device)
                constant = scores.max().item() + 100000

                triangular_mask = torch.ones(scores.size()[1:]).tril(0).unsqueeze(0)
                additive_mask = ((1 - triangular_mask) * -constant).to(scores.device)

                scores1 = scores + additive_mask
                scores2 = scores + additive_mask.permute(0, 2, 1)

                logits1 = torch.nn.functional.log_softmax(scores1, dim=-1)
                logits2 = torch.nn.functional.log_softmax(scores2, dim=-1)

            if scores is not None:
                targets1 = create_coref_target_forward(pred_spans, gold_spans, gold_m2i).to(scores.device)
                targets2 = create_coref_target_backward(pred_spans, gold_spans, gold_m2i).to(scores.device)

                mask = get_mask_from_sequence_lengths(lengths, lengths.max().item()).float()
                loss1 = - logsumexp(logits1 + (1.0 - targets1) * -100000)
                loss2 = - logsumexp(logits2 + (1.0 - targets2) * -100000)
                obj1 = (mask * loss1).sum()
                obj2 = (mask * loss2).sum()
                output['loss'] = self.weight * (obj1 + obj2)
            else:
                raise BaseException("HUH")
                output['loss'] = torch.tensor(0.0).cuda()

            if predict:
                output['pred'] = [convert(m2i_to_clusters(x)[0], y) for x, y in
                                  zip(decode_m2i(logits1, lengths), pred_spans)] if scores is not None else [[] for _ in
                                                                                                             pred_spans]
                output['gold'] = [convert(m2i_to_clusters(x.tolist())[0], y) for x, y in zip(gold_m2i, gold_spans)]
        else:
            output['loss'] = torch.tensor(0.0).cuda()
            output['pred'] = None
            output['gold'] = None

        return output['loss'], output

    def create_metrics(self):
        metrics = [
            MetricCoref(self.task, 'muc', MetricCoref.muc),
            MetricCoref(self.task, 'bcubed', MetricCoref.b_cubed, verbose=False),
            MetricCoref(self.task, 'ceafe', MetricCoref.ceafe, verbose=False),
        ] if self.enabled else []

        out = []
        out.extend(metrics)
        out.append(MetricCorefAverage(self.task, 'avg', metrics))
        out.append(MetricObjective(self.task))
        return out

        # , MetricCoref2(self.task, 'muc2'), MetricCoref2(self.task, 'bcubed2')
        # , MetricObjective(self.task)


def create_target_matrix(clusters):
    cluster2mentions = {}
    for mention, cluster in enumerate(clusters.tolist()):
        if cluster not in cluster2mentions:
            cluster2mentions[cluster] = []
        cluster2mentions[cluster].append(mention)

    number_of_mentions = clusters.size()[0]
    target = torch.zeros(number_of_mentions, number_of_mentions)
    for cluster, mentions in cluster2mentions.items():
        for m1 in mentions:
            for m2 in mentions:
                target[m1, m2] = 1
    return target


def logZ(scores):
    dim = scores.size()[0] + 1

    S = torch.zeros(dim, dim)
    S[1:, 1:] = scores
    A = torch.exp(S + torch.eye(dim) * -10000)
    D = torch.diag(A.sum(0))
    L = D - A

    L[0, 1:] = 1 / scores.size()[0]
    return L.logdet()


class TaskCorefMTT(nn.Module):

    def __init__(self, dim_input, config):
        super(TaskCorefMTT, self).__init__()
        self.module = create_graph(dim_input, 1, config['scorer'])

    def forward(self, mentions, clusters):
        scores = self.module(mentions).squeeze(-1).squeeze(0)
        targets = create_target_matrix(clusters[0]).to(scores.device)

        scores = scores - scores.max()

        g = logZ(scores + (1 - targets) * -10000)
        z = logZ(scores)

        return z - g

import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics.f1 import MetricRelationF1, decode_relations
from metrics.misc import MetricObjective
from metrics.relations import MetricConceptRelationSoftF1
from modules.graph import create_graph
from modules.relations.basic import TaskSpanRelations
from modules.relations.binary import LossRelationsX
from modules.relations.latent import LossRelationsLatentX
from util.sequence import get_mask_from_sequence_lengths


def masked_sum(scores, mask):
    mask = mask.float()
    x = scores * mask.unsqueeze(1).unsqueeze(2)
    x = x.sum(dim=-1) * mask.unsqueeze(1)
    return x.sum()


def inspect(name, x):
    print(name, x.min().item(), x.max().item(), x.mean().item(), x.std().item())


def create_task_relations(name, config, labels):
    if config['type'] == 'binary':
        return LossRelations(name, config, labels)
    elif config['type'] == 'latent-binary':
        return LossRelationsLatent(name, config, labels)
    elif config['type'] == 'latent-x':
        return LossRelationsLatentX(name, config, labels)
    elif config['type'] == 'binary-x':
        return LossRelationsX(name, config, labels)
    elif config['type'] == 'span-binary':
        return TaskSpanRelations(name, config, labels)
    else:
        raise BaseException("no such relation task:", config['type'])


class TaskRelations(nn.Module):

    def __init__(self, dim_input, config, labels):
        super(TaskRelations, self).__init__()
        self.num_relations = len(labels)
        self.module = create_graph(dim_input, self.num_relations, config['scorer'])
        self.loss = nn.BCEWithLogitsLoss(reduction='none')
        self.labels = labels
        W0 = config.get('weight', 1.0)
        self.weight = W0 / len(self.labels) if config.get('normalize', False) else W0
        self.divide_by_number_of_concepts = config.get('divide_by_number_of_concepts', True)
        self.enabled = config['enabled']
        print("Task relations: enabled={} weight={} normalize={} divide_by_number_of_concepts={}".format(self.enabled,
                                                                                                         self.weight,
                                                                                                         config.get(
                                                                                                             'normalize',
                                                                                                             False),
                                                                                                         self.divide_by_number_of_concepts))

    def set_weight(self, W0):
        self.weight = 3 / len(self.labels)
        print("Task {} weight: {}".format('relations', self.weight))
        self.divide_by_number_of_concepts = False

    def forward(self, inputs, targets, concept_lengths, square_mask):
        mask = get_mask_from_sequence_lengths(concept_lengths, concept_lengths.max().item())
        if self.divide_by_number_of_concepts:
            mask = mask.float() / concept_lengths.unsqueeze(1).float()  # dunno if this helps?

        scores = self.module(inputs, square_mask)
        scores = scores.permute(0, 3, 1, 2)

        if targets is not None:
            obj = self.loss(scores, targets)
            obj = self.weight * masked_sum(obj, mask)
            predictions = None
        else:
            obj = None
            predictions = decode_relations(scores, concept_lengths, self.labels)

        # print('obj:', obj.item())

        return obj, scores, predictions

    def create_metrics(self):
        return [MetricRelationF1('relations', self.labels), MetricObjective('relations')] if self.enabled else []


from modules.relations.latent import sum_scores, decode_relations_new


# don't know if we need this
# numerical stable implementation of log(1 - exp(x))
def log1mex(x):
    # print('x:', x.min().item(), x.max().item())
    # print(x[0,0,0,:])

    # expm1(x) = exp(x) - 1
    v1 = torch.log(-torch.expm1(x))
    # print('v1:', v1)
    # log1p(x) = log(1 + x)
    v2 = torch.log1p(-torch.exp(x))
    # print('v2:', v2)

    return v1
    # return torch.where(x > -0.693, v1, v2)


def create_square_mask(lengths):
    mask = get_mask_from_sequence_lengths(lengths, lengths.max().item())
    mask = mask.float()
    square_mask = torch.bmm(mask.unsqueeze(-1), mask.unsqueeze(-2))
    return square_mask


def debug_grad(x, grad):
    for i in range(grad.size(1)):
        for j in range(grad.size(2)):
            if torch.isnan(grad[0, i, j, :]).any():
                print('x:', i, j, x[0, i, j, :])
                print('dx:', i, j, grad[0, i, j, :])


def inspect(x):
    print('x-min:', x.min().item())
    print('x-max:', x.max().item())
    print()


class LossRelationsOld(nn.Module):

    def __init__(self, config, labels):
        super(LossRelationsOld, self).__init__()
        self.num_relations = len(labels)
        self.loss = nn.BCEWithLogitsLoss(reduction='none')
        self.labels = labels
        self.weight = config['weight'] / len(self.labels)
        self.enabled = config['enabled']

    def forward(self, mention_scores, mention_targets, mention_lengths, mention_mask, mapping, concept_lengths):
        if mention_targets is not None:
            obj = self.loss(mention_scores, mention_targets)
            obj = self.weight * (obj * mention_mask.unsqueeze(-1)).sum()
            predictions = None
        else:
            # TODO: fix reordering
            obj = None
            predictions = decode_relations(mention_scores, concept_lengths, self.labels)

        # print('obj:', obj.item())

        return obj, mention_scores.permute(0, 3, 1, 2), predictions

    def create_metrics(self):
        return [MetricRelationF1('relations', self.labels), MetricObjective('relations')] if self.enabled else []


class LossRelations(nn.Module):

    def __init__(self, name, config, labels):
        super(LossRelations, self).__init__()
        self.name = name
        self.num_relations = len(labels)
        self.loss = nn.BCEWithLogitsLoss(reduction='none')
        self.labels = labels
        self.enabled = config['enabled']
        self.weight = config['weight'] / len(self.labels)
        self.debug = config['debug']

    def forward(self, mention_scores, mention_targets, mention_lengths, mention_mask, mapping, coref, relations,
                predict=False):
        output = {}

        if self.enabled and mention_targets is not None:
            obj = self.weight * (self.loss(mention_scores, mention_targets) * mention_mask.unsqueeze(
                -1)).sum() / self.num_relations
        else:
            obj = torch.tensor(0.0).cuda()

        output['loss'] = obj

        concept_targets = (sum_scores(mention_targets, mapping) > 0).float()

        if mention_targets is not None:
            concept_lengths = [len(x) for x in coref['pred']]
            mytargets = decode_relations_new(concept_targets, concept_lengths, self.labels)
            output['target'] = [[(clusters[src], clusters[dst], rel) for src, dst, rel in triples] for clusters, triples
                                in zip(coref['pred'], mytargets)]

        if predict:
            if mention_scores is None:
                output['pred'] = [[] for x in coref['pred']]
            else:
                # print('min:', mention_scores.min().item())
                # print('max:', mention_scores.max().item())
                pred_mentions = (mention_scores > 0).float()
                pred_concepts = sum_scores(pred_mentions, mapping)
                pred_concepts = (pred_concepts > 0).float()

                concept_lengths = [len(x) for x in coref['pred']]
                predictions = decode_relations_new(pred_concepts, concept_lengths, self.labels)
                output['pred'] = [[(clusters[src], clusters[dst], rel) for src, dst, rel in triples] for
                                  clusters, triples in zip(coref['pred'], predictions)]

            output['gold'] = [[(clusters[src], clusters[dst], self.labels[rel]) for src, dst, rel in triples] for
                              clusters, (_, triples, _) in
                              zip(relations['gold_clusters2'], relations['gold_relations'])]

        return output['loss'], output

    def create_metrics(self):
        return [MetricConceptRelationSoftF1(self.name, self.labels, verbose=self.debug),
                MetricObjective(self.name)] if self.enabled else []


class LossRelationsLatent(nn.Module):

    def __init__(self, name, config, labels):
        super(LossRelationsLatent, self).__init__()
        self.name = name
        self.num_relations = len(labels)

        self.labels = labels
        self.enabled = config['enabled']
        self.weight = config['weight'] / len(self.labels)
        self.latent = True
        self.old_implementation = False
        self.debug = config['debug']

    def forward(self, mention_scores, mention_targets, mention_lengths, mention_mask, mapping, coref, relations,
                predict=False):
        output = {}

        # if mention_scores is not None:
        #     print('mention_scores:', mention_scores.size())
        # if mention_targets is not None:
        #     print('mention_targets:', mention_targets.size())

        if mention_targets is not None:
            # print('relations active', mention_targets.sum().item())

            concept_targets = (sum_scores(mention_targets, mapping) > 0).float()

            if self.latent:
                if self.old_implementation:
                    # not all concept pairs have mention pairs
                    mask = (sum_scores(torch.ones(mention_scores.size()).cuda(), mapping) > 0).float()

                    mention_logits = F.logsigmoid(-mention_scores)  # [-inf, 0]
                    concept_logits = sum_scores(mention_logits, mapping)

                    if self.debug:
                        tmp = (concept_logits * mask == 0).float() * mask
                        print('tmp:', tmp.sum().item())  # shouldn't this be zero?

                    # print('->', (concept_logits * mask).min(), (concept_logits * mask).max())
                    # print('->', (concept_logits * (1-mask)).min(), (concept_logits * (1-mask)).max())

                    # print('logits:', concept_logits.sum(-1))

                    # TODO: can we remove this? sign of possible simplification of loss equation?
                    x = concept_logits - 1e-8
                    # x = concept_logits - (1-mask) * 100000 - 1e-8

                    # concept_logits.register_hook(lambda grad: debug_grad(x, grad))

                    inspect(x)

                    loss = concept_targets * log1mex(x)
                    loss += (1 - concept_targets) * concept_logits
                    loss *= mask

                    # print('nan:', torch.isnan(loss).any()) 

                    obj = - self.weight * loss.sum() / self.num_relations
                else:
                    mask = (sum_scores(torch.ones(mention_scores.size()).cuda(), mapping) > 0).float()

                    mention_logits = -torch.log1p(torch.exp(mention_scores.double()))
                    # print('mention_logits:', mention_logits.min().item(), mention_logits.max().item())
                    concept_logits = sum_scores(mention_logits, mapping.double())
                    concept_logits = concept_logits + (1.0 - mask) * -10000
                    # print('concept_logits:', concept_logits.min().item(), concept_logits.max().item())

                    loss = concept_targets * torch.log(-torch.expm1(concept_logits - 1e-100))
                    # print('loss:', loss.min().item(), loss.max().item())
                    loss += (1 - concept_targets) * concept_logits
                    loss *= mask

                    # print('nan:', torch.isnan(loss).any()) 

                    obj = - self.weight * loss.sum() / self.num_relations
            else:
                pos_logits = F.logsigmoid(mention_scores)
                neg_logits = F.logsigmoid(-mention_scores)
                loss = mention_targets * pos_logits + (1 - mention_targets) * neg_logits
                loss = loss * mention_mask.unsqueeze(-1)

                pos_logits2 = sum_scores(pos_logits, mapping)
                neg_logits2 = sum_scores(neg_logits, mapping)
                loss2 = concept_targets * pos_logits2 + (1 - concept_targets) * neg_logits2

                # print('->', (loss.sum()-loss2.sum()).item())

                obj = - self.weight * loss2.sum() / self.num_relations
        else:
            obj = torch.tensor(0.0).cuda()

        output['loss'] = obj

        if mention_targets is not None:
            concept_lengths = [len(x) for x in coref['pred']]
            mytargets = decode_relations_new(concept_targets, concept_lengths, self.labels)
            output['target'] = [[(clusters[src], clusters[dst], rel) for src, dst, rel in triples] for clusters, triples
                                in zip(coref['pred'], mytargets)]

        if predict:
            if mention_scores is None:
                output['pred'] = [[] for x in coref['pred']]
            else:
                # print('min:', mention_scores.min().item())
                # print('max:', mention_scores.max().item())
                pred_mentions = (mention_scores > 0).float()
                pred_concepts = sum_scores(pred_mentions, mapping)
                pred_concepts = (pred_concepts > 0).float()

                concept_lengths = [len(x) for x in coref['pred']]
                predictions = decode_relations_new(pred_concepts, concept_lengths, self.labels)
                output['pred'] = [[(clusters[src], clusters[dst], rel) for src, dst, rel in triples] for
                                  clusters, triples in zip(coref['pred'], predictions)]

            output['gold'] = [[(clusters[src], clusters[dst], self.labels[rel]) for src, dst, rel in triples] for
                              clusters, (_, triples, _) in
                              zip(relations['gold_clusters2'], relations['gold_relations'])]

        # gold = []
        # for clusters, relations in zip(coref['gold'], relations['gold_relations']):
        #     print('clusters:', len(clusters))
        #     for src, dst, rel in relations[1]:
        #         print(src, dst, rel)
        #         gold.append((clusters[src], rel, clusters[dst]))
        # print('gold:', gold)

        # return obj, mention_scores.permute(0, 3, 1, 2), predictions
        return output['loss'], output

    def create_metrics(self):
        # MetricRelationF1('relations', self.labels), 
        # , MetricObjective('relations')
        return [MetricConceptRelationSoftF1(self.name, self.labels, verbose=self.debug),
                MetricObjective(self.name)] if self.enabled else []


class LossRelationsNew(nn.Module):

    def __init__(self, config, labels):
        super(LossRelationsNew, self).__init__()
        self.num_relations = len(labels)
        self.loss = nn.BCEWithLogitsLoss(reduction='none')
        self.labels = labels
        self.weight = config['weight'] / len(self.labels)
        self.enabled = config['enabled']
        self.debug = config['debug']

    def forward(self, scores, targets, clusters, relations, predict=False):
        output = {}

        num_concepts = [len(x) for x in clusters]
        max_concepts = max(num_concepts)

        if scores is not None and targets is not None:
            lengths = torch.LongTensor(num_concepts).to(scores.device)
            mask = get_mask_from_sequence_lengths(lengths, max_concepts).float()
            square_mask = torch.bmm(mask.unsqueeze(-1), mask.unsqueeze(-2))

            obj = self.loss(scores, targets)
            output['loss'] = self.weight * (obj * square_mask.unsqueeze(-1)).sum()
        else:
            output['loss'] = torch.tensor(0.0).cuda()

        if targets is not None:
            mytargets = decode_relations_new(targets, num_concepts, self.labels)
            output['target'] = [[(myclusters[src], myclusters[dst], rel) for src, dst, rel in triples] for
                                myclusters, triples in zip(clusters, mytargets)]

        if predict:
            if scores is None:
                output['pred'] = [[] for _ in clusters]
            else:
                predictions = decode_relations_new(scores, num_concepts, self.labels)
                output['pred'] = [[(myclusters[src], myclusters[dst], rel) for src, dst, rel in triples] for
                                  myclusters, triples in zip(clusters, predictions)]
            output['gold'] = [[(myclusters[src], myclusters[dst], self.labels[rel]) for src, dst, rel in triples] for
                              myclusters, (_, triples, _) in
                              zip(relations['gold_clusters'], relations['gold_relations'])]

        return output

    def create_metrics(self):
        return [MetricConceptRelationSoftF1('relations', self.labels, verbose=self.debug)] if self.enabled else []

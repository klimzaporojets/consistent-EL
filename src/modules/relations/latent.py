import torch
import torch.nn as nn
from metrics.relations import MetricConceptRelationSoftF1
from modules.relations.misc import create_relation_targets_2

from metrics.misc import MetricObjective


def create_mapping(spans, clusters):
    """

    :param spans:
    :param clusters:
    :return:
    """
    num_batch = len(spans)
    max_spans = max([len(x) for x in spans])
    max_concepts = max([len(x) for x in clusters])

    mapping = torch.zeros(num_batch, max_concepts, max_spans)

    for batch, (myspans, myclusters) in enumerate(zip(spans, clusters)):
        span2index = {}
        for idx, span in enumerate(myspans):
            span2index[span] = idx

        for idx, cluster in enumerate(myclusters):
            for span in cluster:
                if span in span2index:  # in case relation pruner != coref pruner
                    mapping[batch, idx, span2index[span]] = 1.0

    return mapping


def sum_scores(scores, u):
    if scores.dim() != 4:
        raise BaseException("scores is not a 4-dimensional tensor")
    if u.dim() != 3:
        raise BaseException("mapping is not a 3-dimensional tensor")
    if scores.size(0) != u.size(0):
        raise BaseException("batch size doesn't match")
    num_batch, num_mentions, num_concepts = u.size() # u.shape: [1,8,14]
    v = u.unsqueeze(1).expand(num_batch, num_concepts, num_mentions, num_concepts) # v.shape: [1, 14, 8, 14]
    o = torch.matmul(v, scores) # scores.shape: [1, 14, 14, 3] , o.shape: [1, 14, 8, 3]
    p = o.view(o.size()[0:2] + (-1,)) # p.shape: [1, 14, 24]
    q = torch.matmul(u, p) # q.shape: [1, 8, 24]
    q = q.view(q.size()[0:2] + o.size()[2:]) # q.shape: 1, 8, 8, 3
    return q


# TODO: move this to cpn utilities?
def decode_relations_new(targets, lengths, labels):
    relations = []
    for b, length in enumerate(lengths):
        rels = []
        for src, dst, rel in torch.nonzero(targets[b, 0:length, 0:length, :] > 0).tolist():
            rels.append((src, dst, labels[rel]))
        relations.append(rels)
    return relations


def filter_gold_clusters(gold_clusters, pred_spans):
    output = []
    for clusters_in, spans in zip(gold_clusters, pred_spans):
        p_spans = set(spans)
        clusters_out = [[span for span in cluster if span in p_spans] for cluster in clusters_in]

        # is this needed?
        g_spans = set()
        for cluster in clusters_in:
            for span in cluster:
                g_spans.add(span)
        clusters_out.extend([[span] for span in p_spans - g_spans])

        output.append(clusters_out)

    return output


# cleaned up version

class LossRelationsLatentX(nn.Module):

    def __init__(self, name, config, labels):
        super(LossRelationsLatentX, self).__init__()
        self.name = name
        self.num_relations = len(labels)

        self.labels = labels
        self.enabled = config['enabled']
        self.weight = config['weight'] / len(self.labels) if config['normalize'] else config['weight']
        self.debug = config['debug']

        print("LossRelationsLatentX:", config['mapping'], config['normalize'])
        self.mapping_binary = config['mapping'] == 'binary'
        self.mapping_gold = config['mapping'] == 'gold'
        self.mapping_pred = config['mapping'] == 'pred'

    def forward(self, relation_filtered, mention_scores, relations, coref, predict=False):
        output = {}

        span_lengths = relation_filtered['span_lengths']
        mention_mask = relation_filtered['square_mask']
        pred_spans = relation_filtered['spans']
        mention_targets = create_relation_targets_2(pred_spans, relations, len(self.labels), span_lengths)

        coref_pred = coref['pred']
        mapping_pred = create_mapping(pred_spans, coref_pred).to(mention_scores.device)

        if self.mapping_binary:
            singletons = [[[span] for span in spans] for spans in pred_spans]
            mapping_loss = create_mapping(pred_spans, singletons).to(mention_scores.device)
        elif self.mapping_gold:
            clusters = filter_gold_clusters(coref['gold'], pred_spans)
            mapping_loss = create_mapping(pred_spans, clusters).to(mention_scores.device)
        elif self.mapping_pred:
            mapping_loss = mapping_pred
        else:
            raise BaseException('no such mapping')

        # TODO: check if mapping is correct in case many negative spans are included
        # print('sum:', mapping_loss.sum(-1).max().item())

        if mention_targets is not None:
            concept_targets = (sum_scores(mention_targets, mapping_loss) > 0).float()

            mask = (sum_scores(torch.ones(mention_scores.size()).cuda(), mapping_loss) > 0).float()

            mention_logits = -torch.log1p(
                torch.exp(mention_scores.double()))  # WARNING: should be unstable but works ?!
            # mention_logits = F.logsigmoid(-mention_scores)
            # print('nan:', torch.isnan(mention_logits).any()) 

            concept_logits = sum_scores(mention_logits, mapping_loss.double())
            concept_logits = concept_logits + (1.0 - mask) * -10000

            # TODO: is this stable?
            loss = concept_targets * torch.log(-torch.expm1(concept_logits - 1e-100))
            loss += (1 - concept_targets) * concept_logits
            loss *= mask

            obj = - self.weight * loss.sum() / self.num_relations
        else:
            obj = torch.tensor(0.0).cuda()

        output['loss'] = obj

        if mention_targets is not None:
            concept_lengths = [len(x) for x in coref_pred]
            mytargets = decode_relations_new(concept_targets, concept_lengths, self.labels)
            output['target'] = [[(clusters[src], clusters[dst], rel) for src, dst, rel in triples] for clusters, triples
                                in zip(coref_pred, mytargets)]

        if predict:
            if mention_scores is None:
                output['pred'] = [[] for x in coref_pred]
            else:
                pred_mentions = (mention_scores > 0).float()
                pred_concepts = sum_scores(pred_mentions, mapping_pred)
                pred_concepts = (pred_concepts > 0).float()

                concept_lengths = [len(x) for x in coref_pred]
                predictions = decode_relations_new(pred_concepts, concept_lengths, self.labels)
                output['pred'] = [[(clusters[src], clusters[dst], rel) for src, dst, rel in triples] for
                                  clusters, triples in zip(coref_pred, predictions)]

            output['gold'] = [[(clusters[src], clusters[dst], self.labels[rel]) for src, dst, rel in triples] for
                              clusters, triples in zip(relations['gold_clusters2'], relations['gold_relations'])]

        return output['loss'], output

    def create_metrics(self):
        return [MetricConceptRelationSoftF1(self.name, self.labels, verbose=self.debug),
                MetricObjective(self.name)] if self.enabled else []

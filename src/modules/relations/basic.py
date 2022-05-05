import torch
import torch.nn as nn
from modules.relations.misc import create_relation_targets_2

from metrics.misc import MetricObjective
from metrics.relations import MetricSpanRelationF1x


def decode_span_relations(scores, spanss, labels):
    relations = []
    for b, spans in enumerate(spanss):
        length = len(spans)
        rels = []
        for src, dst, rel in torch.nonzero(scores[b, 0:length, 0:length, :] > 0).tolist():
            rels.append((spans[src], labels[rel], spans[dst]))
        relations.append(rels)
    return relations


def gold_cluster_to_span_relations(clusters, relations, labels):
    output = []
    for cs, rels in zip(clusters, relations):
        tmp = []
        for src_cluster_idx, dst_cluster_idx, rel_idx in rels:
            for src in cs[src_cluster_idx]:
                for dst in cs[dst_cluster_idx]:
                    tmp.append((src, labels[rel_idx], dst))
        output.append(tmp)
    return output


# relations between spans

class TaskSpanRelations(nn.Module):

    def __init__(self, name, config, labels):
        super(TaskSpanRelations, self).__init__()
        self.name = name
        self.loss = nn.BCEWithLogitsLoss(reduction='none')
        self.labels = labels
        self.enabled = config['enabled']
        self.weight = config['weight'] / len(self.labels) if config.get('normalize', True) else config['weight']
        self.debug = config['debug']
        print("TaskSpanRelations: weight={}".format(self.weight))

    def forward(self, relation_filtered, mention_scores, relations, coref, predict=False):
        output = {}

        span_lengths = relation_filtered['span_lengths']
        mention_mask = relation_filtered['square_mask']
        pred_spans = relation_filtered['spans']

        if self.enabled:
            mention_targets = create_relation_targets_2(pred_spans, relations, len(self.labels), span_lengths)
            obj = self.weight * (self.loss(mention_scores, mention_targets) * mention_mask.unsqueeze(-1)).sum()

            output['loss'] = obj

            output['span-rel-pred'] = decode_span_relations(mention_scores, pred_spans, self.labels)
            output['span-rel-gold'] = gold_cluster_to_span_relations(relations['gold_clusters2'],
                                                                     relations['gold_relations'], self.labels)

            output['pred'] = [None for x in relations['gold_relations']]
            output['gold'] = [None for x in relations['gold_relations']]
        else:
            output['loss'] = 0  # (trainer skips minibatch if zero)

            output['span-rel-pred'] = [None for x in relations['gold_relations']]
            output['span-rel-gold'] = [None for x in relations['gold_relations']]

            output['pred'] = [None for x in relations['gold_relations']]
            output['gold'] = [None for x in relations['gold_relations']]

        return output['loss'], output

    def create_metrics(self):
        return [MetricSpanRelationF1x(self.name, self.labels, verbose=self.debug),
                MetricObjective(self.name)] if self.enabled else []

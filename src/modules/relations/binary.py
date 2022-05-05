import torch.nn as nn

from metrics.misc import MetricObjective
from metrics.relations import MetricConceptRelationSoftF1, MetricConceptRelationToMentionsF1, MetricSpanRelationF1x
from modules.relations.basic import decode_span_relations, gold_cluster_to_span_relations
from modules.relations.latent import create_mapping, sum_scores, decode_relations_new
from modules.relations.misc import create_relation_targets_2


class LossRelationsX(nn.Module):

    def __init__(self, name, config, labels):
        super(LossRelationsX, self).__init__()
        self.name = name
        self.num_relations = len(labels)
        self.loss = nn.BCEWithLogitsLoss(reduction='none')
        self.labels = labels
        self.enabled = config['enabled']
        self.weight = config['weight'] / len(self.labels) if config.get('normalize', True) else config['weight']
        self.debug = config['debug']
        print("LossRelationsX: weight={} (fix-norm)".format(self.weight))
        self.evaluate_mentionwise_predictions = config['mentionwise']

    def forward(self, relation_filtered, mention_scores, relations, coref, predict=False):
        output = {}

        span_lengths = relation_filtered['span_lengths']
        mention_mask = relation_filtered['square_mask']
        pred_spans = relation_filtered['spans']

        if self.enabled:
            mention_targets = create_relation_targets_2(pred_spans, relations, len(self.labels), span_lengths)
            obj = self.weight * (self.loss(mention_scores, mention_targets) * mention_mask.unsqueeze(
                -1)).sum()  # / self.num_relations            # IS THIS A BUG??
        else:
            # obj = torch.tensor(0.0).cuda()
            obj = 0  # (trainer skips minibatch if zero)

        output['loss'] = obj

        if self.enabled:
            mapping = create_mapping(pred_spans, coref['pred']).to(mention_scores.device)
            concept_targets = (sum_scores(mention_targets, mapping) > 0).float()

            # only for debugging
            if mention_targets is not None:
                concept_lengths = [len(x) for x in coref['pred']]
                mytargets = decode_relations_new(concept_targets, concept_lengths, self.labels)
                output['target'] = [[(clusters[src], clusters[dst], rel) for src, dst, rel in triples] for
                                    clusters, triples in zip(coref['pred'], mytargets)]

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
                                  clusters, triples in zip(relations['gold_clusters2'], relations['gold_relations'])]

                if self.evaluate_mentionwise_predictions:
                    output['span-rel-pred'] = decode_span_relations(mention_scores, pred_spans, self.labels)
                    output['span-rel-gold'] = gold_cluster_to_span_relations(relations['gold_clusters2'],
                                                                             relations['gold_relations'], self.labels)
                else:
                    output['span-rel-pred'] = [None for x in relations['gold_relations']]
                    output['span-rel-gold'] = [None for x in relations['gold_relations']]
        else:
            # when api_call, the no gold_relations is coming
            # if relations is None:
            #     gold_rels = []
            # else:
            gold_rels = relations.get('gold_relations')
            output['pred'] = [None for x in gold_rels]
            output['gold'] = [None for x in gold_rels]
            output['span-rel-pred'] = [None for x in gold_rels]
            output['span-rel-gold'] = [None for x in gold_rels]

        return output['loss'], output

    def create_metrics(self):
        if self.enabled:
            metrics = [
                MetricConceptRelationSoftF1(self.name, self.labels, verbose=self.debug),
                MetricConceptRelationToMentionsF1(self.name, self.labels, verbose=self.debug),
                MetricObjective(self.name)
            ]
            if self.evaluate_mentionwise_predictions:
                metrics.append(
                    MetricSpanRelationF1x(self.name, self.labels, verbose=self.debug)
                )
            return metrics
        else:
            return []

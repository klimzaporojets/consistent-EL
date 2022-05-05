import argparse
import json
import os
from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment


# stand-alone evaluation script, should be runnable by external people
# TODO:
# 1. concept wise NER scoring?
# 2. don't throw out  singleton clusters for coref
# 3. overall average score ?

def load_jsonl(filename, tag):
    with open(filename, 'r') as file:
        data = [json.loads(line) for line in file.readlines()]
        data = [x for x in data if tag is None or tag in x['tags']]
        return {x['id']: x for x in data}


def load_json(filename, tag):
    with open(filename, 'r') as file:
        doc = json.load(file)
        return {doc['id']: doc} if tag is None or tag in doc['tags'] else {}


def load_data(path, tag=None):
    if os.path.isdir(path):
        data = {}
        for file in os.listdir(path):
            filename = os.path.join(path, file)
            data.update(load_json(filename, tag))
        return data
    else:
        return load_jsonl(path, tag)


# TODO: make a decision if we allow inconsistent predictions
def decode_spans(instance):
    spans = []
    for mention in instance['mentions']:
        concept = instance['concepts'][mention['concept']]
        concept_tags = concept['tags'] if 'tags' in concept else []
        tags = mention['tags'] if 'tags' in mention else concept_tags
        for tag in tags:
            spans.append((mention['begin'], mention['end'], tag))
    return spans


def decode_spans_clusters(instance):
    spans_cluster = []
    concept_id_to_mentions = dict()
    for mention in instance['mentions']:
        concept_id = mention['concept']
        if concept_id not in concept_id_to_mentions:
            concept_id_to_mentions[concept_id] = list()
        concept_id_to_mentions[concept_id].append((mention['begin'], mention['end']))

    for concept_id, concept_info in enumerate(instance['concepts']):
        if concept_id in concept_id_to_mentions:
            for curr_tag in concept_info['tags']:
                # spans_cluster.append((concept_id_to_mentions[concept_id], curr_tag))
                spans_cluster.append((tuple(concept_id_to_mentions[concept_id]), curr_tag))
    return spans_cluster


def decode_coref(instance):
    concept2cluster = {idx: list() for idx, _ in enumerate(instance['concepts'])}
    for mention in instance['mentions']:
        concept2cluster[mention['concept']].append((mention['begin'], mention['end']))
    return [x for x in concept2cluster.values() if len(x) > 0]


def decode_relations(instance):
    concept2cluster = {idx: list() for idx, _ in enumerate(instance['concepts'])}
    for mention in instance['mentions']:
        concept2cluster[mention['concept']].append((mention['begin'], mention['end']))

    # remove duplicate relations
    relations = set([(relation['s'], relation['p'], relation['o']) for relation in instance['relations']])

    if len(relations) != len(instance['relations']):
        print("WARNING: duplicate relations")

    # relations = [(concept2cluster[s], concept2cluster[o], p) for s, p, o in relations]
    # (kzaporoj) - the tuple would allow to do hard concept level comparisons, the lists can not be hashed in order to
    # do set operations later on
    relations = [(tuple(concept2cluster[s]), tuple(concept2cluster[o]), p) for s, p, o in relations]
    return relations


class MetricF1:

    def __init__(self):
        self.clear()

    def clear(self):
        self.labels = set()
        self.tp = {}
        self.fp = {}
        self.fn = {}
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0

    def add_labels(self, spans):
        for _, _, label in spans:
            if label not in self.labels:
                self.labels.add(label)
                self.tp[label] = 0
                self.fp[label] = 0
                self.fn[label] = 0

    def update(self, preds, golds):
        for pred, gold in zip(preds, golds):
            self.add_labels(pred)
            self.add_labels(gold)

            for _, _, label in [x for x in pred if x in gold]:
                self.tp[label] += 1
                self.total_tp += 1
            for _, _, label in [x for x in pred if x not in gold]:
                self.fp[label] += 1
                self.total_fp += 1
            for _, _, label in [x for x in gold if x not in pred]:
                self.fn[label] += 1
                self.total_fn += 1

    def print(self, details=False):
        for label in self.labels:
            tp, fp, fn = self.tp[label], self.fp[label], self.fn[label]
            pr = tp / (tp + fp) if tp != 0 else 0.0
            re = tp / (tp + fn) if tp != 0 else 0.0
            f1 = 2 * tp / (2 * tp + fp + fn) if tp != 0 else 0.0
            if details:
                print('{:32}    {:5}  {:5}  {:5}    {:6.5f}  {:6.5f}  {:6.5f}'.format(label, tp, fp, fn, pr, re, f1))

        print('{:32}    {:5}  {:5}  {:5}    {:6.5f}  {:6.5f}  {:6.5f}'.format('', self.total_tp, self.total_fp,
                                                                              self.total_fn, self.get_pr(),
                                                                              self.get_re(), self.get_f1()))

    def get_pr(self):
        return self.total_tp / (self.total_tp + self.total_fp) if self.total_tp != 0 else 0.0

    def get_re(self):
        return self.total_tp / (self.total_tp + self.total_fn) if self.total_tp != 0 else 0.0

    def get_f1(self):
        return 2 * self.total_tp / (2 * self.total_tp + self.total_fp + self.total_fn) if self.total_tp != 0 else 0.0


def clusters_to_mentions(cluster_spans):
    flatten_mentions = list()
    for curr_cluster, entity_type in cluster_spans:
        for curr_span in curr_cluster:
            flatten_mentions.append((curr_span, entity_type))
    return set(flatten_mentions)


def cluster_to_mentions(cluster_spans, entity_type):
    flatten_mentions = list()
    for curr_span in cluster_spans:
        flatten_mentions.append((curr_span, entity_type))
    return set(flatten_mentions)


class MetricF1Soft:

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.clear()

    def clear(self):
        self.labels = set()
        self.p_tps = {}
        self.p_fps = {}
        self.r_tps = {}
        self.r_fns = {}

        self.f1 = 0.0
        self.pr = 0.0
        self.re = 0.0

    def add_labels(self, spans_cluster):
        for _, label in spans_cluster:
            if label not in self.labels:
                self.labels.add(label)
                self.p_tps[label] = 0.0
                self.p_fps[label] = 0.0
                self.r_tps[label] = 0.0
                self.r_fns[label] = 0.0

    def update(self, preds, golds):
        for pred, gold in zip(preds, golds):
            self.add_labels(pred)
            self.add_labels(gold)

            P = clusters_to_mentions(pred)
            G = clusters_to_mentions(gold)
            for cluster_p, entity_type in pred:
                pairs = cluster_to_mentions(cluster_p, entity_type)
                if len(pairs) > 0:
                    tp = len(pairs & G) / len(pairs)
                    fp = 1.0 - tp
                    self.p_tps[entity_type] += tp
                    self.p_fps[entity_type] += fp

            for cluster_g, entity_type in gold:
                pairs = cluster_to_mentions(cluster_g, entity_type)
                if len(pairs) > 0:
                    tp = len(pairs & P) / len(pairs)
                    fp = 1.0 - tp
                    self.r_tps[entity_type] += tp
                    self.r_fns[entity_type] += fp

    def calculate_metrics(self, must_print=False):
        total_p_tp, total_p_fp, total_r_tp, total_r_fn = 0, 0, 0, 0

        for label in self.labels:
            p_tp, p_fp = self.p_tps[label], self.p_fps[label]
            r_tp, r_fn = self.r_tps[label], self.r_fns[label]
            pr = p_tp / (p_tp + p_fp) if p_tp != 0 else 0.0
            re = r_tp / (r_tp + r_fn) if r_tp != 0 else 0.0
            f1 = 2.0 * pr * re / (pr + re) if pr * re != 0.0 else 0.0

            if self.verbose and must_print:
                print(
                    '{:24}    {:6.1f} / {:6.1f} = {:6.5f}    {:6.1f} / {:6.1f} = {:6.5f}    {:6.5f}'.format(label, p_tp,
                                                                                                            p_fp, pr,
                                                                                                            r_tp, r_fn,
                                                                                                            re, f1))

            total_p_tp += p_tp
            total_p_fp += p_fp
            total_r_tp += r_tp
            total_r_fn += r_fn

        total_pr = total_p_tp / (total_p_tp + total_p_fp) if total_p_tp != 0 else 0.0
        total_re = total_r_tp / (total_r_tp + total_r_fn) if total_r_tp != 0 else 0.0
        total_f1 = 2.0 * total_pr * total_re / (total_pr + total_re) if total_pr * total_re != 0.0 else 0.0

        if must_print:
            print(
                'SOFT NER {:24}    {:6.1f} / {:6.1f} = {:6.5f}    {:6.1f} / {:6.1f} = {:6.5f}    {:6.5f}'
                    .format('', total_p_tp,
                            total_p_fp,
                            total_pr,
                            total_r_tp,
                            total_r_fn,
                            total_re,
                            total_f1))

        self.f1 = total_f1
        self.pr = total_pr
        self.re = total_re

    def print(self):
        self.calculate_metrics(must_print=True)

    def get_f1(self):
        return self.f1

    def get_pr(self):
        return self.pr

    def get_re(self):
        return self.re


class MetricF1Hard:

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.clear()

    def clear(self):
        self.labels = set()
        self.tps = {}
        self.fps = {}
        self.fns = {}

        self.f1 = 0.0
        self.pr = 0.0
        self.re = 0.0

    def add_labels(self, spans_cluster):
        for _, label in spans_cluster:
            if label not in self.labels:
                self.labels.add(label)
                self.tps[label] = 0.0
                self.fps[label] = 0.0
                self.fns[label] = 0.0

    def update(self, preds, golds):
        for pred, gold in zip(preds, golds):
            self.add_labels(pred)
            self.add_labels(gold)

            for cluster_p in pred:
                predicted_label = cluster_p[1]
                if cluster_p in gold:
                    self.tps[predicted_label] += 1
                else:
                    self.fps[predicted_label] += 1

            for cluster_g in gold:
                gold_label = cluster_g[1]
                if cluster_g not in pred:
                    self.fns[gold_label] += 1

    def calculate_metrics(self, must_print=False):
        total_tps, total_fps, total_fns = 0, 0, 0

        for label in self.labels:
            tps, fps, fns = self.tps[label], self.fps[label], self.fns[label]
            pr = tps / (tps + fps) if tps != 0 else 0.0
            re = tps / (tps + fns) if tps != 0 else 0.0
            f1 = 2.0 * pr * re / (pr + re) if pr * re != 0.0 else 0.0

            if self.verbose and must_print:
                print(
                    '{:24}    {:6.1f} / {:6.1f} = {:6.5f}    {:6.1f} / {:6.1f} = {:6.5f}    {:6.5f}'.format(label, tps,
                                                                                                            fps, pr,
                                                                                                            tps, fns,
                                                                                                            re, f1))

            total_tps += tps
            total_fps += fps
            total_fns += fns

        total_pr = total_tps / (total_tps + total_fps) if total_tps != 0 else 0.0
        total_re = total_tps / (total_tps + total_fns) if total_tps != 0 else 0.0
        total_f1 = 2.0 * total_pr * total_re / (total_pr + total_re) if total_pr * total_re != 0.0 else 0.0

        if must_print:
            print(
                'SOFT NER {:24}    {:6.1f} / {:6.1f} = {:6.5f}    {:6.1f} / {:6.1f} = {:6.5f}    {:6.5f}'
                    .format('', total_tps,
                            total_fps,
                            total_pr,
                            total_tps,
                            total_fns,
                            total_re,
                            total_f1))

        self.f1 = total_f1
        self.pr = total_pr
        self.re = total_re

    def print(self):
        self.calculate_metrics(must_print=True)

    def get_f1(self):
        return self.f1

    def get_pr(self):
        return self.pr

    def get_re(self):
        return self.re


def mention2cluster(clusters):
    clusters = [tuple(tuple(m) for m in gc) for gc in clusters]
    mention_to_cluster = {}
    for cluster in clusters:
        for mention in cluster:
            mention_to_cluster[mention] = cluster
    return mention_to_cluster


class MetricCoref:

    def __init__(self, name, m, verbose=False):
        self.name = name
        self.m = m
        self.verbose = verbose
        self.clear()

    def clear(self):
        self.precision_numerator = 0
        self.precision_denominator = 0
        self.recall_numerator = 0
        self.recall_denominator = 0

    def add(self, pred, gold):
        if pred is not None and gold is not None:
            if self.m == self.ceafe or self.m == self.ceafe_singleton_entities or self.m == self.ceafe_singleton_mentions:
                p_num, p_den, r_num, r_den = self.m(pred, gold)
            else:
                p_num, p_den = self.m(pred, mention2cluster(gold))
                r_num, r_den = self.m(gold, mention2cluster(pred))

            self.precision_numerator += p_num
            self.precision_denominator += p_den
            self.recall_numerator += r_num
            self.recall_denominator += r_den

    def get_f1(self):
        precision = self.get_pr()
        recall = self.get_re()
        return 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    def get_pr(self):
        precision = 0 if self.precision_denominator == 0 else \
            self.precision_numerator / float(self.precision_denominator)
        return precision

    def get_re(self):
        recall = 0 if self.recall_denominator == 0 else \
            self.recall_numerator / float(self.recall_denominator)
        return recall

    def print(self):
        f1 = self.get_f1()

        print("coref\t{}\t{}".format(self.name, f1))

    @staticmethod
    def b_cubed(clusters, mention_to_gold):
        numerator, denominator = 0, 0
        for cluster in clusters:
            if len(cluster) == 1:
                continue
            gold_counts = Counter()
            correct = 0
            for mention in cluster:
                if mention in mention_to_gold:
                    gold_counts[tuple(mention_to_gold[mention])] += 1
            for cluster2, count in gold_counts.items():
                if len(cluster2) != 1:
                    correct += count * count
            numerator += correct / float(len(cluster))
            denominator += len(cluster)
        return numerator, denominator

    @staticmethod
    def b_cubed_singleton_entities(clusters, mention_to_gold):
        numerator, denominator = 0, 0
        for cluster in clusters:
            gold_counts = Counter()
            correct = 0
            for mention in cluster:
                if mention in mention_to_gold:
                    gold_counts[tuple(mention_to_gold[mention])] += 1
            for cluster2, count in gold_counts.items():
                correct += count * count

            # (kzaporoj) - old mention-based:
            # numerator += correct / float(len(cluster))
            # (kzaporoj) - dividing it by the length of the cluster will make it have same weight in scoring than other
            # bigger/smaller clusters
            numerator += correct / float(len(cluster)) / float(len(cluster))

            # (kzaporoj) - old mention-based:
            # denominator += len(cluster)
            # (kzaporoj) - here just sums 1 for each cluster, treating them equally
            denominator += 1
        return numerator, denominator

    @staticmethod
    def b_cubed_singleton_mentions(clusters, mention_to_gold):
        numerator, denominator = 0, 0
        for cluster in clusters:
            gold_counts = Counter()
            correct = 0
            for mention in cluster:
                if mention in mention_to_gold:
                    gold_counts[tuple(mention_to_gold[mention])] += 1
            for cluster2, count in gold_counts.items():
                correct += count * count
            numerator += correct / float(len(cluster))
            denominator += len(cluster)
        return numerator, denominator

    @staticmethod
    def muc(clusters, mention_to_gold):
        true_p, all_p = 0, 0
        for cluster in clusters:
            all_p += len(cluster) - 1
            true_p += len(cluster)
            linked = set()
            for mention in cluster:
                if mention in mention_to_gold:
                    linked.add(mention_to_gold[mention])
                else:
                    true_p -= 1
            true_p -= len(linked)
        return true_p, all_p

    @staticmethod
    def phi4_entity_centric(gold_clustering, predicted_clustering):
        """
        Subroutine for ceafe. Computes the mention F measure between gold and
        predicted mentions in a cluster.
        (kzaporoj) - Entity centric (normalizes by the len of the involved clusters)
        """
        return (
                2
                * len([mention for mention in gold_clustering if mention in predicted_clustering])
                / float(len(gold_clustering) + len(predicted_clustering))
        )

    @staticmethod
    def phi4_mention_centric(gold_clustering, predicted_clustering):
        """
        Subroutine for ceafe. Computes the mention F measure between gold and
        predicted mentions in a cluster.
        (kzaporoj) - Mention centric (sum of the number of mentions in intersected clusters)
        """
        return (
            len([mention for mention in gold_clustering if mention in predicted_clustering])
        )

    @staticmethod
    def ceafe(clusters, gold_clusters):
        """
        Computes the  Constrained EntityAlignment F-Measure (CEAF) for evaluating coreference.
        Gold and predicted mentions are aligned into clusterings which maximise a metric - in
        this case, the F measure between gold and predicted clusters.
        <https://www.semanticscholar.org/paper/On-Coreference-Resolution-Performance-Metrics-Luo/de133c1f22d0dfe12539e25dda70f28672459b99>
        """
        clusters = [cluster for cluster in clusters if len(cluster) != 1]
        gold_clusters = [cluster for cluster in gold_clusters if len(cluster) != 1]  # is this really correct?
        scores = np.zeros((len(gold_clusters), len(clusters)))
        for i, gold_cluster in enumerate(gold_clusters):
            for j, cluster in enumerate(clusters):
                scores[i, j] = MetricCoref.phi4_entity_centric(gold_cluster, cluster)
        # print('pred:', [len(x) for x in clusters])
        # print('gold:', [len(x) for x in gold_clusters])
        row, col = linear_sum_assignment(-scores)
        similarity = sum(scores[row, col])
        return similarity, len(clusters), similarity, len(gold_clusters)

    @staticmethod
    def ceafe_singleton_entities(clusters, gold_clusters):
        """
        Computes the  Constrained EntityAlignment F-Measure (CEAF) for evaluating coreference.
        Gold and predicted mentions are aligned into clusterings which maximise a metric - in
        this case, the F measure between gold and predicted clusters.
        <https://www.semanticscholar.org/paper/On-Coreference-Resolution-Performance-Metrics-Luo/de133c1f22d0dfe12539e25dda70f28672459b99>
        (kzaporoj) - this is entity-centric version where the cost is based on formula (9) of the paper
        """
        clusters = [cluster for cluster in clusters if len(cluster) > 0]
        gold_clusters = [cluster for cluster in gold_clusters if len(cluster) > 0]
        scores = np.zeros((len(gold_clusters), len(clusters)))
        for i, gold_cluster in enumerate(gold_clusters):
            for j, cluster in enumerate(clusters):
                scores[i, j] = MetricCoref.phi4_entity_centric(gold_cluster, cluster)
        row, col = linear_sum_assignment(-scores)
        similarity = sum(scores[row, col])
        return similarity, len(clusters), similarity, len(gold_clusters)

    @staticmethod
    def ceafe_singleton_mentions(clusters, gold_clusters):
        """
        Computes the  Constrained EntityAlignment F-Measure (CEAF) for evaluating coreference.
        Gold and predicted mentions are aligned into clusterings which maximise a metric - in
        this case, the F measure between gold and predicted clusters.
        <https://www.semanticscholar.org/paper/On-Coreference-Resolution-Performance-Metrics-Luo/de133c1f22d0dfe12539e25dda70f28672459b99>
        (kzaporoj) - this is mention-centric version where the cost is based on formula (8) of the paper
        """
        clusters = [cluster for cluster in clusters if len(cluster) > 0]
        gold_clusters = [cluster for cluster in gold_clusters if len(cluster) > 0]
        scores = np.zeros((len(gold_clusters), len(clusters)))
        for i, gold_cluster in enumerate(gold_clusters):
            for j, cluster in enumerate(clusters):
                scores[i, j] = MetricCoref.phi4_mention_centric(gold_cluster, cluster)
        row, col = linear_sum_assignment(-scores)
        similarity = sum(scores[row, col])
        cluster_mentions = [item for sublist in clusters for item in sublist]
        gold_cluster_mentions = [item for sublist in gold_clusters for item in sublist]
        return similarity, len(cluster_mentions), similarity, len(gold_cluster_mentions)


def to_pairwise(rels):
    out = []
    for src_cluster, dst_cluster, rel in rels:
        for src in src_cluster:
            for dst in dst_cluster:
                out.append((src, dst, rel))
    return set(out)


def to_pairs(src_cluster, dst_cluster, rel):
    pairs = []
    for src in src_cluster:
        for dst in dst_cluster:
            pairs.append((src, dst, rel))
    return set(pairs)


def captions(cluster, tokens):
    return [' '.join(tokens[begin:(end + 1)]) for begin, end in cluster]


class MetricRelationF1:

    def __init__(self, verbose):
        self.verbose = verbose
        self.clear()

    def clear(self):
        self.labels = set()
        self.f1 = 0.0
        self.pr = 0.0
        self.re = 0.0
        self.p_tps = {}
        self.p_fps = {}
        self.r_tps = {}
        self.r_fns = {}

    def add_labels(self, triples):
        for _, _, rel in triples:
            if rel not in self.labels:
                self.labels.add(rel)
                self.p_tps[rel] = 0.0
                self.p_fps[rel] = 0.0
                self.r_tps[rel] = 0.0
                self.r_fns[rel] = 0.0

    def add(self, pred, gold):
        self.add_labels(pred)
        self.add_labels(gold)

        P = to_pairwise(pred)
        G = to_pairwise(gold)

        for src_cluster, dst_cluster, rel in pred:
            pairs = to_pairs(src_cluster, dst_cluster, rel)
            if len(pairs) > 0:
                tp = len(pairs & G) / len(pairs)
                fp = 1.0 - tp

                self.p_tps[rel] += tp
                self.p_fps[rel] += fp

        for src_cluster, dst_cluster, rel in gold:
            pairs = to_pairs(src_cluster, dst_cluster, rel)
            if len(pairs) > 0:
                tp = len(pairs & P) / len(pairs)
                fn = 1.0 - tp

                self.r_tps[rel] += tp
                self.r_fns[rel] += fn

    def update(self, args, metadata={}):
        for batch, (pred, gold, identifier, tokens) in enumerate(
                zip(args['pred'], args['gold'], metadata['identifiers'], metadata['tokens'])):
            if self.verbose:
                # print("pred:", pred)
                # print("gold:", gold)
                print("ID:", identifier)
                print("pred:", [(rel, captions(src, tokens), captions(dst, tokens)) for src, dst, rel in pred])
                if 'target' in args:
                    print("target:", [(rel, captions(src, tokens), captions(dst, tokens)) for src, dst, rel in
                                      args['target'][batch]])
                print("gold:", [(rel, captions(src, tokens), captions(dst, tokens)) for src, dst, rel in gold])
                print()

            self.add(pred, gold)

    def calculate_metrics(self, must_print=False):
        total_p_tp, total_p_fp, total_r_tp, total_r_fn = 0, 0, 0, 0

        for label in self.labels:
            p_tp, p_fp = self.p_tps[label], self.p_fps[label]
            r_tp, r_fn = self.r_tps[label], self.r_fns[label]
            pr = p_tp / (p_tp + p_fp) if p_tp != 0 else 0.0
            re = r_tp / (r_tp + r_fn) if r_tp != 0 else 0.0
            f1 = 2.0 * pr * re / (pr + re) if pr * re != 0.0 else 0.0

            if self.verbose and must_print:
                print(
                    '{:24}    {:6.1f} / {:6.1f} = {:6.5f}    {:6.1f} / {:6.1f} = {:6.5f}    {:6.5f}'.format(label, p_tp,
                                                                                                            p_fp, pr,
                                                                                                            r_tp, r_fn,
                                                                                                            re, f1))

            total_p_tp += p_tp
            total_p_fp += p_fp
            total_r_tp += r_tp
            total_r_fn += r_fn

        total_pr = total_p_tp / (total_p_tp + total_p_fp) if total_p_tp != 0 else 0.0
        total_re = total_r_tp / (total_r_tp + total_r_fn) if total_r_tp != 0 else 0.0
        total_f1 = 2.0 * total_pr * total_re / (total_pr + total_re) if total_pr * total_re != 0.0 else 0.0

        if must_print:
            print(
                '{:24}    {:6.1f} / {:6.1f} = {:6.5f}    {:6.1f} / {:6.1f} = {:6.5f}    {:6.5f}'.format('', total_p_tp,
                                                                                                        total_p_fp,
                                                                                                        total_pr,
                                                                                                        total_r_tp,
                                                                                                        total_r_fn,
                                                                                                        total_re,
                                                                                                        total_f1))

        self.f1 = total_f1
        self.pr = total_pr
        self.re = total_re

    # just to be compatible with previous version
    # def print(self, dataset_name, details):
    #     self.print()

    def print(self):
        self.calculate_metrics(must_print=True)

    def get_f1(self):
        return self.f1

    def get_pr(self):
        return self.pr

    def get_re(self):
        return self.re


class MetricRelationF1Hard:

    def __init__(self, verbose):
        self.verbose = verbose
        self.clear()

    def clear(self):
        self.labels = set()
        self.f1 = 0.0
        self.pr = 0.0
        self.re = 0.0
        self.tps = {}
        self.fps = {}
        self.fns = {}

    def add_labels(self, triples):
        for _, _, rel in triples:
            if rel not in self.labels:
                self.labels.add(rel)
                self.tps[rel] = 0.0
                self.fps[rel] = 0.0
                self.fns[rel] = 0.0

    def add(self, pred, gold):
        self.add_labels(pred)
        self.add_labels(gold)

        # tps = pred & gold
        for rel_pred in pred:
            pred_label = rel_pred[2]
            if rel_pred in gold:
                self.tps[pred_label] += 1
            else:
                self.fps[pred_label] += 1

        for rel_gold in gold:
            gold_label = rel_gold[2]
            if rel_gold not in pred:
                self.fns[gold_label] += 1

    def calculate_metrics(self, must_print=False):
        total_tps, total_fps, total_fns = 0, 0, 0

        for label in self.labels:
            tps, fps, fns = self.tps[label], self.fps[label], self.fns[label]
            pr = tps / (tps + fps) if tps != 0 else 0.0
            re = tps / (tps + fns) if tps != 0 else 0.0
            f1 = 2.0 * pr * re / (pr + re) if pr * re != 0.0 else 0.0

            if self.verbose and must_print:
                print(
                    '{:24}    {:6.1f} / {:6.1f} = {:6.5f}    {:6.1f} / {:6.1f} = {:6.5f}    {:6.5f}'.format(label, tps,
                                                                                                            fps, pr,
                                                                                                            tps, fns,
                                                                                                            re, f1))

            total_tps += tps
            total_fps += fps
            total_fns += fns

        total_pr = total_tps / (total_tps + total_fps) if total_tps != 0 else 0.0
        total_re = total_tps / (total_tps + total_fns) if total_tps != 0 else 0.0
        total_f1 = 2.0 * total_pr * total_re / (total_pr + total_re) if total_pr * total_re != 0.0 else 0.0

        if must_print:
            print(
                'HARD RELATION {:24}    {:6.1f} / {:6.1f} = {:6.5f}    {:6.1f} / {:6.1f} = {:6.5f}    {:6.5f}'
                    .format('', total_tps,
                            total_fps,
                            total_pr,
                            total_tps,
                            total_fns,
                            total_re,
                            total_f1))

        self.f1 = total_f1
        self.pr = total_pr
        self.re = total_re

    def print(self):
        self.calculate_metrics(must_print=True)

    def get_f1(self):
        return self.f1

    def get_pr(self):
        return self.pr

    def get_re(self):
        return self.re


class MetricRelationF1Mention:

    def __init__(self, verbose):
        self.verbose = verbose
        self.clear()

    def clear(self):
        self.f1 = 0.0
        self.pr = 0.0
        self.re = 0.0
        self.labels = set()
        self.tps = {}
        self.fps = {}
        # self.r_tps = {}
        self.fns = {}

    def add_labels(self, triples):
        for _, _, rel in triples:
            if rel not in self.labels:
                self.labels.add(rel)
                self.tps[rel] = 0
                self.fps[rel] = 0
                # self.r_tns[rel] = 0
                self.fns[rel] = 0

    def add(self, pred, gold, doc_id=''):
        self.add_labels(pred)
        self.add_labels(gold)

        P = to_pairwise(pred)
        G = to_pairwise(gold)

        for src_cluster, dst_cluster, rel in pred:
            pairs = to_pairs(src_cluster, dst_cluster, rel)
            if len(pairs) > 0:
                tp = len(pairs & G)  # / len(pairs)
                fp = len(pairs) - tp

                self.tps[rel] += tp
                self.fps[rel] += fp

        for src_cluster, dst_cluster, rel in gold:
            pairs = to_pairs(src_cluster, dst_cluster, rel)
            if len(pairs) > 0:
                # tp = len(pairs & P) # / len(pairs)
                fn = len(pairs) - len(pairs & P)
                self.fns[rel] += fn

    def get_if_exists(self, any_dict: dict, key, default):
        if key in any_dict:
            return any_dict[key]
        else:
            return default

    def update(self, args, metadata={}):
        for batch, (pred, gold, identifier, tokens) in enumerate(
                zip(args['pred'], args['gold'], metadata['identifiers'], metadata['tokens'])):
            if self.verbose:
                # print("pred:", pred)
                # print("gold:", gold)
                print("ID:", identifier)
                print("pred:", [(rel, captions(src, tokens), captions(dst, tokens)) for src, dst, rel in pred])
                if 'target' in args:
                    print("target:", [(rel, captions(src, tokens), captions(dst, tokens)) for src, dst, rel in
                                      args['target'][batch]])
                print("gold:", [(rel, captions(src, tokens), captions(dst, tokens)) for src, dst, rel in gold])
                print()

            self.add(pred, gold)

    # def print(self):
    def print(self):
        self.calculate_metrics(must_print=True)

    def calculate_metrics(self, must_print=False):
        total_tp, total_fp, total_fn = 0, 0, 0

        for label in self.labels:
            tp, fp = self.tps[label], self.fps[label]
            fn = self.fns[label]
            pr = tp / (tp + fp) if tp != 0 else 0.0
            re = tp / (tp + fn) if tp != 0 else 0.0
            f1 = 2.0 * pr * re / (pr + re) if pr * re != 0.0 else 0.0

            if self.verbose and must_print:
                print(
                    '{:24}    {:6.1f} / {:6.1f} = {:6.5f}    {:6.1f} / {:6.1f} = {:6.5f}    {:6.5f}'.format(label, tp,
                                                                                                            fp, pr,
                                                                                                            tp, fn,
                                                                                                            re, f1))

            total_tp += tp
            total_fp += fp
            total_fn += fn

        total_pr = total_tp / (total_tp + total_fp) if total_tp != 0 else 0.0
        total_re = total_tp / (total_tp + total_fn) if total_tp != 0 else 0.0
        total_f1 = 2.0 * total_pr * total_re / (total_pr + total_re) if total_pr * total_re != 0.0 else 0.0

        if must_print:
            print('{:24}    {:6.1f} / {:6.1f} = {:6.5f}    {:6.1f} / {:6.1f} = {:6.5f}    {:6.5f}'.format('', total_tp,
                                                                                                          total_fp,
                                                                                                          total_pr,
                                                                                                          total_tp,
                                                                                                          total_fn,
                                                                                                          total_re,
                                                                                                          total_f1))

        self.f1 = total_f1
        self.pr = total_pr
        self.re = total_re

    def get_f1(self):
        return self.f1

    def get_pr(self):
        return self.pr

    def get_re(self):
        return self.re


class EvaluatorCPN:

    def __init__(self):
        self.tags = MetricF1()
        self.tags_soft = MetricF1Soft()
        self.tags_hard = MetricF1Hard()
        self.coref_muc = MetricCoref('muc', MetricCoref.muc)
        self.coref_bcubed = MetricCoref('bcubed', MetricCoref.b_cubed)
        self.coref_bcubed_singleton_men = MetricCoref('bcubed singleton mention-based',
                                                      MetricCoref.b_cubed_singleton_mentions)
        self.coref_bcubed_singleton_ent = MetricCoref('bcubed singleton entity-based',
                                                      MetricCoref.b_cubed_singleton_entities)

        self.coref_ceafe = MetricCoref('ceafe', MetricCoref.ceafe)
        self.coref_ceafe_singleton_men = MetricCoref('ceafe-singleton mention-based',
                                                     MetricCoref.ceafe_singleton_mentions)
        self.coref_ceafe_singleton_ent = MetricCoref('ceafe-singleton entity-based',
                                                     MetricCoref.ceafe_singleton_entities)
        self.rels = MetricRelationF1(True)
        self.rels_mention = MetricRelationF1Mention(True)
        self.rels_hard = MetricRelationF1Hard(True)

    def add(self, pred, gold):
        P = decode_spans(pred)
        G = decode_spans(gold)
        self.tags.update([P], [G])

        P_cluster = decode_spans_clusters(pred)
        G_cluster = decode_spans_clusters(gold)
        self.tags_soft.update([P_cluster], [G_cluster])

        self.tags_hard.update([P_cluster], [G_cluster])

        P = decode_coref(pred)
        G = decode_coref(gold)
        self.coref_muc.add(P, G)
        self.coref_bcubed.add(P, G)
        self.coref_bcubed_singleton_men.add(P, G)
        self.coref_bcubed_singleton_ent.add(P, G)
        self.coref_ceafe.add(P, G)
        self.coref_ceafe_singleton_men.add(P, G)
        self.coref_ceafe_singleton_ent.add(P, G)

        P = decode_relations(pred)
        G = decode_relations(gold)
        self.rels.add(P, G)
        self.rels_hard.add(P, G)
        self.rels_mention.add(P, G)

    def printInfo(self):
        print("# Evalution")
        print("## Multilabel NER")
        self.tags.print(details=True)

        print("## Multilabel Soft NER")
        self.tags_soft.print()
        print()

        print("## Multilabel Hard NER")
        self.tags_hard.print()
        print()

        print("## Coreference")
        self.coref_muc.print()
        self.coref_bcubed.print()
        self.coref_ceafe.print()
        self.coref_ceafe_singleton_ent.print()
        self.coref_ceafe_singleton_men.print()
        avg = sum([self.coref_muc.get_f1(), self.coref_bcubed.get_f1(), self.coref_ceafe.get_f1()]) / 3
        print("coref-avg:", avg)
        print("coref-bcubed pr: ", self.coref_bcubed.get_pr(),
              ", re: ", self.coref_bcubed.get_re(),
              ", f1: ", self.coref_bcubed.get_f1())
        print("coref-bcubed singleton men pr: ", self.coref_bcubed_singleton_men.get_pr(),
              ", re: ", self.coref_bcubed_singleton_men.get_re(),
              ", f1: ", self.coref_bcubed_singleton_men.get_f1())
        print("coref-bcubed singleton ent pr: ", self.coref_bcubed_singleton_ent.get_pr(),
              ", re: ", self.coref_bcubed_singleton_ent.get_re(),
              ", f1: ", self.coref_bcubed_singleton_ent.get_f1())

        print("coref-ceafe pr: ", self.coref_ceafe.get_pr(),
              ", re: ", self.coref_ceafe.get_re(),
              ", f1: ", self.coref_ceafe.get_f1())
        print("coref-ceafe singleton men pr: ", self.coref_ceafe_singleton_men.get_pr(),
              ", re: ", self.coref_ceafe_singleton_men.get_re(),
              ", f1: ", self.coref_ceafe_singleton_men.get_f1())
        print("coref-ceafe singleton ent pr: ", self.coref_ceafe_singleton_ent.get_pr(),
              ", re: ", self.coref_ceafe_singleton_ent.get_re(),
              ", f1: ", self.coref_ceafe_singleton_ent.get_f1())

        print()

        print("## Relations Soft Entity Cluster")
        self.rels.print()
        print()

        print("## Relations Hard Entity Cluster")
        self.rels_hard.print()
        print()

        print("## Relations Mention Based")
        self.rels_mention.print()
        print()

        print('## Summary')
        print('ner pr: ', self.tags.get_pr(), 'ner re: ', self.tags.get_re(), 'ner f1: ', self.tags.get_f1())
        print('soft ner pr: ', self.tags_soft.get_pr(), 'soft ner re:  ', self.tags_soft.get_re(),
              'soft ner f1: ', self.tags_soft.get_f1())
        print('hard ner pr: ', self.tags_hard.get_pr(), 'hard ner re:  ', self.tags_hard.get_re(),
              'hard ner f1: ', self.tags_hard.get_f1())
        print('coref: ', avg)
        print('hard rel pr:  ', self.rels_hard.get_pr(),
              'hard rel re: ', self.rels_hard.get_re(),
              'hard rel f1: ', self.rels_hard.get_f1())
        print('soft rel pr:  ', self.rels.get_pr(), 'soft rel re: ', self.rels.get_re(), 'soft rel f1: ',
              self.rels.get_f1())
        print('rel_mentions pr:  ', self.rels_mention.get_pr(), 'rel re: ', self.rels_mention.get_re(),
              'rel_mentions f1: ', self.rels_mention.get_f1())
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", dest="pred", type=str, default=None, required=True)
    parser.add_argument("--gold", dest="gold", type=str, default=None, required=True)
    parser.add_argument("--pred-filter", dest="pred_filter", type=str, default=None, required=False)
    parser.add_argument("--gold-filter", dest="gold_filter", type=str, default=None, required=False)
    args = parser.parse_args()

    pred = load_data(args.pred, args.pred_filter)
    gold = load_data(args.gold, args.gold_filter)

    print("pred instances:", len(pred))
    print("gold instances:", len(gold))

    evaluator = EvaluatorCPN()
    for identifier in gold.keys():
        evaluator.add(pred[identifier], gold[identifier])
    evaluator.printInfo()

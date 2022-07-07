import argparse
import json
import logging
import os
from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment

# stand-alone evaluation script, should be runnable by external people
# TODO:
# 1. concept wise NER scoring?
# 2. don't throw out  singleton clusters for coref
# 3. overall average score ?

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger()


def load_jsonl(filename, tag):
    logger.info('load jsonl for the following file: %s' % filename)
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
        # (kzaporoj) - previous code that merges purely mention-based predictions with cluster-based predictions
        # The idea is to separate the purely mention-based ('tags' in each mention) from "expanded" (coming from the
        # tags in cluster) - see decode_spans_expanded

        # if mention['concept'] > -1:
        #     concept = instance['concepts'][mention['concept']]
        #     concept_tags = concept['tags'] if 'tags' in concept else []
        #     tags = mention['tags'] if 'tags' in mention else concept_tags
        # else:
        #     tags = mention['tags'] if 'tags' in mention else []

        tags = mention['tags'] if 'tags' in mention else []

        for tag in tags:
            spans.append((mention['begin'], mention['end'], tag))
    return spans


def decode_links(instance, is_pred=False):
    """

    :param instance:
    :return: example of expected output: [(1,1,"Berlin"), (2,2,"NILL"), ...]

    """
    mentions = instance['mentions']
    concepts = instance['concepts']
    decoded_links = []

    # print('the instance in decode_links is ', instance)
    if not is_pred:
        for curr_mention in mentions:
            curr_concept = concepts[curr_mention['concept']]

            if 'link' not in curr_mention:
                if 'candidates' in curr_mention:
                    if 'link' in curr_concept:
                        if curr_concept['link'] is None or curr_concept['link'] == 'NILL':
                            curr_mention['link'] = 'NILL'
                        else:
                            if curr_concept['link'] in set(curr_mention['candidates']):
                                curr_mention['link'] = curr_concept['link']
                            else:
                                curr_mention['link'] = 'NILL'

            if 'link' in curr_mention:
                if curr_mention['link'] is None:
                    curr_mention['link'] = 'NILL'
                decoded_links.append((curr_mention['begin'], curr_mention['end'], curr_mention['link']))
    else:
        for curr_mention in mentions:
            if 'link_pred' in curr_mention and not isinstance(curr_mention['link_pred'], list):
                # if 'link' in curr_mention:
                #     if 'candidates' in curr_mention and curr_mention['link'] is not None and curr_mention['link'] != 'NILL':
                #         if curr_mention['']
                decoded_links.append((curr_mention['begin'], curr_mention['end'], curr_mention['link_pred']))
            elif 'candidates' in curr_mention and 'scores' in curr_mention:
                if 'link_pred' in concepts[curr_mention['concept']] and \
                        concepts[curr_mention['concept']]['link_pred'] is not None \
                        and concepts[curr_mention['concept']]['link_pred'] != 'NILL':
                    raise RuntimeError(
                        'This should not happen, no link_pred in mention but link_pred in respective concept.')
                # print('WARN!!!: no link_pred detected!!! in ', instance['id'])
                # mention_scores = curr_mention['scores']
                # max_ele = mention_scores[0]
                # max_ele_idx = 0
                # for i in range(1, len(mention_scores)):
                #     if mention_scores[i] > max_ele:
                #         max_ele = mention_scores[i]
                #         max_ele_idx = i
                # decoded_links.append((curr_mention['begin'], curr_mention['end'],
                #                       curr_mention['candidates'][max_ele_idx]))

    return decoded_links


def decode_links_from_ents(instance, is_pred=False):
    """

    :param instance:
    :return: example of expected output: [(1,1,"Berlin"), (2,2,"NILL"), ...]

    """
    mentions = instance['mentions']
    concepts = instance['concepts']
    decoded_links = []

    # print('the instance in decode_links is ', instance)
    if not is_pred:
        for curr_mention in mentions:
            curr_concept = concepts[curr_mention['concept']]
            if 'link' in curr_concept:
                if curr_concept['link'] is None:
                    curr_concept['link'] = 'NILL'
                decoded_links.append((curr_mention['begin'], curr_mention['end'], curr_concept['link']))
            else:
                # kzaporoj 14/07/2021 --> needed in order to report NIL results, since we do not predict the entity type
                # the NILL is for all the mentions (including time, and roles)
                # curr_concept['link'] = 'NILL'
                decoded_links.append((curr_mention['begin'], curr_mention['end'], 'NILL'))
    else:
        for curr_mention in mentions:
            if 'link_pred' in curr_mention and not isinstance(curr_mention['link_pred'], list):
                decoded_links.append((curr_mention['begin'], curr_mention['end'], curr_mention['link_pred']))
            # elif 'candidates' in curr_mention and 'scores' in curr_mention:
            else:
                # kzaporoj 14/07/2021 --> needed in order to report NIL results, since we do not predict the entity type
                # the NILL is for all the mentions (including time, and roles)
                decoded_links.append((curr_mention['begin'], curr_mention['end'], 'NILL'))
                # pass  # TODO: see good this situations, doesn't happen with baseline, but happens with coreflinker and mtt
                # print('WARN!!!: no link_pred detected!!! in ', instance['id'], curr_mention['text'],
                #       '(', curr_mention['begin'], ',', curr_mention['end'], ')')
                # mention_scores = curr_mention['scores']
                # max_ele = mention_scores[0]
                # max_ele_idx = 0
                # for i in range(1, len(mention_scores)):
                #     if mention_scores[i] > max_ele:
                #         max_ele = mention_scores[i]
                #         max_ele_idx = i
                # decoded_links.append((curr_mention['begin'], curr_mention['end'],
                #                       curr_mention['candidates'][max_ele_idx]))

    return decoded_links


def decode_link_scores(instance):
    """

    :param instance:
    :return: example of expected output: [[((5, 7), ['NILL'], [1.4783]),
        ((41, 43), ['NILL'], [1.507637619972229]),
        ((45, 47), ['Charité', 'Barack_Obama', 'Angela_Merkel', 'NILL'], [0.5643, 0.8586, 0.7117, 1.3846]),
        ... ]

    """
    mentions = instance['mentions']
    decoded_link_scores = []

    for curr_mention in mentions:
        if 'candidates' in curr_mention and 'scores' in curr_mention:
            decoded_link_scores.append(((curr_mention['begin'], curr_mention['end']),
                                        curr_mention['candidates'],
                                        curr_mention['scores']))

    return decoded_link_scores


def decode_spans_expanded(instance):
    spans = []
    for mention in instance['mentions']:
        if mention['concept'] > -1:
            concept = instance['concepts'][mention['concept']]
            concept_tags = concept['tags'] if 'tags' in concept else []
            tags = concept_tags
        else:
            # tags = mention['tags'] if 'tags' in mention else []
            # can happen for example for ner only scenario that there are no concepts in the predictions
            return []

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
                # sorts by pos and width of span
                sorted_mentions = sorted(concept_id_to_mentions[concept_id], key=lambda x: (x[0], x[1] - x[0]))
                spans_cluster.append((tuple(sorted_mentions), curr_tag))
    return spans_cluster


def decode_links_clusters(instance, is_pred=False):
    link_entry = 'link'
    if is_pred:
        link_entry = 'link_pred'
    spans_cluster = []
    concept_id_to_mentions = dict()
    for mention in instance['mentions']:
        concept_id = mention['concept']
        if concept_id not in concept_id_to_mentions:
            concept_id_to_mentions[concept_id] = list()
        concept_id_to_mentions[concept_id].append((mention['begin'], mention['end']))

    for concept_id, concept_info in enumerate(instance['concepts']):
        if concept_id in concept_id_to_mentions:
            link = None
            if link_entry in concept_info:
                link = concept_info[link_entry]
                if link is None:
                    link = 'NILL'
            else:
                link = 'NILL'  # kzaporoj 14/07/2021 --> we include all the NILLs
            if link is not None:
                # sorts by pos and width of span
                sorted_mentions = sorted(concept_id_to_mentions[concept_id], key=lambda x: (x[0], x[1] - x[0]))
                spans_cluster.append((tuple(sorted_mentions), link))
    return spans_cluster


def decode_links_mention_clusters(instance):
    """
    Only for predicted!

    :param instance:
    :return: example: [((((1, 1), 'L1'), ((2, 2), 'L2'), ((3, 3), 'L1'), ((4, 4), 'L3'),
                   ((5, 5), 'L4')), 'L1'), ((((6, 6), 'L2'),), 'L2')]
    """
    link_entry = 'link'
    spans_cluster = []
    concept_id_to_mentions = dict()
    for mention in instance['mentions']:
        concept_id = mention['concept']
        if concept_id not in concept_id_to_mentions:
            concept_id_to_mentions[concept_id] = list()

        if 'link_pred' in mention and not mention['link_pred'] == 'NONE':
            link_pred = mention['link_pred']
            if link_pred is None:
                link_pred = 'NILL'
            concept_id_to_mentions[concept_id].append(((mention['begin'], mention['end']), link_pred))

    for concept_id, concept_info in enumerate(instance['concepts']):
        if concept_id in concept_id_to_mentions:
            link = None
            if link_entry in concept_info:
                link = concept_info[link_entry]
                if link is None:
                    link = 'NILL'
            if link is not None:
                # sorts by pos and width of span
                sorted_mentions = sorted(concept_id_to_mentions[concept_id], key=lambda x: (x[0][0], x[0][0] - x[0][1]))
                spans_cluster.append((tuple(sorted_mentions), link))
    return spans_cluster


def decode_coref(instance):
    concept2cluster = {idx: list() for idx, _ in enumerate(instance['concepts'])}
    for mention in instance['mentions']:
        if mention['concept'] > -1:
            concept2cluster[mention['concept']].append((mention['begin'], mention['end']))
        else:
            print('WARN!!!: NO CONCEPT FOR MENTION ', instance['id'])
    return [x for x in concept2cluster.values() if len(x) > 0]


def decode_relations_expanded(instance):
    concept2cluster = {idx: list() for idx, _ in enumerate(instance['concepts'])}
    for mention in instance['mentions']:
        if mention['concept'] > -1:
            concept2cluster[mention['concept']].append((mention['begin'], mention['end']))

    # remove duplicate relations
    relations = set([(relation['s'], relation['p'], relation['o']) for relation in instance['relations']])

    if len(relations) != len(instance['relations']):
        # print("WARNING: duplicate relations")
        pass

    # relations = [(concept2cluster[s], concept2cluster[o], p) for s, p, o in relations]
    # (kzaporoj) - the tuple would allow to do hard concept level comparisons, the lists can not be hashed in order to
    # do set operations later on
    relations = [(tuple(concept2cluster[s]), tuple(concept2cluster[o]), p) for s, p, o in relations]
    return relations


def decode_relations(instance):
    # remove duplicate relations
    if 'mention_relations' not in instance:
        return set()

    relations = set([((instance['mentions'][relation['s']]['begin'], instance['mentions'][relation['s']]['end']),
                      (instance['mentions'][relation['o']]['begin'], instance['mentions'][relation['o']]['end']),
                      relation['p']) for relation in instance['mention_relations']])

    if len(relations) != len(instance['mention_relations']):
        logger.warning('WARNING: duplicate relations in mention relations')

    return relations


def exclude_links_mentions(links):
    return {link for link in links if link[2] == 'NILL'}


def exclude_nills_mentions(links):
    return {link for link in links if link[2] != 'NILL'}


def exclude_links_clusters(links):
    return {link for link in links if link[1] == 'NILL'}


def exclude_nills_clusters(links):
    return {link for link in links if link[1] != 'NILL'}


class MetricLinkingF1Mention:

    def __init__(self, mode='default'):
        if mode == 'nills':
            self.filter = exclude_links_mentions
        elif mode == 'links':
            self.filter = exclude_nills_mentions
        else:
            self.filter = lambda x: x
        self.clear()

    def clear(self):
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0

    def add(self, preds, golds):
        for pred, gold in zip(preds, golds):
            if pred is None and gold is None:
                continue
            pred = self.filter(pred)
            gold = self.filter(gold)

            P = set(pred)
            G = set(gold)

            self.total_tp += len(P & G)
            self.total_fp += len(P - G)
            self.total_fn += len(G - P)

    def print_debug(self):
        print('{:32}    {:5}  {:5}  {:5}    {:6.5f}  {:6.5f}  {:6.5f}'.format('', self.total_tp, self.total_fp,
                                                                                    self.total_fn, self.get_pr(),
                                                                                    self.get_re(), self.get_f1()))

    def get_pr(self):
        return self.total_tp / (self.total_tp + self.total_fp) if self.total_tp != 0 else 0.0

    def get_re(self):
        return self.total_tp / (self.total_tp + self.total_fn) if self.total_tp != 0 else 0.0

    def get_f1(self):
        f1 = 2.0 * self.get_pr() * self.get_re() / (
                self.get_pr() + self.get_re()) if self.get_pr() * self.get_re() != 0.0 else 0.0
        return f1


class MetricLinkingF1Hard:
    def __init__(self, mode='default', label_based=False):
        """

        :param mode:
        :param label_based: whether the tp/fp/fn stats are also gathered for each of the labels (links)
        """
        if mode == 'nills':
            self.filter = exclude_links_clusters
        elif mode == 'links':
            self.filter = exclude_nills_clusters
        else:
            self.filter = lambda x: x
        self.label_based = label_based
        self.clear()

    def clear(self):
        self.labels = set()
        self.tps = {}
        self.fps = {}
        self.fns = {}

        self.total_tps = 0
        self.total_fps = 0
        self.total_fns = 0

    def add_labels(self, spans_cluster):
        for _, label in spans_cluster:
            if label not in self.labels:
                self.labels.add(label)
                self.tps[label] = 0.0
                self.fps[label] = 0.0
                self.fns[label] = 0.0

    def add(self, preds, golds):
        for pred, gold in zip(preds, golds):
            if pred is None and gold is None:
                continue
            pred = self.filter(pred)
            gold = self.filter(gold)

            P = set(pred)
            G = set(gold)

            if self.label_based:
                self.add_labels(pred)
                self.add_labels(gold)

                for cluster_p in pred:
                    predicted_label = cluster_p[1]
                    if cluster_p in gold:
                        self.tps[predicted_label] += 1
                        self.total_tps += 1
                    else:
                        self.fps[predicted_label] += 1
                        self.total_fps += 1

                for cluster_g in gold:
                    gold_label = cluster_g[1]
                    if cluster_g not in pred:
                        self.fns[gold_label] += 1
                        self.total_fns += 1
            else:
                self.total_tps += len(P & G)
                self.total_fps += len(P - G)
                self.total_fns += len(G - P)

    def print_debug(self):
        print('{:32}    {:5}  {:5}  {:5}    {:6.5f}  {:6.5f}  {:6.5f}'
                    .format('', self.total_tps, self.total_fps, self.total_fns, self.get_pr(), self.get_re(),
                            self.get_f1()))

    def get_pr(self):
        return self.total_tps / (self.total_tps + self.total_fps) if self.total_tps != 0 else 0.0

    def get_re(self):
        return self.total_tps / (self.total_tps + self.total_fns) if self.total_tps != 0 else 0.0

    def get_f1(self):
        f1 = 2.0 * self.get_pr() * self.get_re() / (
                self.get_pr() + self.get_re()) if self.get_pr() * self.get_re() != 0.0 else 0.0
        return f1


class MetricLinkingF1Soft:
    def __init__(self, mode='default', label_based=False):
        """

        :param mode:
        :param label_based: whether the tp/fp/fn stats are also gathered for each of the labels (links)
        """
        if mode == 'nills':
            self.filter = exclude_links_clusters
        elif mode == 'links':
            self.filter = exclude_nills_clusters
        else:
            self.filter = lambda x: x

        self.label_based = label_based
        self.clear()

    def clear(self):
        self.labels = set()
        self.p_tps = {}
        self.p_fps = {}
        self.r_tps = {}
        self.r_fns = {}

        self.total_p_tp = 0.0
        self.total_p_fp = 0.0
        self.total_r_tp = 0.0
        self.total_r_fn = 0.0

    def add_labels(self, spans_cluster):
        for _, label in spans_cluster:
            if label not in self.labels:
                self.labels.add(label)
                self.p_tps[label] = 0.0
                self.p_fps[label] = 0.0
                self.r_tps[label] = 0.0
                self.r_fns[label] = 0.0

    def add(self, preds, golds):
        for pred, gold in zip(preds, golds):
            pred = self.filter(pred)
            gold = self.filter(gold)

            if self.label_based:
                self.add_labels(pred)
                self.add_labels(gold)

            P = clusters_to_mentions(pred)
            G = clusters_to_mentions(gold)
            for cluster_p, entity_link in pred:
                pairs = cluster_to_mentions(cluster_p, entity_link)
                if len(pairs) > 0:
                    tp = len(pairs & G) / len(pairs)
                    fp = 1.0 - tp
                    if self.label_based:
                        self.p_tps[entity_link] += tp
                        self.p_fps[entity_link] += fp
                    self.total_p_tp += tp
                    self.total_p_fp += fp

            for cluster_g, entity_link in gold:
                pairs = cluster_to_mentions(cluster_g, entity_link)
                if len(pairs) > 0:
                    tp = len(pairs & P) / len(pairs)
                    fp = 1.0 - tp
                    if self.label_based:
                        self.r_tps[entity_link] += tp
                        self.r_fns[entity_link] += fp
                    self.total_r_tp += tp
                    self.total_r_fn += fp

    def print_debug(self):
        print('{:32}    {:5}  {:5}  {:5}  {:5}    {:6.5f}  {:6.5f}  {:6.5f}'
                    .format('', self.total_p_tp, self.total_p_fp, self.total_r_tp, self.total_r_fn, self.get_pr(),
                            self.get_re(), self.get_f1()))

    def get_pr(self):
        return self.total_p_tp / (self.total_p_tp + self.total_p_fp) if self.total_p_tp != 0 else 0.0

    def get_re(self):
        return self.total_r_tp / (self.total_r_tp + self.total_r_fn) if self.total_r_tp != 0 else 0.0

    def get_f1(self):
        f1 = 2.0 * self.get_pr() * self.get_re() / (
                self.get_pr() + self.get_re()) if self.get_pr() * self.get_re() != 0.0 else 0.0
        return f1


class MetricLinkingF1MentionSoft:
    def __init__(self, mode='default', label_based=False):
        """

        :param mode:
        :param label_based: whether the tp/fp/fn stats are also gathered for each of the labels (links)
        """
        if mode == 'nills':
            self.filter = exclude_links_clusters
        elif mode == 'links':
            self.filter = exclude_nills_clusters
        else:
            self.filter = lambda x: x

        self.label_based = label_based
        self.clear()

    def clear(self):
        self.labels = set()
        self.p_tps = {}
        self.p_fps = {}
        self.r_tps = {}
        self.r_fns = {}

        self.total_p_tp = 0.0
        self.total_p_fp = 0.0
        self.total_r_tp = 0.0
        self.total_r_fn = 0.0

    def add_labels(self, spans_cluster):
        for _, label in spans_cluster:
            if label not in self.labels:
                self.labels.add(label)
                self.p_tps[label] = 0.0
                self.p_fps[label] = 0.0
                self.r_tps[label] = 0.0
                self.r_fns[label] = 0.0

    def add(self, preds, golds):
        for pred, gold in zip(preds, golds):
            pred = self.filter(pred)
            gold = self.filter(gold)

            if self.label_based:
                self.add_labels(pred)
                self.add_labels(gold)

            # example of the values of different variables:
            #   pred = <class 'list'>: [((((1, 1), 'L1'), ((2, 2), 'L2'), ((3, 3), 'L1'), ((4, 4), 'L3'),
            #       ((5, 5), 'L4')), 'L1'), ((((6, 6), 'L2'),), 'L2')]
            #
            #   for gold there is no need in having individual links per mention since, by definition, it has a single link per entity
            #   gold = <class 'list'>: [(((1, 1), (2, 2), (3, 3), (4, 4), (5, 5)), 'L1'), (((6, 6),), 'L2')]
            #
            #   pairs = {set} <class 'set'>: {((1, 1), 'L1'), ((2, 2), 'L1'), ((5, 5), 'L1'), ((3, 3), 'L1'), ((4, 4), 'L1')}
            #   G = {set} <class 'set'>: {((1, 1), 'L1'), ((2, 2), 'L1'), ((5, 5), 'L1'), ((3, 3), 'L1'), ((4, 4), 'L1'), ((6, 6), 'L2')}
            #   P = {set} <class 'set'>: {((1, 1), 'L1'), ((2, 2), 'L1'), ((5, 5), 'L1'), ((3, 3), 'L1'), ((4, 4), 'L1'), ((6, 6), 'L2')}

            P = clusters_to_mentions_men_level(pred)
            P = self.filter(P)
            G = clusters_to_mentions(gold)
            for cluster_p, entity_link in pred:
                # pairs = cluster_to_mentions(cluster_p, entity_link)
                pairs = self.filter(cluster_p)
                if len(pairs) > 0:
                    tp = len(pairs & G) / len(pairs)
                    fp = 1.0 - tp
                    if self.label_based:
                        self.p_tps[entity_link] += tp
                        self.p_fps[entity_link] += fp
                    self.total_p_tp += tp
                    self.total_p_fp += fp

            for cluster_g, entity_link in gold:
                pairs = cluster_to_mentions(cluster_g, entity_link)
                if len(pairs) > 0:
                    tp = len(pairs & P) / len(pairs)
                    fp = 1.0 - tp
                    if self.label_based:
                        self.r_tps[entity_link] += tp
                        self.r_fns[entity_link] += fp
                    self.total_r_tp += tp
                    self.total_r_fn += fp

    def print_debug(self):
        print('{:32}    {:5}  {:5}  {:5}  {:5}    {:6.5f}  {:6.5f}  {:6.5f}'
                    .format('', self.total_p_tp, self.total_p_fp, self.total_r_tp, self.total_r_fn, self.get_pr(),
                            self.get_re(), self.get_f1()))

    def get_pr(self):
        return self.total_p_tp / (self.total_p_tp + self.total_p_fp) if self.total_p_tp != 0 else 0.0

    def get_re(self):
        return self.total_r_tp / (self.total_r_tp + self.total_r_fn) if self.total_r_tp != 0 else 0.0

    def get_f1(self):
        f1 = 2.0 * self.get_pr() * self.get_re() / (
                self.get_pr() + self.get_re()) if self.get_pr() * self.get_re() != 0.0 else 0.0
        return f1


# Look for top scoring link, and ignore nills
class MetricLinkingAccuracyMention:

    def __init__(self):
        self.clear()

    def clear(self):
        self.numer = 0
        self.denom = 0

    # def add(self, args):
    def add(self, preds, golds):
        """

        :param preds: example: [[((5, 7), ['NILL'], [1.4783543348312378]),
        ((41, 43), ['NILL'], [1.507637619972229]),
        ((45, 47), ['Charité', 'Barack_Obama', 'Angela_Merkel', 'NILL'], [0.564375638961792, 0.858609139919281, 0.7117089629173279, 1.3846601247787476]),
        ((49, 49), ['NILL', 'Humboldt_University_of_Berlin', 'Berlin'], [1.0195592641830444, 0.0850464478135109, 0.8966303467750549]),
        ((51, 52), ['NILL'], [1.9528312683105469]), ((53, 54), ['NILL'], [2.394974946975708]),
        ((57, 57), ['Humboldt_University_of_Berlin', 'NILL', 'Berlin'], [0.08927180618047714, 1.023781657218933, 0.9006481170654297])]]
        :param golds: example: [(5, 7, 'NILL'), (41, 43, 'NILL'), (45, 47, 'NILL'), (49, 49, 'Berlin'),
        (51, 52, 'NILL'), (53, 54, 'NILL'), (57, 57, 'Berlin')]
        :return:
        """
        # for pred, gold in zip(args['scores'], args['gold']):
        for pred, gold in zip(preds, golds):
            if pred is None and (gold is None or len(gold) == 0):
                continue
            P = []
            G = exclude_nills_mentions(gold)

            for (begin, end), candidates, scores in pred:
                best_score = - float("inf")
                best_candidate = None
                for candidate, score in zip(candidates, scores):
                    if candidate == 'NILL':
                        continue
                    elif score > best_score:
                        best_score = score
                        best_candidate = candidate
                if best_candidate is not None:
                    P.append((begin, end, best_candidate))

            P = set(P)
            G = set(G)

            self.numer += len(P & G)
            self.denom += len(G)

    def get_acc(self):
        acc = self.numer / self.denom if self.numer != 0 else 0.0
        return acc

    def print_debug(self):
        print('{:32}    {:5}  {:5}      {:6.5f}'.format('', self.numer, self.denom, self.get_acc()))


# Look for top scoring link, and ignore nills
class MetricLinkingAccuracyNoCandidatesMention:

    def __init__(self):
        self.clear()

    def clear(self):
        self.numer = 0
        self.denom = 0

    # def add(self, args):
    def add(self, preds, golds):
        """

        :param preds: example: [[(57, 57, 'Berlin'), (41, 43, 'NILL'), (49, 49, 'Berlin'), (45, 47, 'NILL'),
        (51, 52, 'NILL'), (53, 54, 'NILL'), (5, 7, 'NILL')]]
        :param golds: example: [[(5, 7, 'NILL'), (41, 43, 'NILL'), (45, 47, 'NILL'), (49, 49, 'Berlin'),
        (51, 52, 'NILL'), (53, 54, 'NILL'), (57, 57, 'Berlin')]]
        :return:
        """
        # for pred, gold in zip(args['scores'], args['gold']):
        for pred, gold in zip(preds, golds):
            if pred is None and (gold is None or len(gold) == 0):
                continue
            P = exclude_nills_mentions(pred)
            G = exclude_nills_mentions(gold)

            P = set(P)
            G = set(G)

            self.numer += len(P & G)
            self.denom += len(G)

    def get_acc(self):
        acc = self.numer / self.denom if self.numer != 0 else 0.0
        return acc

    def print_debug(self):
        print('{:32}    {:5}  {:5}      {:6.5f}'.format('', self.numer, self.denom, self.get_acc()))


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

    def add(self, preds, golds):
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

    def print_debug(self):
        for label in self.labels:
            tp, fp, fn = self.tp[label], self.fp[label], self.fn[label]
            pr = tp / (tp + fp) if tp != 0 else 0.0
            re = tp / (tp + fn) if tp != 0 else 0.0
            f1 = 2 * tp / (2 * tp + fp + fn) if tp != 0 else 0.0
            print('{:32}    {:5}  {:5}  {:5}    {:6.5f}  {:6.5f}  {:6.5f}'.format(label, tp, fp, fn, pr, re, f1))

        print('{:32}    {:5}  {:5}  {:5}    {:6.5f}  {:6.5f}  {:6.5f}'
                    .format('', self.total_tp, self.total_fp, self.total_fn, self.get_pr(), self.get_re(),
                            self.get_f1()))

    def get_pr(self):
        return self.total_tp / (self.total_tp + self.total_fp) if self.total_tp != 0 else 0.0

    def get_re(self):
        return self.total_tp / (self.total_tp + self.total_fn) if self.total_tp != 0 else 0.0

    def get_f1(self):
        f1 = 2.0 * self.get_pr() * self.get_re() / (
                self.get_pr() + self.get_re()) if self.get_pr() * self.get_re() != 0.0 else 0.0
        return f1


def clusters_to_mentions(cluster_spans):
    flatten_mentions = list()
    for curr_cluster, entity_type in cluster_spans:
        for curr_span in curr_cluster:
            flatten_mentions.append((curr_span, entity_type))
    return set(flatten_mentions)


def clusters_to_mentions_men_level(cluster_spans):
    flatten_mentions = list()
    for curr_cluster, _ in cluster_spans:
        for curr_span, span_entity_type in curr_cluster:
            flatten_mentions.append((curr_span, span_entity_type))
    return set(flatten_mentions)


def cluster_to_mentions(cluster_spans, entity_type):
    flatten_mentions = list()
    for curr_span in cluster_spans:
        flatten_mentions.append((curr_span, entity_type))
    return set(flatten_mentions)


class MetricF1Soft:

    def __init__(self):
        self.clear()

    def clear(self):
        self.labels = set()
        self.p_tps = {}
        self.p_fps = {}
        self.r_tps = {}
        self.r_fns = {}

        self.total_p_tp = 0.0
        self.total_p_fp = 0.0
        self.total_r_tp = 0.0
        self.total_r_fn = 0.0

    def add_labels(self, spans_cluster):
        for _, label in spans_cluster:
            if label not in self.labels:
                self.labels.add(label)
                self.p_tps[label] = 0.0
                self.p_fps[label] = 0.0
                self.r_tps[label] = 0.0
                self.r_fns[label] = 0.0

    def add(self, preds, golds):
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
                    self.total_p_tp += tp
                    self.total_p_fp += fp

            for cluster_g, entity_type in gold:
                pairs = cluster_to_mentions(cluster_g, entity_type)
                if len(pairs) > 0:
                    tp = len(pairs & P) / len(pairs)
                    fp = 1.0 - tp
                    self.r_tps[entity_type] += tp
                    self.r_fns[entity_type] += fp
                    self.total_r_tp += tp
                    self.total_r_fn += fp

    def print_debug(self):
        total_p_tp, total_p_fp, total_r_tp, total_r_fn = 0, 0, 0, 0

        for label in self.labels:
            p_tp, p_fp = self.p_tps[label], self.p_fps[label]
            r_tp, r_fn = self.r_tps[label], self.r_fns[label]
            pr = p_tp / (p_tp + p_fp) if p_tp != 0 else 0.0
            re = r_tp / (r_tp + r_fn) if r_tp != 0 else 0.0
            f1 = 2.0 * pr * re / (pr + re) if pr * re != 0.0 else 0.0

            print('{:24}    {:6.1f} / {:6.1f} = {:6.5f}    {:6.1f} / {:6.1f} = {:6.5f}    {:6.5f}'
                        .format(label, p_tp, p_fp, pr, r_tp, r_fn, re, f1))

            total_p_tp += p_tp
            total_p_fp += p_fp
            total_r_tp += r_tp
            total_r_fn += r_fn

        total_pr = total_p_tp / (total_p_tp + total_p_fp) if total_p_tp != 0 else 0.0
        total_re = total_r_tp / (total_r_tp + total_r_fn) if total_r_tp != 0 else 0.0
        total_f1 = 2.0 * total_pr * total_re / (total_pr + total_re) if total_pr * total_re != 0.0 else 0.0

        print('SOFT NER {:24}    {:6.1f} / {:6.1f} = {:6.5f}    {:6.1f} / {:6.1f} = {:6.5f}    {:6.5f}'
                    .format('', total_p_tp, total_p_fp, total_pr, total_r_tp, total_r_fn, total_re, total_f1))

    def get_pr(self):
        return self.total_p_tp / (self.total_p_tp + self.total_p_fp) if self.total_p_tp != 0 else 0.0

    def get_re(self):
        return self.total_r_tp / (self.total_r_tp + self.total_r_fn) if self.total_r_tp != 0 else 0.0

    def get_f1(self):
        f1 = 2.0 * self.get_pr() * self.get_re() / (
                self.get_pr() + self.get_re()) if self.get_pr() * self.get_re() != 0.0 else 0.0
        return f1


class MetricF1Hard:

    def __init__(self):
        self.clear()

    def clear(self):
        self.labels = set()
        self.tps = {}
        self.fps = {}
        self.fns = {}

        self.total_tps = 0.0
        self.total_fps = 0.0
        self.total_fns = 0.0

    def add_labels(self, spans_cluster):
        for _, label in spans_cluster:
            if label not in self.labels:
                self.labels.add(label)
                self.tps[label] = 0.0
                self.fps[label] = 0.0
                self.fns[label] = 0.0

    def add(self, preds, golds):
        for pred, gold in zip(preds, golds):
            self.add_labels(pred)
            self.add_labels(gold)

            for cluster_p in pred:
                predicted_label = cluster_p[1]
                if cluster_p in gold:
                    self.tps[predicted_label] += 1
                    self.total_tps += 1
                else:
                    self.fps[predicted_label] += 1
                    self.total_fps += 1

            for cluster_g in gold:
                gold_label = cluster_g[1]
                if cluster_g not in pred:
                    self.fns[gold_label] += 1
                    self.total_fns += 1

    def print_debug(self):
        total_tps, total_fps, total_fns = 0, 0, 0

        for label in self.labels:
            tps, fps, fns = self.tps[label], self.fps[label], self.fns[label]
            pr = tps / (tps + fps) if tps != 0 else 0.0
            re = tps / (tps + fns) if tps != 0 else 0.0
            f1 = 2.0 * pr * re / (pr + re) if pr * re != 0.0 else 0.0

            print('{:24}    {:6.1f} / {:6.1f} = {:6.5f}    {:6.1f} / {:6.1f} = {:6.5f}    {:6.5f}'
                        .format(label, tps, fps, pr, tps, fns, re, f1))

            total_tps += tps
            total_fps += fps
            total_fns += fns

        total_pr = total_tps / (total_tps + total_fps) if total_tps != 0 else 0.0
        total_re = total_tps / (total_tps + total_fns) if total_tps != 0 else 0.0
        total_f1 = 2.0 * total_pr * total_re / (total_pr + total_re) if total_pr * total_re != 0.0 else 0.0

        print('SOFT NER {:24}    {:6.1f} / {:6.1f} = {:6.5f}    {:6.1f} / {:6.1f} = {:6.5f}    {:6.5f}'
                    .format('', total_tps, total_fps, total_pr, total_tps, total_fns, total_re, total_f1))

    def get_pr(self):
        return self.total_tps / (self.total_tps + self.total_fps) if self.total_tps != 0 else 0.0

    def get_re(self):
        return self.total_tps / (self.total_tps + self.total_fns) if self.total_tps != 0 else 0.0

    def get_f1(self):
        f1 = 2.0 * self.get_pr() * self.get_re() / (
                self.get_pr() + self.get_re()) if self.get_pr() * self.get_re() != 0.0 else 0.0
        return f1


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

        print('coref\t{}\t{}'.format(self.name, f1))

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

    def __init__(self):
        self.clear()

    def clear(self):
        self.labels = set()
        self.p_tps = {}
        self.p_fps = {}
        self.r_tps = {}
        self.r_fns = {}

        self.total_p_tp = 0.0
        self.total_p_fp = 0.0
        self.total_r_tp = 0.0
        self.total_r_fn = 0.0

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
                self.total_p_tp += tp
                self.total_p_fp += fp

        for src_cluster, dst_cluster, rel in gold:
            pairs = to_pairs(src_cluster, dst_cluster, rel)
            if len(pairs) > 0:
                tp = len(pairs & P) / len(pairs)
                fn = 1.0 - tp

                self.r_tps[rel] += tp
                self.r_fns[rel] += fn
                self.total_r_tp += tp
                self.total_r_fn += fn

    def print_debug(self):
        total_p_tp, total_p_fp, total_r_tp, total_r_fn = 0, 0, 0, 0

        for label in self.labels:
            p_tp, p_fp = self.p_tps[label], self.p_fps[label]
            r_tp, r_fn = self.r_tps[label], self.r_fns[label]
            pr = p_tp / (p_tp + p_fp) if p_tp != 0 else 0.0
            re = r_tp / (r_tp + r_fn) if r_tp != 0 else 0.0
            f1 = 2.0 * pr * re / (pr + re) if pr * re != 0.0 else 0.0

            print('{:24}    {:6.1f} / {:6.1f} = {:6.5f}    {:6.1f} / {:6.1f} = {:6.5f}    {:6.5f}'
                        .format(label, p_tp, p_fp, pr, r_tp, r_fn, re, f1))

            total_p_tp += p_tp
            total_p_fp += p_fp
            total_r_tp += r_tp
            total_r_fn += r_fn

        total_pr = total_p_tp / (total_p_tp + total_p_fp) if total_p_tp != 0 else 0.0
        total_re = total_r_tp / (total_r_tp + total_r_fn) if total_r_tp != 0 else 0.0
        total_f1 = 2.0 * total_pr * total_re / (total_pr + total_re) if total_pr * total_re != 0.0 else 0.0

        print('{:24}    {:6.1f} / {:6.1f} = {:6.5f}    {:6.1f} / {:6.1f} = {:6.5f}    {:6.5f}'
                    .format('', total_p_tp, total_p_fp, total_pr, total_r_tp, total_r_fn, total_re, total_f1))

    def get_pr(self):
        return self.total_p_tp / (self.total_p_tp + self.total_p_fp) if self.total_p_tp != 0 else 0.0

    def get_re(self):
        return self.total_r_tp / (self.total_r_tp + self.total_r_fn) if self.total_r_tp != 0 else 0.0

    def get_f1(self):
        f1 = 2.0 * self.get_pr() * self.get_re() / (
                self.get_pr() + self.get_re()) if self.get_pr() * self.get_re() != 0.0 else 0.0
        return f1


class MetricRelationF1Hard:

    def __init__(self):
        self.clear()

    def clear(self):
        self.labels = set()
        self.tps = {}
        self.fps = {}
        self.fns = {}
        self.total_tps = 0
        self.total_fps = 0
        self.total_fns = 0

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
                self.total_tps += 1
            else:
                self.fps[pred_label] += 1
                self.total_fps += 1

        for rel_gold in gold:
            gold_label = rel_gold[2]
            if rel_gold not in pred:
                self.fns[gold_label] += 1
                self.total_fns += 1

    def print_debug(self):
        total_tps, total_fps, total_fns = 0, 0, 0

        for label in self.labels:
            tps, fps, fns = self.tps[label], self.fps[label], self.fns[label]
            pr = tps / (tps + fps) if tps != 0 else 0.0
            re = tps / (tps + fns) if tps != 0 else 0.0
            f1 = 2.0 * pr * re / (pr + re) if pr * re != 0.0 else 0.0

            print('{:24}    {:6.1f} / {:6.1f} = {:6.5f}    {:6.1f} / {:6.1f} = {:6.5f}    {:6.5f}'
                        .format(label, tps, fps, pr, tps, fns, re, f1))

            total_tps += tps
            total_fps += fps
            total_fns += fns

        total_pr = total_tps / (total_tps + total_fps) if total_tps != 0 else 0.0
        total_re = total_tps / (total_tps + total_fns) if total_tps != 0 else 0.0
        total_f1 = 2.0 * total_pr * total_re / (total_pr + total_re) if total_pr * total_re != 0.0 else 0.0

        print(
            'HARD RELATION {:24}    {:6.1f} / {:6.1f} = {:6.5f}    {:6.1f} / {:6.1f} = {:6.5f}    {:6.5f}'
                .format('', total_tps,
                        total_fps,
                        total_pr,
                        total_tps,
                        total_fns,
                        total_re,
                        total_f1))

    def get_f1(self):
        f1 = 2.0 * self.get_pr() * self.get_re() / (
                self.get_pr() + self.get_re()) if self.get_pr() * self.get_re() != 0.0 else 0.0
        return f1

    def get_pr(self):
        return self.total_tps / (self.total_tps + self.total_fps) if self.total_tps != 0 else 0.0

    def get_re(self):
        return self.total_tps / (self.total_tps + self.total_fns) if self.total_tps != 0 else 0.0


class MetricRelationF1Mention:

    def __init__(self):
        self.clear()

    def clear(self):
        self.labels = set()
        self.tps = {}
        self.fps = {}
        # self.r_tps = {}
        self.fns = {}
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0

    def add_labels(self, triples):
        for _, _, rel in triples:
            if rel not in self.labels:
                self.labels.add(rel)
                self.tps[rel] = 0
                self.fps[rel] = 0
                # self.r_tns[rel] = 0
                self.fns[rel] = 0

    def add_cluster_relations(self, pred, gold, doc_id=''):
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
                self.total_tp += tp
                self.total_fp += fp

        for src_cluster, dst_cluster, rel in gold:
            pairs = to_pairs(src_cluster, dst_cluster, rel)
            if len(pairs) > 0:
                # tp = len(pairs & P) # / len(pairs)
                fn = len(pairs) - len(pairs & P)
                self.fns[rel] += fn
                self.total_fn += fn

    def add_mention_relations(self, pred, gold, doc_id=''):
        self.add_labels(pred)
        self.add_labels(gold)

        # P = to_pairwise(pred)
        G = to_pairwise(gold)

        for src_mention, dst_mention, rel in pred:
            # pairs = to_pairs(src_cluster, dst_cluster, rel)
            # if len(pairs) > 0:
            tp = len({(src_mention, dst_mention, rel)} & G)
            fp = 1 - tp

            self.tps[rel] += tp
            self.fps[rel] += fp
            self.total_tp += tp
            self.total_fp += fp

        for src_cluster, dst_cluster, rel in gold:
            pairs = to_pairs(src_cluster, dst_cluster, rel)
            if len(pairs) > 0:
                fn = len(pairs) - len(pairs & pred)
                self.fns[rel] += fn
                self.total_fn += fn

    def get_if_exists(self, any_dict: dict, key, default):
        if key in any_dict:
            return any_dict[key]
        else:
            return default

    def print_debug(self):
        total_tp, total_fp, total_fn = 0, 0, 0

        for label in self.labels:
            tp, fp = self.tps[label], self.fps[label]
            fn = self.fns[label]
            pr = tp / (tp + fp) if tp != 0 else 0.0
            re = tp / (tp + fn) if tp != 0 else 0.0
            f1 = 2.0 * pr * re / (pr + re) if pr * re != 0.0 else 0.0

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

        print('{:24}    {:6.1f} / {:6.1f} = {:6.5f}    {:6.1f} / {:6.1f} = {:6.5f}    {:6.5f}'.format('', total_tp,
                                                                                                      total_fp,
                                                                                                      total_pr,
                                                                                                      total_tp,
                                                                                                      total_fn,
                                                                                                      total_re,
                                                                                                      total_f1))

    def get_f1(self):
        f1 = 2.0 * self.get_pr() * self.get_re() / (
                self.get_pr() + self.get_re()) if self.get_pr() * self.get_re() != 0.0 else 0.0
        return f1

    def get_pr(self):
        return self.total_tp / (self.total_tp + self.total_fp) if self.total_tp != 0 else 0.0

    def get_re(self):
        return self.total_tp / (self.total_tp + self.total_fn) if self.total_tp != 0 else 0.0


class EvaluatorCPN:

    def __init__(self, debug=False):
        self.debug = debug
        self.tags_soft = MetricF1Soft()
        self.tags_hard = MetricF1Hard()
        self.tags_mention = MetricF1()
        self.tags_mention_expanded = MetricF1()

        self.coref_muc = MetricCoref('muc', MetricCoref.muc)
        self.coref_bcubed = MetricCoref('bcubed', MetricCoref.b_cubed)
        self.coref_bcubed_singleton_men = MetricCoref('bcubed singleton mention-based',
                                                      MetricCoref.b_cubed_singleton_mentions)
        self.coref_bcubed_singleton_ent = MetricCoref('bcubed singleton entity-based',
                                                      MetricCoref.b_cubed_singleton_entities)

        self.coref_ceafe = MetricCoref('ceafe', MetricCoref.ceafe)
        self.coref_ceafm_singleton_men = MetricCoref('ceafm-singleton mention-based',
                                                     MetricCoref.ceafe_singleton_mentions)
        self.coref_ceafe_singleton_ent = MetricCoref('ceafe-singleton entity-based',
                                                     MetricCoref.ceafe_singleton_entities)

        self.rels_soft = MetricRelationF1()
        self.rels_hard = MetricRelationF1Hard()
        self.rels_mention = MetricRelationF1Mention()
        self.rels_mention_expanded = MetricRelationF1Mention()

        self.links_mention_all = MetricLinkingF1Mention(mode='default')
        self.links_mention_nill = MetricLinkingF1Mention(mode='nills')
        self.links_mention_links = MetricLinkingF1Mention(mode='links')

        self.links_mention_ent_all = MetricLinkingF1Mention(mode='default')
        self.links_mention_ent_nill = MetricLinkingF1Mention(mode='nills')
        self.links_mention_ent_links = MetricLinkingF1Mention(mode='links')

        self.links_hard_ent_links = MetricLinkingF1Hard(mode='links')
        self.links_nill_hard_ent_links = MetricLinkingF1Hard(mode='nills')
        self.links_soft_ent_links = MetricLinkingF1Soft(mode='links')
        self.links_mentionsoft_ent_links = MetricLinkingF1MentionSoft(mode='links')

        self.links_accuracy = MetricLinkingAccuracyMention()
        self.links_accuracy_no_candidates = MetricLinkingAccuracyNoCandidatesMention()
        self.links_accuracy_ent_no_cand = MetricLinkingAccuracyNoCandidatesMention()

    def add(self, pred, gold):
        P_links = decode_links(pred, is_pred=True)  # TODO: we are here thinking about this!
        G_links = decode_links(gold)

        P_links_from_ents = decode_links_from_ents(pred, is_pred=True)
        G_links_from_ents = decode_links_from_ents(gold)

        P_links_scores = decode_link_scores(pred)

        self.links_mention_all.add([P_links], [G_links])
        self.links_mention_nill.add([P_links], [G_links])
        self.links_mention_links.add([P_links], [G_links])

        self.links_mention_ent_all.add([P_links_from_ents], [G_links_from_ents])
        self.links_mention_ent_nill.add([P_links_from_ents], [G_links_from_ents])
        self.links_mention_ent_links.add([P_links_from_ents], [G_links_from_ents])

        self.links_accuracy.add([P_links_scores], [G_links])

        self.links_accuracy_ent_no_cand.add([P_links_from_ents], [G_links_from_ents])

        self.links_accuracy_no_candidates.add([P_links], [G_links])

        P = decode_spans(pred)
        G = decode_spans_expanded(gold)
        P_expanded = decode_spans_expanded(pred)

        if self.debug:
            print(pred['id'])
            content = pred['content']
            print("P:", P)
            print("G:", G)
            print("P:", [[content[begin:end] for begin, end in cluster] for cluster in P])
            print("G:", [[content[begin:end] for begin, end in cluster] for cluster in G])

        self.tags_mention.add([P], [G])
        self.tags_mention_expanded.add([P_expanded], [G])

        P_links_cluster = decode_links_clusters(pred, is_pred=True)
        G_links_cluster = decode_links_clusters(gold, is_pred=False)
        self.links_soft_ent_links.add([P_links_cluster], [G_links_cluster])

        P_links_mention_cluster = decode_links_mention_clusters(pred)
        # G_links_mention_cluster = decode_links_mention_clusters(gold, is_pred=False)

        self.links_mentionsoft_ent_links.add([P_links_mention_cluster], [G_links_cluster])
        self.links_hard_ent_links.add([P_links_cluster], [G_links_cluster])
        self.links_nill_hard_ent_links.add([P_links_cluster], [G_links_cluster])

        P_cluster = decode_spans_clusters(pred)
        G_cluster = decode_spans_clusters(gold)
        self.tags_soft.add([P_cluster], [G_cluster])

        self.tags_hard.add([P_cluster], [G_cluster])

        P = decode_coref(pred)
        G = decode_coref(gold)
        self.coref_muc.add(P, G)
        self.coref_bcubed.add(P, G)
        self.coref_bcubed_singleton_men.add(P, G)
        self.coref_bcubed_singleton_ent.add(P, G)
        self.coref_ceafe.add(P, G)
        self.coref_ceafm_singleton_men.add(P, G)
        self.coref_ceafe_singleton_ent.add(P, G)

        P_expanded = decode_relations_expanded(pred)
        P = decode_relations(pred)
        G = decode_relations_expanded(gold)
        self.rels_soft.add(P_expanded, G)
        self.rels_hard.add(P_expanded, G)
        self.rels_mention_expanded.add_cluster_relations(P_expanded, G)
        self.rels_mention.add_mention_relations(P, G)

        # (kzaporoj) - previous code that doesn't differentiate between expanded and mention-based relation predictions
        # if 'mention_relations' not in pred:
        #     # uses relation mentions on cluster level if no "relation_mentions" exist in predicted file, this is done
        #     # by expanding the cluster-based relations into mention-based relations by doing cartesian product between
        #     # mentions of related clusters.
        #     self.rels_mention.add_cluster_relations(P, G)
        # else:
        #     P = decode_relations(pred)
        #     self.rels_mention.add_mention_relations(P, G)

    def printInfo(self):
        print("## Coreference")
        print('\t{:.<30} {}'.format('muc-f1:', self.coref_muc.get_f1()))
        print()
        avg = sum([self.coref_muc.get_f1(), self.coref_bcubed.get_f1(), self.coref_ceafe.get_f1()]) / 3
        print('\t{:.<30} {}'.format('not-singleton-b-cubed-f1:', self.coref_bcubed.get_f1()))

        print('\t{:.<30} {}'.format('not-singleton-ceafe-f1:', self.coref_ceafe.get_f1()))
        print('\t{:.<30} {}'.format('not-singleton-avg-f1:', avg))
        print()
        print('\t{:.<30} {}'.format('singleton-b-cubed-f1:', self.coref_bcubed_singleton_men.get_f1()))
        print('\t{:.<30} {}'.format('singleton-ceafe-f1:', self.coref_ceafe_singleton_ent.get_f1()))
        avg = sum([self.coref_muc.get_f1(), self.coref_bcubed_singleton_men.get_f1(),
                   self.coref_ceafe_singleton_ent.get_f1()]) / 3
        print('\t{:.<30} {}'.format('singleton-avg-f1:', avg))

        print('## NER')
        print('\t{:.<30} {}'.format('mention-ner-f1:', self.tags_mention.get_f1()))
        print('\t{:.<30} {}'.format('mention-ner-f1 (expanded):', self.tags_mention_expanded.get_f1()))
        print('\t{:.<30} {}'.format('soft-ner-f1:', self.tags_soft.get_f1()))
        print('\t{:.<30} {}'.format('hard-ner-f1:', self.tags_hard.get_f1()))

        print('## Relations')
        print('\t{:.<30} {}'.format('mention-rel-f1:', self.rels_mention.get_f1()))
        print('\t{:.<30} {}'.format('mention-rel-f1 (expanded):', self.rels_mention_expanded.get_f1()))
        print('\t{:.<30} {}'.format('hard-rel-f1:', self.rels_hard.get_f1()))
        print('\t{:.<30} {}'.format('soft-rel-f1:', self.rels_soft.get_f1()))

        print('## Links')
        print('\t{:.<30} {}'.format('mention-links-f1 (all):', self.links_mention_all.get_f1()))
        print('\t{:.<30} {}'.format('mention-links-f1 (nills):', self.links_mention_nill.get_f1()))
        print('\t{:.<30} {}'.format('mention-links-f1 (links):', self.links_mention_links.get_f1()))

        print('\t{:.<30} {}'.format('mention-links-f1 ent (all):', self.links_mention_ent_all.get_f1()))
        print('\t{:.<30} {}'.format('mention-links-f1 ent (nills):', self.links_mention_ent_nill.get_f1()))
        print('\t{:.<30} {}'.format('mention-links-f1 ent (links):', self.links_mention_ent_links.get_f1()))

        print('\t{:.<30} {}'.format('hard-links-f1 ent (links):', self.links_hard_ent_links.get_f1()))
        print('\t{:.<30} {}'.format('hard-nill-f1 ent (links):', self.links_nill_hard_ent_links.get_f1()))
        print('\t{:.<30} {}'.format('soft-links-f1 ent (links):', self.links_soft_ent_links.get_f1()))
        print('\t{:.<30} {}'.format('softmention-links-f1 ent (links):', self.links_mentionsoft_ent_links.get_f1()))

        print('\t{:.<30} {}'.format('mention-accuracy:', self.links_accuracy.get_acc()))
        print('\t{:.<30} {}'.format('mention-accuracy (no candidates method):',
                                    self.links_accuracy_no_candidates.get_acc()))
        print('\t{:.<30} {}'.format('mention-accuracy (no candidates method) ent:',
                                    self.links_accuracy_ent_no_cand.get_acc()))

        if self.debug:
            print('## debug mention-based ner')
            self.tags_mention.print_debug()
            print('## debug soft ner')
            self.tags_soft.print_debug()
            print('## debug hard ner')
            self.tags_hard.print_debug()

            print('## debug mention-based relations')
            self.rels_mention.print_debug()
            print('## debug soft relations')
            self.rels_soft.print_debug()
            print('## debug hard relations')
            self.rels_hard.print_debug()

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

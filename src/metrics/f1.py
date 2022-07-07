import logging

import numpy as np
import torch

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger()


def decode_multiner(targets, sequence_lengths, labels):
    decoded = []
    tmp = targets.data.cpu()
    for b, length in enumerate(sequence_lengths.tolist()):
        segments = []

        enabled = np.argwhere(tmp[b, 0:length, :].numpy() > 0).tolist()
        state_begin = {}
        state_end = {}
        for i, (pos, idx) in enumerate(enabled):
            label = labels[idx][2:]
            if labels[idx].startswith('B-'):
                if label in state_begin:
                    segments.append((state_begin[label], state_end[label], label))
                state_begin[label] = pos
                state_end[label] = pos + 1
            elif labels[idx].startswith('I-'):
                if label in state_end and state_end[label] == pos:
                    state_end[label] = pos + 1
        for label in state_begin.keys():
            segments.append((state_begin[label], state_end[label], label))

        decoded.append(segments)
    return decoded


class MetricF1:

    def __init__(self, labels):
        self.labels = labels
        self.clear()

    def clear(self):
        self.tp = {l: 0 for l in self.labels}
        self.fp = {l: 0 for l in self.labels}
        self.fn = {l: 0 for l in self.labels}
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0

    def update(self, preds, golds):
        for pred, gold in zip(preds, golds):
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
                logger.debug(
                    '{:32}    {:5}  {:5}  {:5}    {:6.5f}  {:6.5f}  {:6.5f}'.format(label, tp, fp, fn, pr, re, f1))

        logger.info('{:32}    {:5}  {:5}  {:5}    {:6.5f}  {:6.5f}  {:6.5f}'.format('', self.total_tp, self.total_fp,
                                                                                    self.total_fn, self.pr(), self.re(),
                                                                                    self.f1()))

    def pr(self):
        return self.total_tp / (self.total_tp + self.total_fp) if self.total_tp != 0 else 0.0

    def re(self):
        return self.total_tp / (self.total_tp + self.total_fn) if self.total_tp != 0 else 0.0

    def f1(self):
        return 2 * self.total_tp / (2 * self.total_tp + self.total_fp + self.total_fn) if self.total_tp != 0 else 0.0


def decision_function_softmax(logits, sequence_lengths, mask):
    return torch.max(logits, 2)[1]


#

def decode_segments(targets, sequence_lengths, labels):
    if isinstance(targets, (list,)):
        indices = targets
    else:
        indices = []
        for length, data in zip(sequence_lengths.tolist(), targets.tolist()):
            indices.append(data[:length])

    outputs = []
    for lst in indices:
        data = [labels[x] for x in lst]

        output = []
        start = -1
        type = None
        for pos, target in enumerate(data):
            if target.startswith('B-'):
                if start >= 0:
                    output.append((start, pos, type))
                start = pos
                type = target[2:]
            elif target == 'O':
                if start >= 0:
                    output.append((start, pos, type))
                    start = -1
                    type = None

        if start >= 0:
            output.append((start, len(data), type))
        outputs.append(output)
    return outputs


class MetricNERF1:

    def __init__(self, task, bio_labels, verbose=True, decision_function=decision_function_softmax):
        self.task = task
        self.evaluator = MetricF1([label[2:] for label in bio_labels if label.startswith('B-')])
        self.bio_labels = bio_labels
        self.epoch = 0
        self.max_f1 = 0
        self.max_iter = 0
        self.decision_function = decision_function
        self.verbose = verbose

    def step(self):
        self.evaluator.clear()
        self.epoch += 1

    def update(self, logits, targets, sequence_lengths, mask):
        predictions = self.decision_function(logits, sequence_lengths, mask)
        p = decode_segments(predictions, sequence_lengths, self.bio_labels)
        g = decode_segments(targets, sequence_lengths, self.bio_labels)
        self.evaluator.update(p, g)

    def update2(self, args, metadata={}):
        self.update(args['scores'], args['targets'], args['sequence_lengths'], args['mask'])

    def print(self, dataset_name, details=False):
        f1 = self.evaluator.f1()

        if f1 > self.max_f1:
            self.max_f1 = f1
            self.max_iter = self.epoch
        stall = self.epoch - self.max_iter

        self.evaluator.print(self.verbose)
        logger.info('EVAL-NER\t{}-{}\tcurr-iter: {}\tcurr-f1: {}\tmax-iter: {}\tmax-f1: {}\tstall: {}'
                    .format(dataset_name, self.task, self.epoch, f1, self.max_iter, self.max_f1, stall))

    def log(self, tb_logger, dataset_name):
        tb_logger.log_value('{}-{}/f1'.format(dataset_name, self.task), self.evaluator.f1(), self.epoch)


class MetricMultiNERF1:

    def __init__(self, task, bi_labels):
        self.task = task
        self.evaluator = MetricF1([label[2:] for label in bi_labels if label.startswith('B-')])
        self.bi_labels = bi_labels
        self.epoch = 0
        self.max_f1 = 0
        self.max_iter = 0

    def step(self):
        self.evaluator.clear()
        self.epoch += 1

    def update(self, logits, targets, args, metadata={}):
        sequence_lengths, mask = args['sequence_lengths'], args['mask']
        p = decode_multiner(logits, sequence_lengths, self.bi_labels)
        g = decode_multiner(targets, sequence_lengths, self.bi_labels)
        self.evaluator.update(p, g)

    def update2(self, args, metadata={}):
        self.update(args['pred'], args['gold'], metadata=metadata)

    def print(self, dataset_name, details=False):
        f1 = self.evaluator.f1()

        if f1 > self.max_f1:
            self.max_f1 = f1
            self.max_iter = self.epoch
        stall = self.epoch - self.max_iter

        self.evaluator.print(details)
        logger.info('EVAL-NER\t{}-{}\tcurr-iter: {}\tcurr-f1: {}\tmax-iter: {}\tmax-f1: {}\tstall: {}'
                    .format(dataset_name, self.task, self.epoch, f1, self.max_iter, self.max_f1, stall))

    def log(self, tb_logger, dataset_name):
        tb_logger.log_value('{}/{}-f1'.format(dataset_name, self.task), self.evaluator.f1(), self.epoch)


def decode_relations(targets, sequence_lengths, labels):
    tmp = targets.data.cpu()
    relations = []
    for b, length in enumerate(sequence_lengths.tolist()):
        rels = []
        for r, s, o in np.argwhere(tmp[b, :, 0:length, 0:length].numpy() > 0).tolist():
            rels.append((s, o, labels[r]))
        relations.append(rels)
    return relations


class MetricRelationF1:

    def __init__(self, name, labels):
        self.task = name
        self.evaluator = MetricF1(labels)
        self.labels = labels
        self.iter = 0
        self.max_f1 = 0
        self.max_iter = 0

    def step(self):
        self.evaluator.clear()
        self.iter += 1

    def update(self, logits, targets, args, metadata={}):
        if logits.size() != targets.size():
            raise BaseException('invalid dims %s %s' % (logits.size(), targets.size()))
        concept_lengths = args['concept_lengths']
        p = decode_relations(logits, concept_lengths, self.labels)
        g = decode_relations(targets, concept_lengths, self.labels)
        self.evaluator.update(p, g)

    def update2(self, args, metadata={}):
        logger.warning('TODO')

    def print(self, dataset_name, details=False):
        f1 = self.evaluator.f1()

        if f1 > self.max_f1:
            self.max_f1 = f1
            self.max_iter = self.iter
        stall = self.iter - self.max_iter

        self.evaluator.print(details)
        logger.info('EVAL-REL\tdataset: {}\tcurr-iter: {}\tcurr-f1: {}\tmax-iter: {}\tmax-f1: {}\tstall: {}'.format(
            dataset_name, self.iter, f1, self.max_iter, self.max_f1, stall))

    def log(self, tb_logger, dataset_name):
        tb_logger.log_value('{}/{}-f1'.format(dataset_name, self.task), self.evaluator.f1(), self.iter)


class MetricSpanNER:

    def __init__(self, task, verbose=False, labels=None, bio_labels=None):
        if labels is not None:
            self.labels = labels
        elif bio_labels is not None:
            self.labels = [label[2:] for label in bio_labels if label.startswith('B-')]
        else:
            raise BaseException("no labels")

        self.task = task
        self.evaluator = MetricF1(self.labels)
        self.iter = 0
        self.max_f1 = 0
        self.max_iter = 0
        self.verbose = verbose

    def step(self):
        self.evaluator.clear()
        self.iter += 1

    def update(self, pred, gold, metadata={}):
        self.evaluator.update(pred, gold)

    def update2(self, args, metadata={}):
        self.update(args['pred'], args['gold'], metadata=metadata)

    def print(self, dataset_name, details=False):
        f1 = self.evaluator.f1()

        if f1 > self.max_f1:
            self.max_f1 = f1
            self.max_iter = self.iter
        stall = self.iter - self.max_iter

        self.evaluator.print(self.verbose)

        logger.info('EVAL-NER\t{}-{}\tcurr-iter: {}\tcurr-f1: {}\tmax-iter: {}\tmax-f1: {}\tstall: {}'
                    .format(dataset_name, self.task, self.iter, f1, self.max_iter, self.max_f1, stall))

    def log(self, tb_logger, dataset_name):
        tb_logger.log_value('metrics/f1', self.evaluator.f1(), self.iter)

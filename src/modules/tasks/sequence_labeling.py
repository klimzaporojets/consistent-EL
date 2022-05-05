import torch
import torch.nn as nn
import numpy as np
from metrics.f1 import MetricNERF1, MetricMultiNERF1, decode_multiner, MetricSpanNER
from metrics.misc import MetricObjective
from modules.conditional_random_field import ConditionalRandomField
from util.sequence import get_mask_from_sequence_lengths

class TaskCRF(nn.Module):

    def __init__(self, name, dim_input, labels):
        super(TaskCRF, self).__init__()
        self.name = name
        self.linear = nn.Linear(dim_input, labels.size)
        self.crf = ConditionalRandomField(labels.size)
        self.labels = labels.tolist()

    def forward(self, inputs, targets, sequence_lengths, mask):
        logits = self.linear(inputs)

        values = self.crf(logits, targets, sequence_lengths, mask)

        return values, logits

    def create_metrics(self):
        def decision_function(logits, sequence_lengths, mask):
            out = self.crf.predict(logits, sequence_lengths, mask)
            # print("PRED:", out)
            return out

        return MetricNERF1(self.name, self.labels, decision_function)


class TaskNerSoftmax(nn.Module):

    def __init__(self, name, dim_input, labels, config):
        super(TaskNerSoftmax, self).__init__()
        self.name = name
        self.linear = nn.Linear(dim_input, labels.size)
        self.loss = nn.CrossEntropyLoss(reduce=False)
        self.labels = labels.tolist()
        self.weight = config.get('weight', 1.0) / len(self.labels) if config['normalize'] else config.get('weight', 1.0)
        self.enabled = config['enabled']
        self.verbose = config.get('verbose', True)

    def forward(self, inputs, targets=None, sequence_lengths=None, mask=None, predict=False):
        output = {}

        logits = self.linear(inputs)

        if targets is not None:
            batch_size, sequence_length, num_tags = logits.size()

            values = self.loss(logits.view(-1, num_tags), targets.view(-1)).view_as(targets)
            masked_values = values * mask.float()
            output['loss'] = masked_values.sum()

        if predict:
            # TODO: put decode code here, and evaluate on spans
            output['sequence_lengths'] = sequence_lengths
            output['mask'] = mask
            output['scores'] = logits
            output['targets'] = targets
            # _, predictions = torch.max(logits, 2)
            # output = []
            # for length, data in zip(sequence_lengths.tolist(), predictions.tolist()):
            #     output.append(data[:length])
            # print('pred:', output)

        return output

    def create_metrics(self):
        return [MetricNERF1(self.name, self.labels, verbose=self.verbose), MetricObjective(self.name)] if self.enabled else []


class TaskNerMulti(nn.Module):

    def __init__(self, name, dim_input, labels, config):
        super(TaskNerMulti, self).__init__()
        self.name = name
        self.linear = nn.Linear(dim_input, labels.size)
        self.loss = nn.BCEWithLogitsLoss(reduce=False)
        self.labels = labels.tolist()
        W0 = config.get('weight', 1.0)
        self.weight = W0 / len(self.labels) if 'normalize' in config else W0
        self.enabled = config['enabled']
        self.verbose = config.get('verbose', True)
        print("Task {}: enabled={} weight={}".format(self.name, self.enabled, self.weight))

    def set_weight(self, W0):
        self.weight = W0 / len(self.labels)
        print("Task {} weight: {}".format(self.name, self.weight))

    def forward(self, inputs, targets, sequence_lengths, mask, predict=False):
        output = {}

        logits = self.linear(inputs)
        
        if targets is not None:
            num_tags = logits.size(2)

            losses = self.loss(logits.view(-1, num_tags), targets.view(-1, num_tags)).view_as(targets)
            losses = losses.sum(-1)
            ner_masked_losses = losses * mask.float()
            output['loss'] = self.weight * ner_masked_losses.sum()

        if predict:
            output['pred'] = decode_multiner(logits, sequence_lengths, self.labels)
            output['gold'] = decode_multiner(targets, sequence_lengths, self.labels)

        return output

    # def predict(self, logits, sequence_lengths, mask):
    #     tmp = logits.data.cpu()
    #     output = []
    #     for b, length in enumerate(sequence_lengths.tolist()):
    #         print(np.argwhere(tmp[b,0:length,:].numpy()>0))
    #         sequence = []
    #         for i in range(length):
    #             sequence.append([])
    #         output.append(sequence)
    #     return output

    def create_metrics(self):
        # return [MetricMultiNERF1(self.name, self.labels), MetricObjective(self.name)]  if self.enabled else []
        return [MetricSpanNER(self.name, bio_labels=self.labels, verbose=self.verbose), MetricObjective(self.name)]  if self.enabled else []

# don't model recursive spans
def decode_naive(sequence):
    output = []
    span_begin = None
    for pos, label in enumerate(sequence):
        if label == 'B':
            if span_begin is not None:
                output.append((span_begin, pos))
            span_begin = pos
        elif label == 'I':
            if span_begin is None:
                span_begin = pos
        elif label == 'O':
            if span_begin is not None:
                output.append((span_begin, pos))
            span_begin = None
        else:
            # do as if it is an I
            if span_begin is None:
                span_begin = pos

    if span_begin is not None:
        output.append((span_begin, len(sequence)))
    return output

def decode_recursive(indices, sequence_lengths, labels):
    predictions = [indices[b,0:length] for b, length in enumerate(sequence_lengths.tolist())]
    predictions = [[labels[l] for l in prediction] for prediction in predictions]
    spans = [decode_naive(x) for x in predictions]
    return spans

class TaskNerRecursive(nn.Module):

    def __init__(self, name, dim_input, labels, config):
        super(TaskNerRecursive, self).__init__()
        self.name = name
        self.linear = nn.Linear(dim_input, labels.size)
        self.loss = nn.CrossEntropyLoss(reduce=False)
        self.labels = labels.tolist()
        self.weight = config.get('weight', 1.0) / len(self.labels)
        self.enabled = config['enabled']
        print("TaskNerRecursive {}: enabled={} weight={}".format(self.name, self.enabled, self.weight))

    def forward(self, inputs, targets, sequence_lengths, mask, predict=False):
        output = {}

        logits = self.linear(inputs)
        
        if targets is not None:
            batch_size, sequence_length, num_tags = logits.size()

            values = self.loss(logits.view(-1, num_tags), targets.view(-1)).view_as(targets)
            masked_values = values * mask.float()
            output['loss'] = masked_values.sum()

        pred_spans = decode_recursive(logits.max(dim=-1)[1], sequence_lengths, self.labels)
        gold_spans = decode_recursive(targets, sequence_lengths, self.labels)

        output['pred'] = [[(b,e,'other') for b,e in spans] for spans in pred_spans]
        output['gold'] = [[(b,e,'other') for b,e in spans] for spans in gold_spans]

        # if predict:
        #     output['pred'] = decode_multiner(logits, sequence_lengths, self.labels)
        #     output['gold'] = decode_multiner(targets, sequence_lengths, self.labels)

        return output

    def create_metrics(self):
        # return [MetricMultiNERF1(self.name, self.labels), MetricObjective(self.name)]  if self.enabled else []
        # MetricSpanNER(self.name, bio_labels=self.labels),
        return [ MetricObjective(self.name) ]  if self.enabled else []

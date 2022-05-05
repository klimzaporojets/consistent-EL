import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics.f1 import MetricSpanNER
from metrics.misc import MetricObjective


def create_bie_filters(max_span_length):
    filters = torch.zeros(max_span_length, 3, max_span_length)
    for width in range(1, max_span_length + 1):
        filters[width - 1, 0, 0] = 1
        for i in range(1, width - 1):
            filters[width - 1, 1, i] = 1
        filters[width - 1, 2, width - 1] = 1
    return filters


def create_span_mask(length, width, sequence_lengths):
    b = torch.arange(length)
    w = torch.arange(width)
    mask = b.unsqueeze(1) + w.unsqueeze(0)
    mask = mask.unsqueeze(0).to(sequence_lengths) < sequence_lengths.unsqueeze(-1).unsqueeze(-1)
    return mask


# enable more spans by tweaking a weight on positive class
class TaskSpanGenerator1(nn.Module):

    def __init__(self, name, dim_input, config):
        super(TaskSpanGenerator1, self).__init__()
        self.name = name
        self.enabled = config['enabled']
        self.max_span_length = config['max_span_length']
        self.bie = nn.Linear(dim_input, 3)
        self.filters = create_bie_filters(self.max_span_length).cuda()
        self.weight = config['weight']
        pos_weights = torch.ones([1]) * config['pos_weight']
        print('->', pos_weights)
        self.loss = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weights)
        self.bias = nn.Parameter(torch.Tensor(self.max_span_length))

        self.total_pred_spans = 0
        self.total_gold_spans = 0

    def forward(self, inputs, targets, sequence_lengths, mask, gold_spans, predict=False):
        output = {}

        scores = self.bie(inputs)
        logits = F.conv1d(F.pad(scores.permute(0, 2, 1), (0, self.max_span_length - 1)), self.filters,
                          bias=self.bias.data)
        logits = logits.permute(0, 2, 1)
        # print(scores.size(), '->', out.size())
        # logits = self.test(inputs).permute(0,2,1)

        # logits = self.hihi(inputs.permute(0,2,1)).permute(0,2,1)

        mask = create_span_mask(inputs.size(1), self.max_span_length, sequence_lengths).float()

        t = torch.zeros_like(logits)
        for i, instance in enumerate(gold_spans):
            self.total_gold_spans += len(instance)
            for begin, end in instance:
                if end - begin < self.max_span_length:
                    t[i, begin, end - begin] = 1

        obj = self.loss((logits * mask).view(-1, 1), t.view(-1, 1))

        output['loss'] = obj.sum() * self.weight

        if True:
            predictions = torch.nonzero(logits * mask > 0).tolist()
            preds = [list() for _ in gold_spans]
            for batch, begin, width in predictions:
                preds[batch].append((begin, begin + width))
            self.total_pred_spans += len(predictions)
            pred_spans = preds

        output['pred'] = [[(b, e + 1, 'span') for b, e in spans] for spans in pred_spans]
        output['gold'] = [[(b, e + 1, 'span') for b, e in spans] for spans in gold_spans]

        # print('p:', output['pred'][0])
        # print('g:', output['gold'][0])

        # pos = 1
        # print("span-0:", scores[0,pos,0].item() + scores[0,pos,2].item(), out[0,0,pos].item())
        # print("span-1:", scores[0,pos,0].item() + scores[0,pos+1,2].item(), out[0,1,pos].item())
        # print("span-2:", scores[0,pos,0].item() + scores[0,pos+1,1].item() + scores[0,pos+2,2].item(), out[0,2,pos].item())
        # print("span-3:", scores[0,pos,0].item() + scores[0,pos+1,1].item() + scores[0,pos+2,1].item() + scores[0,pos+3,2].item(), out[0,3,pos].item())
        # print("span-4:", scores[0,pos,0].item() + scores[0,pos+1,1].item() + scores[0,pos+2,1].item() + scores[0,pos+3,1].item() + scores[0,pos+4,2].item(), out[0,4,pos].item())

        return output

    def create_metrics(self):
        return [MetricSpanNER(self.name, labels=['span']), MetricObjective(self.name)]

    def tick(self, dataset_name):
        print("{}-span-generator: {} / {} = {}".format(dataset_name, self.total_pred_spans, self.total_gold_spans,
                                                       self.total_pred_spans / self.total_gold_spans))
        self.total_pred_spans = 0
        self.total_gold_spans = 0


# enable fraction based on number of tokens
class TaskSpanGenerator2(nn.Module):

    def __init__(self, name, dim_input, config):
        super(TaskSpanGenerator2, self).__init__()
        self.name = name
        self.enabled = config['enabled']
        self.max_span_length = config['max_span_length']
        self.bie = nn.Linear(dim_input, 3)
        self.filters = create_bie_filters(self.max_span_length).cuda()
        self.weight = config['weight']
        self.loss = nn.BCEWithLogitsLoss(reduction='none')
        self.bias = nn.Parameter(torch.Tensor(self.max_span_length))
        self.spans_per_token = config['spans_per_token']

        self.total_obj = 0
        self.total_pred_spans = 0
        self.total_gold_spans = 0

    def forward(self, inputs, targets, sequence_lengths, mask, gold_spans, predict=False):
        output = {}

        scores = self.bie(inputs)
        logits = F.conv1d(F.pad(scores.permute(0, 2, 1), (0, self.max_span_length - 1)), self.filters,
                          bias=self.bias.data)
        logits = logits.permute(0, 2, 1)
        # print(scores.size(), '->', out.size())
        # logits = self.test(inputs).permute(0,2,1)

        # logits = self.hihi(inputs.permute(0,2,1)).permute(0,2,1)

        mask = create_span_mask(inputs.size(1), self.max_span_length, sequence_lengths).float()

        t = torch.zeros_like(logits)
        for i, instance in enumerate(gold_spans):
            for begin, end in instance:
                if end - begin < self.max_span_length:
                    t[i, begin, end - begin] = 1

        obj = self.loss(logits * mask, t)

        output['loss'] = obj.sum() * self.weight
        self.total_obj += output['loss'].item()

        # if True:
        #     predictions = torch.nonzero(logits * mask > 0).tolist()
        #     preds = [list() for _ in gold_spans]
        #     for batch, begin, width in predictions:
        #         preds[batch].append( (begin, begin+width) )
        #     pred_spans = preds
        # print("pred-1:", [len(x) for x in pred_spans])

        masked_logits = logits + (1.0 - mask) * -10000.0
        # masked_logits = t + (1.0 - mask) * -10000.0
        sorted_indices = torch.argsort(-masked_logits.contiguous().view(logits.size(0), -1))
        begins = sorted_indices / self.max_span_length
        widths = torch.fmod(sorted_indices, self.max_span_length)

        # tops = [len(x) for x in gold_spans]
        tops = (sequence_lengths * self.spans_per_token).int().tolist()

        preds = [list() for _ in gold_spans]
        for batch, top in enumerate(tops):
            for begin, width in zip(begins[batch, 0:top].tolist(), widths[batch, 0:top].tolist()):
                preds[batch].append((begin, begin + width))
        pred_spans = preds

        # print("pred-2:", [len(x) for x in preds])

        output['pred'] = [[(b, e + 1, 'span') for b, e in spans] for spans in pred_spans]
        output['gold'] = [[(b, e + 1, 'span') for b, e in spans] for spans in gold_spans]

        # print("gold:", [len(x) for x in gold_spans])

        self.total_pred_spans += sum([len(x) for x in pred_spans])
        self.total_gold_spans += sum([len(x) for x in gold_spans])

        # print('p:', output['pred'][0])
        # print('g:', output['gold'][0])

        # pos = 1
        # print("span-0:", scores[0,pos,0].item() + scores[0,pos,2].item(), out[0,0,pos].item())
        # print("span-1:", scores[0,pos,0].item() + scores[0,pos+1,2].item(), out[0,1,pos].item())
        # print("span-2:", scores[0,pos,0].item() + scores[0,pos+1,1].item() + scores[0,pos+2,2].item(), out[0,2,pos].item())
        # print("span-3:", scores[0,pos,0].item() + scores[0,pos+1,1].item() + scores[0,pos+2,1].item() + scores[0,pos+3,2].item(), out[0,3,pos].item())
        # print("span-4:", scores[0,pos,0].item() + scores[0,pos+1,1].item() + scores[0,pos+2,1].item() + scores[0,pos+3,1].item() + scores[0,pos+4,2].item(), out[0,4,pos].item())

        return output

    def create_metrics(self):
        return [MetricSpanNER(self.name, labels=['span']), MetricObjective(self.name)]

    def tick(self, dataset_name):
        print("{}-obj: {}".format(dataset_name, self.total_obj))
        print("{}-span-generator: {} / {} = {}".format(dataset_name, self.total_pred_spans, self.total_gold_spans,
                                                       self.total_pred_spans / self.total_gold_spans))
        self.total_pred_spans = 0
        self.total_gold_spans = 0

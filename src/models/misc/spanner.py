import logging
from collections import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from metrics.f1 import MetricSpanNER
from metrics.misc import MetricObjective
from misc import settings
from models.misc.misc import batched_index_select, FeedForward

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger()


def create_all_spans(batch_size, length, width):
    """

    :param batch_size: example: {int} 1
    :param length: example: {int} 69
    :param width: example: {int} 5
    :return:
    """
    b = torch.arange(length, dtype=torch.long)
    w = torch.arange(width, dtype=torch.long)
    e = b.unsqueeze(-1) + w.unsqueeze(0)
    b = b.unsqueeze(-1).expand_as(e)

    b = b.unsqueeze(0).expand((batch_size,) + b.size())
    e = e.unsqueeze(0).expand((batch_size,) + e.size())
    return b, e


def create_span_targets(ref, instances):
    targets = torch.zeros(ref.size())
    max_span_length = targets.size(2)
    for i, spans in enumerate(instances):
        for begin, end, label in spans:
            if end - begin < max_span_length:
                targets[i, begin, end - begin, label] = 1.0
    return targets


def decode_span_predictions(logits, labels):
    predictions = torch.nonzero(logits > 0)
    preds = [list() for _ in range(logits.size(0))]
    if predictions.size(0) < logits.size(0) * logits.size(1):
        for batch, begin, width, l in predictions.tolist():
            preds[batch].append((begin, begin + width + 1, labels[l]))
    return preds


def create_span_extractor(dim_input, max_span_length, config):
    se_type = config['type']

    if se_type == 'endpoint':
        return SpanEndpoint(dim_input, max_span_length, config)
    elif se_type == 'endpoint_spanbert':
        return SpanEndpointSpanBert(dim_input, max_span_length, config)
    elif se_type == 'average':
        return SpanAverage(dim_input, max_span_length, config)
    else:
        raise BaseException("no such span extractor:", se_type)


class SpanEndpoint(nn.Module):

    def make_linear(self, in_features, out_features, bias=True, std=0.02):
        linear = nn.Linear(in_features, out_features, bias)
        init.normal_(linear.weight, std=std)
        if bias:
            init.zeros_(linear.bias)
        return linear

    def make_ffnn(self, feat_size, hidden_size, output_size):
        if hidden_size is None or hidden_size == 0 or hidden_size == [] or hidden_size == [0]:
            return self.make_linear(feat_size, output_size)

        if not isinstance(hidden_size, Iterable):
            hidden_size = [hidden_size]
        ffnn = [self.make_linear(feat_size, hidden_size[0]), nn.ReLU(), self.dropout]
        for i in range(1, len(hidden_size)):
            ffnn += [self.make_linear(hidden_size[i - 1], hidden_size[i]), nn.ReLU(), self.dropout]
        ffnn.append(self.make_linear(hidden_size[-1], output_size))
        return nn.Sequential(*ffnn)

    def __init__(self, dim_input, max_span_length, config):
        super(SpanEndpoint, self).__init__()
        self.span_embed = 'span_embed' in config
        self.dim_output = 2 * dim_input
        self.dim_input = dim_input
        self.span_average = config['average']

        if self.span_embed:
            self.embed = nn.Embedding(max_span_length, config['span_embed'])
            self.dim_output += config['span_embed']

        if self.span_average:
            self.dim_output += dim_input

        if 'ff_dim' in config:
            self.ff = nn.Sequential(
                nn.Linear(self.dim_output, config['ff_dim']),
                nn.ReLU(),
                nn.Dropout(config['ff_dropout'])
            )
            self.dim_output = config['ff_dim']
        else:
            self.ff = nn.Sequential()

        if 'attention_heads' in config and config['attention_heads']:
            self.attention_heads = True
            self.mention_token_attn = self.make_ffnn(self.dim_input, 0, output_size=1)
            self.dim_output += dim_input
        else:
            self.attention_heads = False

    def forward(self, inputs, b, e, max_width, span_mask=None):
        b_vec = batched_index_select(inputs, b)
        e_vec = batched_index_select(inputs, torch.clamp(e, max=inputs.size(1) - 1))

        vecs = [b_vec, e_vec]

        if self.span_embed:
            vecs.append(self.embed(e - b))

        if self.attention_heads:
            span_mask = span_mask > 0.9
            # TODO: only will work for batch size of 1!
            curr_batch = 0
            candidate_starts, candidate_ends = b[curr_batch][span_mask[curr_batch]], e[curr_batch][
                span_mask[curr_batch]]
            num_candidates = candidate_starts.shape[0]

            token_attn = torch.squeeze(self.mention_token_attn(inputs[curr_batch]), 1)

            num_subtokens = inputs[curr_batch].shape[0]  # num_words
            candidate_tokens = torch.unsqueeze(torch.arange(0, num_subtokens,
                                                            device=settings.device), 0).repeat(num_candidates, 1)
            candidate_tokens_mask = (candidate_tokens >= torch.unsqueeze(candidate_starts, 1)) & \
                                    (candidate_tokens <= torch.unsqueeze(candidate_ends, 1))

            candidate_tokens_attn_raw = torch.log(candidate_tokens_mask.float()) + \
                                        torch.unsqueeze(token_attn, 0)

            candidate_tokens_attn = nn.functional.softmax(candidate_tokens_attn_raw, dim=1)

            head_attn_emb = torch.matmul(candidate_tokens_attn, inputs[curr_batch])

            att_vec = torch.zeros_like(b_vec, device=settings.device)
            att_vec[span_mask] = head_attn_emb
            vecs.append(att_vec)

        if self.span_average:
            vecs.append(span_average(inputs, b, e, max_width))

        vec = torch.cat(vecs, -1)
        return self.ff(vec)


class SpanEndpointSpanBert(nn.Module):
    """
    The original idea is that, unlike SpanEndpoint, it accepts directly the masked spans.
    Also some extra stuff from https://github.com/lxucs/coref-hoi
    """

    def make_linear(self, in_features, out_features, bias=True, std=0.02):
        linear = nn.Linear(in_features, out_features, bias)
        init.normal_(linear.weight, std=std)
        if bias:
            init.zeros_(linear.bias)
        return linear

    def make_ffnn(self, feat_size, hidden_size, output_size):
        if hidden_size is None or hidden_size == 0 or hidden_size == [] or hidden_size == [0]:
            return self.make_linear(feat_size, output_size)

        if not isinstance(hidden_size, Iterable):
            hidden_size = [hidden_size]
        ffnn = [self.make_linear(feat_size, hidden_size[0]), nn.ReLU(), self.dropout]
        for i in range(1, len(hidden_size)):
            ffnn += [self.make_linear(hidden_size[i - 1], hidden_size[i]), nn.ReLU(), self.dropout]
        ffnn.append(self.make_linear(hidden_size[-1], output_size))
        return nn.Sequential(*ffnn)

    def __init__(self, dim_input, max_span_length, config):
        super(SpanEndpointSpanBert, self).__init__()
        self.span_embed = 'span_embed' in config
        self.dim_output = 2 * dim_input
        self.dim_input = dim_input
        self.span_average = config['average']
        self.dropout = nn.Dropout(p=config['dropout'])

        if self.span_embed:
            self.embed = nn.Embedding(max_span_length, config['span_embed'])
            # adapted from hoi, initialization with std 0.02
            init.normal_(self.embed.weight, std=0.02)
            self.dim_output += config['span_embed']

        if self.span_average:
            self.dim_output += dim_input

        if 'ff_dim' in config:
            self.ff = self.make_ffnn(self.dim_output, 0, output_size=config['ff_dim'])
            self.dim_output = config['ff_dim']
        else:
            self.ff = nn.Sequential()

        if 'attention_heads' in config and config['attention_heads']:
            self.attention_heads = True
            # self.mention_token_attn = self.make_ffnn(self.bert_emb_size, 0, output_size=1) if config[
            self.mention_token_attn = self.make_ffnn(self.dim_input, 0, output_size=1)
            self.dim_output += dim_input
        else:
            self.attention_heads = False

    def forward(self, inputs, b, e, max_width, span_mask=None):
        # inputs.shape --> [1, 96, 768]; b.shape --> [1, 315]; e.shape --> [1, 315]; max_width --> 15
        b_vec = batched_index_select(inputs, b)
        # b_vec.shape --> [1, 315, 768]

        # e_vec = batched_index_select(inputs, torch.clamp(e, max=inputs.size(1) - 1))
        e_vec = batched_index_select(inputs, torch.clamp(e, max=inputs.size(1) - 1))
        # e_vec.shape --> [1, 315, 768]

        vecs = [b_vec, e_vec]

        if self.span_embed:
            candidate_width_idx = e - b
            candidate_width_emb = self.embed(e - b)
            candidate_width_emb = self.dropout(candidate_width_emb)
            vecs.append(candidate_width_emb)

        if self.attention_heads:
            # num_candidates
            # span_mask = span_mask > 0.9
            # TODO: only will work for batch size of 1!
            curr_batch = 0
            # candidate_starts, candidate_ends = b[curr_batch][span_mask[curr_batch]], e[curr_batch][
            #     span_mask[curr_batch]]
            num_candidates = b[curr_batch].shape[0]

            token_attn = torch.squeeze(self.mention_token_attn(inputs[curr_batch]), 1)

            num_subtokens = inputs[curr_batch].shape[0]  # num_words
            candidate_tokens = torch.unsqueeze(torch.arange(0, num_subtokens,
                                                            device=settings.device), 0).repeat(num_candidates, 1)
            candidate_tokens_mask = (candidate_tokens >= torch.unsqueeze(b[curr_batch], 1)) & \
                                    (candidate_tokens <= torch.unsqueeze(e[curr_batch], 1))

            candidate_tokens_attn_raw = torch.log(candidate_tokens_mask.float()) + \
                                        torch.unsqueeze(token_attn, 0)

            candidate_tokens_attn = nn.functional.softmax(candidate_tokens_attn_raw, dim=1)

            head_attn_emb = torch.matmul(candidate_tokens_attn, inputs[curr_batch])
            head_attn_emb.unsqueeze_(0)
            vecs.append(head_attn_emb)

        if self.span_average:
            vecs.append(span_average(inputs, b, e, max_width))

        vec = torch.cat(vecs, -1)
        return self.ff(vec), candidate_width_idx


def span_average(inputs, b, e, max_width):
    w = torch.arange(max_width).to(b.device)
    indices = b.unsqueeze(-1) + w.unsqueeze(0).unsqueeze(0)
    vectors = batched_index_select(inputs, torch.clamp(indices, max=inputs.size(1) - 1))

    mask = (indices <= e.unsqueeze(-1)).float()
    lengths = mask.sum(-1)
    probs = mask / lengths.unsqueeze(-1)
    output = torch.matmul(probs.unsqueeze(-2), vectors).squeeze(-2)
    return output


class SpanAverage(nn.Module):

    def __init__(self, dim_input, max_span_length, config):
        super(SpanAverage, self).__init__()
        self.span_embed = 'span_embed' in config
        self.dim_output = dim_input

        if self.span_embed:
            self.embed = nn.Embedding(max_span_length, config['span_embed'])
            self.dim_output += config['span_embed']

    def forward(self, inputs, b, e, max_width):
        output = span_average(inputs, b, e, max_width)

        if self.span_embed:
            emb = self.embed(e - b)
            return torch.cat((output, emb), -1)
        else:
            return output


class SpanSelfAttention(nn.Module):

    def __init__(self, dim_input, config):
        super(SpanSelfAttention, self).__init__()
        self.ff = FeedForward(dim_input, config['attention'])
        self.out = nn.Linear(self.ff.dim_output, 1)
        self.dim_output = dim_input

    def forward(self, inputs, b, e, max_width):
        w = torch.arange(max_width).to(b.device)
        indices = b.unsqueeze(-1) + w.unsqueeze(0).unsqueeze(0)
        vectors = batched_index_select(inputs, torch.clamp(indices, max=inputs.size(1) - 1))

        mask = indices <= e.unsqueeze(-1)

        scores = self.out(self.ff(vectors)).squeeze(-1)
        scores = scores - (1.0 - mask.float()) * 1e38
        probs = F.softmax(scores, -1)

        output = torch.matmul(probs.unsqueeze(-2), vectors).squeeze(-2)
        return output


class TaskSpan1(nn.Module):

    def __init__(self, name, dim_input, dictionary, config):
        super(TaskSpan1, self).__init__()
        self.name = name
        self.enabled = config['enabled']
        self.max_span_length = config['max_span_length']
        self.weight = config['weight']
        self.loss = nn.BCEWithLogitsLoss(reduction='none')
        self.labels = dictionary.tolist()

        span_type = config['span-extractor']['type']
        if span_type == 'endpoint':
            self.span_extractor = SpanEndpoint(dim_input, self.max_span_length, config['span-extractor'])
        elif span_type == 'self-attention':
            self.span_extractor = SpanSelfAttention(dim_input, config['span-extractor'])
        else:
            raise BaseException("no such span extractor:", span_type)
        dim_span = self.span_extractor.dim_output

        self.net = FeedForward(dim_span, config['network'])
        self.out = nn.Linear(self.net.dim_output, dictionary.size)

        self.total_obj = 0
        self.total_pred_spans = 0
        self.total_gold_spans = 0

    def forward(self, inputs, targets, sequence_lengths, mask, gold_spans, metadata, predict=False):
        output = {}

        inputs = inputs.contiguous()

        b, e = create_all_spans(inputs.size(0), inputs.size(1), self.max_span_length)
        b = b.cuda()
        e = e.cuda()

        vec = self.span_extractor(inputs, b, e, self.max_span_length)
        logits = self.out(self.net(vec))

        mask = (e < sequence_lengths.unsqueeze(-1).unsqueeze(-1)).float().unsqueeze(-1)

        span_targets = create_span_targets(logits, metadata['gold_tags_indices'])

        obj = self.loss(logits, span_targets.to(logits.device)) * mask
        output['loss'] = obj.sum() * self.weight

        output['pred'] = decode_span_predictions(logits * mask, self.labels)
        output['gold'] = [[(b, e + 1, self.labels[l]) for b, e, l in spans] for spans in metadata['gold_tags_indices']]

        return output

    def create_metrics(self):
        return [MetricSpanNER(self.name, labels=self.labels), MetricObjective(self.name)]

    def tick(self, dataset_name):
        return


class TaskSpan1x(nn.Module):

    def __init__(self, name, dim_span, dictionary, config):
        super(TaskSpan1x, self).__init__()
        self.enabled = config['enabled']
        self.name = name
        if self.enabled:
            self.loss = nn.BCEWithLogitsLoss(reduction='none')
            self.labels = dictionary.tolist()
            self.weight = config['weight']
            self.add_pruner_scores = config['add_pruner_scores']
            self.add_pruner_loss = config['add_pruner_loss']
            self.mask_target = config['mask_target']

            if config['divide_by_number_of_labels']:
                self.weight /= len(self.labels)

            logger.info('TaskSpan1x: weight= %s add_pruner_scores= %s mask_target= %s'
                        % (self.weight, self.add_pruner_scores, self.mask_target))

            self.net = FeedForward(dim_span, config['network'])
            self.out = nn.Linear(self.net.dim_output, dictionary.size)

    def forward(self, spans_all, sequence_lengths, gold_tags_indices, api_call=False):
        output = {}

        if self.enabled:
            span_vecs = spans_all['span_vecs']
            span_end = spans_all['span_end']

            logits = self.out(self.net(span_vecs))

            if self.add_pruner_scores:
                logits = logits + spans_all['span_scores']

            span_targets = None
            if not api_call:
                span_targets = create_span_targets(logits, gold_tags_indices).to(logits.device)

            if self.mask_target and not api_call:
                mask = (span_targets.sum(-1) > 0).float().unsqueeze(-1)
            else:
                mask = (span_end < sequence_lengths.unsqueeze(-1).unsqueeze(-1)).float().unsqueeze(-1)

            if not api_call:
                obj = self.loss(logits, span_targets) * mask

                if self.add_pruner_loss:
                    pruner_target = (span_targets.sum(-1) > 0).float().unsqueeze(-1)
                    obj_pruner = self.loss(spans_all['span_scores'], pruner_target) * mask
                    obj = obj + obj_pruner

                output['loss'] = obj.sum() * self.weight
                output['gold'] = [[(b, e + 1, self.labels[l]) for b, e, l in spans] for spans in gold_tags_indices]
            else:
                output['loss'] = 0

            output['pred'] = decode_span_predictions(logits * mask, self.labels)
        else:
            output['loss'] = 0  # torch.tensor(0.0).cuda() (trainer skips minibatch if zero)
            num_batch = spans_all['span_vecs'].size(0)
            output['pred'] = [[] for x in range(num_batch)]
            output['gold'] = [[] for x in range(num_batch)]

        return output['loss'], output

    def create_metrics(self):
        return [MetricSpanNER(self.name, labels=self.labels), MetricObjective(self.name)] if self.enabled else []

import logging

import torch
import torch.nn as nn

from misc import settings
from data_processing.dictionary import Dictionary
from models.misc.misc import batched_index_select, bucket_values

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger()
import torch.nn.init as init


def inspect(name, x):
    x_norm = x.norm(dim=-1)
    logger.info('{}\tsize: {}'.format(name, x.size()))
    logger.info('{}\tnorm: {} - {}'.format(name, x_norm.min().item(), x_norm.max().item()))
    logger.info('{}\tdata: {}'.format(name, x_norm))


def predict_scores(scores, linker_spans, linker_candidates, candidate_lengths, labels):
    """
    Used by linkers
    :param scores:
    :param linker_spans:
    :param linker_candidates:
    :param candidate_lengths:
    :param labels:
    :return:
    """
    output = [list() for _ in linker_spans]
    for batch_idx, span_idx in torch.nonzero(candidate_lengths).tolist():
        num_candidates = candidate_lengths[batch_idx, span_idx].item()
        span = linker_spans[batch_idx][span_idx]
        c_ = [labels[c] for c in linker_candidates[batch_idx, span_idx, :num_candidates].tolist()]
        s_ = scores[batch_idx, span_idx, :num_candidates].tolist()
        output[batch_idx].append((span, c_, s_))
    return output


def predict_scores_mtt(scores, linker_spans, linker_candidates, candidate_lengths, entity_dictionary: Dictionary):
    """
    Used by linkers
    :param scores:
    :param linker_spans:
    :param linker_candidates:
    :param candidate_lengths:
    :param labels:
    :return:
    """
    output = [list() for _ in linker_spans]
    for batch_idx, span_idx in torch.nonzero(candidate_lengths).tolist():
        num_candidates = candidate_lengths[batch_idx, span_idx].item()
        span = linker_spans[batch_idx][span_idx]
        c_ = [entity_dictionary.get(c) for c in linker_candidates[batch_idx, span_idx, :num_candidates].tolist()]

        s_ = scores[batch_idx, span_idx, 1:num_candidates + 1].tolist()
        output[batch_idx].append((span, c_, s_))
    return output


def prune_spans(span_scores, sequence_lengths, sort_aft_pruning, prune_ratio=0.2, no_cross_overlap=False,
                span_mask=None):
    span_lengths = (sequence_lengths.float() * prune_ratio + 1).long()
    span_scores = span_scores.view(span_scores.size(0), -1)  # 24/03/2021 TODO: we are here, what is the order of this??

    values, top_indices = torch.topk(span_scores, span_lengths.max().item(), largest=True, sorted=True)

    # span_scores.shape --> [1,1440]
    # span_lengths --> [14]
    # values.shape --> [1,14]  ; values --> tensor([[0.0408, 0.0314, 0.0245, 0.0239, 0.0232, 0.0209, 0.0202, 0.0170, 0.0112,
    #          0.0080, 0.0074, 0.0054, 0.0049, 0.0045]], grad_fn=<TopkBackward>)
    # top_indices.shape --> torch.Size([1, 14]); top_indices --> tensor([[ 272,  274, 1170,   94, 1098,  226,  515,  228, 1111,  621,  360,  285,
    #          1380,  831]])
    if sort_aft_pruning:
        top_indices, _ = sort_after_pruning(top_indices, span_lengths, span_scores)
    return top_indices, span_lengths


def _extract_top_spans(candidate_idx_sorted, candidate_starts, candidate_ends, num_top_spans, no_cross_overlap):
    """ Keep top non-cross-overlapping candidates ordered by scores; compute on CPU because of loop

    len(candidate_idx_sorted) --> 13484; len(set(candidate_idx_sorted)) --> 13484
    min(candidate_idx_sorted) --> 0 ; max(candidate_idx_sorted) --> 13483
    ----
    len(candidate_starts) --> 13484; len(set(candidate_starts)) --> 1121
    min(candidate_starts) --> 0; max(candidate_starts) --> 1120
    ----
    len(candidate_ends) --> 13484; len(set(candidate_ends)) --> 1121
    min(candidate_ends) --> 0; max(candidate_ends) --> 1120
    num_top_spans 448
    """
    selected_candidate_idx = []
    start_to_max_end, end_to_min_start = {}, {}
    for candidate_idx in candidate_idx_sorted:
        if len(selected_candidate_idx) >= num_top_spans:
            break
        # Perform overlapping check
        span_start_idx = candidate_starts[candidate_idx]
        span_end_idx = candidate_ends[candidate_idx]
        cross_overlap = False
        for token_idx in range(span_start_idx, span_end_idx + 1):
            max_end = start_to_max_end.get(token_idx, -1)
            if token_idx > span_start_idx and max_end > span_end_idx and no_cross_overlap:
                cross_overlap = True
                break
            min_start = end_to_min_start.get(token_idx, -1)
            if token_idx < span_end_idx and 0 <= min_start < span_start_idx and no_cross_overlap:
                cross_overlap = True
                break
        if not cross_overlap:
            # Pass check; select idx and update dict stats
            selected_candidate_idx.append(candidate_idx)
            max_end = start_to_max_end.get(span_start_idx, -1)
            if span_end_idx > max_end:
                start_to_max_end[span_start_idx] = span_end_idx
            min_start = end_to_min_start.get(span_end_idx, -1)
            if min_start == -1 or span_start_idx < min_start:
                end_to_min_start[span_end_idx] = span_start_idx
    # Sort selected candidates by span idx
    selected_candidate_idx = sorted(selected_candidate_idx,
                                    key=lambda idx: (candidate_starts[idx], candidate_ends[idx]))

    if len(selected_candidate_idx) < num_top_spans:  # Padding
        logger.warning('length of selected candidates (' + str(len(selected_candidate_idx)) +
                       ') lower than num_top_spans (' + str(num_top_spans) + ')')
        selected_candidate_idx += ([selected_candidate_idx[0]] * (num_top_spans - len(selected_candidate_idx)))
    return selected_candidate_idx


def sort_after_pruning(indices, span_lengths, span_scores):
    """
    :return: sorted matrix by span position
    """
    for b, l in enumerate(span_lengths.tolist()):
        indices[b, l:] = span_scores.size(1) - 1
    top_indices, reindex = torch.sort(indices)

    return top_indices, reindex


def filter_spans(span_vecs, span_indices):
    """
    This function ....
    :param span_vecs:
    :param span_indices:
    :return:
    """
    # (kzaporoj) - added .contiguous()
    tmp = span_vecs.contiguous().view(span_vecs.size(0), -1, span_vecs.size(-1))

    # (kzaporoj) returns a vector of [1, 9, 1676] where 9 is the number of spans and 1676 is the dimension
    # span_indices are tensor([[ 16,  27, 207, 227, 245, 250, 256, 266, 285]])
    return batched_index_select(tmp, span_indices)


def create_masks(num_mentions, max_mentions):
    mask = get_mask_from_sequence_lengths(num_mentions, max_mentions).float()
    square_mask = torch.bmm(mask.unsqueeze(-1), mask.unsqueeze(-2))

    triangular_mask = torch.ones(max_mentions, max_mentions).tril(0).unsqueeze(0).to(num_mentions.device)
    return square_mask, square_mask * triangular_mask


def spans_to_indices(spans, max_span_width):
    b = spans[:, :, 0]
    e = spans[:, :, 1]
    i = b * max_span_width + (e - b)
    return i


def indices_to_spans(top_indices, span_lengths, max_span_width):
    b = top_indices // max_span_width
    w = top_indices % max_span_width
    e = b + w
    return [list(zip(b[i, 0:length].tolist(), e[i, 0:length].tolist())) for i, length in
            enumerate(span_lengths.tolist())]


def overwrite_spans(span_vecs, span_pruned_indices, span_lengths, values):
    output = span_vecs.clone()
    # output.shape --> torch.Size([1, 96, 15, 2324])
    # span_vecs.shape --> torch.Size([1, 96, 15, 2324])
    tmp = output.view(output.size(0), -1, output.size(-1))
    # tmp.shape --> torch.Size([1, 1440, 2324])

    for b, length in enumerate(span_lengths.tolist()):
        # span_lengths.shape --> [21]
        #
        if length > 0:
            indices = span_pruned_indices[b, :length]
            # span_pruned_indices.shape --> [1, 21]
            #
            tmp[b, indices, :] = values[b, :indices.size(0), :]
            # values.shape --> torch.Size([1, 21, 2324])
            # torch.Size([21, 2324]).shape --> [21, 2324] ; values[b, :indices.size(0), :].shape --> [21, 2324]

    return output  # output.shape --> torch.Size([1, 96, 15, 2324])


def overwrite_spans_hoi(span_vecs, span_pruned_indices, span_lengths, values):
    # span_vecs.shape --> [1, 315, 2324]
    # span_pruned_indices.shape --> [1, 21] --> tensor([[  5,   7,  12,  16,  20,  34,  82,  84,  86,  90, 100, ...]])
    # span_lengths --> tensor([21])
    # values.shape --> torch.Size([1, 21, 2324])
    assert span_lengths.shape[0] == 1  # no batch
    output = span_vecs.clone()

    length = span_lengths[0].item()
    if length > 0:
        output[0, span_pruned_indices[0], :] = values[0, :, :]

    return output  # output.shape --> [1, 315, 2324]


def relation_add_scores(relation_scores, filtered_prune_scores):
    scores_left = filtered_prune_scores
    scores_right = filtered_prune_scores.squeeze(-1).unsqueeze(-2)
    return relation_scores + scores_left.unsqueeze(-1) + scores_right.unsqueeze(-1)


def coref_add_scores(coref_scores, filtered_prune_scores):
    scores_left = filtered_prune_scores
    scores_right = filtered_prune_scores.squeeze(-1).unsqueeze(-2)
    coref_scores = coref_scores + scores_left + scores_right

    # zero-out self references (without this pruner doesn't work)
    eye = torch.eye(coref_scores.size(1)).unsqueeze(0).to(coref_scores)
    coref_scores = coref_scores * (1.0 - eye)
    return coref_scores


def coref_add_scores_hoi(coref_scores, filtered_prune_scores):
    scores_left = filtered_prune_scores
    scores_right = filtered_prune_scores.squeeze(-1).unsqueeze(-2)
    coref_scores = coref_scores + scores_left + scores_right

    # zero-out self references (without this pruner doesn't work)
    eye = torch.eye(coref_scores.size(1)).unsqueeze(0).to(coref_scores)
    coref_scores = coref_scores * (1.0 - eye)
    return coref_scores


def coref_add_scores_coreflinker(coref_scores, filtered_prune_scores, filter_singletons_with_matrix,
                                 subtract_pruner_for_singletons=True):
    scores_left = filtered_prune_scores  # .shape --> torch.Size([1, 21, 1])
    scores_right = filtered_prune_scores.squeeze(-1).unsqueeze(-2)  # .shape --> torch.Size([1, 1, 21])
    coref_scores = coref_scores + scores_left

    coref_scores[:, :, -scores_right.size(-1):] = coref_scores[:, :, -scores_right.size(-1):] + scores_right
    # coref_scores.shape --> torch.Size([1, 21, 37])
    # scores_right.shape --> torch.Size([1, 1, 21])
    # scores_right.size(-1) --> 21

    # zero-out self references (without this pruner doesn't work)
    eye = torch.eye(coref_scores.size(1), device=settings.device).unsqueeze(0).to(coref_scores)
    #
    #
    coref_scores[:, :, -coref_scores.size(1):] = coref_scores[:, :, -coref_scores.size(1):] * (1.0 - eye)
    # coref_scores.shape --> torch.Size([1, 21, 37])
    # eye.shape --> torch.Size([1, 21, 21]) ; coref_scores[:, :, -coref_scores.size(1):].shape --> torch.Size([1, 21, 21])
    if filter_singletons_with_matrix:
        # also adds (TODO substracts???) the pruner scores to the mentions in the positions to be ignored
        # has to subtract twice because it was added using scores_left already (see above)
        if subtract_pruner_for_singletons:
            coref_scores[:, :, :1] = coref_scores[:, :, :1] - scores_left - scores_left
    return coref_scores
    # coref_scores.shape --> torch.Size([1, 21, 37])


class MyGate(nn.Module):

    def __init__(self, dim_input, init_weights_std=None):
        super(MyGate, self).__init__()
        self.linear = nn.Linear(2 * dim_input, dim_input)
        if init_weights_std is not None:
            init.normal_(self.linear.weight, std=init_weights_std)
            init.zeros_(self.linear.bias)

    def forward(self, left, right):
        g = self.linear(torch.cat((left, right), -1))
        g = torch.sigmoid(g)
        return g * left + (1 - g) * right


def span_distance_tokens(span_begin, span_end):
    span_begin = span_begin.view(span_begin.size(0), -1)
    span_end = span_end.view(span_end.size(0), -1)
    span_dist = torch.relu(span_begin.unsqueeze(-1) - span_end.unsqueeze(-2))
    span_dist = span_dist + span_dist.permute(0, 2, 1)

    return span_dist


def span_distance_ordering(span_begin, span_end):
    span_index = torch.arange(span_begin.size(1)).unsqueeze(0)
    span_index = span_index.expand(span_begin.size()[0:2])
    span_dist = torch.abs(span_index.unsqueeze(-1) - span_index.unsqueeze(-2))
    return span_dist.to(span_begin.device)


class SpanPairs(nn.Module):

    def __init__(self, dim_input, config):
        super(SpanPairs, self).__init__()
        self.num_distance_buckets = config['num_distance_buckets']
        self.dim_distance_embedding = config['dim_distance_embedding']
        self.distance_embeddings = nn.Embedding(self.num_distance_buckets,
                                                self.dim_distance_embedding) if self.dim_distance_embedding > 0 else None
        self.init_embeddings_std = config['init_embeddings_std']
        if self.init_embeddings_std is not None:
            init.normal_(self.distance_embeddings.weight, std=self.init_embeddings_std)
        self.dim_output = dim_input * 2 + self.dim_distance_embedding
        self.span_product = config['span_product']
        if self.span_product:
            self.dim_output += dim_input

        if config['distance_function'] == 'tokens':
            self.distance_function = span_distance_tokens
            self.requires_sorted_spans = False
        elif config['distance_function'] == 'ordering':
            self.distance_function = span_distance_ordering
            self.requires_sorted_spans = True
        elif self.dim_distance_embedding > 0:
            raise BaseException("no such distance function")
        else:
            self.requires_sorted_spans = False

    def forward(self, span_vecs, span_begin, span_end):
        num_batch, num_spans, dim_vector = span_vecs.size()
        left = span_vecs.unsqueeze(-2).expand(num_batch, num_spans, num_spans, dim_vector)
        right = span_vecs.unsqueeze(-3).expand(num_batch, num_spans, num_spans, dim_vector)

        tmp = [left, right]

        if self.span_product:
            tmp.append(left * right)

        if self.dim_distance_embedding > 0:
            span_dist = self.distance_function(span_begin, span_end)
            span_dist = bucket_values(span_dist, num_total_buckets=self.num_distance_buckets)
            tmp.append(self.distance_embeddings(span_dist))

        return torch.cat(tmp, -1)

    def get_product_embedding(self, span_vecs):
        num_batch, num_spans, dim_vector = span_vecs.size()
        left = span_vecs.unsqueeze(-2).expand(num_batch, num_spans, num_spans, dim_vector)
        right = span_vecs.unsqueeze(-3).expand(num_batch, num_spans, num_spans, dim_vector)
        return left * right

    def get_distance_embedding(self, span_begin, span_end):
        span_dist = self.distance_function(span_begin, span_end)
        span_dist = bucket_values(span_dist, num_total_buckets=self.num_distance_buckets)
        return self.distance_embeddings(span_dist)


def get_mask_from_sequence_lengths(sequence_lengths, max_length=None):
    """
    Given a variable of shape ``(batch_size,)`` that represents the sequence lengths of each batch
    element, this function returns a ``(batch_size, max_length)`` mask variable.  For example, if
    our input was ``[2, 2, 3]``, with a ``max_length`` of 4, we'd return
    ``[[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]]``.
    We require ``max_length`` here instead of just computing it from the input ``sequence_lengths``
    because it lets us avoid finding the max, then copying that value from the GPU to the CPU so
    that we can use it to construct a new tensor.
    """
    # (batch_size, max_length)
    ones = sequence_lengths.new_ones(sequence_lengths.size(0), max_length)
    range_tensor = ones.cumsum(dim=1)
    return (sequence_lengths.unsqueeze(1) >= range_tensor).long()

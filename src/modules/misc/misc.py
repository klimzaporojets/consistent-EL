import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


def create_activation_function(name):
    if name == 'relu':
        return nn.ReLU()
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'glu':
        return nn.GLU()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    else:
        raise BaseException("no such activation function:", name)


class FeedForward(nn.Module):

    def __init__(self, dim_input, config):
        super(FeedForward, self).__init__()
        self.dim_output = dim_input
        self.layers = []

        if 'type' not in config:
            self.create_default(config)
        elif config['type'] == 'ffnn':
            self.create_ffnn(config)
        else:
            raise BaseException("no such type: ", config['type'])

        self.layers = nn.Sequential(*self.layers)

    def create_default(self, config):
        if config['ln']:
            # from modules.misc.misc import LayerNorm
            self.layers.append(LayerNorm(self.dim_output))
        if config['dropout'] != 0.0:
            self.layers.append(nn.Dropout(config["dropout"]))

    def create_ffnn(self, config):
        if 'dp_in' in config:
            self.layers.append(nn.Dropout(config['dp_in']))
        for dim in config['dims']:
            self.layers.append(nn.Linear(self.dim_output, dim))
            if 'actfnc' in config:
                self.layers.append(create_activation_function(config['actfnc']))
            if 'dp_h' in config:
                self.layers.append(nn.Dropout(config['dp_h']))
            self.dim_output = dim

    def forward(self, tensor):
        return self.layers(tensor)


class Seq2Seq(nn.Module):

    def __init__(self, dim_input, config):
        super(Seq2Seq, self).__init__()
        self.module = seq2seq_create(dim_input, config)
        self.dim_output = self.module.dim_output

    def forward(self, inputs, seqlens, indices=None):
        return self.module(inputs, seqlens, indices)


class ConfigurationError(Exception):
    """
    The exception raised by any AllenNLP object when it's misconfigured
    (e.g. missing properties, invalid properties, unknown properties).
    """

    def __init__(self, message):
        super(ConfigurationError, self).__init__()
        self.message = message

    def __str__(self):
        return repr(self.message)


def get_range_vector(size: int, device: int) -> torch.Tensor:
    """
    Returns a range vector with the desired size, starting at 0. The CUDA implementation
    is meant to avoid copy data from CPU to GPU.
    """
    if device > -1:
        return torch.cuda.LongTensor(size, device=device).fill_(1).cumsum(0) - 1
    else:
        return torch.arange(0, size, dtype=torch.long)


def get_device_of(tensor: torch.Tensor) -> int:
    """
    Returns the device of the tensor.
    """
    if not tensor.is_cuda:
        return -1
    else:
        return tensor.get_device()


def flatten_and_batch_shift_indices(indices: torch.Tensor,
                                    sequence_length: int) -> torch.Tensor:
    """
    This is a subroutine for :func:`~batched_index_select`. The given ``indices`` of size
    ``(batch_size, d_1, ..., d_n)`` indexes into dimension 2 of a target tensor, which has size
    ``(batch_size, sequence_length, embedding_size)``. This function returns a vector that
    correctly indexes into the flattened target. The sequence length of the target must be
    provided to compute the appropriate offsets.

    .. code-block:: python

        indices = torch.ones([2,3], dtype=torch.long)
        # Sequence length of the target tensor.
        sequence_length = 10
        shifted_indices = flatten_and_batch_shift_indices(indices, sequence_length)
        # Indices into the second element in the batch are correctly shifted
        # to take into account that the target tensor will be flattened before
        # the indices are applied.
        assert shifted_indices == [1, 1, 1, 11, 11, 11]

    Parameters
    ----------
    indices : ``torch.LongTensor``, required.
    sequence_length : ``int``, required.
        The length of the sequence the indices index into.
        This must be the second dimension of the tensor.

    Returns
    -------
    offset_indices : ``torch.LongTensor``
    """
    # Shape: (batch_size)
    if torch.max(indices) >= sequence_length or torch.min(indices) < 0:
        raise ConfigurationError(f"All elements in indices should be in range (0, {sequence_length - 1})")
    offsets = get_range_vector(indices.size(0), get_device_of(indices)) * sequence_length
    for _ in range(len(indices.size()) - 1):
        offsets = offsets.unsqueeze(1)

    # Shape: (batch_size, d_1, ..., d_n)
    offset_indices = indices + offsets

    # Shape: (batch_size * d_1 * ... * d_n)
    offset_indices = offset_indices.view(-1)
    return offset_indices


def batched_index_select(target: torch.Tensor,
                         indices: torch.LongTensor,
                         flattened_indices: Optional[torch.LongTensor] = None) -> torch.Tensor:
    """
    The given ``indices`` of size ``(batch_size, d_1, ..., d_n)`` indexes into the sequence
    dimension (dimension 2) of the target, which has size ``(batch_size, sequence_length,
    embedding_size)``.

    This function returns selected values in the target with respect to the provided indices, which
    have size ``(batch_size, d_1, ..., d_n, embedding_size)``. This can use the optionally
    precomputed :func:`~flattened_indices` with size ``(batch_size * d_1 * ... * d_n)`` if given.

    An example use case of this function is looking up the start and end indices of spans in a
    sequence tensor. This is used in the
    :class:`~allennlp.models.coreference_resolution.CoreferenceResolver`. Model to select
    contextual word representations corresponding to the start and end indices of mentions. The key
    reason this can't be done with basic torch functions is that we want to be able to use look-up
    tensors with an arbitrary number of dimensions (for example, in the coref model, we don't know
    a-priori how many spans we are looking up).

    Parameters
    ----------
    target : ``torch.Tensor``, required.
        A 3 dimensional tensor of shape (batch_size, sequence_length, embedding_size).
        This is the tensor to be indexed.
    indices : ``torch.LongTensor``
        A tensor of shape (batch_size, ...), where each element is an index into the
        ``sequence_length`` dimension of the ``target`` tensor.
    flattened_indices : Optional[torch.Tensor], optional (default = None)
        An optional tensor representing the result of calling :func:~`flatten_and_batch_shift_indices`
        on ``indices``. This is helpful in the case that the indices can be flattened once and
        cached for many batch lookups.

    Returns
    -------
    selected_targets : ``torch.Tensor``
        A tensor with shape [indices.size(), target.size(-1)] representing the embedded indices
        extracted from the batch flattened target tensor.
    """
    if flattened_indices is None:
        # Shape: (batch_size * d_1 * ... * d_n)
        flattened_indices = flatten_and_batch_shift_indices(indices, target.size(1))

    # Shape: (batch_size * sequence_length, embedding_size)
    flattened_target = target.view(-1, target.size(-1))

    # Shape: (batch_size * d_1 * ... * d_n, embedding_size)
    flattened_selected = flattened_target.index_select(0, flattened_indices)
    selected_shape = list(indices.size()) + [target.size(-1)]
    # Shape: (batch_size, d_1, ..., d_n, embedding_size)
    selected_targets = flattened_selected.view(*selected_shape)
    return selected_targets


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


# from allennlp.nn.util import bucket_values


def bucket_values(distances: torch.Tensor,
                  num_identity_buckets: int = 4,
                  num_total_buckets: int = 10) -> torch.Tensor:
    """
    Places the given values (designed for distances) into ``num_total_buckets``semi-logscale
    buckets, with ``num_identity_buckets`` of these capturing single values.

    The default settings will bucket values into the following buckets:
    [0, 1, 2, 3, 4, 5-7, 8-15, 16-31, 32-63, 64+].

    Parameters
    ----------
    distances : ``torch.Tensor``, required.
        A Tensor of any size, to be bucketed.
    num_identity_buckets: int, optional (default = 4).
        The number of identity buckets (those only holding a single value).
    num_total_buckets : int, (default = 10)
        The total number of buckets to bucket values into.

    Returns
    -------
    A tensor of the same shape as the input, containing the indices of the buckets
    the values were placed in.
    """
    # Chunk the values into semi-logscale buckets using .floor().
    # This is a semi-logscale bucketing because we divide by log(2) after taking the log.
    # We do this to make the buckets more granular in the initial range, where we expect
    # most values to fall. We then add (num_identity_buckets - 1) because we want these indices
    # to start _after_ the fixed number of buckets which we specified would only hold single values.
    logspace_index = (distances.float().log() / math.log(2)).floor().long() + (num_identity_buckets - 1)
    # create a mask for values which will go into single number buckets (i.e not a range).
    use_identity_mask = (distances <= num_identity_buckets).long()
    use_buckets_mask = 1 + (-1 * use_identity_mask)
    # Use the original values if they are less than num_identity_buckets, otherwise
    # use the logspace indices.
    combined_index = use_identity_mask * distances + use_buckets_mask * logspace_index
    # Clamp to put anything > num_total_buckets into the final bucket.
    return combined_index.clamp(0, num_total_buckets - 1)


def create_masks(num_mentions, max_mentions):
    mask = get_mask_from_sequence_lengths(num_mentions, max_mentions).float()
    square_mask = torch.bmm(mask.unsqueeze(-1), mask.unsqueeze(-2))

    triangular_mask = torch.ones(max_mentions, max_mentions).tril(0).unsqueeze(0).to(num_mentions.device)
    return square_mask, square_mask * triangular_mask


def filter_spans(span_vecs, span_indices):
    # tmp = span_vecs.view(span_vecs.size(0), -1, span_vecs.size(-1))
    tmp = span_vecs.contiguous().view(span_vecs.size(0), -1, span_vecs.size(-1))
    return batched_index_select(tmp, span_indices)


def prune_spans(span_scores, sequence_lengths, sort_after_pruning, prune_ratio=0.2):
    span_lengths = (sequence_lengths * prune_ratio + 1).long()
    span_scores = span_scores.view(span_scores.size(0), -1)
    values, top_indices = torch.topk(span_scores, span_lengths.max().item(), largest=True, sorted=True)
    if sort_after_pruning:
        for b, l in enumerate(span_lengths.tolist()):
            top_indices[b, l:] = span_scores.size(1) - 1
        top_indices, _ = torch.sort(top_indices)
    return top_indices, span_lengths


def indices_to_spans(top_indices, span_lengths, max_span_width):
    b = top_indices // max_span_width
    w = top_indices % max_span_width
    e = b + w
    return [list(zip(b[i, 0:length].tolist(), e[i, 0:length].tolist())) for i, length in
            enumerate(span_lengths.tolist())]


def spans_to_indices(spans, max_span_width):
    b = spans[:, :, 0]
    e = spans[:, :, 1]
    i = b * max_span_width + (e - b)
    return i


def coref_add_scores(coref_scores, filtered_prune_scores):
    scores_left = filtered_prune_scores
    scores_right = filtered_prune_scores.squeeze(-1).unsqueeze(-2)
    coref_scores = coref_scores + scores_left + scores_right

    # zero-out self references (without this pruner doesn't work)
    eye = torch.eye(coref_scores.size(1)).unsqueeze(0).to(coref_scores)
    coref_scores = coref_scores * (1.0 - eye)
    return coref_scores


def create_all_spans(batch_size, length, width):
    b = torch.arange(length, dtype=torch.long)
    w = torch.arange(width, dtype=torch.long)
    e = b.unsqueeze(-1) + w.unsqueeze(0)
    b = b.unsqueeze(-1).expand_as(e)

    b = b.unsqueeze(0).expand((batch_size,) + b.size())
    e = e.unsqueeze(0).expand((batch_size,) + e.size())
    return b, e


def span_intersection(pred, gold):
    numer = 0
    for p, g in zip(pred, gold):
        numer += len(set(p) & set(g))
    return numer


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


def seq2seq_create(dim_input, config):
    if config['type'] == 'none':
        return LayerNone(dim_input, config)
    elif config['type'] == 'lstm':
        return Seq2seq(dim_input, config)
    else:
        raise BaseException("seq2seq_create: no such type", config['type'])


class LayerNone(nn.Module):

    def __init__(self, dim_input, config):
        super(LayerNone, self).__init__()
        self.dim_output = dim_input

    def forward(self, inputs, seqlens, indices=None):
        return inputs


class Seq2seq(nn.Module):

    def __init__(self, dim_input, config):
        super(Seq2seq, self).__init__()
        if 'i_dp' in config:
            self.idp = nn.Dropout(config['i_dp'])
        else:
            self.idp = nn.Sequential()

        if config['type'] == 'lstm':
            self.rnn = nn.LSTM(dim_input, config['dim'], bidirectional=True, num_layers=config['layers'],
                               dropout=config['dropout'], batch_first=True)
        else:
            raise RuntimeError('ERROR in Seq2Seq, config type ' + config['type'] + ' not defined')

        print("WARNING:WDROP COMMENTED OUT")

        self.dim = config['dim'] * 2
        self.dim_output = self.dim

    def forward(self, inputs, seqlens, indices=None):
        inputs = self.idp(inputs)
        packed_inputs = rnn_utils.pack_padded_sequence(inputs, seqlens.cpu(), batch_first=True)
        packed_outputs, _ = self.rnn(packed_inputs)
        outputs, _ = rnn_utils.pad_packed_sequence(packed_outputs, batch_first=True)
        return outputs


class CNNMaxpool(nn.Module):

    def __init__(self, dim_input, config):
        super(CNNMaxpool, self).__init__()
        self.cnns = nn.ModuleList([nn.Conv1d(dim_input, config['dim'], k) for k in config['kernels']])
        self.dim_output = config['dim'] * len(config['kernels'])
        self.max_kernel = max(config['kernels'])

    def forward(self, inputs):
        inp = inputs.view(inputs.size(0) * inputs.size(1), inputs.size(2), inputs.size(3))
        inp = inp.transpose(1, 2)
        outputs = []
        for cnn in self.cnns:
            maxpool, _ = torch.max(cnn(inp), -1)
            outputs.append(maxpool)
        outputs = torch.cat(outputs, -1)
        result = outputs.view(inputs.size(0), inputs.size(1), -1)
        return result


class OptFFpairs(nn.Module):

    def __init__(self, dim_input, dim_output, config, span_pair_generator):
        super(OptFFpairs, self).__init__()
        hidden_dim = config['hidden_dim']  # 150
        hidden_dp = config['hidden_dropout']  # 0.4

        self.span_pair_generator = span_pair_generator
        self.left = nn.Linear(dim_input, hidden_dim)
        self.right = nn.Linear(dim_input, hidden_dim)
        self.prod = nn.Linear(dim_input, hidden_dim)
        self.dist = nn.Linear(span_pair_generator.dim_distance_embedding, hidden_dim)
        self.dp1 = nn.Dropout(hidden_dp)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.dp2 = nn.Dropout(hidden_dp)
        self.out = nn.Linear(hidden_dim, dim_output)

    def forward(self, span_vecs, span_begin, span_end):
        # product of span representations
        p = self.span_pair_generator.get_product_embedding(span_vecs)
        # embedding of difference between the two spans (the distance) (this has dim 20)
        d = self.span_pair_generator.get_distance_embedding(span_begin, span_end)

        # size of hidden dim is only 150 see config file (or see above)
        h = self.left(span_vecs).unsqueeze(-2) + self.right(span_vecs).unsqueeze(-3) + self.prod(p) + self.dist(d)
        h = self.dp1(torch.relu(h))
        h = self.layer2(h)
        h = self.dp2(torch.relu(h))
        # resize to dim output which is = #labels or relations
        return self.out(h)

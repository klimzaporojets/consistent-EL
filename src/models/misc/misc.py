import math
from typing import Optional

import torch
import torch.nn as nn


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


# def get_mask_from_sequence_lengths(sequence_lengths, max_length=None):
#     """
#     Given a variable of shape ``(batch_size,)`` that represents the sequence lengths of each batch
#     element, this function returns a ``(batch_size, max_length)`` mask variable.  For example, if
#     our input was ``[2, 2, 3]``, with a ``max_length`` of 4, we'd return
#     ``[[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]]``.
#     We require ``max_length`` here instead of just computing it from the input ``sequence_lengths``
#     because it lets us avoid finding the max, then copying that value from the GPU to the CPU so
#     that we can use it to construct a new tensor.
#     """
#     # (batch_size, max_length)
#     ones = sequence_lengths.new_ones(sequence_lengths.size(0), max_length)
#     range_tensor = ones.cumsum(dim=1)
#     return (sequence_lengths.unsqueeze(1) >= range_tensor).long()


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


import torch
import torch.nn as nn

# from modules.seq2seq import seq2seq_create
from models.misc.text_field import TextFieldEmbedderTokens, TextFieldEmbedderCharacters
from models.misc.transformers import WrapperBERT, WrapperSpanBERT, WrapperSpanBERTSubtoken, WrapperSpanBERT_X
from models.utils.debug import Wrapper1


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
        raise BaseException('no such activation function: %s' % name)


class TextEmbedder(nn.Module):
    def __init__(self, dictionaries, config):
        super(TextEmbedder, self).__init__()
        self.config = config
        self.dim_output = 0
        if 'char_embedder' in config:
            self.char_embedder = TextFieldEmbedderCharacters(dictionaries, config['char_embedder'])
            self.dim_output += self.char_embedder.dim_output
            self.do_char_embedding = True
        else:
            self.do_char_embedding = False
        if 'text_field_embedder' in config:
            self.word_embedder = TextFieldEmbedderTokens(dictionaries, config['text_field_embedder'])
            self.dim_output += self.word_embedder.dim
        if 'bert_embedder' in config:
            self.bert_embedder = WrapperBERT(dictionaries, config['bert_embedder'])
            self.dim_output += self.bert_embedder.dim_output
        if 'spanbert_embedder' in config:
            self.spanbert_embedder = WrapperSpanBERT(dictionaries, config['spanbert_embedder'])
            self.dim_output += self.spanbert_embedder.dim_output
        if 'spanbert_embedder_subtoken' in config:
            self.spanbert_embedder = WrapperSpanBERTSubtoken(dictionaries, config['spanbert_embedder_subtoken'])
            self.dim_output += self.spanbert_embedder.dim_output

        if 'spanbert_embedder_x' in config:
            self.spanbert_embedder = WrapperSpanBERT_X(config['spanbert_embedder_x'])
            self.dim_output += self.spanbert_embedder.dim_output

    def forward(self, data):
        outputs = []
        if 'char_embedder' in self.config:
            outputs.append(self.char_embedder(data['characters']))
        if 'text_field_embedder' in self.config:
            outputs.append(self.word_embedder(data['tokens']))
        if 'whitespace_embedder' in self.config:
            outputs.append(self.whitespace_embedder(data))
        if 'elmo_embedder' in self.config:
            outputs.append(self.ctxt_embedder(data['text']))
        if 'bert_embedder' in self.config:
            outputs.append(self.bert_embedder(data['text']))
            # outputs.append(self.bert_embedder(data['tokens']))
        if 'spanbert_embedder' in self.config:
            outputs.append(self.spanbert_embedder(data['text']))
        if 'spanbert_embedder_subtoken' in self.config:
            outputs.append(self.spanbert_embedder(data['text']))

        if 'spanbert_embedder_x' in self.config:
            outputs.append(self.spanbert_embedder(data['bert_segments'], data['bert_segments_mask']))

        return torch.cat(outputs, -1)


class SpanExtractor(nn.Module):

    def __init__(self, dim_input, config):
        super(SpanExtractor, self).__init__()
        self.span_extractor1 = AverageSpanExtractor(dim_input) if config['avg'] else None
        self.span_extractor2 = None
        self.span_extractor3 = None
        self.dim_output = self.get_output_dims()

    def forward(self, inputs, token2mention, span_indices):
        mentions = []
        if self.span_extractor1 is not None:
            mentions.append(self.span_extractor1(inputs, token2mention))
        if self.span_extractor2 is not None:
            mentions.append(self.span_extractor2(inputs, span_indices)),
        if self.span_extractor3 is not None:
            mentions.append(self.span_extractor3(inputs, span_indices))
        return torch.cat(mentions, -1)

    def get_output_dims(self):
        dims = 0
        if self.span_extractor1 is not None:
            dims += self.span_extractor1.dim_output
        if self.span_extractor2 is not None:
            dims += self.span_extractor2.get_output_dim()
        if self.span_extractor3 is not None:
            dims += self.span_extractor3.get_output_dim()
        return dims


class AverageSpanExtractor(nn.Module):

    def __init__(self, dim_input):
        super(AverageSpanExtractor, self).__init__()
        self.dim_output = dim_input

    def forward(self, sequence_tensor, span_matrix):
        num_batch = sequence_tensor.size()[0]
        y = sequence_tensor.view(-1, self.dim_output)
        spans = torch.matmul(span_matrix, y)
        spans = spans.view(num_batch, -1, self.dim_output)
        return spans


class ResLayerX(nn.Module):

    def __init__(self, dim_input, config):
        super(ResLayerX, self).__init__()
        self.layer = Wrapper1('res', FeedForward(dim_input, config['layer']))
        self.out = nn.Linear(self.layer.dim_output, dim_input)

    def forward(self, tensor):
        return tensor + self.out(self.layer(tensor))


class ResLayer(nn.Module):

    def __init__(self, dim_input, config):
        super(ResLayer, self).__init__()
        self.dp = nn.Dropout(config['dropout'])
        self.input = nn.Linear(dim_input, config['dim'])
        self.fnc = create_activation_function(config['actfnc'])
        self.output = nn.Linear(config['dim'], dim_input)

    def forward(self, tensor):
        h = self.dp(tensor)
        h = self.input(h)
        h = self.fnc(h)
        h = self.output(h)
        return tensor + h


class FeedForward(nn.Module):

    def __init__(self, dim_input, config):
        super(FeedForward, self).__init__()
        self.dim_output = dim_input
        self.layers = []

        if 'type' not in config:
            self.create_default(config)
        elif config['type'] == 'ffnn':
            self.create_ffnn(config)
        elif config['type'] == 'res':
            self.create_res(config)
        elif config['type'] == 'resnet':
            self.create_resnet(config)
        elif config['type'] == 'glu':
            self.create_glu(config)
        else:
            raise BaseException("no such type: ", config['type'])

        self.layers = nn.Sequential(*self.layers)

    def create_default(self, config):
        if config['ln']:
            # from modules.misc import LayerNorm
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

    def create_glu(self, config):
        if 'dp_in' in config:
            self.layers.append(nn.Dropout(config['dp_in']))
        for dim in config['dims']:
            self.layers.append(nn.Linear(self.dim_output, 2 * dim))
            self.layers.append(nn.GLU())
            if 'dp_h' in config:
                self.layers.append(nn.Dropout(config['dp_h']))
            self.dim_output = dim

    def create_res(self, config):
        for _ in range(config['layers']):
            self.layers.append(ResLayerX(self.dim_output, config))

    def create_resnet(self, config):
        for _ in range(config['layers']):
            self.layers.append(ResLayer(self.dim_output, config))

    def forward(self, tensor):
        return self.layers(tensor)


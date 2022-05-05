from typing import List, Optional
import torch
# from allennlp.common.checks import ConfigurationError
from model import logger
from modules.misc.misc import ConfigurationError


def viterbi_decode(tag_sequence: torch.Tensor,
                   transition_matrix: torch.Tensor,
                   tag_observations: Optional[List[int]] = None):
    """
    Perform Viterbi decoding in log space over a sequence given a transition matrix
    specifying pairwise (transition) potentials between tags and a matrix of shape
    (sequence_length, num_tags) specifying unary potentials for possible tags per
    timestep.

    Parameters
    ----------
    tag_sequence : torch.Tensor, required.
        A tensor of shape (sequence_length, num_tags) representing scores for
        a set of tags over a given sequence.
    transition_matrix : torch.Tensor, required.
        A tensor of shape (num_tags, num_tags) representing the binary potentials
        for transitioning between a given pair of tags.
    tag_observations : Optional[List[int]], optional, (default = None)
        A list of length ``sequence_length`` containing the class ids of observed
        elements in the sequence, with unobserved elements being set to -1. Note that
        it is possible to provide evidence which results in degenerate labellings if
        the sequences of tags you provide as evidence cannot transition between each
        other, or those transitions are extremely unlikely. In this situation we log a
        warning, but the responsibility for providing self-consistent evidence ultimately
        lies with the user.

    Returns
    -------
    viterbi_path : List[int]
        The tag indices of the maximum likelihood tag sequence.
    viterbi_score : torch.Tensor
        The score of the viterbi path.
    """
    sequence_length, num_tags = list(tag_sequence.size())
    if tag_observations:
        if len(tag_observations) != sequence_length:
            raise ConfigurationError("Observations were provided, but they were not the same length "
                                     "as the sequence. Found sequence of length: {} and evidence: {}"
                                     .format(sequence_length, tag_observations))
    else:
        tag_observations = [-1 for _ in range(sequence_length)]

    path_scores = []
    path_indices = []

    if tag_observations[0] != -1:
        one_hot = torch.zeros(num_tags)
        one_hot[tag_observations[0]] = 100000.
        path_scores.append(one_hot)
    else:
        path_scores.append(tag_sequence[0, :])

    # Evaluate the scores for all possible paths.
    for timestep in range(1, sequence_length):
        # Add pairwise potentials to current scores.
        summed_potentials = path_scores[timestep - 1].unsqueeze(-1) + transition_matrix
        scores, paths = torch.max(summed_potentials, 0)

        # If we have an observation for this timestep, use it
        # instead of the distribution over tags.
        observation = tag_observations[timestep]
        # Warn the user if they have passed
        # invalid/extremely unlikely evidence.
        if tag_observations[timestep - 1] != -1:
            if transition_matrix[tag_observations[timestep - 1], observation] < -10000:
                logger.warning("The pairwise potential between tags you have passed as "
                               "observations is extremely unlikely. Double check your evidence "
                               "or transition potentials!")
        if observation != -1:
            one_hot = torch.zeros(num_tags)
            one_hot[observation] = 100000.
            path_scores.append(one_hot)
        else:
            path_scores.append(tag_sequence[timestep, :] + scores.squeeze())
        path_indices.append(paths.squeeze())

    # Construct the most likely sequence backwards.
    viterbi_score, best_path = torch.max(path_scores[-1], 0)
    viterbi_path = [int(best_path.numpy())]
    for backward_timestep in reversed(path_indices):
        viterbi_path.append(int(backward_timestep[viterbi_path[-1]]))
    # Reverse the backward path.
    viterbi_path.reverse()
    return viterbi_path, viterbi_score

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

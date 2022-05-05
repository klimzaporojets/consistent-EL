import torch


def logsumexp(tensor: torch.Tensor,
              dim: int = -1,
              keepdim: bool = False) -> torch.Tensor:
    """
    A numerically stable computation of logsumexp. This is mathematically equivalent to
    `tensor.exp().sum(dim, keep=keepdim).log()`.  This function is typically used for summing log
    probabilities.

    Parameters
    ----------
    tensor : torch.FloatTensor, required.
        A tensor of arbitrary size.
    dim : int, optional (default = -1)
        The dimension of the tensor to apply the logsumexp to.
    keepdim: bool, optional (default = False)
        Whether to retain a dimension of size one at the dimension we reduce over.
    """
    max_score, _ = tensor.max(dim, keepdim=keepdim)
    if keepdim:
        stable_vec = tensor - max_score
    else:
        stable_vec = tensor - max_score.unsqueeze(dim)
    return max_score + (stable_vec.exp().sum(dim, keepdim=keepdim)).log()


# WARNING: copy from allennlp
def masked_log_softmax(vector: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if mask is not None:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        vector = vector + (mask + 1e-45).log()
    return torch.nn.functional.log_softmax(vector, dim=dim)

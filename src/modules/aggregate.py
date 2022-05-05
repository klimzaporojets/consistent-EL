import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.spirit import FeedForward
from util.sequence import get_mask_from_sequence_lengths


def Aggregate(dim_mention, config):
    print('->', config)
    if config['type'] == 'average':
        return AggregateAverage(dim_mention, config)
    elif config['type'] == 'self-attention':
        return AggregateSelfAttention(dim_mention, config)
    elif config['type'] == 'max-pool':
        return AggregateMaxPool(dim_mention, config)
    else:
        raise BaseException("no such type:", config['type'])


class AggregateAverage(nn.Module):

    def __init__(self, dim_mention, config):
        super(AggregateAverage, self).__init__()
        self.dim_output = dim_mention

    def forward(self, mention_vectors, mention2concept):
        num_batch = mention_vectors.size(0)
        return torch.matmul(mention2concept, mention_vectors.view(-1, self.dim_output)).view(num_batch, -1, self.dim_output)


class AggregateSelfAttention(nn.Module):

    def __init__(self, dim_mention, config):
        super(AggregateSelfAttention, self).__init__()
        self.ff = FeedForward(dim_mention, config['attention'])
        self.out = nn.Linear(self.ff.dim_output, 1)
        self.dim_output = dim_mention

    def forward(self, mention_vectors, mention2concept):
        num_batch = mention_vectors.size(0)
        mention_vectors = mention_vectors.view(-1, self.dim_output)

        vectors, concept_lengths = AggregateSelfAttention.to_att(mention_vectors, mention2concept)
        scores = self.out(self.ff(vectors)).squeeze(-1)

        mask = get_mask_from_sequence_lengths(concept_lengths, concept_lengths.max().item()).float().to(mention_vectors.device)
        scores = scores - (1.0 - mask) * 1e38
        probs = F.softmax(scores, -1)
        output = torch.matmul(probs.unsqueeze(-2), vectors)
        return output.unsqueeze(-2).view(num_batch, -1, self.dim_output)

    @staticmethod
    def to_att(mention_vectors, mention2concept):
        indices = mention2concept._indices().t().tolist()
        clusters = list([list() for _ in range(mention2concept.size(0))])
        for concept, mention in indices:
            clusters[concept].append(mention)
        maxlen = max([len(c) for c in clusters])
        # add padding
        lengths = []
        for c in clusters:
            lengths.append(len(c))
            c.extend([0 for _ in range(maxlen-len(c))])
        concept_indices = torch.LongTensor(clusters).to(mention_vectors.device)
        vectors = torch.index_select(mention_vectors, 0, concept_indices.view(-1)).view(concept_indices.size() + (-1,))
        return vectors, torch.LongTensor(lengths)

class AggregateMaxPool(nn.Module):

    def __init__(self, dim_mention, config):
        super(AggregateMaxPool, self).__init__()
        self.dim_output = dim_mention

    def forward(self, mention_vectors, mention2concept):
        vectors, concept_lengths = AggregateSelfAttention.to_att(mention_vectors, mention2concept)

        mask = get_mask_from_sequence_lengths(concept_lengths, concept_lengths.max().item()).float().to(mention_vectors.device)
        output, _ = torch.max(vectors - (1.0 - mask).unsqueeze(-1) * 1e38, -2)
        return output

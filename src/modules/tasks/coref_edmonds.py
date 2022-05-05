import torch
import torch.nn as nn
from util.edmonds import mst

from metrics.coref import MetricCoref
from metrics.misc import MetricObjective


class LossCorefEdmonds(nn.Module):

    def __init__(self, task, config):
        super(LossCorefEdmonds, self).__init__()
        self.task = task
        self.weight = config.get('weight', 1.0)
        self.enabled = config['enabled']

    def forward(self, scores, targets, lengths):
        # WARNING: targets not correct

        P = mst(scores, lengths)
        G = mst(scores * targets, lengths)

        margin = ((P - G) * scores).view(scores.size(0), -1).sum(-1)
        print('margin:', margin)
        output = torch.relu(margin).sum()

        return output, P, None

    def create_metrics(self):
        return [MetricCoref(self.task, 'muc', MetricCoref.muc), MetricCoref(self.task, 'bcubed', MetricCoref.b_cubed),
                MetricObjective(self.task)] if self.enabled else []

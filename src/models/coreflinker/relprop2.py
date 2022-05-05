import torch
import torch.nn as nn

from modules.utils.misc import MyGate, overwrite_spans
from models.coreflinker.scorers import OptFFpairs


class ModuleRelProp2(nn.Module):

    def __init__(self, dim_span, span_pair_generator, labels, config):
        super(ModuleRelProp2, self).__init__()
        self.rel_prop = config['rel_prop']
        self.residual = config['residual']
        self.ctxt_ff = config['ctxt_ff']

        self.scorer = OptFFpairs(dim_span, len(labels), config, span_pair_generator)
        # self.A = nn.Linear(len(labels), dim_span, bias=False)
        # self.B = nn.Linear(len(labels), dim_span, bias=False)
        if self.ctxt_ff:
            self.ff = nn.Sequential(
                nn.Linear(dim_span * 3, dim_span, bias=False),
                nn.Tanh(),
                nn.Linear(dim_span, dim_span, bias=False)
            )
        self.gate = MyGate(dim_span)

    def forward(self, all_spans, filtered_spans, sequence_lengths):
        update = filtered_spans['span_vecs']
        filtered_span_begin = filtered_spans['span_begin']
        filtered_span_end = filtered_spans['span_end']
        square_mask = filtered_spans['square_mask']

        relation_scores = self.scorer(update, filtered_span_begin, filtered_span_end)

        update_all = all_spans.copy()
        update_filtered = filtered_spans.copy()

        if self.rel_prop > 0:
            # print('before:', update.norm().item())

            for _ in range(self.rel_prop):
                # probs = torch.sigmoid(relation_scores) * square_mask.unsqueeze(-1)
                # ctxt1 = (self.A(probs) * update.unsqueeze(-2)).sum(-3) / span_lengths.unsqueeze(-1).unsqueeze(-1)
                # ctxt2 = (self.B(probs) * update.unsqueeze(-3)).sum(-2) / span_lengths.unsqueeze(-1).unsqueeze(-1)
                # update = self.gate(update, ctxt1+ctxt2)

                probs1 = torch.sigmoid(relation_scores.max(-1)[0]) * square_mask
                probs2 = probs1.permute(0, 2, 1)
                ctxt1 = torch.matmul(probs1, update) / (probs1.sum(-1) + 1e-7).unsqueeze(-1)
                ctxt2 = torch.matmul(probs2, update) / (probs2.sum(-1) + 1e-7).unsqueeze(-1)
                ctxt = self.ff(torch.cat((update, ctxt1, ctxt2), -1)) if self.ctxt_ff else ctxt1 + ctxt2
                update = self.gate(update, ctxt)

                if self.residual:
                    relation_scores = relation_scores + self.scorer(update, filtered_span_begin, filtered_span_end)
                else:
                    relation_scores = self.scorer(update, filtered_span_begin, filtered_span_end)

            # print('after:', update.norm().item())

            update_filtered['span_vecs'] = update
            update_all['span_vecs'] = overwrite_spans(update_all['span_vecs'], filtered_spans['prune_indices'],
                                                      filtered_spans['span_lengths'], update)

        return update_all, update_filtered, relation_scores

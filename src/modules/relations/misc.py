import torch


def create_relation_targets_2(pred_spans, relations, num_relations, span_lengths):
    gold_spans = relations['gold_spans']
    gold_m2i = relations['gold_m2i']
    gold_relations = relations['gold_relations']
    num_concepts = relations['num_concepts']

    num_batch = span_lengths.size(0)
    max_spans = span_lengths.max().item()

    targets = torch.zeros(num_batch, max_spans, max_spans, num_relations)

    for batch, (p_spans, g_spans, m2i, rels, max_clusters) in enumerate(
            zip(pred_spans, gold_spans, gold_m2i, gold_relations, num_concepts)):
        if len(rels) > 0:
            # max_clusters = max([max(src,dst) for src,dst,_ in rels])+1

            gold2index = {span: idx for span, idx in zip(g_spans, m2i)}
            pred2cluster = torch.LongTensor([gold2index.get(span, max_clusters) for span in p_spans])

            rels = torch.LongTensor(rels)
            cluster_targets = torch.zeros(max_clusters + 1, max_clusters + 1, num_relations)
            cluster_targets[rels[:, 0], rels[:, 1], rels[:, 2]] = torch.ones(rels.size(0))

            dim = (pred2cluster.size(0), pred2cluster.size(0))
            r = pred2cluster.unsqueeze(-1).expand(dim).reshape(-1)
            c = pred2cluster.unsqueeze(-2).expand(dim).reshape(-1)

            indices = torch.arange(pred2cluster.size(0))
            rr = indices.unsqueeze(-1).expand(dim).reshape(-1)
            cc = indices.unsqueeze(-2).expand(dim).reshape(-1)
            targets[batch, rr, cc, :] = cluster_targets[r, c, :]

    return targets.to(span_lengths.device)

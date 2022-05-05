import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from modules.aggregate import Aggregate

from datass.collate import collate_character, collate_sparse2, collate_sparse_to_dense_4
from modules.graph import create_graph
from modules.seq2seq import sequence_mask
from modules.spirit import TextEmbedder, FeedForward, Seq2Seq
from modules.tasks import create_ner_task
from modules.tasks.coref import LossCoref
from modules.tasks.relations import LossRelationsNew
from util.sequence import get_mask_from_sequence_lengths


# import modules.graphnet.corefnet (kzaporoj) - not used


def collate_ncr3(batch, device, char_padding, nertasks, num_relations):
    batch.sort(key=lambda x: x['tokens'].size()[0], reverse=True)

    last_idx = max([len(x['tokens']) for x in batch]) - 1

    sequence_lengths = torch.LongTensor([x['tokens'].size()[0] for x in batch])
    characters = collate_character([x['characters'] for x in batch], 50, char_padding)
    tokens = rnn_utils.pad_sequence([x['tokens'] for x in batch], batch_first=True)
    indices = rnn_utils.pad_sequence([x['tokens-indices'] for x in batch], batch_first=True, padding_value=last_idx)
    labels = {task: rnn_utils.pad_sequence([x[task] for x in batch], batch_first=True).to(device) for task in nertasks}

    # relations
    max_tokens = tokens.size()[1]
    max_mentions = max([x['token2mention'][0] for x in batch])
    max_concepts = max([x['mention2concept'][0] for x in batch])
    token2mention = collate_sparse2([x['token2mention'] for x in batch], max_mentions, max_tokens)
    mention2concept = collate_sparse2([x['mention2concept'] for x in batch], max_concepts, max_mentions)
    relations = torch.zeros(len(batch), max_concepts, max_concepts, num_relations)
    collate_sparse_to_dense_4(relations, [x['relations'] for x in batch])
    mention_lengths = torch.LongTensor([x['coref'][0] for x in batch])
    concept_lengths = torch.LongTensor([x['mention2concept'][0] for x in batch])

    # labels['coref'] = collate_sparse_to_dense_3([x['coref'] for x in batch]).to(device)
    # labels['relations'] = relations.to(device).permute(0, 3, 1, 2)

    gold_spans = [[(m[0], m[1]) for m in x['spans']] for x in batch]

    gold_clusters = []
    for spans, m2c in zip(gold_spans, [x['mention2concept'] for x in batch]):
        clusters = [list() for _ in range(m2c[0])]
        for mention, concept in zip(m2c[3], m2c[2]):
            clusters[concept].append(spans[mention])
        gold_clusters.append(clusters)

    metadata = {
        'identifiers': [x['id'] for x in batch],
        'mentions': [x['mentions'] for x in batch],
        'tokens': [x['text'] for x in batch]
    }

    inputs = {
        'tokens': tokens.to(device),
        'characters': characters.to(device),
        'sequence_lengths': sequence_lengths.to(device),
        'token_indices': indices.to(device)
    }

    coref = {
        'gold_spans': gold_spans,
        'gold_clusters': gold_clusters,
        'gold_m2i': [x['clusters'] for x in batch],
    }

    relations = {
        'gold_spans': gold_spans,
        'gold_clusters': gold_clusters,
        'gold_relations': [x['relations'] for x in batch],
    }

    return {
        'inputs': inputs,
        'labels': labels,
        'coref': coref,
        'relations': relations,
        'metadata': metadata
    }


# class CombineNet(nn.Module):

#     def __init__(self, dim_input, config):
#         super(CombineNet, self).__init__()
#         self.dim_input = dim_input
#         self.net = FeedForward(dim_input, config)
#         self.dim_output = self.net.dim_output

#     def forward(self, inputs, input2output):
#         num_batch = inputs.size()[0]
#         y = inputs.contiguous().view(-1, self.dim_input)
#         outputs = torch.matmul(input2output, y)
#         outputs = self.net(outputs)
#         return outputs.view(num_batch, -1, self.dim_output)


class SpanBuilder:

    def __init__(self, batch_size):
        self.spans = [set() for _ in range(batch_size)]

    def add(self, predictions):
        for spans, prediction in zip(self.spans, predictions):
            spans.update([(token_begin, token_end - 1) for token_begin, token_end, tag in prediction])

    def build(self):
        return [list(x) for x in self.spans]

    def get_max_len(self):
        return max([len(x) for x in self.spans])


def create_token2mention(spans, sequence_lengths, device):
    max_spans = max(len(x) for x in spans)
    max_tokens = sequence_lengths.max().item()

    if max_spans > 0:
        rows, cols, vals = [], [], []
        row_offs, col_offs = 0, 0
        for myspans in spans:
            for span_idx, (begin, end) in enumerate(myspans):
                for pos in range(begin, end + 1):
                    rows.append(row_offs + span_idx)
                    cols.append(col_offs + pos)
                    vals.append(1.0 / (end - begin + 1))
            row_offs += max_spans
            col_offs += max_tokens

        token2mention = torch.sparse.FloatTensor(torch.LongTensor([rows, cols]), torch.FloatTensor(vals),
                                                 torch.Size([row_offs, col_offs])).to(device)
    else:
        token2mention = None

    return token2mention


class CheckGradient:
    iter = 0

    def __init__(self, tb_logger):
        self.gradients = {}
        self.num = 0
        self.tb_logger = tb_logger

    def run(self, name, inputs):
        self.num += 1

        def myhook(layer, grad_input, grad_output):
            self.gradients[name] = grad_input[0]
            if len(self.gradients) == self.num:
                self.inspect()

        tmp = nn.Sequential()
        tmp.register_backward_hook(myhook)
        return tmp(inputs)

    def inspect(self):
        sum = None
        for value in self.gradients.values():
            if sum is None:
                sum = value
            else:
                sum = sum + value

        snorm = sum.norm().item()
        for key, value in self.gradients.items():
            vnorm = value.norm().item()
            proj = (value * sum).sum().item() / snorm if vnorm != 0 else 0.0
            # print('key: {:12}   norm: {:14}   proj: {:14}'.format(key, vnorm, proj))
            self.tb_logger.log_value('norm/{}'.format(key), vnorm, CheckGradient.iter)
            self.tb_logger.log_value('proj/{}'.format(key), proj, CheckGradient.iter)
        self.tb_logger.log_value('norm/sum', snorm, CheckGradient.iter)
        CheckGradient.iter += 1
        # print('snorm:', snorm)


def create_mention2concept(spans, clusters, device):
    num_batch = len(spans)
    max_spans = max([len(x) for x in spans])
    max_clusters = max([len(x) for x in clusters])
    mentions = []
    concepts = []
    vals = []
    offset_mentions = 0
    offset_concepts = 0

    for myspans, myclusters in zip(spans, clusters):
        span2index = {span: idx for idx, span in enumerate(myspans)}

        for idx, cluster in enumerate(myclusters):
            for span in cluster:
                mentions.append(offset_mentions + span2index[span])
                concepts.append(offset_concepts + idx)
                vals.append(1.0 / len(cluster))

        offset_mentions += max_spans
        offset_concepts += max_clusters

    return torch.sparse.FloatTensor(torch.LongTensor([concepts, mentions]), torch.FloatTensor(vals),
                                    torch.Size([offset_concepts, offset_mentions])).to(device)


def create_relation_targets(pred_clusters, gold_clusters, gold_relations, num_relations, device):
    # print('pred_clusters:', pred_clusters)
    # print('gold_relations:', gold_relations)
    # print('num_relations:', num_relations)

    num_batch = len(pred_clusters)
    max_clusters = max([len(x) for x in pred_clusters])

    targets = torch.zeros(num_batch, max_clusters, max_clusters, num_relations)

    for batch, (pred, gold, relations) in enumerate(zip(pred_clusters, gold_clusters, gold_relations)):
        pred = [set(x) for x in pred]
        gold = [set(x) for x in gold]

        rels = []
        len_gold = len(gold)
        for src, dst, rel in relations[1]:
            # if src < len_gold and dst < len_gold:       # ARG!!!
            rels.append((gold[src], gold[dst], rel))

        for src, src_cluster in enumerate(pred):
            for dst, dst_cluster in enumerate(pred):
                for r in rels:
                    if src_cluster <= r[0] and dst_cluster <= r[1]:
                        targets[batch, src, dst, r[2]] = 1.0

    return targets.to(device)


def span_intersection(pred, gold):
    numer = 0
    for p, g in zip(pred, gold):
        numer += len(set(p) & set(g))
    return numer


class NCR3(nn.Module):

    def __init__(self, dictionaries, config):
        super(NCR3, self).__init__()

        print("NCR3: ner + coref + rel")
        self.embedder = TextEmbedder(dictionaries, config)

        self.shared_seq = None
        self.ner_seq = None
        self.coref_seq = None
        self.relation_seq = None

        if 'shared_seq2seq' in config:
            self.shared_seq = Seq2Seq(self.embedder.dim_output, config['shared_seq2seq'])
            shared_dim = self.shared_seq.dim_output
        else:
            shared_dim = 0
        ner_dim = shared_dim
        coref_dim = shared_dim
        rel_dim = shared_dim

        if 'ner_seq2seq' in config:
            self.ner_seq = Seq2Seq(self.embedder.dim_output, config['ner_seq2seq'])
            ner_dim += self.ner_seq.dim_output
        self.ner_ff = FeedForward(ner_dim, config['ner_ff'])
        self.ner_tasks = nn.ModuleList(
            [create_ner_task(task, self.ner_ff.dim_output, dictionaries) for task in config['tasks-ner']])

        if 'coref_seq2seq' in config:
            self.coref_seq = Seq2Seq(self.embedder.dim_output, config['coref_seq2seq'])
            coref_dim += self.coref_seq.dim_output
        self.coref_ff = FeedForward(ner_dim, config['coref_ff'])
        # self.coref_mention_net = CombineNet(self.coref_ff.dim_output, config['mention_net'])
        self.coref_mention_agr = Aggregate(self.coref_ff.dim_output, config['mention_agr'])
        self.coref_mention_net = FeedForward(self.coref_mention_agr.dim_output, config['mention_net'])

        self.coref_scorer = create_graph(self.coref_mention_net.dim_output, 1, config['coref_scorer'])
        self.coref_task = LossCoref('coref', config['task-coref'])

        if 'rel_seq2seq' in config:
            self.relation_seq = Seq2Seq(self.embedder.dim_output, config['rel_seq2seq'])
            rel_dim += self.relation_seq.dim_output
        self.relation_ff = FeedForward(rel_dim, config['rel_ff'])
        self.relation_mention_agr = Aggregate(self.relation_ff.dim_output, config['mention_agr'])
        self.relation_mention_net = FeedForward(self.relation_mention_agr.dim_output, config['mention_net'])
        self.relation_concept_agr = Aggregate(self.relation_mention_net.dim_output, config['concept_agr'])
        self.relation_concept_net = FeedForward(self.relation_concept_agr.dim_output, config['concept_net'])
        self.relation_labels = dictionaries['relations'].tolist()
        self.relation_scorer = create_graph(self.relation_concept_net.dim_output, len(self.relation_labels),
                                            config['relation_scorer'])
        self.relation_task = LossRelationsNew(config['task-relations'], self.relation_labels)

        self.weight = config.get('weight', 1.0)
        print("global weight:", self.weight)
        self.eval_correct = config['correct-evaluation']
        self.train_on_predicted_spans = config['train-on-predicted-spans']
        self.span_upperbound_factor = config['span-upperbound-factor']
        self.debug = False

    def collate_func(self, datasets, device):
        char_padding = self.embedder.char_embedder.padding
        print("CHAR PADDING:", char_padding)
        return lambda x: collate_ncr3(x, device, char_padding, [t.name for t in self.ner_tasks],
                                      len(self.relation_labels))

    def begin_epoch(self):
        self.span_recall_numer = 0
        self.span_recall_denom = 0
        self.obj = {'tags': 0.0, 'auto': 0.0, 'coref': 0.0, 'relations': 0.0}

    def end_epoch(self, dataset_name):
        print("{}-span-recall: {} / {} = {}".format(dataset_name, self.span_recall_numer, self.span_recall_denom,
                                                    self.span_recall_numer / self.span_recall_denom))
        for key, value in self.obj.items():
            print('{}-{}-loss: {}\n'.format(dataset_name, key, value))

    def forward(self, inputs, labels, coref, relations, metadata, metrics=[]):
        total_obj = torch.tensor(0.0).cuda()
        output = {}

        tokens = inputs['tokens']
        characters = inputs['characters']
        sequence_lengths = inputs['sequence_lengths']
        token_indices = inputs['token_indices']

        embeddings = self.embedder(characters, tokens)

        ner_outputs = []
        coref_outputs = []
        relation_outputs = []

        if self.shared_seq is not None:
            shared_outputs = self.shared_seq(embeddings, sequence_lengths, token_indices)
            if isinstance(shared_outputs, tuple):
                n, c, r = shared_outputs
                # n, c, r = shared_outputs.unbind(-1)
            else:
                n = shared_outputs
                c = shared_outputs
                r = shared_outputs
            ner_outputs.append(n)
            coref_outputs.append(c)
            relation_outputs.append(r)

        if self.ner_seq is not None:
            x = self.ner_seq(embeddings, sequence_lengths, token_indices)
            ner_outputs.append(x)

        # checker = CheckGradient(self.tb_logger)

        ner_outputs = torch.cat(ner_outputs, -1)
        # ner_outputs = checker.run('ner', ner_outputs)
        ner_outputs = self.ner_ff(ner_outputs)

        if self.train_on_predicted_spans or (self.eval_correct and not self.training):
            num_batch = tokens.size(0)
            spanbuilder = SpanBuilder(num_batch)
        else:
            spanbuilder = None

        seq_mask = sequence_mask(sequence_lengths).to(embeddings.device)

        for task in self.ner_tasks:
            if task.enabled:
                ner_output = task(ner_outputs, labels[task.name], sequence_lengths, seq_mask,
                                  predict=(not self.training or self.train_on_predicted_spans))
                output[task.name] = ner_output
                total_obj += ner_output['loss']

                self.obj[task.name] += ner_output['loss'].item()

                if 'pred' in ner_output and spanbuilder is not None:
                    spanbuilder.add(ner_output['pred'])

        gold_spans = coref['gold_spans']

        if spanbuilder is not None:
            max_pred_spans = spanbuilder.get_max_len()
            max_gold_spans = max([len(x) for x in gold_spans])

            if self.training and (max_pred_spans > max_gold_spans * self.span_upperbound_factor or max_pred_spans == 0):
                print("WARNING: not enough or too many predicted spans. Backing off to gold spans.")
                print('max_pred_spans:', max_pred_spans)
                print('max_gold_spans:', max_gold_spans)
                pred_spans = gold_spans
            else:
                pred_spans = spanbuilder.build()
                print('pred_spans:', [len(x) for x in pred_spans])

                self.span_recall_numer += span_intersection(pred_spans, gold_spans)
        else:
            pred_spans = gold_spans

        self.span_recall_denom += sum([len(x) for x in gold_spans])

        token2mention = create_token2mention(pred_spans, sequence_lengths, embeddings.device)

        if self.coref_seq is not None:
            x = self.coref_seq(embeddings, sequence_lengths, token_indices)
            coref_outputs.append(x)

        if token2mention is not None:
            coref_outputs = torch.cat(coref_outputs, -1)
            # coref_outputs = checker.run('coref', coref_outputs)
            coref_outputs = self.coref_ff(coref_outputs)
            coref_mentions = self.coref_mention_net(self.coref_mention_agr(coref_outputs, token2mention))
        else:
            coref_mentions = None

        if self.coref_task.enabled:
            gold_m2i = coref['gold_m2i']

            span_lengths = torch.LongTensor([len(x) for x in pred_spans]).to(embeddings.device)

            if coref_mentions is not None:
                mask = get_mask_from_sequence_lengths(span_lengths, span_lengths.max()).float()
                square_mask = torch.bmm(mask.unsqueeze(-1), mask.unsqueeze(-2))
                scores = self.coref_scorer(coref_mentions, square_mask, span_lengths).squeeze(-1)
            else:
                scores = None

            if not self.training:
                coref_targets = None
            else:
                coref_targets = None
                # coref_targets = create_coref_target(pred_spans, gold_spans, gold_m2i).to(embeddings.device)

            coref_output = self.coref_task(scores, coref_targets, gold_m2i=gold_m2i, pred_spans=pred_spans,
                                           gold_spans=gold_spans, predict=True)
            output['coref'] = coref_output
            total_obj += coref_output['loss']

            self.obj['coref'] += coref_output['loss'].item()

        if self.relation_seq is not None:
            x = self.relation_seq(embeddings, sequence_lengths, token_indices)
            relation_outputs.append(x)

        if self.relation_task.enabled:
            pred_clusters = output['coref']['pred']
            # gold_clusters = output['coref']['gold'] # WHY IS THIS INCORRECT?
            gold_clusters = relations['gold_clusters']

            if token2mention is not None:
                mention2concept = create_mention2concept(pred_spans, pred_clusters, embeddings.device)

                relation_outputs = torch.cat(relation_outputs, -1)
                # relation_outputs = checker.run('relations', relation_outputs)
                relation_outputs = self.relation_ff(relation_outputs)
                relation_mentions = self.coref_mention_net(self.relation_mention_agr(relation_outputs, token2mention))
                relation_concepts = self.relation_concept_net(
                    self.relation_concept_agr(relation_mentions, mention2concept))

                num_concepts = torch.LongTensor([len(x) for x in pred_clusters]).to(embeddings.device)
                mask = get_mask_from_sequence_lengths(num_concepts, num_concepts.max()).float()
                square_mask = torch.bmm(mask.unsqueeze(-1), mask.unsqueeze(-2))
                rel_scores = self.relation_scorer(relation_concepts, square_mask, num_concepts)

                if self.debug:
                    print('min:', rel_scores.min().item())
                    print('max:', rel_scores.max().item())
            else:
                rel_scores = None

            rel_targets = create_relation_targets(pred_clusters, gold_clusters, relations['gold_relations'],
                                                  len(self.relation_labels), device=embeddings.device)
            if self.debug:
                print('relations active:', rel_targets.sum().item())

            relation_output = self.relation_task(rel_scores, rel_targets, output['coref']['pred'], relations,
                                                 predict=not self.training)
            output['relations'] = relation_output
            total_obj += relation_output['loss']

            self.obj['relations'] += relation_output['loss'].item()

        for m in metrics:
            if m.task in output:
                m.update2(output[m.task], metadata)

        return total_obj * self.weight, output, None

    def create_metrics(self):
        metrics = []
        for task in self.ner_tasks:
            metrics.extend(task.create_metrics())
        metrics.extend(self.relation_task.create_metrics())
        metrics.extend(self.coref_task.create_metrics())
        return metrics

    def write_model(self, filename):
        print("write model:", filename)
        mydict = {}
        for k, v in self.state_dict().items():
            if k.startswith('word_embeddings'):
                print("skip:", k)
                continue
            else:
                mydict[k] = v
        torch.save(mydict, filename)

    def load_model(self, filename, config, to_cpu=False):
        if to_cpu:
            partial = torch.load(filename, map_location=torch.device('cpu'))
        else:
            partial = torch.load(filename)

        # del partial['seq2seq.rnn1.rnn.module.weight_hh_l0']
        # del partial['seq2seq.rnn1.rnn.module.weight_hh_l0_reverse']
        # del partial['seq2seq.rnn2.rnn.module.weight_hh_l0']
        # del partial['seq2seq.rnn2.rnn.module.weight_hh_l0_reverse']
        # del partial['seq2seq.rnn3.rnn.module.weight_hh_l0']
        # del partial['seq2seq.rnn3.rnn.module.weight_hh_l0_reverse']
        # print(partial.keys())

        # update because word_embed is missing
        state = self.state_dict()
        state.update(partial)
        self.load_state_dict(state)

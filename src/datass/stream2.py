from collections import Counter

import json
import time
import torch
import torch.nn.utils.rnn as rnn_utils

from datass.collate import collate_mentions_sparse, collate_table, decode_table
from datass.dictionary import Dictionary
from torch.utils.data import Dataset
from tqdm import tqdm


class SubsetDataset(Dataset):
    def __init__(self, name, dataset, indices):
        self.name = name
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def load_array_tokens(tokens, dict):
    return torch.LongTensor([dict.lookup(x) for x in tokens])


def load_unique_indices(tokens):
    tmp = Dictionary()
    return torch.LongTensor([tmp.lookup(x) for x in tokens])


def load_characters(tokens, dict):
    output = []
    for token in tokens:
        output.append([dict.lookup(c) for c in token])
    return output


def load_characters2(tokens, dict):
    output = []
    for token in tokens:
        token = '<' + token + '>'
        output.append([dict.lookup(c) for c in token])
    return output


def load_sparse_tensor(data):
    dims = int(data[0])
    shape = [int(x) for x in data[1:dims + 1]]
    I = []
    V = []
    for i in range(int((len(data) - 1) / (dims + 1))):
        begin = (i + 1) * (dims + 1)
        I.append([int(x) for x in data[begin:begin + dims]])
        V.append(data[begin + dims])
    return (shape, I, V)


def transform_copy(stream2, config):
    src = config['src']
    dst = config['dst']
    for instance in stream2.data:
        instance[dst] = instance[src]


def transform_characters(stream2, config):
    print("RUN transform_characters:", config)
    field = config['field']
    dict = stream2.dictionaries_all[config['dict']]
    threshold = config['threshold']
    padding = config['padding']

    for instance in stream2.data:
        instance[field] = load_characters2(instance['tokens'], dict)

    if dict.update == False:
        print("WARNING: CHARACTER DICTIONARY IS ALREADY INITIALIZED")
    elif config['train']:
        c = Counter()
        for instance in stream2.data:
            for ch in instance[field]:
                c.update(ch)

        chars = []
        items = dict.tolist()
        for ch, count in c.items():
            if count >= threshold:
                chars.append(items[ch])

        print("WARNING: NOT CLEARING DICTIONARY")
        # dict.clear()
        # dict.add(padding)
        for ch in chars:
            dict.add(ch)
        dict.update = False

        for instance in stream2.data:
            instance[field] = load_characters2(instance['tokens'], dict)
    else:
        print("WARNING: not thresholding character dictionary")


def transform_indices(stream2, config):
    print("RUN transform_indices:", config)
    for instance in stream2.data:
        instance['tokens-indices'] = load_unique_indices(instance['tokens'])


def transform_lookup_tokens(stream2, config):
    print("RUN transform_lookup_tokens:", config)
    field = config['field']
    dict = stream2.dictionaries_all[config['dict']]
    old_size = dict.size
    for instance in stream2.data:
        instance[field] = load_array_tokens(instance[field], dict)
    print("new tokens added to dictionary:", dict.size - old_size)


# def transform_seqlabels_multi_tags(stream2, config):
#     print("RUN transform_seqlabels_multi_tags:", config)
#     field = config['field']
#     dict = stream2.dictionaries_all[config['dict']]
#     for i, instance in enumerate(stream2.data):
#         # print("labels:", instance[field])
#         for labels in instance[field]:
#             [dict.lookup(x) for x in labels]
#     for i, instance in enumerate(stream2.data):
#         length = len(instance[field])
#         array = np.zeros((length, dict.size))
#         for pos, labels in enumerate(instance[field]):
#             idx = [dict.lookup(x) for x in labels]
#             array[pos, idx] = 1
#         instance[field] = torch.FloatTensor(array)

def transform_spans(stream2, config):
    field = config['field']
    dict = stream2.dictionaries_all[config['dict']]
    for i, instance in enumerate(stream2.data):
        instance[field] = [(start, end, dict.lookup(label)) for start, end, label in instance[field]]


# decode spans from sparse matrix
def transform_decode_spans(stream2, config):
    field = config['field']
    for instance in stream2.data:
        rows, cols, irows, icols, vals = instance['token2mention']
        spans = [list() for i in range(rows)]
        for irow, icol, val in zip(irows, icols, vals):
            spans[irow].append(icol)
        span_start = [min(x) for x in spans]
        span_end = [max(x) for x in spans]
        spans = [span_start, span_end]
        instance[field] = list(map(list, zip(*spans)))


def transform_decode_spans_with_types(stream2, config):
    field = config['field']
    bi_labels = stream2.dictionaries_all[config['srcdict']].tolist()
    dictionary = stream2.dictionaries_all[config['dstdict']]

    lengths = []

    for instance in stream2.data:
        from metrics.f1 import decode_multiner
        sequence_lengths = torch.LongTensor([instance['tokens'].size()[0]])
        instance[field] = decode_multiner(instance['tags'].unsqueeze(0), sequence_lengths, bi_labels)[0]

        for begin, end, tag in instance[field]:
            dictionary.lookup(tag)
            lengths.append(end - begin)

    print("max span length:", max(lengths))


def create_data_transformer(config):
    if config['type'] == 'characters':
        return lambda stream2: transform_characters(stream2, config)
    elif config['type'] == "indices":
        return lambda stream2: transform_indices(stream2, config)
    elif config['type'] == "lookup-tokens":
        return lambda stream2: transform_lookup_tokens(stream2, config)
    # elif config['type'] == "seqlabels-multi-tags":
    #     return lambda stream2: transform_seqlabels_multi_tags(stream2, config)
    elif config['type'] == "spans":
        return lambda stream2: transform_spans(stream2, config)
    elif config['type'] == "decode-spans":
        return lambda stream2: transform_decode_spans(stream2, config)
    elif config['type'] == "decode-spans-with-types":
        return lambda stream2: transform_decode_spans_with_types(stream2, config)
    elif config['type'] == 'copy':
        return lambda stream2: transform_copy(stream2, config)
    else:
        raise BaseException("no such data transformer:", config)


class Stream2Dataset(Dataset):

    def __init__(self, name, filename, config, dictionaries):
        self.name = name
        self.data = []
        self.dictionaries = {k: dictionaries[v] for k, v in config['fields'].items()}
        self.dictionaries_all = dictionaries
        self.load = set(config['load'])

        has_candidates = False

        tic = time.time()
        print("Loading {}".format(filename))
        print("   char prefix/suffix enabled")
        instance = None
        count = 0
        with open(filename) as file:
            for line in file:
                line = line.strip()

                if line.startswith("s id "):
                    instance = {'id': line[5:]}
                    self.data.append(instance)

                elif line.startswith("i[] token2mention "):
                    if 'token2mention' in self.load:
                        instance['token2mention'] = self.convertArray(line.split(" ")[3:])

                elif line.startswith("i[] targets "):
                    if 'targets' in self.load:
                        instance['targets'] = self.convertArray(line.split(" ")[3:])
                        has_candidates = True

                # elif line.startswith("sm token2mention "):
                #     if 'token2mention' in self.load:
                #         instance['token2mention'] = self.convertSparseMatrix(line.split(" ")[2:])

                elif line.startswith("x mention2candidate "):
                    if 'mention2candidate' in self.load:
                        instance['mention2candidate'] = self.convertArray(line.split(" ")[2:])

                elif len(line) == 0:
                    count += 1

                elif line.startswith("json "):
                    sep = line.find(" ", 5)
                    field = line[5:sep]
                    instance[field] = json.loads(line[sep:])

                elif line.startswith("s[] "):
                    sep = line.find(" ", 4)
                    field = line[4:sep]
                    tokens = line.split(" ")[3:]

                    if field in self.load:
                        instance[field] = tokens

                elif line.startswith("sm "):
                    data = line.split(" ")
                    field = data[1]
                    if field in self.load:
                        instance[field] = self.convertSparseMatrix(data[2:])

                elif line.startswith("st "):
                    data = line.split(" ")
                    field = data[1]
                    if field in self.load:
                        instance[field] = load_sparse_tensor(data[2:])

                elif line.startswith("i[] "):
                    data = line.split(" ")
                    field = data[1]
                    if field in self.load:
                        instance[field] = self.convertArray(data[3:])

                elif line.startswith("s "):
                    sep = line.find(" ", 2)
                    field = line[2:sep]

                    if field in self.load:
                        instance[field] = line[(sep + 1):].replace('\\r', '\r').replace('\\n', '\n').replace('\\\\',
                                                                                                             '\\')

                else:
                    raise BaseException("invalid line", line)

        for transformer in [create_data_transformer(cfg) for cfg in config['transformers']]:
            transformer(self)

        skips = 0
        if has_candidates:
            print("WARNING: candidates enabled")
            docs = []
            for i, instance in enumerate(self.data):
                try:
                    num_candidates = instance['candidates'].size()[0]
                    if num_candidates > 0:
                        targets = torch.zeros(num_candidates)
                        for x in instance['targets'].tolist():
                            targets[x] = 1.0
                        instance['targets'] = targets
                        docs.append(i)
                    elif config.get('remove-no-links', True):
                        # print("skip instance without candidates")
                        skips += 1
                    else:
                        targets = torch.zeros(0)
                        instance['targets'] = targets
                        docs.append(i)
                except Exception as e:
                    print('Error in:', instance)
                    print("num_candidates:", num_candidates)
                    raise e
            self.docs = docs
        else:
            self.docs = list(range(len(self.data)))

        if config.get('remove-long-instances', 0) > 0:
            maxlen = config['remove-long-instances']
            mydocs = []
            for i in self.docs:
                instance = self.data[i]['tokens']
                if instance.size()[0] <= maxlen:
                    mydocs.append(i)
                else:
                    skips += 1
            self.docs = mydocs

        if config.get('remove-empty', False):
            print("WARNING: removing empty sequences")
            docs = []
            for i, instance in enumerate(self.data):
                if instance['tokens'].size()[0] > 0:
                    docs.append(i)
                else:
                    skips += 1
            self.docs = docs

        print("   instances: {} {}".format(len(self.data), count))
        print("   docs: {}".format(len(self.docs)))
        print("   skipped: {}".format(skips))
        print("done. ({})".format(time.time() - tic))
        print()

    def convertTokens(self, tokens):
        return torch.LongTensor([self.words.lookup(x) for x in tokens])

    def convertTargets(self, tokens):
        return torch.LongTensor([self.entities.lookup(x) for x in tokens])

    def convertMatrixTargets(self, tokens):
        rows, cols = int(tokens[0]), int(tokens[1])
        matrix = []
        ptr = 2
        for r in range(rows):
            row = []
            for c in range(cols):
                row.append(self.entities.lookup(tokens[ptr]))
                ptr += 1
            matrix.append(row)
        return torch.LongTensor(matrix)

    def convertArray(self, tokens):
        return torch.IntTensor([int(x) for x in tokens])

    def convertSparseMatrix(self, data):
        rows, cols, size = int(data[0]), int(data[1]), int(data[2])
        irow = []
        icol = []
        vals = []
        offs = 3
        for i in range(size):
            irow.append(int(data[offs]))
            icol.append(int(data[offs + 1]))
            vals.append(float(data[offs + 2]))
            offs += 3
        return rows, cols, irow, icol, vals

    def __getitem__(self, index):
        return self.data[self.docs[index]]

    def __len__(self):
        return len(self.docs)

    def get_histogram(self, field):
        dict = self.dictionaries[field]
        print("counting histogram of {} (size: {})".format(field, dict.size))
        histogram = torch.LongTensor(dict.size)
        for instance in tqdm(self):
            for idx in instance[field].tolist():
                histogram[idx] += 1
        return histogram.float()


def collate_stream2(batch):
    batch.sort(key=lambda x: x['tokens'].size()[0], reverse=True)

    # tmp = []
    # tmp.extend(batch)
    # tmp.extend(batch)
    # batch = tmp

    seqlens = [x['tokens'].size()[0] for x in batch]
    tokens = rnn_utils.pad_sequence([x['tokens'] for x in batch], batch_first=True)
    maxlen = tokens.size()[1]
    token2mention = collate_mentions_sparse([x['token2mention'] for x in batch], maxlen)
    # mention2candidate = collate_sparse([decode_table(x['mention2candidate']) for x in batch])
    mention2candidate = collate_table([x['mention2candidate'] for x in batch])
    targets = torch.cat([x['targets'] for x in batch], 0)
    candidates = torch.cat([x['candidates'] for x in batch], 0)

    return {
        'seqlens': seqlens,
        'tokens': tokens,
        'token2mention': token2mention,
        'mention2candidate': decode_table(mention2candidate),
        'candidates': candidates,
        'targets': targets,
        'table': mention2candidate
    }


def collate_stream2_negative_samples(batch):
    batch.sort(key=lambda x: x['tokens'].size()[0], reverse=True)

    seqlens = [x['tokens'].size()[0] for x in batch]
    tokens = rnn_utils.pad_sequence([x['tokens'] for x in batch], batch_first=True)
    maxlen = tokens.size()[1]
    token2mention = collate_mentions_sparse([x['token2mention'] for x in batch], maxlen)
    targets = torch.cat([x['targets'] for x in batch], 0)

    return {
        'seqlens': seqlens,
        'tokens': tokens,
        'token2mention': token2mention,
        'targets': targets
    }


# def collate_mentions_dense(batch, maxlen):
#     token2mention = torch.zeros([targets.size()[0], tokens.size()[0] * maxlen])
#     for row, a in enumerate([x['token2mention'] for x in batch]):
#         for col, b in enumerate(a):
#             if b >= 0:
#                 token2mention[row, row * maxlen + col] = 1.0


def collate_text1(batch):
    batch.sort(key=lambda x: x.size()[0], reverse=True)
    packed = rnn_utils.pad_sequence(batch)
    return packed


def collate_text2(batch):
    batch.sort(key=lambda x: x.size()[0], reverse=True)
    packed = rnn_utils.pad_sequence(batch)
    token2mention = torch.zeros([64, packed.size()[0] * packed.size()[1]])
    return packed, token2mention

from collections import Counter

import numpy as np
import torch

from cpn.tokenizer import TokenizerCPN


# improved implementation: delay transformation after loading in the data
def create_data_transformer2(config, dictionaries):
    # kzaporoj - no transformer so far, lets see if errors arise
    raise BaseException("no such data transformer:", config)
    # if config['type'] == 'tokenize':
    #     return TranformTokenize(config)
    # elif config['type'] == 'lookup-tokens':
    #     return TranformLookupTokens(config, dictionaries)
    # elif config['type'] == 'token-indices':
    #     return TranformTokenIndices(config)
    # elif config['type'] == "seqlabels-multi-tags":
    #     return TransformSeqlabelsToMultitags(config, dictionaries)
    # elif config['type'] == "seqlabels-spans":
    #     return TransformSpansToTensors(config, dictionaries)
    # elif config['type'] == "lookup-spans":
    #     return TransformLookupSpans(config, dictionaries)
    # elif config['type'] == "characters":
    #     return TransformCharacters(config, dictionaries)
    # elif config['type'] == "fix-spans":
    #     return TransformFixSpans(config, dictionaries)
    # else:
    #     raise BaseException("no such data transformer:", config)


class Transform:

    def initialize(self, stream2):
        raise BaseException("NOT IMPLEMENTED")

    def transform(self, stream2):
        raise BaseException("NOT IMPLEMENTED")


class TranformTokenize(Transform):

    def __init__(self, config):
        self.field_input = config['input']
        self.tokenizer = TokenizerCPN()

    def initialize(self, stream2):
        return

    def transform(self, stream2):
        print("RUN TranformTokenize")
        for i, instance in enumerate(stream2.data):
            instance['old_begin'] = instance['begin']
            instance['old_end'] = instance['end']

            text = instance[self.field_input]
            tokens = self.tokenizer.tokenize(text)
            instance['tokens'] = [x['token'] for x in tokens]
            instance['begin'] = torch.IntTensor([x['offset'] for x in tokens])
            instance['end'] = torch.IntTensor([x['offset'] + x['length'] for x in tokens])
            instance['text'] = instance['tokens']


def get_token_buckets(tokens):
    token2idx = {}
    for token in tokens:
        token = token.lower()
        if token not in token2idx:
            token2idx[token] = len(token2idx)
    return [token2idx[token.lower()] for token in tokens]


class TranformTokenIndices(Transform):

    def __init__(self, config):
        self.field_input = config['input']
        self.field_output = config['output']

    def initialize(self, stream2):
        return

    def transform(self, stream2):
        print("RUN TranformTokenIndices")
        for i, instance in enumerate(stream2.data):
            tokens = instance[self.field_input]
            instance[self.field_output] = torch.LongTensor(get_token_buckets(tokens))


class TranformLookupTokens(Transform):

    def __init__(self, config, dictionaries):
        self.config = config
        self.field_input = config['input']
        self.field_output = config['output']
        self.dictionary = dictionaries[config['dict']]

    def initialize(self, stream2):
        return

    def transform(self, stream2):
        print("RUN TranformLookupTokens:", self.config)
        old_size = self.dictionary.size
        for i, instance in enumerate(stream2.data):
            instance[self.field_output] = torch.LongTensor(
                [self.dictionary.lookup(x) for x in instance[self.field_input]])
        print("new tokens added to dictionary:", self.dictionary.size - old_size)


class TransformSeqlabelsToMultitags(Transform):

    def __init__(self, config, dictionaries):
        self.config = config
        self.field = config['field']
        self.dictionary = dictionaries[config['dict']]

    def initialize(self, stream2):
        print("INIT transform_seqlabels_multi_tags:", self.config)

        # add labels to dictionary
        for i, instance in enumerate(stream2.data):
            for labels in instance[self.field]:
                [self.dictionary.lookup(x) for x in labels]

    def transform(self, stream2):
        print("RUN transform_seqlabels_multi_tags:", self.config, "labels=", self.dictionary.size)
        for i, instance in enumerate(stream2.data):
            length = len(instance[self.field])
            array = np.zeros((length, self.dictionary.size))
            for pos, labels in enumerate(instance[self.field]):
                idx = [self.dictionary.lookup(x) for x in labels]
                array[pos, idx] = 1
            instance[self.field] = torch.FloatTensor(array)


class TransformSpansToTensors(Transform):

    def __init__(self, config, dictionaries):
        self.config = config
        self.field = config['field']
        self.dictionary = dictionaries[config['dict']]
        self.output = config['output']

    def initialize(self, stream2):
        for instance in stream2.data:
            [self.dictionary.lookup(tag) for _, _, tag in instance[self.field]]
        return

    def transform(self, stream2):
        for i, instance in enumerate(stream2.data):
            span2tensor = {}
            for begin, end, tag in instance[self.field]:
                key = (begin, end)
                if key not in span2tensor:
                    span2tensor[key] = torch.zeros(self.dictionary.size)
                span2tensor[key][self.dictionary.lookup(tag)] = 1.0
            instance[self.output] = span2tensor


class TransformLookupSpans(Transform):

    def __init__(self, config, dictionaries):
        self.field = config['field']
        self.dictionary = dictionaries[config['dict']]
        self.output = config['output']

    def initialize(self, stream2):
        return

    def transform(self, stream2):
        for i, instance in enumerate(stream2.data):
            instance[self.output] = [(start, end, self.dictionary.lookup(label)) for start, end, label in
                                     instance[self.field]]


class TransformCharacters(Transform):

    def __init__(self, config, dictionaries):
        self.config = config
        self.field_input = config['input']
        self.field_output = config['output']
        self.dictionary = dictionaries[config['dict']]
        self.threshold = config['threshold']
        self.padding = config['padding']
        self.debug = config.get('debug', False)

        self.fit = set(config['fit'])
        self.counter = Counter() if len(self.fit) > 0 else None

    def initialize(self, stream2):
        if stream2.name in self.fit:
            print("TransformCharacters: count characters on {}".format(stream2.name))
            for instance in stream2.data:
                for chars in self.process(instance[self.field_input]):
                    self.counter.update(chars)

    def initialized(self):
        if self.counter is not None:
            chars = []
            items = self.dictionary.tolist()
            for ch, count in self.counter.items():
                if self.debug:
                    print("CHAR", ch, ": ", count)
                if count >= self.threshold:
                    chars.append(items[ch])

            self.dictionary.clear()
            self.dictionary.add(self.padding)
            for ch in chars:
                self.dictionary.add(ch)
            self.dictionary.update = False

            self.counter = None

    def transform(self, stream2):
        self.initialized()

        for i, instance in enumerate(stream2.data):
            instance[self.field_output] = self.process(instance[self.field_input])

    def process(self, tokens):
        output = []
        for token in tokens:
            token = '<' + token + '>'
            output.append([self.dictionary.lookup(c) for c in token])
        return output


# fix spans with new tokenizer
class TransformFixSpans(Transform):

    def __init__(self, config, dictionaries):
        return

    def initialize(self, stream2):
        return

    def transform(self, stream2):
        for i, instance in enumerate(stream2.data):
            old_begin = instance['old_begin'].tolist()
            old_end = instance['old_end'].tolist()
            new_begin = {pos: idx for idx, pos in enumerate(instance['begin'].tolist())}
            new_end = {pos: idx for idx, pos in enumerate(instance['end'].tolist())}

            instance['spans'] = [(new_begin.get(old_begin[begin], -1), new_end.get(old_end[end], -1)) for begin, end in
                                 instance['spans']]
            instance['gold_tags_indices'] = [(new_begin.get(old_begin[begin], -1), new_end.get(old_end[end], -1), l) for
                                             begin, end, l in instance['gold_tags_indices']]
            instance['gold_tags_indices'] = [(b, e, l) if b >= 0 and e >= 0 else (0, 1000, l) for b, e, l in
                                             instance['gold_tags_indices']]

            # print('gold_tags_indices:', instance['gold_tags_indices'])
            # print('mention2concept:', instance['mention2concept'])
            # print('clusters:', instance['clusters'])

            del instance['tags']
            del instance['tagspans']
            del instance['tags-x']
            del instance['auto']
            del instance['autospans']
            del instance['token2mention']
            # del instance['mention2concept']
            del instance['coref']
            # del instance['clusters']
            del instance['tags_target']

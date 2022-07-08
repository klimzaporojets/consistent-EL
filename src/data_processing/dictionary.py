import json
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger()


class Dictionary:

    def __init__(self):
        self.rewriter = lambda t: t
        self.debug = False
        self.token_unknown = -1
        self.update = True
        self.prefix = ''
        self.tmp_unknown = None

        self.clear()

    def clear(self):
        self.word2idx = {}
        self.idx2word = {}
        self.matrix = False
        self.size = 0
        self.out_of_voc = 0
        self.oov = set()

        if self.tmp_unknown is not None:
            self.token_unknown = self.lookup(self.tmp_unknown)

    def load_spirit_dictionary(self, filename, threshold_doc_freq=0):
        self.update = True
        with open(filename) as file:
            for line in file:
                data = line.strip().split("\t")
                if len(data) == 3:
                    df, tf, term = data
                    if int(df) >= threshold_doc_freq:
                        self.lookup(term)
        self.update = False

    def load_wordpiece_vocab(self, filename):
        self.update = True
        with open(filename) as file:
            for line in file:
                term, _ = line.split('\t')
                self.lookup(term)
        self.update = False

    def load_json(self, filename):
        with open(filename) as file:
            data = json.load(file)
            if isinstance(data, (list,)):
                for idx, word in enumerate(data):
                    if self.lookup(word) != idx:
                        logger.warning('WARNING: invalid dictionary')
            else:
                for word, idx in data.items():
                    if self.lookup(word) != idx:
                        logger.warning('WARNING: invalid dictionary')

    def lookup(self, token):
        token = self.prefix + self.rewriter(token)
        if not token in self.word2idx:
            if self.update:
                self.word2idx[token] = self.size
                self.idx2word[self.size] = token
                self.size += 1
            else:
                if self.debug:
                    logger.info('oov: "{}" -> {}'.format(token, self.token_unknown))
                self.out_of_voc += 1
                return self.token_unknown
        return self.word2idx[token]

    def add(self, token):
        if not token in self.word2idx:
            self.word2idx[token] = self.size
            self.idx2word[self.size] = token
            self.size += 1
        return self.word2idx[token]

    def set_unknown_token(self, unknown_token):
        self.tmp_unknown = unknown_token
        self.token_unknown = self.word2idx[self.prefix + unknown_token]
        logger.info('%s -> %s' % (self.get(self.token_unknown), self.token_unknown))

    def write(self, filename):
        import json
        with open(filename, 'w') as file:
            json.dump(self.word2idx, file)

    def get(self, index):
        return self.idx2word.get(index, None)

    def tolist(self):
        list = [None] * self.size
        for word, idx in self.word2idx.items():
            list[idx] = word
        return list

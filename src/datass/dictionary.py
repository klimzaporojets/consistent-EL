import json


# def load_word2vec_text(filename):
# 	with gzip.open(filename, 'r') as f:
# 		vec_n, vec_size = map(int, f.readline().split())
#
# 		word2idx = {}
# 		matrix = np.zeros((vec_n, vec_size), dtype=np.float32)
#
# 		for n in range(vec_n):
# 			line = f.readline().split()
# 			word=line[0].decode('utf-8')
# 			vec=np.array([float(x) for x in line[1:]], dtype=np.float32)
# 			word2idx[word] = n
# 			matrix[n] = vec
# 		return word2idx, matrix, vec_n


class Dictionary:

    def __init__(self):
        self.rewriter = lambda t: t
        self.debug = False
        self.token_unknown = -1
        self.update = True
        self.prefix = ''
        self.tmp_unknown = None

        self.clear()

    # if filename is not None:
    # 	print("loading %s" % filename)
    # 	self.load_embedding_matrix_text(filename)
    # 	print("done.")
    # 	print()

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
            # else:
            # 	print("skip:", line)
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
                        print("WARNING: invalid dictionary")
            else:
                for word, idx in data.items():
                    if self.lookup(word) != idx:
                        print("WARNING: invalid dictionary")

    # def load_embedding_matrix_text(self, filename):
    # 	self.word2idx, self.matrix, self.size = load_word2vec_text(filename)
    # 	self.update = False

    def lookup(self, token):
        token = self.prefix + self.rewriter(token)
        if not token in self.word2idx:
            if self.update:
                self.word2idx[token] = self.size
                self.idx2word[self.size] = token
                self.size += 1
            else:
                if self.debug:
                    print("oov: '{}' -> {}".format(token, self.token_unknown))
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
        # self.token_unknown = self.lookup(unknown_token) #NOT CORRECT
        self.token_unknown = self.word2idx[self.prefix + unknown_token]
        print(self.get(self.token_unknown), "->", self.token_unknown)

    def write(self, filename):
        import json
        with open(filename, 'w') as file:
            json.dump(self.word2idx, file)

    def get(self, index):
        # for word, idx in self.word2idx.items():
        #     if idx == index:
        #         return word
        # return None
        return self.idx2word.get(index, None)

    def tolist(self):
        list = [None] * self.size
        for word, idx in self.word2idx.items():
            list[idx] = word
        return list

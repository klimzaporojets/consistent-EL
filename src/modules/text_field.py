import math
import random
from time import sleep

import numpy as np
import torch
import torch.nn as nn

import settings
from datass.wordvec import load_wordembeddings, load_wordembeddings_words, load_wordembeddings_with_random_unknowns
from modules.seq2vec import CNNMaxpool


class TextFieldEmbedderList(nn.Module):
    def __init__(self, list):
        super(TextFieldEmbedderList, self).__init__()
        self.embedders = nn.ModuleList(list)
        self.dim = sum([embedder.dim for embedder in self.embedders])

    def forward(self, inputs):
        outputs = [embedder(inputs) for embedder in self.embedders]
        return torch.cat(outputs, 1)


class TextFieldEmbedderTokens(nn.Module):

    def __init__(self, dictionaries, config):
        super(TextFieldEmbedderTokens, self).__init__()
        self.dictionary = dictionaries[config['dict']]
        self.dim = config['dim']
        self.embed = nn.Embedding(self.dictionary.size, self.dim)
        self.dropout = nn.Dropout(config['dropout'], inplace=True)
        self.normalize = 'norm' in config
        self.freeze = config.get('freeze', False)

        if 'embed_file' in config:
            self.init_unknown = config['init_unknown']
            self.init_random = config['init_random']
            self.backoff_to_lowercase = config['backoff_to_lowercase']
            # self.filtered_file = config['filtered_file']
            # self.use_filtered = config['use_filtered']

            self.load_embeddings(config['embed_file'], config['filtered_file'], config['use_filtered'],
                                 what_load=config['what_load'], load_type=config['load_type'])
        else:
            print("WARNING: training word vectors from scratch")

        if self.dictionary.size == 0:
            print("WARNING: empty dictionary")
            return

        if 'fill' in config:
            print("WARNING: filling vector to constant value:", config['fill'])
            index = self.dictionary.lookup(config['fill'])
            self.embed.weight.data[index, :] = 1.0 / math.sqrt(self.dim)
            print(config['fill'], '->', self.embed.weight.data[index, :])

        nrms = self.embed.weight.norm(p=2, dim=1, keepdim=True)
        print("norms: min={} max={} avg={}".format(nrms.min().item(), nrms.max().item(), nrms.mean().item()))

    def load_all_wordvecs(self, filename):
        print("LOADING ALL WORDVECS")
        words = load_wordembeddings_words(filename)
        for word in words:
            self.dictionary.add(word)
        self.load_embeddings(filename)
        print("DONE")

    def load_embeddings(self, filename, filtered_file=None, use_filtered=False, retry=0, what_load='dictionary',
                        load_type='wordvec'):
        if self.init_random:
            try:
                embeddings = load_wordembeddings_with_random_unknowns(filename, dictionary=self.dictionary,
                                                                      dim=self.dim,
                                                                      backoff_to_lowercase=self.backoff_to_lowercase,
                                                                      filtered_file=filtered_file,
                                                                      use_filtered=use_filtered,
                                                                      what_load=what_load,
                                                                      load_type=load_type)
            except OSError as exept:
                if retry < 10:
                    print('following exept in load_embeddings: ', exept.strerror)
                    sleep(random.randint(5, 10))
                    self.load_embeddings(filename, filtered_file=filtered_file,
                                         use_filtered=use_filtered, retry=retry + 1, what_load=what_load,
                                         load_type=load_type)
                    return
                else:
                    print('NO MORE RETRIES LEFT, FAILING in load_embeddings')
                    raise exept

        else:
            unknown_vec = np.ones((self.dim)) / np.sqrt(self.dim) if self.init_unknown else None

            try:
                word_vectors = load_wordembeddings(filename, dictionary=self.dictionary, dim=self.dim,
                                                   out_of_voc_vector=unknown_vec,
                                                   filtered_file=filtered_file, use_filtered=use_filtered,
                                                   what_load=what_load, load_type=load_type)
            except OSError as exept:
                if retry < 10:
                    print('following exept in load_embeddings: ', exept.strerror)
                    sleep(random.randint(5, 10))
                    self.load_embeddings(filename, filtered_file=filtered_file, use_filtered=use_filtered,
                                         retry=retry + 1, what_load=what_load, load_type=load_type)
                    return
                else:
                    print('NO MORE RETRIES LEFT, FAILING in load_embeddings')
                    raise exept

            if self.normalize:
                norms = np.einsum('ij,ij->i', word_vectors, word_vectors)
                np.sqrt(norms, norms)
                norms += 1e-8
                word_vectors /= norms[:, np.newaxis]

            embeddings = torch.from_numpy(word_vectors)

        # device = next(self.embed.parameters()).device
        # self.embed = nn.Embedding(self.dictionary.size, self.dim).to(device)
        self.embed = nn.Embedding(self.dictionary.size, self.dim).to(settings.device)
        self.embed.weight.data.copy_(embeddings)
        self.embed.weight.requires_grad = not self.freeze

    def forward(self, inputs):
        return self.dropout(self.embed(inputs))


class TextFieldEmbedderCharacters(nn.Module):

    def __init__(self, dictionaries, config):
        super(TextFieldEmbedderCharacters, self).__init__()
        self.embedder = TextFieldEmbedderTokens(dictionaries, config['embedder'])
        self.padding = self.embedder.dictionary.lookup('PADDING')
        self.seq2vec = CNNMaxpool(self.embedder.dim, config['encoder'])
        self.dropout = nn.Dropout(config['dropout'])
        self.dim_output = self.seq2vec.dim_output
        self.min_word_length = self.seq2vec.max_kernel
        # self.min_word_length = 50
        print("TextFieldEmbedderCharacters:", self.min_word_length)

    def forward(self, characters):
        # print('chars:', characters[0,0,:])

        char_vec = self.embedder(characters)
        # print('char_embed', char_vec.size(), char_vec.sum().item())
        char_vec = self.seq2vec(char_vec)
        return self.dropout(torch.relu(char_vec))


class TextFieldEmbedderWhitespace(nn.Module):

    def __init__(self, dictionaries, config):
        super(TextFieldEmbedderWhitespace, self).__init__()
        self.dictionary = dictionaries[config['dict']]
        self.embed = nn.Embedding(self.dictionary.size, config['dim'])
        self.dropout = nn.Dropout(config['dropout'], inplace=False)
        self.dim = 2 * config['dim']

    def forward(self, data):
        inputs = data['whitespace']
        emb = self.embed(inputs)
        left = emb[:, :-1, :]
        right = emb[:, 1:, :]
        output = torch.cat((left, right), -1)
        return self.dropout(output)

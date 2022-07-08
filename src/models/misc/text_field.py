import logging
import math
import random
import traceback
from time import sleep

import numpy as np
import torch
import torch.nn as nn

from misc import settings
from models.misc.wordvec import load_wordembeddings, load_wordembeddings_words, load_wordembeddings_with_random_unknowns
from models.models.seq2vec import CNNMaxpool

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger()


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

            self.load_embeddings(config['embed_file'], config['filtered_file'], config['use_filtered'],
                                 what_load=config['what_load'], load_type=config['load_type'])
        else:
            logger.warning('WARNING: training word vectors from scratch')

        if self.dictionary.size == 0:
            logger.warning('WARNING: empty dictionary')
            return

        if 'fill' in config:
            logger.warning('WARNING: filling vector to constant value: %s' % config['fill'])
            index = self.dictionary.lookup(config['fill'])
            self.embed.weight.data[index, :] = 1.0 / math.sqrt(self.dim)
            logger.warning('%s -> %s)' % (config['fill'], self.embed.weight.data[index, :]))

        nrms = self.embed.weight.norm(p=2, dim=1, keepdim=True)
        logger.info('norms: min={} max={} avg={}'.format(nrms.min().item(), nrms.max().item(), nrms.mean().item()))

    def load_all_wordvecs(self, filename):
        logger.info('LOADING ALL WORDVECS')
        words = load_wordembeddings_words(filename)
        for word in words:
            self.dictionary.add(word)
        self.load_embeddings(filename)
        logger.info('DONE')

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
                    logger.error('following exept in load_embeddings: %s' % exept.strerror)
                    traceback.print_exc()
                    sleep(random.randint(5, 10))
                    self.load_embeddings(filename, filtered_file=filtered_file,
                                         use_filtered=use_filtered, retry=retry + 1, what_load=what_load,
                                         load_type=load_type)
                    return
                else:
                    logger.error('NO MORE RETRIES LEFT, FAILING in load_embeddings')
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
                    logger.error('following exept in load_embeddings: %s' % exept.strerror)
                    traceback.print_exc()
                    sleep(random.randint(5, 10))
                    self.load_embeddings(filename, filtered_file=filtered_file, use_filtered=use_filtered,
                                         retry=retry + 1, what_load=what_load, load_type=load_type)
                    return
                else:
                    logger.error('NO MORE RETRIES LEFT, FAILING in load_embeddings')
                    raise exept

            if self.normalize:
                norms = np.einsum('ij,ij->i', word_vectors, word_vectors)
                np.sqrt(norms, norms)
                norms += 1e-8
                word_vectors /= norms[:, np.newaxis]

            embeddings = torch.from_numpy(word_vectors)

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
        logger.info('TextFieldEmbedderCharacters: %s' % self.min_word_length)

    def forward(self, characters):
        char_vec = self.embedder(characters)
        char_vec = self.seq2vec(char_vec)
        return self.dropout(torch.relu(char_vec))

import logging
import math
import os

import numpy as np
import psutil
import torch
from torch import nn

from data_processing.dictionary import Dictionary
from misc import settings

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger()


class KolitsasEntityEmbeddings(nn.Module):
    def load_kolitsas_embeddings(self):
        pass

    def load_kolitsas_ent_id_to_wiki_id(self, ent_id_to_wiki_id_file, ent_id_to_wiki_id, offset=0):
        with open(ent_id_to_wiki_id_file) as infile:
            for line in infile:
                sp = line.split('\t')
                assert len(sp) == 2
                ent_id = int(sp[0])
                wiki_id = int(sp[1])
                # it is 0-based, this is why -1
                assert (ent_id - 1 + offset) not in ent_id_to_wiki_id
                ent_id_to_wiki_id[ent_id - 1 + offset] = wiki_id
        return ent_id_to_wiki_id

    def load_kolitsas_wiki_id_to_wiki_link(self):
        to_ret = dict()
        with open(self.wiki_id_to_wiki_link_file) as infile:
            for line in infile:
                sp = line.split('\t')
                assert len(sp) == 2
                wiki_id = int(sp[1])
                wiki_link = sp[0].strip()
                wiki_link = wiki_link.replace(' ', '_')
                if wiki_link not in self.dictionary.word2idx:
                    continue
                to_ret[wiki_id] = wiki_link
        return to_ret

    def load_kolitsas_wiki_id_to_wiki_link_universe(self, entities_universe_file, wiki_id_to_wiki_link):
        with open(entities_universe_file) as infile:
            for line in infile:
                sp = line.split('\t')
                assert len(sp) == 2
                wiki_id = int(sp[0])
                wiki_link = sp[1].strip()
                if wiki_link not in self.dictionary.word2idx:
                    continue
                if wiki_id in wiki_id_to_wiki_link:
                    # checks the match of link, but doesn't add anything
                    assert wiki_id_to_wiki_link[wiki_id] == wiki_link
                    continue
                wiki_id_to_wiki_link[wiki_id] = wiki_link
        return wiki_id_to_wiki_link

    def load_kolitsas_embeddings_random_unknowns(self):
        # entity_embeddings_nparray = np.load(config.base_folder + "data/entities/ent_vecs/ent_vecs.npy")
        process = psutil.Process(os.getpid())

        entity_embeddings_nparray = np.load(self.embed_file)
        logger.debug('kolitsas emb used mem 1: %s' % (process.memory_info().rss / 1024 / 1024 / 1024))

        entity_embeddings_nparray[0] = 0
        ent_id_to_wiki_id = dict()
        ent_id_to_wiki_id = self.load_kolitsas_ent_id_to_wiki_id(self.ent_id_to_wiki_id_file, ent_id_to_wiki_id,
                                                                 offset=0)

        logger.debug('kolitsas emb used mem 2: %s' % (process.memory_info().rss / 1024 / 1024 / 1024))

        wiki_id_to_wiki_link = self.load_kolitsas_wiki_id_to_wiki_link()

        logger.debug('kolitsas emb used mem 3: %s' % (process.memory_info().rss / 1024 / 1024 / 1024))

        if self.load_extension:
            entity_extension = np.load(self.extension_embed_file)
            ent_id_to_wiki_id = self.load_kolitsas_ent_id_to_wiki_id(self.extension_ent_id_to_wiki_id_file,
                                                                     ent_id_to_wiki_id,
                                                                     offset=len(ent_id_to_wiki_id))

            logger.debug('kolitsas emb used mem 4: %s' % (process.memory_info().rss / 1024 / 1024 / 1024))
            entity_embeddings_nparray = np.vstack((entity_embeddings_nparray, entity_extension))
            logger.debug('kolitsas emb used mem 5: %s' % (process.memory_info().rss / 1024 / 1024 / 1024))

        wiki_id_to_wiki_link = self.load_kolitsas_wiki_id_to_wiki_link_universe(self.entities_universe_file,
                                                                                wiki_id_to_wiki_link)

        logger.debug('kolitsas emb used mem 6: %s' % (process.memory_info().rss / 1024 / 1024 / 1024))

        if self.load_extension:
            wiki_id_to_wiki_link = self.load_kolitsas_wiki_id_to_wiki_link_universe(
                self.extension_entities_universe_file,
                wiki_id_to_wiki_link)

            logger.debug('kolitsas emb used mem 7: %s ' % (process.memory_info().rss / 1024 / 1024 / 1024))

        accept = self.dictionary.word2idx
        found = {}
        for idx, entity_embedding in enumerate(entity_embeddings_nparray):
            wiki_id = ent_id_to_wiki_id[idx]
            wiki_link = None
            if wiki_id in wiki_id_to_wiki_link:
                wiki_link = wiki_id_to_wiki_link[wiki_id]
            if wiki_link in accept:
                found[accept[wiki_link]] = entity_embedding

        logger.debug('kolitsas emb used mem 8: %s' % (process.memory_info().rss / 1024 / 1024 / 1024))

        # BEGIN: the following loop for debugging purposes only to understand better what links are missed
        for wiki_link_id in accept.values():
            if wiki_link_id not in found:
                logger.debug('following wiki link was not found: %s' % self.dictionary.get(wiki_link_id))
        # END: the following loop for debugging purposes only to understand better what links are missed

        all_embeddings = np.asarray(list(found.values()))

        logger.debug('kolitsas emb used mem 9: %s' % (process.memory_info().rss / 1024 / 1024 / 1024))

        logger.info('shape of all_embeddings: %s' % str(all_embeddings.shape))
        embeddings_mean = np.mean(all_embeddings)
        embeddings_std = np.std(all_embeddings)

        # kzaporoj - in case of the mean and/or std come on nan
        if math.isnan(embeddings_mean):
            embeddings_mean = 0.0

        if math.isnan(embeddings_std):
            embeddings_std = 0.5

        embeddings = torch.FloatTensor(len(accept), self.dim).normal_(
            embeddings_mean, embeddings_std
        )
        for key, value in found.items():
            embeddings[key] = torch.FloatTensor(value)

        if self.refit_ratio is not None:
            embeddings = embeddings * self.refit_ratio

        if self.norm_clip is not None:
            embeddings = (self.norm_clip * embeddings) / torch.linalg.norm(embeddings)

        embeddings_mean = embeddings.mean().item()
        embeddings_std = embeddings.std().item()
        logger.debug('kolitsas emb used mem 10: %s' % (process.memory_info().rss / 1024 / 1024 / 1024))

        logger.info('found: {} / {} = {}'.format(len(found), len(accept), len(found) / len(accept)))
        logger.debug('words/embedding entries randomly initialized: %s' % (len(accept) - len(found)))
        logger.debug('the embeddings norm is: %s' % torch.linalg.norm(embeddings))
        logger.debug('the embeddings mean is: %s' % embeddings_mean)
        logger.debug('the embeddings std is: %s' % embeddings_std)

        return embeddings

    def __init__(self, dictionaries, config):
        super(KolitsasEntityEmbeddings, self).__init__()
        self.dictionary: Dictionary = dictionaries[config['dict']]
        self.dim = config['dim']
        self.dropout = nn.Dropout(config['dropout'], inplace=True)
        self.normalize = 'norm' in config
        self.freeze = config.get('freeze', False)

        self.embed_file = config['embed_file']
        self.extension_embed_file = config['extension_embed_file']
        self.extension_ent_id_to_wiki_id_file = config['extension_ent_id_to_wiki_id_file']
        self.what_load = config['what_load']
        self.load_extension = config['load_extension']
        self.ent_id_to_wiki_id_file = config['ent_id_to_wiki_id_file']
        self.wiki_id_to_wiki_link_file = config['wiki_id_to_wiki_link_file']
        self.entities_universe_file = config['entities_universe_file']
        self.extension_entities_universe_file = config['extension_entities_universe_file']
        self.refit_ratio = config['refit_ratio']
        self.norm_clip = config['norm_clip']

        self.load_embeddings()

        if self.dictionary.size == 0:
            logger.warning('WARNING: empty dictionary')
            return

        nrms = self.embed.weight.norm(p=2, dim=1, keepdim=True)
        logger.info('norms: min={} max={} avg={}'.format(nrms.min().item(), nrms.max().item(), nrms.mean().item()))

    def load_embeddings(self):
        process = psutil.Process(os.getpid())

        logger.debug('starting loading kolitsas entity embeddings, used memory: %s' %
                     (process.memory_info().rss / 1024 / 1024 / 1024))
        entity_embeddings = self.load_kolitsas_embeddings_random_unknowns()

        logger.debug('starting loading kolitsas entity embeddings, used memory: %s' %
                     (process.memory_info().rss / 1024 / 1024 / 1024))

        if self.normalize:
            norms = np.einsum('ij,ij->i', entity_embeddings, entity_embeddings)
            np.sqrt(norms, norms)
            norms += 1e-8
            entity_embeddings /= norms[:, np.newaxis]

        self.embed = nn.Embedding(self.dictionary.size, self.dim).to(settings.device)
        self.embed.weight.data.copy_(entity_embeddings)
        self.embed.weight.requires_grad = not self.freeze

    def forward(self, inputs):
        return self.dropout(self.embed(inputs))

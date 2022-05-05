# import math
#
# import numpy as np
# import torch
# import torch.nn as nn
#
# from data.entityvec import load_entity_embeddings_with_random_unknowns, load_entity_embeddings
#
#
# class EntityFieldEmbedderTokens(nn.Module):
#
#     def __init__(self, dictionaries, config):
#         super(EntityFieldEmbedderTokens, self).__init__()
#         self.dictionary = dictionaries[config['dict']]
#         self.dim = config['dim']
#         self.embed = nn.Embedding(self.dictionary.size, self.dim)
#         self.dropout = nn.Dropout(config['dropout'], inplace=True)
#         self.normalize = 'norm' in config
#         self.freeze = config.get('freeze', False)
#
#         if 'embed_file' in config:
#             self.init_unknown = config['init_unknown']
#             self.init_random = config['init_random']
#             self.backoff_to_lowercase = config['backoff_to_lowercase']
#
#             self.load_embeddings(config['embed_file'])
#         else:
#             print("WARNING: training word vectors from scratch")
#
#         if self.dictionary.size == 0:
#             print("WARNING: empty dictionary")
#             return
#
#         if 'fill' in config:
#             print("WARNING: filling vector to constant value:", config['fill'])
#             index = self.dictionary.lookup(config['fill'])
#             self.embed.weight.data[index, :] = 1.0 / math.sqrt(self.dim)
#             print(config['fill'], '->', self.embed.weight.data[index, :])
#
#         nrms = self.embed.weight.norm(p=2, dim=1, keepdim=True)
#         print("norms: min={} max={} avg={}".format(nrms.min().item(), nrms.max().item(), nrms.mean().item()))
#
#     def load_all_wordvecs(self, filename):
#         print("LOADING ALL ENTITY VECS")
#         words = load_entity_embeddings(filename)
#         for word in words:
#             self.dictionary.add(word)
#         self.load_embeddings(filename)
#         print("DONE")
#
#     def load_embeddings(self, filename):
#         if self.init_random:
#             embeddings = load_entity_embeddings_with_random_unknowns(filename, accept=self.dictionary.word2idx,
#                                                                      dim=self.dim,
#                                                                      backoff_to_lowercase=self.backoff_to_lowercase)
#         else:
#             unknown_vec = np.ones((self.dim)) / np.sqrt(self.dim) if self.init_unknown else None
#
#             word_vectors = load_entity_embeddings(filename, accept=self.dictionary.word2idx, dim=self.dim,
#                                                   out_of_voc_vector=unknown_vec)
#             if self.normalize:
#                 norms = np.einsum('ij,ij->i', word_vectors, word_vectors)
#                 np.sqrt(norms, norms)
#                 norms += 1e-8
#                 word_vectors /= norms[:, np.newaxis]
#
#             embeddings = torch.from_numpy(word_vectors)
#
#         device = next(self.embed.parameters()).device
#         self.embed = nn.Embedding(self.dictionary.size, self.dim).to(device)
#         self.embed.weight.data.copy_(embeddings)
#         self.embed.weight.requires_grad = not self.freeze
#
#     def forward(self, inputs):
#         return self.dropout(self.embed(inputs))

import gzip
import math

import numpy as np
import torch


def load_entity_embeddings(filename, accept={}, dim=300, out_of_voc_vector=None):
    embedding_matrix = np.zeros((len(accept), dim))
    if out_of_voc_vector is not None:
        print("WARNING: initialize word embeddings with ", out_of_voc_vector)
        embedding_matrix = embedding_matrix + out_of_voc_vector

    print("loading word vectors:", filename)
    found = 0

    file = gzip.open(filename, 'rt') if filename.endswith('.gz') else open(filename)
    for line in file:
        # Note: use split(' ') instead of split() if you get an error.
        values = line.rstrip().split(' ')
        word = values[0]
        if word in accept:
            coefs = np.asarray(values[1:], dtype='float32')
            embedding_matrix[accept[word]] = coefs
            found += 1
    file.close()

    print("found: {} / {} = {}".format(found, len(accept), found / len(accept) if found != 0 else 0.0))

    return embedding_matrix


def load_entity_embeddings_with_random_unknowns(filename, accept={}, dim=300, debug=False, backoff_to_lowercase=False):
    print("loading entity vectors:", filename)
    found = {}

    backoff = {}
    if backoff_to_lowercase:
        print("WARNING: backing off to lowercase")
        for x, idx in accept.items():
            backoff[x.lower()] = None
            backoff[x.casefold()] = None

    file = gzip.open(filename, 'rt') if filename.endswith('.gz') else open(filename)
    for line in file:
        values = line.rstrip().split(' ')
        word = values[0]
        if word in accept:
            found[accept[word]] = np.asarray(values[1:], dtype='float32')
        if word in backoff:
            backoff[word] = np.asarray(values[1:], dtype='float32')
    file.close()

    all_embeddings = np.asarray(list(found.values()))
    embeddings_mean = float(np.mean(all_embeddings))
    embeddings_std = float(np.std(all_embeddings))

    # kzaporoj - in case of the mean and/or std come on nan
    if math.isnan(embeddings_mean):
        embeddings_mean = 0.0

    if math.isnan(embeddings_std):
        embeddings_std = 0.5

    embeddings = torch.FloatTensor(len(accept), dim).normal_(
        embeddings_mean, embeddings_std
    )
    for key, value in found.items():
        embeddings[key] = torch.FloatTensor(value)

    print("found: {} / {} = {}".format(len(found), len(accept), len(found) / len(accept)))
    print("entities randomly initialized:", len(accept) - len(found))

    if debug:
        counter = 0
        for word in accept.keys():
            if accept[word] not in found:
                print("no such pretrained word: {} ({})".format(word, counter))
                counter += 1

    if backoff_to_lowercase:
        num_backoff = 0
        for word, idx in accept.items():
            if accept[word] not in found:
                if word.lower() in backoff and backoff[word.lower()] is not None:
                    print("backoff {} -> {}".format(word, word.lower()))
                    embeddings[idx, :] = torch.FloatTensor(backoff[word.lower()])
                    num_backoff += 1
                elif word.casefold() in backoff and backoff[word.casefold()] is not None:
                    print("casefold {} -> {}".format(word, word.lower()))
                    embeddings[idx, :] = torch.FloatTensor(backoff[word.casefold()])
                    num_backoff += 1
        print("num_backoff:", num_backoff)

    return embeddings


def load_entity_embeddings(filename):
    entities = []

    print("loading words:", filename)
    file = gzip.open(filename, 'rt') if filename.endswith('.gz') else open(filename)
    for line in file:
        # Note: use split(' ') instead of split() if you get an error.
        values = line.rstrip().split(' ')
        entities.append(values[0])
    file.close()

    return entities

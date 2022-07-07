import gzip
import logging
import math
import os

import numpy as np
import torch

from data_processing.dictionary import Dictionary

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger()


def load_wordembeddings(filename, dictionary: Dictionary, dim=300, out_of_voc_vector=None, filtered_file=None,
                        use_filtered=False, what_load='dictionary', load_type='wordvec'):
    logger.info('loading word vectors: %s' % filename)
    # logger.info('load_wordembeddings use_filtered: ', use_filtered, ' what_load: ', what_load, ' load_type: ', load_type)
    logger.info('load_wordembeddings use_filtered: %s what_load %s load_type %s' % (use_filtered, what_load, load_type))
    if what_load == 'allvecs':
        file = open_embedding_file(use_filtered, filtered_file, filename)
        for line in file:
            if is_valid_word_line(line, load_type):
                values = parse_line(line, load_type)
                word = values[0]
                dictionary.add(word)
    accept = dictionary.word2idx
    embedding_matrix = np.zeros((len(accept), dim))
    if out_of_voc_vector is not None:
        logger.warning('WARNING: initialize word embeddings with %s' % out_of_voc_vector)
        embedding_matrix = embedding_matrix + out_of_voc_vector

    found = 0

    file = open_embedding_file(use_filtered, filtered_file, filename)

    if use_filtered and (not os.path.isfile(filtered_file)):
        file_found_out = open(filtered_file, 'wt')
    else:
        file_found_out = None

    for line in file:
        # Note: use split(' ') instead of split() if you get an error.
        if is_valid_word_line(line, load_type):
            values = parse_line(line, load_type)
            word = values[0]
            if word in accept:
                # if word in accept or True: # kzaporoj - just to see how much space it occupies
                coefs = np.asarray(values[1], dtype='float32')
                embedding_matrix[accept[word]] = coefs
                found += 1
                if file_found_out is not None:
                    file_found_out.write(line)

    file.close()

    if file_found_out is not None:
        file_found_out.close()

    logger.info('found: {} / {} = {}'.format(found, len(accept), found / len(accept) if found != 0 else 0.0))

    return embedding_matrix


def open_embedding_file(use_filtered, filtered_file, filename):
    # print('IN OPEN_EMBEDDING_FILE with use_filtered: ', use_filtered, '  filtered_file: ', filtered_file,
    #       ' filename: ', filename)
    logger.info('IN OPEN_EMBEDDING_FILE with use_filtered: %s filtered_file: %s filename %s' %
                (use_filtered, filtered_file, filename))
    if use_filtered and os.path.isfile(filtered_file):
        logger.info('FOR SOME REASON IN use_filtered PART OF open_embedding_file!! %s' % use_filtered)
        file = gzip.open(filtered_file, 'rt') if filtered_file.endswith('.gz') else open(filtered_file)
    else:
        logger.info('NOT IN use_filtered PART OF open_embedding_file!!: %s' % use_filtered)
        file = gzip.open(filename, 'rt') if filename.endswith('.gz') else open(filename)

    return file


def is_valid_word_line(line: str, load_type='wordvec'):
    if load_type == 'wordvec':
        return True
    elif load_type == 'word_wordentvec' and line.startswith('1#'):
        return True
    elif load_type == 'ent_wordentvec' and (not line.startswith('1#')):
        return True
    else:
        return False


def parse_line(line: str, load_type='wordvec'):
    line_splitted = line.rstrip().split(' ')
    if load_type == 'wordvec' or load_type == 'ent_wordentvec':
        if line.startswith('1#'):
            token = line_splitted[0][1:]
        else:
            token = line_splitted[0]
        embedding_vector = line_splitted[1:]
    elif load_type == 'word_wordentvec' and line.startswith('1#'):
        token = line_splitted[0][2:]
        embedding_vector = line_splitted[1:]
    else:
        raise Exception('Something wrong in parse_line for line: ' + line +
                        ' and load_type: ' + load_type)

    return token, embedding_vector


def load_wordembeddings_with_random_unknowns(filename, dictionary: Dictionary = None, dim=300, debug=False,
                                             backoff_to_lowercase=False,
                                             filtered_file=None, use_filtered=False, what_load='dictionary',
                                             load_type='wordvec'):
    logger.info('loading word vectors with random unknowns: %s' % filename)
    logger.info('load_wordembeddings use_filtered: %s what_load: %s load_type: %s' %
                (use_filtered, what_load, load_type))
    found = {}

    if what_load == 'allvecs':
        file = open_embedding_file(use_filtered, filtered_file, filename)
        for line in file:
            if is_valid_word_line(line, load_type):
                values = parse_line(line, load_type)
                # values = line.rstrip().split(' ')
                word = values[0]
                dictionary.add(word)

    accept = dictionary.word2idx
    logger.info('LENGTH OF ACCEPT: %s' % len(accept))
    backoff = {}
    if backoff_to_lowercase:
        logger.info('WARNING: backing off to lowercase')
        for x, idx in accept.items():
            if x == 'UNKNOWN':
                continue
            backoff[x.lower()] = None
            backoff[x.casefold()] = None

    file = open_embedding_file(use_filtered, filtered_file, filename)

    if use_filtered and (not os.path.isfile(filtered_file)):
        file_found_out = open(filtered_file, 'wt')
    else:
        file_found_out = None

    first_dim = 0
    nr_lines_file = 0
    for line in file:
        nr_lines_file += 1
        if is_valid_word_line(line, load_type):
            values = parse_line(line, load_type)
            # values = line.rstrip().split(' ')
            word = values[0]
            if word in accept:
                np_found = np.asarray(values[1], dtype='float32')
                if first_dim == 0:
                    first_dim = np_found.shape
                elif first_dim != np_found.shape:
                    logger.warning('================================')
                    logger.warning('!!!WARNING WITH SHAPE, first shape %s current shape: %s word: %s' %
                                   (first_dim, np_found.shape, word))
                    logger.warning('!!!WARNING WITH SHAPE line: %s' % line)
                    logger.warning('!!!WARNING WITH SHAPE IGNORING THIS LINE IN THE EMBEDDINGS')
                    continue
                found[accept[word]] = np_found
                if file_found_out is not None:
                    file_found_out.write(line)

            if word in backoff:
                backoff[word] = np.asarray(values[1], dtype='float32')
                if file_found_out is not None:
                    file_found_out.write(line)
    file.close()

    if file_found_out is not None:
        file_found_out.close()

    all_embeddings = np.asarray(list(found.values()))
    logger.info('nr of lines in file: %s' % nr_lines_file)
    logger.info('shape of all_embeddings: %s' % str(all_embeddings.shape))
    embeddings_mean = np.mean(all_embeddings)
    embeddings_std = np.std(all_embeddings)

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

    logger.info('found: {} / {} = {}'.format(len(found), len(accept), len(found) / len(accept)))
    logger.info('words/embedding entries randomly initialized: %s' % (len(accept) - len(found)))
    logger.debug('the embeddings norm is: %s' % torch.linalg.norm(embeddings))
    logger.debug('the embeddings mean is: %s' % embeddings_mean)
    logger.debug('the embeddings std is: %s' % embeddings_std)

    if debug:
        counter = 0
        for word in accept.keys():
            if accept[word] not in found:
                logger.info('no such pretrained word: {} ({})'.format(word, counter))
                counter += 1

    if backoff_to_lowercase:
        num_backoff = 0
        for word, idx in accept.items():
            if word == 'UNKNOWN':
                continue
            if accept[word] not in found:
                if word.lower() in backoff and backoff[word.lower()] is not None:
                    logger.info('backoff {} -> {}'.format(word, word.lower()))
                    embeddings[idx, :] = torch.FloatTensor(backoff[word.lower()])
                    num_backoff += 1
                elif word.casefold() in backoff and backoff[word.casefold()] is not None:
                    logger.info('casefold {} -> {}'.format(word, word.lower()))
                    embeddings[idx, :] = torch.FloatTensor(backoff[word.casefold()])
                    num_backoff += 1
        logger.info('num_backoff: %s' % num_backoff)

    return embeddings


def load_wordembeddings_words(filename):
    words = []

    logger.info('loading words: %s' % filename)
    file = gzip.open(filename, 'rt') if filename.endswith('.gz') else open(filename)
    for line in file:
        # Note: use split(' ') instead of split() if you get an error.
        values = line.rstrip().split(' ')
        words.append(values[0])
    file.close()

    return words

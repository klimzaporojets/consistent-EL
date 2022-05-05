import gzip
import math
import os

import numpy as np
import torch

from datass.dictionary import Dictionary


def load_wordembeddings(filename, dictionary: Dictionary, dim=300, out_of_voc_vector=None, filtered_file=None,
                        use_filtered=False, what_load='dictionary', load_type='wordvec'):
    # if use_filtered and os.path.isfile(filtered_file):
    #     file = gzip.open(filtered_file, 'rt') if filtered_file.endswith('.gz') else open(filtered_file)
    # else:
    #     file = gzip.open(filename, 'rt') if filename.endswith('.gz') else open(filename)
    # file = open_embedding_file(use_filtered, filtered_file, filename)

    print("loading word vectors:", filename)
    print('load_wordembeddings use_filtered: ', use_filtered, ' what_load: ', what_load, ' load_type: ', load_type)
    if what_load == 'allvecs':
        file = open_embedding_file(use_filtered, filtered_file, filename)
        for line in file:
            if is_valid_word_line(line, load_type):
                values = parse_line(line, load_type)
                # values = line.rstrip().split(' ')
                word = values[0]
                dictionary.add(word)
    # else:
    accept = dictionary.word2idx
    embedding_matrix = np.zeros((len(accept), dim))
    if out_of_voc_vector is not None:
        print("WARNING: initialize word embeddings with ", out_of_voc_vector)
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
            # values = line.rstrip().split(' ')
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

    print("found: {} / {} = {}".format(found, len(accept), found / len(accept) if found != 0 else 0.0))

    return embedding_matrix


def open_embedding_file(use_filtered, filtered_file, filename):
    print('IN OPEN_EMBEDDING_FILE with use_filtered: ', use_filtered, '  filtered_file: ', filtered_file,
          ' filename: ', filename)
    if use_filtered and os.path.isfile(filtered_file):
        print('FOR SOME REASON IN use_filtered PART OF open_embedding_file!!: ', use_filtered)
        file = gzip.open(filtered_file, 'rt') if filtered_file.endswith('.gz') else open(filtered_file)
    else:
        print('NOT IN use_filtered PART OF open_embedding_file!!: ', use_filtered)
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
    # token = None
    # embedding_vector = None
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
    print("loading word vectors with random unknowns:", filename)
    print('load_wordembeddings use_filtered: ', use_filtered, ' what_load: ', what_load, ' load_type: ', load_type)
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
    print('LENGTH OF ACCEPT: ', len(accept))
    backoff = {}
    if backoff_to_lowercase:
        print("WARNING: backing off to lowercase")
        for x, idx in accept.items():
            if x == 'UNKNOWN':
                continue
            backoff[x.lower()] = None
            backoff[x.casefold()] = None

    # file = gzip.open(filename, 'rt') if filename.endswith('.gz') else open(filename)

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
                    print('================================')
                    print('!!!WARNING WITH SHAPE, first shape ', first_dim, ' current shape: ', np_found.shape,
                          '  word: ', word)
                    print('!!!WARNING WITH SHAPE line: ', line)
                    print('!!!WARNING WITH SHAPE IGNORING THIS LINE IN THE EMBEDDINGS')
                    continue
                found[accept[word]] = np_found
                if file_found_out is not None:
                    file_found_out.write(line)
            # else:
            #     print('(kzaporoj 13/04/2021) NOT FOUND debugging: ', word)

            if word in backoff:
                backoff[word] = np.asarray(values[1], dtype='float32')
                if file_found_out is not None:
                    file_found_out.write(line)
        # else:
        #     print('(kzaporoj 13/04/2021) not valid word debugging: ', line)
    file.close()

    # BEGIN: code for debugging purposes only
    # for word, word_id in accept.items():
    #     if word_id not in found:
    #         print('the following embedding element not found: ', word)

    # END: code for debugging purposes only

    if file_found_out is not None:
        file_found_out.close()

    all_embeddings = np.asarray(list(found.values()))
    print('nr of lines in file: ', nr_lines_file)
    print('shape of all_embeddings: ', all_embeddings.shape)
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

    print("found: {} / {} = {}".format(len(found), len(accept), len(found) / len(accept)))
    print("words/embedding entries randomly initialized:", len(accept) - len(found))
    print('the embeddings norm is: ', torch.linalg.norm(embeddings))
    print('the embeddings mean is: ', embeddings_mean)
    print('the embeddings std is: ', embeddings_std)

    if debug:
        counter = 0
        for word in accept.keys():
            if accept[word] not in found:
                print("no such pretrained word: {} ({})".format(word, counter))
                counter += 1

    if backoff_to_lowercase:
        num_backoff = 0
        for word, idx in accept.items():
            if word == 'UNKNOWN':
                continue
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


def load_wordembeddings_words(filename):
    words = []

    print("loading words:", filename)
    file = gzip.open(filename, 'rt') if filename.endswith('.gz') else open(filename)
    for line in file:
        # Note: use split(' ') instead of split() if you get an error.
        values = line.rstrip().split(' ')
        words.append(values[0])
    file.close()

    return words

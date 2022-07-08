import json
import logging

import torch
from transformers import BertTokenizer

from data_processing.dictionary import Dictionary
from models import model_create

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger()


def load_dictionary(config, path):
    type = config['type']
    filename = config['filename']
    filename = filename if ('/' in filename) else '{}/{}'.format(path, filename)
    logger.debug('load_dictionary of config %s and path %s and filename %s' % (config, path, filename))
    if type == 'word2vec':
        dictionary = Dictionary()
    elif type == 'spirit':
        dictionary = Dictionary()
        dictionary.load_spirit_dictionary(filename, config['threshold'])
    elif type == 'vocab':
        dictionary = Dictionary()
        dictionary.load_wordpiece_vocab(filename)
    elif type == 'json':
        dictionary = Dictionary()
        dictionary.load_json(filename)
    elif type == 'bert':
        dictionary = BertTokenizer.from_pretrained(config['filename'])
    else:
        raise BaseException('no such type', type)

    return dictionary


def create_linking_candidates(config, entity_dictionary: Dictionary):
    candidates_path = config['file']
    max_link_candidates = config['max_link_candidates']
    span_text_to_candidates = dict()
    for curr_line in open(candidates_path):
        curr_span_candidates = json.loads(curr_line)
        span_text = curr_span_candidates['text'].strip()  # TODO: makes sense lowercasing, or will make it worse???
        span_candidates = curr_span_candidates['candidates']
        span_scores = curr_span_candidates['scores']
        # candidates should come sorted by score, but just in case sorts again
        sorted_candidates = sorted(zip(span_candidates, span_scores), key=lambda x: x[1], reverse=True)
        if max_link_candidates > -1:
            sorted_candidates = sorted_candidates[:max_link_candidates]

        span_text_to_candidates[span_text] = dict()

        scores_list = list()
        candidates_list = list()
        for curr_candidate, curr_score in sorted_candidates:
            candidates_list.append(entity_dictionary.add(curr_candidate))
            scores_list.append(curr_score)
        # passes to torch.tensor in order to decrease the memory footprint - the lists consume too much memory in python
        span_text_to_candidates[span_text]['candidates'] = torch.tensor(candidates_list, dtype=torch.int)
        span_text_to_candidates[span_text]['scores'] = torch.tensor(scores_list, dtype=torch.float)

    return span_text_to_candidates


def create_dictionaries(config, training):
    path = config['dictionaries_path']

    logger.info('Loading dictionaries (training={}) from path {}'.format(training, path))

    if 'dictionaries' in config:
        dictionaries = {}
        for name, dict_config in config['dictionaries'].items():
            if training:
                if 'init' in dict_config:
                    dictionary = load_dictionary(dict_config['init'], path)
                    if isinstance(dictionary, Dictionary):
                        logger.info('init {}: size={}'.format(name, dictionary.size))
                    elif isinstance(dictionary, BertTokenizer):
                        logger.info('init {}: size={}'.format(name, dictionary.vocab_size))
                    else:
                        raise Exception('not recognized dictionary: ', dictionary)
                else:
                    logger.info('init {} (blank)'.format(name))
                    dictionary = Dictionary()
            else:
                if 'init' in dict_config:
                    dictionary = load_dictionary(dict_config['init'], path)
                    if isinstance(dictionary, Dictionary):
                        logger.info('init {}: size={}'.format(name, dictionary.size))
                    elif isinstance(dictionary, BertTokenizer):
                        logger.info('init {}: size={}'.format(name, dictionary.vocab_size))
                    else:
                        raise Exception('not recognized dictionary: ', dictionary)
                else:
                    dictionary = load_dictionary(dict_config, path)
                    logger.info('load {}: size={}'.format(name, dictionary.size))

            dictionary.prefix = dict_config['prefix'] if 'prefix' in dict_config else ''

            if 'rewriter' in dict_config:
                if dict_config['rewriter'] == 'lowercase':
                    dictionary.rewriter = lambda t: t.lower()
                elif dict_config['rewriter'] == 'none':
                    logger.info('rewriter: none')
                else:
                    raise BaseException('no such rewriter', dict_config['rewriter'])

            if 'append' in dict_config:
                for x in dict_config['append']:
                    idx = dictionary.add(x)
                    logger.info('   add token %s -> %s' % (x, idx))

            if 'unknown' in dict_config:
                dictionary.set_unknown_token(dict_config['unknown'])

            if 'debug' in dict_config:
                dictionary.debug = dict_config['debug']

            if 'update' in dict_config:
                dictionary.update = dict_config['update']

            if isinstance(dictionary, Dictionary):
                logger.info('   update: %s' % dictionary.update)
                logger.info('   debug: %s' % dictionary.debug)

            dictionaries[name] = dictionary

        return dictionaries
    else:
        logger.info('WARNING: using wikipedia dictionary')
        words = Dictionary()
        entities = Dictionary()

        words.set_unknown_token('UNKNOWN')
        words.load_spirit_dictionary('data/tokens.dict', 5)
        entities.set_unknown_token('UNKNOWN')
        entities.load_spirit_dictionary('data/entities.dict', 5)
        return {
            'words': words,
            'entities': entities
        }


def create_model(config, dictionaries):
    model = model_create(config['model'], dictionaries)

    logger.debug('Model: %s' % model)

    regularization = config['optimizer']['regularization'] if 'regularization' in config['optimizer'] else {}

    logger.debug('Parameters:')
    parameters = []
    num_params = 0
    for key, value in dict(model.named_parameters()).items():
        if not value.requires_grad:
            logger.debug('skip %s' % key)
            continue
        else:
            if key in regularization:
                logger.debug('param {} size={} l2={}'.format(key, value.numel(), regularization[key]))
                parameters += [{'params': value, 'weight_decay': regularization[key]}]
            else:
                logger.debug('param {} size={}'.format(key, value.numel()))
                parameters += [{'params': value}]
        num_params += value.numel()
    logger.debug('total number of params: {} = {}M'.format(num_params, num_params / 1024 / 1024 * 4))

    init_cfg = config['optimizer']['initializer'] if 'initializer' in config['optimizer'] else {}

    logger.debug('Initializaing parameters')
    for key, param in dict(model.named_parameters()).items():
        for initializer in [y for x, y in init_cfg.items() if x in key]:
            if initializer == 'orthogonal':
                logger.debug('ORTHOGONAL %s %s' % (key, param.data.size()))
                torch.nn.init.orthogonal_(param.data)
            elif initializer == 'rnn-orthogonal':
                logger.debug('before: %s %s' % (param.data.size(), param.data.sum().item()))
                for tmp in torch.split(param.data, param.data.size(1), dim=0):
                    torch.nn.init.orthogonal_(tmp)
                    logger.debug('RNN-ORTHOGONAL %s %s %s' % (key, tmp.size(), param.data.sum().item()))
                logger.debug('after: %s %s ' % (param.data.size(), param.data.sum().item()))
            elif initializer == 'xavier_normal':
                before = param.data.norm().item()
                torch.nn.init.xavier_normal_(param.data)
                after = param.data.norm().item()
                logger.debug('XAVIER_NORMAL %s %s %s -> %s' % (key, param.data.size(), before, after))
            break

    return model, parameters

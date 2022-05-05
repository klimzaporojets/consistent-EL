import json

import torch
# from data.dataset import create_datasets
from transformers import BertTokenizer

from data.dictionary import Dictionary
from models import model_create


def load_dictionary(config, path):
    type = config['type']
    filename = config['filename']
    filename = filename if filename.startswith("/") else "{}/{}".format(path, filename)

    if type == 'word2vec':
        print("init {} with {}".format(path, filename))
        dictionary = Dictionary(filename)
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
        raise BaseException("no such type", type)

    return dictionary


def create_linking_candidates(config, entity_dictionary: Dictionary):
    # print('STARTING LOADING LINKING CANDIDATES')
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

    # print('END LOADING LINKING CANDIDATES')

    return span_text_to_candidates


def create_dictionaries(config, training):
    path = config['path']

    print("Loading dictionaries (training={})".format(training))

    if 'dictionaries' in config:
        dictionaries = {}
        for name, dict_config in config['dictionaries'].items():
            if training:
                if "init" in dict_config:
                    dictionary = load_dictionary(dict_config['init'], path)
                    if isinstance(dictionary, Dictionary):
                        print('init {}: size={}'.format(name, dictionary.size))
                    elif isinstance(dictionary, BertTokenizer):
                        print('init {}: size={}'.format(name, dictionary.vocab_size))
                    else:
                        raise Exception('not recognized dictionary: ', dictionary)
                else:
                    print("init {} (blank)".format(name))
                    dictionary = Dictionary()
            else:
                dictionary = load_dictionary(dict_config, path)
                print('load {}: size={}'.format(name, dictionary.size))

            dictionary.prefix = dict_config['prefix'] if 'prefix' in dict_config else ''

            if 'rewriter' in dict_config:
                if dict_config['rewriter'] == 'lowercase':
                    dictionary.rewriter = lambda t: t.lower()
                elif dict_config['rewriter'] == 'none':
                    print("rewriter: none")
                else:
                    raise BaseException("no such rewriter", dict_config['rewriter'])

            if 'append' in dict_config:
                for x in dict_config['append']:
                    idx = dictionary.add(x)
                    print("   add token", x, "->", idx)

            if 'unknown' in dict_config:
                dictionary.set_unknown_token(dict_config['unknown'])

            if 'debug' in dict_config:
                dictionary.debug = dict_config['debug']

            if 'update' in dict_config:
                dictionary.update = dict_config['update']

            # kzaporoj 20/12/2020 - I comment this update to false, reason: let's say we want to try on a different
            # domain corpus such as AIDA Conll a particular model trained on DWIE. There will be many words non-existent
            # in DWIE, but whose embedding can give extra information if they are close enough to the embeddings of
            # words in DWIE.
            # if not training:
            #     dictionary.update = False

            if isinstance(dictionary, Dictionary):
                print("   update:", dictionary.update)
                print("   debug:", dictionary.debug)

            dictionaries[name] = dictionary

        return dictionaries
    else:
        print("WARNING: using wikipedia dictionary")
        words = Dictionary()
        entities = Dictionary()

        words.set_unknown_token("UNKNOWN")
        words.load_spirit_dictionary('data/tokens.dict', 5)
        entities.set_unknown_token("UNKNOWN")
        entities.load_spirit_dictionary('data/entities.dict', 5)
        return {
            'words': words,
            'entities': entities
        }


def create_model(config, dictionaries):
    # model_name = config['model']['name']
    # if model_name == "model3":
    #     model = MyModel3(dictionaries, config['model'])
    # elif model_name == "model4":
    #     model = MyModel4(dictionaries, config['model'])
    #     # model.load_tensorflow_model()
    # elif model_name == "model5":
    #     model = MyModel5(dictionaries, config['model'])
    # elif model_name == "entbydecr1":
    #     model = entybydesc.MyModel1(dictionaries, config['model'])
    # elif model_name == "model6":
    #     model = MyModel6(dictionaries, config['model'])
    # elif model_name == "lm_1":
    #     model = LM1(dictionaries, config['model'])
    # elif model_name == "lm_2":
    #     model = LM2(dictionaries, config['model'])
    # elif model_name == "ner_1":
    #     model = Ner1(dictionaries, config['model'])
    # elif model_name == "ner_2":
    #     model = Ner2(dictionaries, config['model'])
    # elif model_name == "linker_1":
    #     model = Linker1(dictionaries, config['model'])
    # elif model_name == "nerlink_1":
    #     model = NerLink1(dictionaries, config['model'])
    # elif model_name == "nerlink_2":
    #     model = NerLink2(dictionaries, config['model'])
    # elif model_name == "linker_adv_1":
    #     model = LinkerAdv1(dictionaries, config['model'])
    # else:
    #     raise BaseException("no such model: ", model_name)
    # from models import model_create
    model = model_create(config['model'], dictionaries)

    print("Model:", model)

    regularization = config['optimizer']['regularization'] if 'regularization' in config['optimizer'] else {}

    print("Parameters:")
    parameters = []
    num_params = 0
    for key, value in dict(model.named_parameters()).items():
        # 		print(key)
        if not value.requires_grad:
            print("skip ", key)
            continue
        else:
            if key in regularization:
                print("param {} size={} l2={}".format(key, value.numel(), regularization[key]))
                parameters += [{"params": value, "weight_decay": regularization[key]}]
            else:
                print("param {} size={}".format(key, value.numel()))
                parameters += [{"params": value}]
        num_params += value.numel()
    print("total number of params: {} = {}M".format(num_params, num_params / 1024 / 1024 * 4))
    print()

    init_cfg = config['optimizer']['initializer'] if 'initializer' in config['optimizer'] else {}

    print("Initializaing parameters")
    for key, param in dict(model.named_parameters()).items():
        for initializer in [y for x, y in init_cfg.items() if x in key]:
            if initializer == 'orthogonal':
                # is this correct for RNNs, don't think so ?
                print("ORTHOGONAL", key, param.data.size())
                torch.nn.init.orthogonal_(param.data)
            elif initializer == 'rnn-orthogonal':
                print("before:", param.data.size(), param.data.sum().item())
                for tmp in torch.split(param.data, param.data.size(1), dim=0):
                    torch.nn.init.orthogonal_(tmp)
                    print("RNN-ORTHOGONAL", key, tmp.size(), param.data.sum().item())
                print("after:", param.data.size(), param.data.sum().item())
            elif initializer == 'xavier_normal':
                before = param.data.norm().item()
                torch.nn.init.xavier_normal_(param.data)
                after = param.data.norm().item()
                print("XAVIER_NORMAL", key, param.data.size(), before, "->", after)
            break
    print()

    return model, parameters

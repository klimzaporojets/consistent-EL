import csv
import json
import logging
import os
import pickle
import shutil
from statistics import stdev, mean
from typing import Dict, List

# import third_party/cpn-evaluator/python/cpn_eval.py
# import cpn_eval
from misc.python_cpn_eval import load_jsonl, EvaluatorCPN, load_json
from stats.results.utils.print_results_tables import get_csv_line_v2, print_latex_table_coreflinker, \
    print_latex_table_coreflinker_nill

max_coref_props = 4
max_att_props = 4
max_rel_props = 4

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger()


def get_dict_recursive_key_value(dictionary: Dict, curr_entry=''):
    curr_set_entry_values = set()
    for curr_key, curr_value in dictionary.items():
        if isinstance(curr_value, dict):
            if curr_entry == '':
                curr_values = get_dict_recursive_key_value(curr_value, curr_key)
            else:
                curr_values = get_dict_recursive_key_value(curr_value, curr_entry + '.' + curr_key)
            curr_set_entry_values = curr_set_entry_values.union(curr_values)
        else:
            if isinstance(curr_value, list):
                curr_value = tuple(curr_value)

            if curr_entry == '':
                curr_set_entry_values.add((curr_key, curr_value))
            else:
                curr_set_entry_values.add((curr_entry + '.' + curr_key, curr_value))

    return curr_set_entry_values


def is_matching_config(json_config: Dict, filter_json: Dict):
    """ Example of filter_json: {'relation_types':{'binary_x'}} """

    if 'mentionwise_rels' in filter_json and filter_json['mentionwise_rels'] is True:
        if 'mentionwise' not in json_config['model']['relations'] or \
                json_config['model']['relations']['mentionwise'] is False:
            return False

    if 'relation_types' in filter_json:
        if not (json_config['model']['relations']['type'] in filter_json['relation_types']):
            return False

    if 'is_joint' in filter_json:
        is_ner = 0
        is_coref = 0
        is_rel = 0

        if json_config['model']['ner']['enabled']:
            is_ner = 1

        if json_config['model']['coref']['enabled']:
            is_coref = 1

        if json_config['model']['relations']['enabled']:
            is_rel = 1

        if filter_json['is_joint'] and is_ner + is_coref + is_rel <= 2:
            return False
        elif not filter_json['is_joint'] and is_ner + is_coref + is_rel > 1:
            return False

    if 'is_merged_with_ner_only' in filter_json:
        if 'merged_with_ner_only' not in json_config:
            if filter_json['is_merged_with_ner_only']:
                return False
        else:
            if filter_json['is_merged_with_ner_only'] != json_config['merged_with_ner_only']:
                return False

    if 'is_bert' in filter_json:

        if 'bert_embedder' in json_config['model']['text_embedder']:
            field_is_bert = True
        else:
            field_is_bert = False
        if filter_json['is_bert'] != field_is_bert:
            return False

    if 'is_tag' in filter_json:
        if filter_json['is_tag'] != json_config['model']['ner']['enabled']:
            return False

    if 'is_coref' in filter_json:
        if filter_json['is_coref'] != json_config['model']['coref']['enabled']:
            return False

    if 'is_relation' in filter_json:
        if filter_json['is_relation'] != json_config['model']['relations']['enabled']:
            return False

    if 'coref_prop' in filter_json:
        if 'coref_prop' not in json_config['model']['corefprop'] and filter_json['coref_prop'] != 0:
            return False

        if 'coref_prop' in json_config['model']['corefprop']:
            if filter_json['coref_prop'] > -1:
                if filter_json['coref_prop'] != json_config['model']['corefprop']['coref_prop']:
                    return False
            else:
                if not json_config['model']['corefprop']['coref_prop'] > 0 or \
                        json_config['model']['corefprop']['coref_prop'] > filter_json['max_props']:
                    return False

    if 'rel_prop' in filter_json:
        if 'rel_prop' not in json_config['model']['relprop'] and filter_json['rel_prop'] != 0:
            return False

        if 'rel_prop' in json_config['model']['relprop']:
            if filter_json['rel_prop'] > -1:
                if filter_json['rel_prop'] != json_config['model']['relprop']['rel_prop']:
                    return False
            else:
                if not json_config['model']['relprop']['rel_prop'] > 0 or \
                        json_config['model']['relprop']['rel_prop'] > filter_json['max_props']:
                    return False

    if 'att_prop' in filter_json:
        if 'spanprop' not in json_config['model'] and filter_json['att_prop'] != 0:
            return False

        if 'spanprop' in json_config['model']:
            if filter_json['att_prop'] > -1:
                if filter_json['att_prop'] != json_config['model']['spanprop']['att_prop']:
                    return False
            else:
                if not json_config['model']['spanprop']['att_prop'] > 0 or \
                        json_config['model']['spanprop']['att_prop'] > filter_json['max_props']:
                    return False
    return True


def get_best_joint(results: List, specific_config: List[str] = None):
    best_f1_avg = -1.0
    best_result = None
    for curr_result in results:
        f1_ner = curr_result['experiment_results']['tags']['tags_soft']['f1'][0]
        f1_rel = curr_result['experiment_results']['relations']['rels_soft']['f1'][0]
        f1_coref_avg = (curr_result['experiment_results']['coref']['muc']['f1'][0] +
                        curr_result['experiment_results']['coref']['b_cubed_singleton_men_conll']['f1'][0] +
                        curr_result['experiment_results']['coref']['ceafe_singleton']['f1'][0]) / 3
        if specific_config is None:
            if (f1_ner + f1_rel + f1_coref_avg) / 3 > best_f1_avg:
                best_f1_avg = ((f1_ner + f1_rel + f1_coref_avg) / 3)
                best_result = curr_result
        else:
            for curr_specific_config in specific_config:
                if curr_specific_config in curr_result['experiment_id']:
                    if (f1_ner + f1_rel + f1_coref_avg) / 3 > best_f1_avg:
                        best_f1_avg = ((f1_ner + f1_rel + f1_coref_avg) / 3)
                        best_result = curr_result

    return best_result


def get_best_single(results: List, type='ner', specific_config: List = None):
    """

    :param specific_config:
    :param results:
    :param type:
    :return:
    """
    best_f1 = -1.0
    best_result = None
    for curr_result in results:
        f1_compare = 0
        if type == 'ner':
            f1_compare = curr_result['experiment_results']['tags']['tags_mention']['f1'][0]

        elif type == 'coref':
            f1_compare = (curr_result['experiment_results']['coref']['muc']['f1'][0] +
                          curr_result['experiment_results']['coref']['b_cubed_singleton_men_conll']['f1'][0] +
                          curr_result['experiment_results']['coref']['ceafe_singleton']['f1'][0]) / 3
        elif type == 'rel':
            f1_compare = curr_result['experiment_results']['relations']['rels_mention']['f1'][0]

        if specific_config is None:
            if f1_compare > best_f1:
                best_f1 = f1_compare
                best_result = curr_result
        else:
            for curr_specific_config in specific_config:
                if curr_specific_config in curr_result['experiment_id']:
                    if f1_compare > best_f1:
                        best_f1 = f1_compare
                        best_result = curr_result

    return best_result


def get_csv_title_line():
    return ['', '', '']


def my_stdev(numbers: List):
    if len(numbers) < 2:
        return -1.0
    else:
        return stdev(numbers)


def merge_ner_with_coref(coref_test_path, ner_test_path, merged_coref_ner_path, id_exp, create_clusters=False):
    ner_test_file = '{}-{}/test.json'.format(ner_test_path, id_exp)
    coref_test_file = '{}-{}/test.json'.format(coref_test_path, id_exp)
    loaded_ner_json = load_jsonl(ner_test_file, None)
    loaded_coref_json = load_jsonl(coref_test_file, None)
    merged_coref_ner_json = dict()
    for id_file, ner_content in loaded_ner_json.items():
        mention_pos_to_concept = dict()
        concept_to_mention = dict()
        coref_content = loaded_coref_json[id_file]
        for curr_mention in coref_content['mentions']:
            mention_pos_to_concept[(curr_mention['begin'], curr_mention['end'])] = curr_mention['concept']
            if curr_mention['concept'] not in concept_to_mention:
                concept_to_mention[curr_mention['concept']] = [curr_mention]
            else:
                concept_to_mention[curr_mention['concept']].append(curr_mention)

        max_concept_id = len(coref_content['concepts']) - 1
        for curr_mention in ner_content['mentions']:
            pred_tag_types = curr_mention['tags']
            if (curr_mention['begin'], curr_mention['end']) not in mention_pos_to_concept:
                if create_clusters:
                    curr_mention['concept'] = max_concept_id + 1
                    coref_content['mentions'].append(curr_mention)
                    coref_content['concepts'].append({'concept': max_concept_id + 1, 'text': curr_mention['text'],
                                                      'count': 1, 'tags': pred_tag_types})
                    max_concept_id += 1
            else:
                coref_concept_id = mention_pos_to_concept[(curr_mention['begin'], curr_mention['end'])]
                if coref_concept_id >= len(coref_content['concepts']):
                    logger.warning('something wrong here the concept id is bigger than the nr of concepts!!')

                for curr_men in concept_to_mention[coref_concept_id]:
                    curr_men['tags'] = list(set(curr_men['tags'] + pred_tag_types))

                coref_content['concepts'][coref_concept_id]['tags'].extend(pred_tag_types)
                coref_content['concepts'][coref_concept_id]['tags'] = \
                    list(set(coref_content['concepts'][coref_concept_id]['tags']))

        merged_coref_ner_json[id_file] = coref_content

    # if path doesn't exist, creates it, also copies the config of coref there
    os.makedirs('{}-{}/'.format(merged_coref_ner_path, id_exp), exist_ok=True)
    shutil.copyfile('{}-{}/config.json'.format(coref_test_path, id_exp),
                    '{}-{}/config.json'.format(merged_coref_ner_path, id_exp))

    with open('{}-{}/test.json'.format(merged_coref_ner_path, id_exp), 'w') as out_merged_file:
        for curr_json_merged in merged_coref_ner_json.values():
            out_merged_file.write(json.dumps(curr_json_merged) + '\n')


def merge_rel_with_coref(coref_test_path, rel_test_path, merged_coref_rel_path, id_exp, create_clusters=False):
    rel_test_file = '{}-{}/test.json'.format(rel_test_path, id_exp)
    coref_test_file = '{}-{}/test.json'.format(coref_test_path, id_exp)
    loaded_rel_json = load_jsonl(rel_test_file, None)
    loaded_coref_json = load_jsonl(coref_test_file, None)
    merged_coref_ner_json = dict()
    for id_file, rel_content in loaded_rel_json.items():
        mention_pos_to_concept = dict()
        concept_to_mention = dict()
        coref_content = loaded_coref_json[id_file]
        for curr_mention in coref_content['mentions']:
            mention_pos_to_concept[(curr_mention['begin'], curr_mention['end'])] = curr_mention['concept']
            if curr_mention['concept'] not in concept_to_mention:
                concept_to_mention[curr_mention['concept']] = [curr_mention]
            else:
                concept_to_mention[curr_mention['concept']].append(curr_mention)
        max_concept_id = len(coref_content['concepts']) - 1
        already_added = set()
        for curr_relation in rel_content['mention_relations']:
            curr_mention_s = rel_content['mentions'][curr_relation['s']]
            curr_mention_o = rel_content['mentions'][curr_relation['o']]

            curr_concept_s = -1
            curr_concept_o = -1
            if (curr_mention_s['begin'], curr_mention_s['end']) not in mention_pos_to_concept:
                if create_clusters:
                    curr_mention_s['concept'] = max_concept_id + 1
                    coref_content['mentions'].append(curr_mention_s)
                    coref_content['concepts'].append({'concept': max_concept_id + 1, 'text': curr_mention_s['text'],
                                                      'count': 1, 'tags': []})
                    curr_concept_s = max_concept_id
                    max_concept_id += 1
            else:
                curr_concept_s = mention_pos_to_concept[(curr_mention_s['begin'], curr_mention_s['end'])]

            if (curr_mention_o['begin'], curr_mention_o['end']) not in mention_pos_to_concept:
                if create_clusters:
                    curr_mention_o['concept'] = max_concept_id + 1
                    coref_content['mentions'].append(curr_mention_o)
                    coref_content['concepts'].append({'concept': max_concept_id + 1, 'text': curr_mention_o['text'],
                                                      'count': 1, 'tags': []})
                    curr_concept_o = max_concept_id
                    max_concept_id += 1
            else:
                curr_concept_o = mention_pos_to_concept[(curr_mention_o['begin'], curr_mention_o['end'])]

            if curr_concept_s > -1 and curr_concept_o > -1:
                # checks if it already exists:
                already_exists = False
                if (curr_concept_s, curr_concept_o, curr_relation['p']) in already_added:
                    already_exists = True
                if not already_exists:
                    coref_content['relations'].append(
                        {'s': curr_concept_s, 'o': curr_concept_o, 'p': curr_relation['p']})
                    already_added.add((curr_concept_s, curr_concept_o, curr_relation['p']))

        merged_coref_ner_json[id_file] = coref_content

    # if path doesn't exist, creates it, also copies the config of coref there
    os.makedirs('{}-{}/'.format(merged_coref_rel_path, id_exp), exist_ok=True)
    shutil.copyfile('{}-{}/config.json'.format(coref_test_path, id_exp),
                    '{}-{}/config.json'.format(merged_coref_rel_path, id_exp))

    with open('{}-{}/test.json'.format(merged_coref_rel_path, id_exp), 'w') as out_merged_file:
        for curr_json_merged in merged_coref_ner_json.values():
            out_merged_file.write(json.dumps(curr_json_merged) + '\n')


if __name__ == "__main__":

    # some experiments have to be excluded since the setup is not entirely correct or were just small tests, etc.
    to_exclude = {
        # 'selected',
        '20200923-base-linker',
        '20200922-dwie20200908-coreflinker',
        '20200922-dwie-only-linker',
        '20200922-dwie20200908-base',
        '20200921-coreflinker',
        '20200923-coreflinker',
        '20200923-dwie20200908-coreflinker',
        '20200924-dwie20200908-coreflinker',
        '20200924-dwie20200908-base-linker',
        '20200923-dwie20200908-base-linker',
        '20200925-base-linker',
        '20201006-base-coref-scores',
        '20201006-coreflinker-scores',
        '20201006-dwie20200908-base-coref-scores',
        '20201006-dwie20200908-coreflinker-scores'
    }

    # {'experiment_name': '20201207-coref_e2e-ap0', 'test_tag': 'test', 'config_file':'config.json'},
    # {'experiment_name': '20201207-coreflinker_e2e-ap0', 'test_tag': 'test', 'config_file':'config.json'},
    # {'experiment_name': '20201207-linker_e2e_all-ap0', 'test_tag': 'test', 'config_file':'config.json'},
    # {'experiment_name': '20201207-linker_e2e_pruner-ap0', 'test_tag': 'test', 'config_file':'config.json'},
    # {'experiment_name': '20201208-cl_e2e_singl_matr_snnet_ignore_chains_pr_loss-ap0', 'test_tag': 'test', 'config_file':'config.json'},
    # {'experiment_name': '20201208-cl_e2e_singl_matr_snnet_ignore_chains-ap0', 'test_tag': 'test', 'config_file':'config.json'},
    # {'experiment_name': '20201208-cl_e2e_singl_matr_ennet_ignore_chains_pr_loss-ap0', 'test_tag': 'test', 'config_file':'config.json'},
    # {'experiment_name': '20201208-cl_e2e_singl_matr_ennet_ignore_chains-ap0', 'test_tag': 'test', 'config_file':'config.json'},
    # {'experiment_name': '20201214-linker_e2e_pruner_coref-ap0', 'test_tag': 'test', 'config_file':'config.json'},
    # {'experiment_name': '20210219-cl_e2e_mtt_float32_arsinh_dwie_j-ap0', 'test_tag': 'test',
    #  'config_file': 'config.json','alias':''},
    # {'experiment_name': '20201203-coreflinker_e2e-ap0', 'test_tag': 'test', 'config_file': 'config.json'},

    # {'experiment_name': '20210223-cl_mtt_float32_arsinh_no_multihead', 'test_tag': 'test',
    #  'config_file': 'config.json', 'alias': 'MTT arh', 'sort_order': 9},
    # {'experiment_name': '20210223-cl_mtt_float32_multihead', 'test_tag': 'test', 'config_file': 'config.json',
    #  'alias': 'MTT (mh)', 'sort_order': 8},
    # {'experiment_name': '20210223-cl_mtt_float64_arsinh_multihead-ap0', 'test_tag': 'test',
    #  'config_file': 'config.json', 'alias': 'MTT arh (mh)', 'sort_order': 6},
    # {'experiment_name': '20210225-cl_edmonds_softmax', 'test_tag': 'test', 'config_file': 'config.json',
    #  'alias': 'Edmonds Soft', 'sort_order': 2},
    # {'experiment_name': '20210223-cl_mtt_float32_no_multihead', 'test_tag': 'test', 'config_file': 'config.json',
    #  'alias': 'MTT', 'sort_order': 5},
    # {'experiment_name': '20210222-cl_e2e_mtt_float32_arsinh-ap0', 'test_tag': 'test', 'config_file': 'config.json',
    #  'alias': 'MTT arh (mh)', 'sort_order': 6},

    only_include_these = [  # if this is empty, then includes all except the ones in to_exclude
        # {'experiment_name': '20210222b-cl_e2e_mtt_float32_arsinh-ap0', 'test_tag': 'test',
        #  'config_file': 'config.json', 'alias': 'MTT arh (mh)', 'sort_order': 5},
        # {'experiment_name': '20210223-cl_mtt_float64_arsinh_no_multihead', 'test_tag': 'test',
        #  'config_file': 'config.json', 'alias': 'MTT arh', 'sort_order': 3},
        # {'experiment_name': '20210223-cl_mtt_float64_multihead', 'test_tag': 'test', 'config_file': 'config.json',
        #  'alias': 'MTT (mh)', 'sort_order': 4},
        # {'experiment_name': '20210223-cl_mtt_float64_no_multihead', 'test_tag': 'test', 'config_file': 'config.json',
        #  'alias': 'MTT', 'sort_order': 2},
        # {'experiment_name': '20201207-coreflinker_e2e-ap0', 'test_tag': 'test', 'config_file': 'config.json',
        #  'alias': 'CorefLkr', 'sort_order': 2},
        # {'experiment_name': '20201214-linker_e2e_pruner_coref-ap0', 'test_tag': 'test', 'config_file': 'config.json',
        #  'alias': 'Base+Coref', 'sort_order': 1},
        # {'experiment_name': '20201207-linker_e2e_pruner-ap0', 'test_tag': 'test', 'config_file': 'config.json',
        #  'alias': 'Base', 'sort_order': 0},
        # {'experiment_name': '20210311-cl_mtt_arsh_bert_plain-ap0', 'test_tag': 'test', 'config_file': 'config.json',
        #  'alias': 'BERT v1', 'sort_order': 7},
        # {'experiment_name': '20210316-cl_mtt_arsh_bert_w_lstm-ap0', 'test_tag': 'test', 'config_file': 'config.json',
        #  'alias': 'BERT v2', 'sort_order': 8},
        # {'experiment_name': '20210316-cl_mtt_arsh_bert_w_lstm_nwcemb-ap0', 'test_tag': 'test', 'config_file': 'config.json',
        #  'alias': 'BERT v3', 'sort_order': 9},
        # {'experiment_name': '20210316-cl_mtt_arsh_bert_w_lstm_nwemb-ap0', 'test_tag': 'test', 'config_file': 'config.json',
        #  'alias': 'BERT v4', 'sort_order': 10},
        # {'experiment_name': '20210316-cl_mtt_arsh_spanbert_w_lstm-ap0', 'test_tag': 'test', 'config_file': 'config.json',
        #  'alias': 'BERT v5', 'sort_order': 11},
        # {'experiment_name': '20210316-cl_mtt_arsh_spanbert_w_lstm_nwcemb-ap0', 'test_tag': 'test', 'config_file': 'config.json',
        #  'alias': 'BERT v6', 'sort_order': 12},
        # {'experiment_name': '20210316-cl_mtt_arsh_spanbert_w_lstm_nwemb-ap0', 'test_tag': 'test', 'config_file': 'config.json',
        #  'alias': 'BERT v7', 'sort_order': 13},
        # {'experiment_name': '20210318-cl_mtt_arsh_spanbert_plain-ap0', 'test_tag': 'test', 'config_file': 'config.json',
        #  'alias': 'SP BERT 1', 'sort_order': 14},
        # {'experiment_name': '20210325-cl_mtt_arsh_spanbert_subtoks-ap0', 'test_tag': 'test',
        #  'config_file': 'config.json', 'alias': 'SP BERT 2', 'sort_order': 15},
        # {'experiment_name': '20210326-cl_mtt_arsh_sbert_hatt-ap0', 'test_tag': 'test',
        #  'config_file': 'config.json', 'alias': 'SBERT ATT', 'sort_order': 16}
        # {'experiment_name': '20210327-cl_mtt_sbert_hatt_ft_e-4-ap0', 'test_tag': 'test',
        #  'config_file': 'config.json', 'alias': 'SBERT tune', 'sort_order': 17},
        # {'experiment_name': '20210327-cl_mtt_sbert_hatt_finetune-ap0', 'test_tag': 'test',
        #  'config_file': 'config.json', 'alias': 'SBERT tune2', 'sort_order': 19},
        # {'experiment_name': '20210327-cl_mtt_sbert_hatt_200iters-ap0', 'test_tag': 'test',
        #  'config_file': 'config.json', 'alias': 'SBERT 200 iters', 'sort_order': 18},
        # {'experiment_name': '20210328-cl_mtt_sbert_large_hatt-ap0', 'test_tag': 'test',
        #  'config_file': 'config.json', 'alias': 'SBERT large', 'sort_order': 20},
        # {'experiment_name': '20210328-mtt_sbert_hatt_s384-ap0', 'test_tag': 'test',
        #  'config_file': 'config.json', 'alias': 'SBERT s384', 'sort_order': 21},
        # {'experiment_name': '20210328-mtt_sbert_hatt_s128-ap0', 'test_tag': 'test',
        #  'config_file': 'config.json', 'alias': 'SBERT s128', 'sort_order': 22},
        ### baseline
        # {'experiment_name': '20210328-linker_sbert_hatt-ap0', 'test_tag': 'test',
        #  'config_file': 'config.json', 'alias': 'SB b', 'sort_order': 23},
        # {'experiment_name': '20210401-linker_sbert_ftune_1e-5-ap0', 'test_tag': 'test',
        #  'config_file': 'config.json', 'alias': 'SB b 1e-5', 'sort_order': 24},
        # {'experiment_name': '20210401-linker_sbert_ftune_5e-5-ap0', 'test_tag': 'test',
        #  'config_file': 'config.json', 'alias': 'SB b 5e-5', 'sort_order': 25},
        # #####
        # {'experiment_name': '20210402-cl_sbert', 'test_tag': 'test',
        #  'config_file': 'config.json', 'alias': 'SB CL', 'sort_order': 26},
        # {'experiment_name': '20210402-cl_sbert_ftune_5e-5', 'test_tag': 'test',
        #  'config_file': 'config.json', 'alias': 'SB CL 5e-5', 'sort_order': 27},
        # {'experiment_name': '20210402-cl_sbert_ftune_1e-4', 'test_tag': 'test',
        #  'config_file': 'config.json', 'alias': 'SB CL 1e-4', 'sort_order': 28},
        # {'experiment_name': '20210402-cl_sbert_ftune_1e-5', 'test_tag': 'test',
        #  'config_file': 'config.json', 'alias': 'SB CL 1e-5', 'sort_order': 29},
        # {'experiment_name': '20210402-cl_sbert_large', 'test_tag': 'test',
        #  'config_file': 'config.json', 'alias': 'SB l CL', 'sort_order': 30},
        # {'experiment_name': '20210402-cl_sbert_large_ftune_1e-4', 'test_tag': 'test',
        #  'config_file': 'config.json', 'alias': 'SB l CL 1e-4', 'sort_order': 31},
        # {'experiment_name': '20210402-cl_sbert_large_ftune_1e-5', 'test_tag': 'test',
        #  'config_file': 'config.json', 'alias': 'SB l CL 1e-5', 'sort_order': 32},
        # {'experiment_name': '20210402-cl_sbert_large_ftune_5e-5', 'test_tag': 'test',
        #  'config_file': 'config.json', 'alias': 'SB l CL 5e-5', 'sort_order': 33},
        # {'experiment_name': '20210402-l_sbert_ftune_1e-4', 'test_tag': 'test',
        #  'config_file': 'config.json', 'alias': 'SB b 1e-4', 'sort_order': 34},
        # {'experiment_name': '20210402-l_sbert_large', 'test_tag': 'test',
        #  'config_file': 'config.json', 'alias': 'SB l b', 'sort_order': 35},
        # {'experiment_name': '20210402-l_sbert_large_ftune_1e-4', 'test_tag': 'test',
        #  'config_file': 'config.json', 'alias': 'SB l b 1e-4', 'sort_order': 36},
        # {'experiment_name': '20210402-l_sbert_large_ftune_1e-5', 'test_tag': 'test',
        #  'config_file': 'config.json', 'alias': 'SB l b 1e-5', 'sort_order': 37},
        # {'experiment_name': '20210402-l_sbert_large_ftune_5e-5', 'test_tag': 'test',
        #  'config_file': 'config.json', 'alias': 'SB l b 5e-5', 'sort_order': 38},
        # {'experiment_name': '20210402-mtt_sbert_ftune_5e-5', 'test_tag': 'test',
        #  'config_file': 'config.json', 'alias': 'SB mtt 5e-5', 'sort_order': 39}
        # {'experiment_name': '20210406-l_sbert_hoi-t-', 'test_tag': 'test',
        #  'config_file': 'config.json', 'alias': 'SB L HOI', 'sort_order': 40},
        # {'experiment_name': '20210408-l_sbert_hoi-t-', 'test_tag': 'test',
        #  'config_file': 'config.json', 'alias': 'SB L HOI 2', 'sort_order': 41},
        # {'experiment_name': '20210409-cl_sbert_hoi-t', 'test_tag': 'test',
        #  'config_file': 'config.json', 'alias': 'SB CL HOI', 'sort_order': 42},
        # {'experiment_name': '20210409-mtt_sbert_hoi-t', 'test_tag': 'test',
        #  'config_file': 'config.json', 'alias': 'SB MTT HOI', 'sort_order': 43},
        # {'experiment_name': '20210412-l_sb_hoi_dpt0.3_s256-', 'test_tag': 'test',
        #  'config_file': 'config.json', 'alias': 'L s256 s', 'sort_order': 44},
        # {'experiment_name': '20210412-l_sb_hoi_dpt0.3_s256_sssFalse-', 'test_tag': 'test',
        #  'config_file': 'config.json', 'alias': 'L s256', 'sort_order': 45},
        # {'experiment_name': '20210412-l_sb_hoi_dpt0.3_s256_hd300_fd2', 'test_tag': 'test',
        #  'config_file': 'config.json', 'alias': 'L s256 d300 s', 'sort_order': 46},
        # {'experiment_name': '20210412-l_sb_hoi_dpt0.3_s256_hd300_fd2_sssFalse', 'test_tag': 'test',
        #  'config_file': 'config.json', 'alias': 'L s256 d300', 'sort_order': 47},
        # {'experiment_name': '20210412-l_sb_hoi_dpt0.3_s256_hd3000_fd1-', 'test_tag': 'test',
        #  'config_file': 'config.json', 'alias': 'L s256 d3000 s', 'sort_order': 48},
        # {'experiment_name': '20210412-l_sb_hoi_dpt0.3_s256_hd3000_fd1_sssFalse-', 'test_tag': 'test',
        #  'config_file': 'config.json', 'alias': 'L s256 d3000', 'sort_order': 49},
        #
        # {'experiment_name': '20210412-l_sb_hoi_dpt0.3_s384-', 'test_tag': 'test',
        #  'config_file': 'config.json', 'alias': 'L s384 s', 'sort_order': 50},
        # {'experiment_name': '20210412-l_sb_hoi_dpt0.3_s384_sssFalse-', 'test_tag': 'test',
        #  'config_file': 'config.json', 'alias': 'L s384', 'sort_order': 51},
        # {'experiment_name': '20210412-l_sb_hoi_dpt0.3_s384_hd300_fd2', 'test_tag': 'test',
        #  'config_file': 'config.json', 'alias': 'L s384 d300 s', 'sort_order': 52},
        # {'experiment_name': '20210412-l_sb_hoi_dpt0.3_s384_hd300_fd2_sssFalse', 'test_tag': 'test',
        #  'config_file': 'config.json', 'alias': 'L s384 d300', 'sort_order': 53},
        # {'experiment_name': '20210412-l_sb_hoi_dpt0.3_s384_hd3000_fd1-', 'test_tag': 'test',
        #  'config_file': 'config.json', 'alias': 'L s384 d3000 s', 'sort_order': 54},
        # {'experiment_name': '20210412-l_sb_hoi_dpt0.3_s384_hd3000_fd1_sssFalse-', 'test_tag': 'test',
        #  'config_file': 'config.json', 'alias': 'L s384 d3000', 'sort_order': 55}
        # {'experiment_name': '20210418-l_sb_hoi_init0.02', 'test_tag': 'test',
        #  'config_file': 'config.json', 'alias': 'SB BASE', 'sort_order': 55},
        # {'experiment_name': '20210418-cl_sb_hoi_init0.02', 'test_tag': 'test',
        #  'config_file': 'config.json', 'alias': 'SB CLinker', 'sort_order': 56},
        # {'experiment_name': '20210419-mtt_sb_hoi_init0.02', 'test_tag': 'test',
        #  'config_file': 'config.json', 'alias': 'SB MTT', 'sort_order': 57}
        # {'experiment_name': '20210418-aida_l_sb_hoi_dpt0.3_init0.02_sssFalse_yamada_emb', 'test_tag': 'testa',
        #  'config_file': 'config.json', 'alias': 'SB BASE', 'sort_order': 58},
        # {'experiment_name': '20210418-aida_l_sb_hoi_dpt0.3_init0.02_sssFalse_yamada_emb', 'test_tag': 'testb',
        #  'config_file': 'config.json', 'alias': 'SB BASE', 'sort_order': 59},
        # {'experiment_name': '20210418-aida_cl_sb_init0.02_sssFalse_yamada_emb', 'test_tag': 'testa',
        #  'config_file': 'config.json', 'alias': 'SB CLinker', 'sort_order': 60},
        # {'experiment_name': '20210418-aida_cl_sb_init0.02_sssFalse_yamada_emb', 'test_tag': 'testb',
        #  'config_file': 'config.json', 'alias': 'SB CLinker', 'sort_order': 61},
        # {'experiment_name': '20210426c-mtt_50eps_t_1e-3_1e-4_50eps_b2e-5_0_20eps_no_zeros_to_cl',
        #  'test_tag': 'test', 'config_file': 'config.json', 'alias': 'SB MTT 1', 'sort_order': 62},
        # {'experiment_name': '20210426c-mtt_50eps_t_1e-3_1e-4_50eps_b2e-5_0_20eps_no_zeros_to_cl_srs',
        #  'test_tag': 'test', 'config_file': 'config.json', 'alias': 'SB MTT 2', 'sort_order': 63}
        # {'experiment_name': '20210427b-cl',
        #  'test_tag': 'test', 'config_file': 'config.json', 'alias': 'SB CL', 'sort_order': 65},
        # {'experiment_name': '20210427b-l',
        #  'test_tag': 'test', 'config_file': 'config.json', 'alias': 'SB BASE', 'sort_order': 64},

        #  "Final" for DWIE
        # {'experiment_name': '20210428a-l',
        #  'test_tag': 'test', 'config_file': 'config.json', 'alias': 'BASE', 'sort_order': 64},
        # {'experiment_name': '20210428a-cl-clip_norm10',
        #  'test_tag': 'test', 'config_file': 'config.json', 'alias': 'CLinker', 'sort_order': 65},
        # {'experiment_name': '20210428b-mtt_root_init_std_clip_norm10',
        #  'test_tag': 'test', 'config_file': 'config.json', 'alias': 'MTT 1', 'sort_order': 66},
        # {'experiment_name': '20210428b-mtt_root_init_zeros_clip_norm10',
        #  'test_tag': 'test', 'config_file': 'config.json', 'alias': 'MTT 2', 'sort_order': 67},
        # {'experiment_name': '20210429c-mtt_init_std_zeros_to_cl',
        #  'test_tag': 'test', 'config_file': 'config.json', 'alias': 'MTT 3', 'sort_order': 68},
        # "Final" for AIDA
        # {'experiment_name': '20210428a-aida_l_clip_norm10',
        #  'test_tag': 'testa', 'config_file': 'config.json', 'alias': 'BASE', 'sort_order': 68},
        # {'experiment_name': '20210428a-aida-cl_clip_norm10',
        #  'test_tag': 'testa', 'config_file': 'config.json', 'alias': 'CLinker', 'sort_order': 69},
        # {'experiment_name': '20210428b-mtt_aida_init_std_clip_norm10',
        #  'test_tag': 'testa', 'config_file': 'config.json', 'alias': 'MTT 1', 'sort_order': 70}
        # {'experiment_name': '20210428a-aida_l_clip_norm10',
        #  'test_tag': 'testb', 'config_file': 'config.json', 'alias': 'BASE', 'sort_order': 71},
        # {'experiment_name': '20210428a-aida-cl_clip_norm10',
        #  'test_tag': 'testb', 'config_file': 'config.json', 'alias': 'CLinker', 'sort_order': 72},
        # {'experiment_name': '20210428b-mtt_aida_init_std_clip_norm10',
        #  'test_tag': 'testb', 'config_file': 'config.json', 'alias': 'MTT 1', 'sort_order': 73}

        # {'experiment_name': '20210430b-mtt_init_ones_no_add_pruner_to_root',
        #  'test_tag': 'test', 'config_file': 'config.json', 'alias': 'MTT 1', 'sort_order': 74},
        # {'experiment_name': '20210430b-mtt_init_ones_zeros_to_links',
        #  'test_tag': 'test', 'config_file': 'config.json', 'alias': 'MTT 2', 'sort_order': 75},
        # {'experiment_name': '20210430b-mtt_init_std_no_add_pruner_to_root',
        #  'test_tag': 'test', 'config_file': 'config.json', 'alias': 'MTT 3', 'sort_order': 76},
        # {'experiment_name': '20210430b-mtt_init_std_zeros_to_links',
        #  'test_tag': 'test', 'config_file': 'config.json', 'alias': 'MTT 4', 'sort_order': 77},
        # {'experiment_name': '20210430b-mtt_init_zeros_no_add_pruner_to_root',
        #  'test_tag': 'test', 'config_file': 'config.json', 'alias': 'MTT 5', 'sort_order': 78},
        # {'experiment_name': '20210430b-mtt_init_zeros_zeros_to_links',
        #  'test_tag': 'test', 'config_file': 'config.json', 'alias': 'MTT 6', 'sort_order': 79}

        # "Final" for AIDA kolitsas
        # {'experiment_name': '20210429d-aida_l_kolitsas_emb',
        #  'test_tag': 'testa', 'config_file': 'config.json', 'alias': 'BASE', 'sort_order': 68},
        # {'experiment_name': '20210429d-aida-cl_kolitsas_emb',
        #  'test_tag': 'testa', 'config_file': 'config.json', 'alias': 'CLinker', 'sort_order': 69},
        # {'experiment_name': '20210430a-mtt_aida_init_std_kolitsas',
        #  'test_tag': 'testa', 'config_file': 'config.json', 'alias': 'MTT 1', 'sort_order': 70},
        # {'experiment_name': '20210430a-mtt_aida_zeros_to_cl_kolitsas',
        #  'test_tag': 'testa', 'config_file': 'config.json', 'alias': 'MTT 2', 'sort_order': 71}

        # {'experiment_name': '20210429d-aida_l_kolitsas_emb',
        #  'test_tag': 'testb', 'config_file': 'config.json', 'alias': 'BASE', 'sort_order': 68},
        # {'experiment_name': '20210429d-aida-cl_kolitsas_emb',
        #  'test_tag': 'testb', 'config_file': 'config.json', 'alias': 'CLinker', 'sort_order': 69},
        # {'experiment_name': '20210430a-mtt_aida_init_std_kolitsas',
        #  'test_tag': 'testb', 'config_file': 'config.json', 'alias': 'MTT 1', 'sort_order': 70},
        # {'experiment_name': '20210430a-mtt_aida_zeros_to_cl_kolitsas',
        #  'test_tag': 'testb', 'config_file': 'config.json', 'alias': 'MTT 2', 'sort_order': 71}

        # {'experiment_name': '20210428a-aida_l_clip_norm10',
        #  'test_tag': 'testb', 'config_file': 'config.json', 'alias': 'BASE', 'sort_order': 71},
        # {'experiment_name': '20210428a-aida-cl_clip_norm10',
        #  'test_tag': 'testb', 'config_file': 'config.json', 'alias': 'CLinker', 'sort_order': 72},
        # {'experiment_name': '20210428b-mtt_aida_init_std_clip_norm10',
        #  'test_tag': 'testb', 'config_file': 'config.json', 'alias': 'MTT 1', 'sort_order': 73}

        # {'experiment_name': '20210501a-mtt_zeros_to_cl_50eps',
        #  'test_tag': 'test', 'config_file': 'config.json', 'alias': 'MTT 1 50', 'sort_order': 74},
        # {'experiment_name': '20210502a-mtt_max_spans_to_link_50_eps',
        #  'test_tag': 'test', 'config_file': 'config.json', 'alias': 'MTT 2 50', 'sort_order': 75},
        # {'experiment_name': '20210502a-mtt_max_spans_to_link_min_zero_50_eps',
        #  'test_tag': 'test', 'config_file': 'config.json', 'alias': 'MTT 3 50', 'sort_order': 76},
        # {'experiment_name': '20210502a-mtt_max_spans_to_link_50_eps_all_to_0.0',
        #  'test_tag': 'test', 'config_file': 'config.json', 'alias': 'MTT 4 50/0', 'sort_order': 77},
        # {'experiment_name': '20210502a-mtt_max_spans_to_link_min_zero_50_eps_all_to_0.0',
        #  'test_tag': 'test', 'config_file': 'config.json', 'alias': 'MTT 5 50/0', 'sort_order': 78},
        # {'experiment_name': '20210501a-cl-50eps',
        #  'test_tag': 'test', 'config_file': 'config.json', 'alias': 'CL 50', 'sort_order': 79},
        # {'experiment_name': '20210501a-l_50eps',
        #  'test_tag': 'test', 'config_file': 'config.json', 'alias': 'BASE 50', 'sort_order': 79},

        # {'experiment_name': '20210504a-l_50eps_all_to_0.0',
        #  'test_tag': 'test', 'config_file': 'config.json', 'alias': 'BASE', 'sort_order': 74},
        # {'experiment_name': '20210504a-cl-50eps_all_to_0.0',
        #  'test_tag': 'test', 'config_file': 'config.json', 'alias': 'CLinker', 'sort_order': 75},
        # {'experiment_name': '20210502a-mtt_max_spans_to_link_min_zero_50_eps_all_to_0.0',
        #  'test_tag': 'test', 'config_file': 'config.json', 'alias': 'MTT 1', 'sort_order': 76},
        # {'experiment_name': '20210502a-mtt_max_spans_to_link_50_eps_all_to_0.0',
        #  'test_tag': 'test', 'config_file': 'config.json', 'alias': 'MTT 2', 'sort_order': 77},
        # {'experiment_name': '20210504a-mtt_zeros_to_cl_50eps_all_to_0.0',
        #  'test_tag': 'test', 'config_file': 'config.json', 'alias': 'MTT 3', 'sort_order': 78}
        # {'experiment_name': '20210501a-l_50eps',
        #  'test_tag': 'test', 'config_file': 'config.json', 'alias': 'BASE', 'sort_order': 74},
        # {'experiment_name': '20210501a-cl-50eps',
        #  'test_tag': 'test', 'config_file': 'config.json', 'alias': 'CLinker', 'sort_order': 74},
        # {'experiment_name': '20210506a-mtt_0_to_links_trainable_rnd_root_50_eps',
        #  'test_tag': 'test', 'config_file': 'config.json', 'alias': 'MTT', 'sort_order': 75},
        {'experiment_name': '20210505a-aida_l_kolitsas_emb',
         'test_tag': 'testb', 'config_file': 'config.json', 'alias': 'Base', 'sort_order': 75},
        # {'experiment_name': '20210505a-aida-cl_kolitsas_emb_50eps',
        #  'test_tag': 'testb', 'config_file': 'config.json', 'alias': 'CLinker', 'sort_order': 76},
        # {'experiment_name': '20210507a-mtt_aida_0_to_links_trainable_rnd_root_kolitsas',
        #  'test_tag': 'testb', 'config_file': 'config.json', 'alias': 'MTT', 'sort_order': 77}
        # {'experiment_name': '20210507a-mtt_aida_0_to_links_0_to_root_kolitsas',
        #  'test_tag': 'testb', 'config_file': 'config.json', 'alias': 'MTT', 'sort_order': 78}
        # {'experiment_name': '20210506a-mtt_0_to_links_0_root_50_eps',
        #  'test_tag': 'test', 'config_file': 'config.json', 'alias': 'MTT', 'sort_order': 78}
        # {'experiment_name': '20210508a-only_coref_50eps',
        #  'test_tag': 'test', 'config_file': 'config.json', 'alias': 'Coref', 'sort_order': 78},
        # {'experiment_name': '20210508a-only_linking_50eps',
        #  'test_tag': 'test', 'config_file': 'config.json', 'alias': 'Linking', 'sort_order': 79},
        # {'experiment_name': '20210506a-mtt_0_to_links_0_root_50_eps',
        #  'test_tag': 'test', 'config_file': 'config.json', 'alias': 'MTT', 'sort_order': 80}
        # {'experiment_name': '20210508a-aida_only_coref_kolitsas_emb',
        #  'test_tag': 'testb', 'config_file': 'config.json', 'alias': 'Coref', 'sort_order': 78},
        # {'experiment_name': '20210508a-aida_only_linker_kolitsas_emb',
        #  'test_tag': 'testb', 'config_file': 'config.json', 'alias': 'Linking', 'sort_order': 79},
        # {'experiment_name': '20210507a-mtt_aida_0_to_links_trainable_rnd_root_kolitsas',
        #  'test_tag': 'testb', 'config_file': 'config.json', 'alias': 'MTT', 'sort_order': 80}

        # {'experiment_name': '20210513a-cl_singletons-50eps',
        #  'test_tag': 'test', 'config_file': 'config.json', 'alias': 'CLinker', 'sort_order': 80},
        # {'experiment_name': '20210513a-mtt_0_to_links_trainable_rnd_root_50_eps',
        #  'test_tag': 'test', 'config_file': 'config.json', 'alias': 'MTT 1', 'sort_order': 81},

        # {'experiment_name': '20210508a-aida_only_linker_kolitsas_emb',
        #  'test_tag': 'testa', 'config_file': 'config.json', 'alias': 'CLinker', 'sort_order': 79},
        {'experiment_name': '20210513a-aida-cl_kolitsas_emb_singletons_50eps',
         'test_tag': 'testb', 'config_file': 'config.json', 'alias': 'CLinker', 'sort_order': 80},
        {'experiment_name': '20210513a-mtt_aida_0_to_l_tr_rnd_root_singletons_kolitsas',
         'test_tag': 'testb', 'config_file': 'config.json', 'alias': 'MTT 1', 'sort_order': 81}
    ]

    test_tag = 'test'  # by default test
    config_file = 'config.json'
    alias = ''
    sort_order = 0

    # max_props -> maximum number of propagations to take because some configurations were run with 4 props (ex: individual tasks)
    # some others with 3 (ex: joint)
    max_props = 3

    # path to the results obtained using johannes dygie model
    # path_results = 'models/selected/'
    path_results = 'models/selected_aida/'
    # path_results = 'models_studied/'
    # path_results = 'models/20200924-dwie20200908-coreflinker-1'
    # this has to be extracted from test set of each config.json file
    # gold_test_path = '/home/ibcn044/work_files/ugent/phd_work/repositories/projectcpn/third_party/cpn_spirit/data/' \
    #                  'cpn-1.0'
    out_all_results_path = 'results/results_all.csv'
    os.makedirs('results', exist_ok=True)
    # out_all_prop_graph_path = '/home/ibcn044/work_files/ugent/phd_work/repositories/projectcpn/research_notes/' \
    #                        'results/cpn-1.0-results/prop_graph_plots_data.csv'

    pickle_path = 'results/pickled/pickled_loaded_results.bin'

    # whether the ner/relation mention scores in joint setup are taked from expansions from the clusters.
    ner_expand_from_clusters = False
    rel_expand_from_clusters = False
    use_mention_wise_rels = True
    has_to_load = True
    # whether to merge the independent tasks
    # produce_merged = False

    if not os.path.exists(pickle_path):
        has_to_load = True

    if has_to_load:
        cnt = 0
        config_to_results = dict()
        debug_lengths_experiments = -1
        debug_length_docs = -1
        for (dirpath, dirnames, filenames) in os.walk(path_results, followlinks=True):
            if -1 < debug_lengths_experiments < cnt:
                break

            has_to_exclude = False
            for curr_exclude in to_exclude:
                if curr_exclude in dirpath:
                    has_to_exclude = True

            has_to_include = True
            if len(only_include_these) > 0:
                has_to_include = False
                for curr_include in only_include_these:
                    if curr_include['experiment_name'] in dirpath:
                        has_to_include = True
                        test_tag = curr_include['test_tag']
                        config_file = curr_include['config_file']
                        alias = curr_include['alias']
                        sort_order = curr_include['sort_order']

            if has_to_exclude or not has_to_include:
                # print('=====EXCLUDING====', dirpath)
                continue

            # test_path = os.path.join(dirpath, 'test.jsonl')
            test_path = os.path.join(dirpath, '{}.jsonl'.format(test_tag))
            if not os.path.exists(test_path):
                logger.warning('PATH DOES NOT EXIST!!!: %s' % test_path)
                continue

            for filename in filenames:
                # if filename == 'config.json':
                if filename == config_file:
                    cnt += 1
                    if cnt % 50 == 0:
                        logger.info('processed %s experiment executions' % cnt)
                    logger.info('=========PROCESSING======= %s' % test_path)
                    try:
                        loaded_predicted_test = load_jsonl(test_path, None)
                    except:
                        logger.error('!!!!!!!!error in loading the following test path: %s' % test_path)
                        continue
                    # loaded_predicted_test = load_jsonl(test_path, None)
                    # print('here have to interpret the results in ', filename, ': ', os.path.join(dirpath, filename))
                    cpn_evaluator = EvaluatorCPN()
                    config_json = json.load(open(os.path.join(dirpath, filename), 'r'))

                    # now loads the test
                    # gold_test_path = config_json['datasets']['test']['filename']
                    gold_test_path = config_json['datasets'][test_tag]['filename']
                    loaded_gold_test = dict()
                    for curr_file in os.listdir(gold_test_path):
                        # if 'DW_' in curr_file:
                        if 'main' not in curr_file:
                            curr_file_path = os.path.join(gold_test_path, curr_file)
                            loaded_json = load_json(curr_file_path, test_tag)
                            loaded_gold_test = dict(loaded_gold_test, **loaded_json)

                    if len(loaded_gold_test) != len(loaded_predicted_test):
                        logger.warning('WARN: differences between predicted (%s) and gold (%s) '
                                       'for follwing dir: %s' %
                                       (len(loaded_predicted_test), len(loaded_gold_test), dirpath))
                        continue
                    for idx, identifier in enumerate(loaded_gold_test.keys()):
                        if -1 < debug_length_docs < idx:
                            break
                        cpn_evaluator.add(loaded_predicted_test[identifier], loaded_gold_test[identifier])

                    execution_results = dict()

                    execution_results['relations'] = {'rels_mention': {'f1': cpn_evaluator.rels_mention.get_f1(),
                                                                       'pr': cpn_evaluator.rels_mention.get_pr(),
                                                                       're': cpn_evaluator.rels_mention.get_re()},
                                                      'rels_mention_expanded': {
                                                          'f1': cpn_evaluator.rels_mention_expanded.get_f1(),
                                                          'pr': cpn_evaluator.rels_mention_expanded.get_pr(),
                                                          're': cpn_evaluator.rels_mention_expanded.get_re()},
                                                      'rels_soft': {'f1': cpn_evaluator.rels_soft.get_f1(),
                                                                    'pr': cpn_evaluator.rels_soft.get_pr(),
                                                                    're': cpn_evaluator.rels_soft.get_re()},
                                                      'rels_hard': {'f1': cpn_evaluator.rels_hard.get_f1(),
                                                                    'pr': cpn_evaluator.rels_hard.get_pr(),
                                                                    're': cpn_evaluator.rels_hard.get_re()}
                                                      }
                    execution_results['tags'] = {'tags_mention': {'f1': cpn_evaluator.tags_mention.get_f1(),
                                                                  'pr': cpn_evaluator.tags_mention.get_pr(),
                                                                  're': cpn_evaluator.tags_mention.get_re()},
                                                 'tags_mention_expanded': {
                                                     'f1': cpn_evaluator.tags_mention_expanded.get_f1(),
                                                     'pr': cpn_evaluator.tags_mention_expanded.get_pr(),
                                                     're': cpn_evaluator.tags_mention_expanded.get_re()},
                                                 'tags_soft': {'f1': cpn_evaluator.tags_soft.get_f1(),
                                                               'pr': cpn_evaluator.tags_soft.get_pr(),
                                                               're': cpn_evaluator.tags_soft.get_re()},
                                                 'tags_hard': {'f1': cpn_evaluator.tags_hard.get_f1(),
                                                               'pr': cpn_evaluator.tags_hard.get_pr(),
                                                               're': cpn_evaluator.tags_hard.get_re()}}

                    execution_results['coref'] = {'ceafe_not_singleton': {'f1': cpn_evaluator.coref_ceafe.get_f1(),
                                                                          'pr': cpn_evaluator.coref_ceafe.get_pr(),
                                                                          're': cpn_evaluator.coref_ceafe.get_re()},
                                                  'ceafe_singleton': {
                                                      'f1': cpn_evaluator.coref_ceafe_singleton_ent.get_f1(),
                                                      'pr': cpn_evaluator.coref_ceafe_singleton_ent.get_pr(),
                                                      're': cpn_evaluator.coref_ceafe_singleton_ent.get_re()},
                                                  'ceafm_singleton': {
                                                      'f1': cpn_evaluator.coref_ceafm_singleton_men.get_f1(),
                                                      'pr': cpn_evaluator.coref_ceafm_singleton_men.get_pr(),
                                                      're': cpn_evaluator.coref_ceafm_singleton_men.get_re()},
                                                  'b_cubed_not_singleton': {
                                                      'f1': cpn_evaluator.coref_bcubed.get_f1(),
                                                      'pr': cpn_evaluator.coref_bcubed.get_pr(),
                                                      're': cpn_evaluator.coref_bcubed.get_re()},
                                                  'b_cubed_singleton_ent': {
                                                      'f1': cpn_evaluator.coref_bcubed_singleton_ent.get_f1(),
                                                      'pr': cpn_evaluator.coref_bcubed_singleton_ent.get_pr(),
                                                      're': cpn_evaluator.coref_bcubed_singleton_ent.get_re()},
                                                  'b_cubed_singleton_men_conll': {
                                                      'f1': cpn_evaluator.coref_bcubed_singleton_men.get_f1(),
                                                      'pr': cpn_evaluator.coref_bcubed_singleton_men.get_pr(),
                                                      're': cpn_evaluator.coref_bcubed_singleton_men.get_re()
                                                  },
                                                  'muc': {
                                                      'f1': cpn_evaluator.coref_muc.get_f1(),
                                                      'pr': cpn_evaluator.coref_muc.get_pr(),
                                                      're': cpn_evaluator.coref_muc.get_re()
                                                  }
                                                  }

                    execution_results['links'] = {
                        'links-all': {
                            'f1': cpn_evaluator.links_mention_all.get_f1(),
                            'pr': cpn_evaluator.links_mention_all.get_pr(),
                            're': cpn_evaluator.links_mention_all.get_re(),
                        },
                        'links-links': {
                            'f1': cpn_evaluator.links_mention_links.get_f1(),
                            'pr': cpn_evaluator.links_mention_links.get_pr(),
                            're': cpn_evaluator.links_mention_links.get_re(),
                        },
                        'links-nill': {
                            'f1': cpn_evaluator.links_mention_nill.get_f1(),
                            'pr': cpn_evaluator.links_mention_nill.get_pr(),
                            're': cpn_evaluator.links_mention_nill.get_re(),
                        },
                        'links-accuracy': {
                            'acc_candidates': cpn_evaluator.links_accuracy.get_acc(),
                            'acc_no_candidates': cpn_evaluator.links_accuracy_no_candidates.get_acc()
                        },
                        'links-all-from-ents': {
                            'f1': cpn_evaluator.links_mention_ent_all.get_f1(),
                            'pr': cpn_evaluator.links_mention_ent_all.get_pr(),
                            're': cpn_evaluator.links_mention_ent_all.get_re(),
                        },
                        'links-links-from-ents': {
                            'f1': cpn_evaluator.links_mention_ent_links.get_f1(),
                            'pr': cpn_evaluator.links_mention_ent_links.get_pr(),
                            're': cpn_evaluator.links_mention_ent_links.get_re(),
                        },
                        'links-nill-from-ents': {
                            'f1': cpn_evaluator.links_mention_ent_nill.get_f1(),
                            'pr': cpn_evaluator.links_mention_ent_nill.get_pr(),
                            're': cpn_evaluator.links_mention_ent_nill.get_re(),
                        },
                        'links-accuracy-from-ents': {
                            'acc_no_candidates': cpn_evaluator.links_accuracy_ent_no_cand.get_acc()
                        },
                        'links-links-soft': {
                            'f1': cpn_evaluator.links_soft_ent_links.get_f1(),
                            'pr': cpn_evaluator.links_soft_ent_links.get_pr(),
                            're': cpn_evaluator.links_soft_ent_links.get_re()
                        },
                        'links-links-mentionsoft': {
                            'f1': cpn_evaluator.links_mentionsoft_ent_links.get_f1(),
                            'pr': cpn_evaluator.links_mentionsoft_ent_links.get_pr(),
                            're': cpn_evaluator.links_mentionsoft_ent_links.get_re()
                        },
                        'links-links-hard': {
                            'f1': cpn_evaluator.links_hard_ent_links.get_f1(),
                            'pr': cpn_evaluator.links_hard_ent_links.get_pr(),
                            're': cpn_evaluator.links_hard_ent_links.get_re(),
                        },
                        'links-nill-hard': {
                            'f1': cpn_evaluator.links_nill_hard_ent_links.get_f1(),
                            'pr': cpn_evaluator.links_nill_hard_ent_links.get_pr(),
                            're': cpn_evaluator.links_nill_hard_ent_links.get_re(),
                        }
                    }

                    # kzaporoj (04/12/2020) - for the recursive values, we ignore the "evaluate" because in some runs
                    # of the same experiment I evaluate on both test and train (usually the run 1), and others
                    # (usually runs 2-5) I evaluate only on test in order to gain speed.
                    config_json_key = config_json.copy()

                    del config_json_key['trainer']['evaluate']
                    if 'evaluation_frequency' in config_json_key['trainer']:
                        del config_json_key['trainer']['evaluation_frequency']

                    recursive_values = get_dict_recursive_key_value(config_json_key, '')
                    freeze_rec_values = frozenset(recursive_values)
                    # sometimes, several experiments can have been run for a particular configuration, we divide them
                    # by experiment_id
                    experiment_id = os.path.basename(os.path.normpath(dirpath))
                    experiment_id = experiment_id[:experiment_id.rindex('-')]
                    if freeze_rec_values not in config_to_results:
                        config_to_results[freeze_rec_values] = {'experiments': dict(), 'config': config_json}
                    if experiment_id not in config_to_results[freeze_rec_values]['experiments']:
                        config_to_results[freeze_rec_values]['experiments'][experiment_id] = {'results': list()}

                    config_to_results[freeze_rec_values]['alias'] = alias
                    config_to_results[freeze_rec_values]['sort_order'] = sort_order
                    config_to_results[freeze_rec_values]['experiments'][experiment_id]['results'].append({
                        'execution_path': dirpath,
                        'execution_results': execution_results
                    })
    else:
        with open(pickle_path, 'rb') as handle:
            config_to_results = pickle.load(handle)

    out_csv_file = open(out_all_results_path, 'w')

    csv_writer = csv.writer(out_csv_file)

    csv_title_line = get_csv_line_v2({}, {}, '', only_f1=False, only_title=True)
    csv_writer.writerow(csv_title_line)

    results_joint = []
    results_base_joint = []
    results_joint_with_coref_prop = []
    results_joint_with_att_prop = []
    results_joint_with_rel_prop = []

    results_joint_bert = []
    results_joint_bert_with_rel_prop = []
    results_joint_bert_with_coref_prop = []
    results_joint_bert_with_att_prop = []
    results_joint_bert_base = []

    results_single_coref = []
    results_single_coref_merged_ner = []
    results_single_ner = []
    results_single_rel = []

    results_single_coref_with_att_prop = []
    results_single_coref_with_coref_prop = []
    results_single_ner_with_att_prop = []
    results_single_rel_with_rel_prop = []
    results_single_rel_with_att_prop = []

    results_corefprop_joint = []
    results_relprop_joint = []
    results_attprop_joint = []
    results_corefrelprop_joint = []
    results_base_bert_joint = []
    results_corefprop_bert_joint = []
    results_attprop_bert_joint = []
    results_relprop_bert_joint = []
    results_corefrelprop_bert_joint = []

    results_single_coref_by_coref_prop = {i: list() for i in range(max_coref_props)}
    results_single_coref_by_att_prop = {i: list() for i in range(max_att_props)}
    results_single_ner_by_att_prop = {i: list() for i in range(max_att_props)}
    results_single_rel_by_att_prop = {i: list() for i in range(max_att_props)}
    results_single_rel_by_rel_prop = {i: list() for i in range(max_rel_props)}

    results_coref_prop = {i: list() for i in range(max_coref_props)}
    results_att_prop = {i: list() for i in range(max_att_props)}
    results_rel_prop = {i: list() for i in range(max_rel_props)}

    results_bert_coref_prop = {i: list() for i in range(max_coref_props)}
    results_bert_att_prop = {i: list() for i in range(max_att_props)}
    results_bert_rel_prop = {i: list() for i in range(max_rel_props)}

    logger.info('=================FROM NOW ON SHOULD BE FAST==================')

    best_res_rel_merged_single = dict()
    best_res_rel_merged_rel_prop = dict()
    best_res_rel_merged_coref_prop = dict()
    best_res_rel_merged_att_prop = dict()

    best_res_ner_merged_single = dict()
    best_res_ner_merged_rel_prop = dict()
    best_res_ner_merged_coref_prop = dict()
    best_res_ner_merged_att_prop = dict()

    to_print_results_latex = list()
    # loop to get stdev of the results
    for key_config, res_values in config_to_results.items():

        if 1 < len(res_values['experiments'].keys()):
            logger.warning('------WARNING TWO EQUAL EXPERIMENTS: %s' % res_values['experiments'].keys())

        for curr_experiment in res_values['experiments'].keys():
            # also here exclude to have effect even if it was cached (ex: I realized to put something in exclude and don't
            # want (have time) to re-cache everything again).
            logger.info('CURR EXPERIMENT IS OF: %s' % curr_experiment)

            has_to_exclude = False
            for curr_exclude in to_exclude:
                if curr_exclude in curr_experiment:
                    has_to_exclude = True

            if has_to_exclude:
                logger.info('=========excluding from cached========= %s' % curr_experiment)
                continue

            rels_mention_f1 = []
            rels_mention_pr = []
            rels_mention_re = []

            rels_mention_expanded_f1 = []
            rels_mention_expanded_pr = []
            rels_mention_expanded_re = []

            rels_soft_f1 = []
            rels_soft_pr = []
            rels_soft_re = []

            rels_hard_f1 = []
            rels_hard_pr = []
            rels_hard_re = []

            tags_mention_f1 = []
            tags_mention_pr = []
            tags_mention_re = []

            tags_mention_expanded_f1 = []
            tags_mention_expanded_pr = []
            tags_mention_expanded_re = []

            tags_soft_f1 = []
            tags_soft_pr = []
            tags_soft_re = []

            tags_hard_f1 = []
            tags_hard_pr = []
            tags_hard_re = []

            coref_muc_f1 = []
            coref_muc_pr = []
            coref_muc_re = []

            coref_avg_f1 = []
            coref_avg_pr = []
            coref_avg_re = []

            coref_ceafe_not_singleton_f1 = []
            coref_ceafe_not_singleton_pr = []
            coref_ceafe_not_singleton_re = []

            coref_ceafe_singleton_f1 = []
            coref_ceafe_singleton_pr = []
            coref_ceafe_singleton_re = []

            coref_ceafm_singleton_f1 = []
            coref_ceafm_singleton_pr = []
            coref_ceafm_singleton_re = []

            coref_bcubed_not_singleton_f1 = []
            coref_bcubed_not_singleton_pr = []
            coref_bcubed_not_singleton_re = []

            coref_bcubed_singleton_ent_f1 = []
            coref_bcubed_singleton_ent_pr = []
            coref_bcubed_singleton_ent_re = []

            coref_bcubed_singleton_men_conll_f1 = []
            coref_bcubed_singleton_men_conll_pr = []
            coref_bcubed_singleton_men_conll_re = []

            links_all_f1 = []
            links_all_pr = []
            links_all_re = []

            links_links_f1 = []
            links_links_pr = []
            links_links_re = []

            links_nill_f1 = []
            links_nill_pr = []
            links_nill_re = []

            links_acc_cand = []
            links_acc_no_cand = []

            links_all_from_ent_f1 = []
            links_all_from_ent_pr = []
            links_all_from_ent_re = []

            links_links_from_ent_f1 = []
            links_links_from_ent_pr = []
            links_links_from_ent_re = []

            links_nill_from_ent_f1 = []
            links_nill_from_ent_pr = []
            links_nill_from_ent_re = []

            links_acc_no_cand_from_ent = []

            links_links_soft_f1 = []
            links_links_soft_pr = []
            links_links_soft_re = []
            links_links_mentionsoft_f1 = []
            links_links_mentionsoft_pr = []
            links_links_mentionsoft_re = []
            links_links_hard_f1 = []
            links_links_hard_pr = []
            links_links_hard_re = []
            links_nill_hard_f1 = []
            links_nill_hard_pr = []
            links_nill_hard_re = []

            curr_exp_seed_results = res_values['experiments'][curr_experiment]['results']

            for curr_exp_result in curr_exp_seed_results:
                rels_mention_expanded_f1.append(
                    curr_exp_result['execution_results']['relations']['rels_mention_expanded']['f1'])
                rels_mention_expanded_pr.append(
                    curr_exp_result['execution_results']['relations']['rels_mention_expanded']['pr'])
                rels_mention_expanded_re.append(
                    curr_exp_result['execution_results']['relations']['rels_mention_expanded']['re'])
                rels_mention_f1.append(curr_exp_result['execution_results']['relations']['rels_mention']['f1'])
                rels_mention_pr.append(curr_exp_result['execution_results']['relations']['rels_mention']['pr'])
                rels_mention_re.append(curr_exp_result['execution_results']['relations']['rels_mention']['re'])

                rels_soft_f1.append(curr_exp_result['execution_results']['relations']['rels_soft']['f1'])
                rels_soft_pr.append(curr_exp_result['execution_results']['relations']['rels_soft']['pr'])
                rels_soft_re.append(curr_exp_result['execution_results']['relations']['rels_soft']['re'])

                rels_hard_f1.append(curr_exp_result['execution_results']['relations']['rels_hard']['f1'])
                rels_hard_pr.append(curr_exp_result['execution_results']['relations']['rels_hard']['pr'])
                rels_hard_re.append(curr_exp_result['execution_results']['relations']['rels_hard']['re'])

                tags_mention_expanded_f1.append(
                    curr_exp_result['execution_results']['tags']['tags_mention_expanded']['f1'])
                tags_mention_expanded_pr.append(
                    curr_exp_result['execution_results']['tags']['tags_mention_expanded']['pr'])
                tags_mention_expanded_re.append(
                    curr_exp_result['execution_results']['tags']['tags_mention_expanded']['re'])

                tags_mention_f1.append(curr_exp_result['execution_results']['tags']['tags_mention']['f1'])
                tags_mention_pr.append(curr_exp_result['execution_results']['tags']['tags_mention']['pr'])
                tags_mention_re.append(curr_exp_result['execution_results']['tags']['tags_mention']['re'])

                tags_soft_f1.append(curr_exp_result['execution_results']['tags']['tags_soft']['f1'])
                tags_soft_pr.append(curr_exp_result['execution_results']['tags']['tags_soft']['pr'])
                tags_soft_re.append(curr_exp_result['execution_results']['tags']['tags_soft']['re'])

                tags_hard_f1.append(curr_exp_result['execution_results']['tags']['tags_hard']['f1'])
                tags_hard_pr.append(curr_exp_result['execution_results']['tags']['tags_hard']['pr'])
                tags_hard_re.append(curr_exp_result['execution_results']['tags']['tags_hard']['re'])

                coref_muc_f1.append(curr_exp_result['execution_results']['coref']['muc']['f1'])
                coref_muc_pr.append(curr_exp_result['execution_results']['coref']['muc']['pr'])
                coref_muc_re.append(curr_exp_result['execution_results']['coref']['muc']['re'])

                coref_ceafe_not_singleton_f1.append(curr_exp_result['execution_results']['coref']
                                                    ['ceafe_not_singleton']['f1'])
                coref_ceafe_not_singleton_pr.append(curr_exp_result['execution_results']['coref']
                                                    ['ceafe_not_singleton']['pr'])
                coref_ceafe_not_singleton_re.append(curr_exp_result['execution_results']['coref']
                                                    ['ceafe_not_singleton']['re'])

                coref_ceafe_singleton_f1.append(curr_exp_result['execution_results']['coref']
                                                ['ceafe_singleton']['f1'])
                coref_ceafe_singleton_pr.append(curr_exp_result['execution_results']['coref']
                                                ['ceafe_singleton']['pr'])
                coref_ceafe_singleton_re.append(curr_exp_result['execution_results']['coref']
                                                ['ceafe_singleton']['re'])

                coref_ceafm_singleton_f1.append(curr_exp_result['execution_results']['coref']
                                                ['ceafm_singleton']['f1'])
                coref_ceafm_singleton_pr.append(curr_exp_result['execution_results']['coref']
                                                ['ceafm_singleton']['pr'])
                coref_ceafm_singleton_re.append(curr_exp_result['execution_results']['coref']
                                                ['ceafm_singleton']['re'])

                coref_bcubed_not_singleton_f1.append(curr_exp_result['execution_results']['coref']
                                                     ['b_cubed_not_singleton']['f1'])
                coref_bcubed_not_singleton_pr.append(curr_exp_result['execution_results']['coref']
                                                     ['b_cubed_not_singleton']['pr'])
                coref_bcubed_not_singleton_re.append(curr_exp_result['execution_results']['coref']
                                                     ['b_cubed_not_singleton']['re'])

                coref_bcubed_singleton_ent_f1.append(curr_exp_result['execution_results']['coref']
                                                     ['b_cubed_singleton_ent']['f1'])
                coref_bcubed_singleton_ent_pr.append(curr_exp_result['execution_results']['coref']
                                                     ['b_cubed_singleton_ent']['pr'])
                coref_bcubed_singleton_ent_re.append(curr_exp_result['execution_results']['coref']
                                                     ['b_cubed_singleton_ent']['re'])

                coref_bcubed_singleton_men_conll_f1.append(curr_exp_result['execution_results']['coref']
                                                           ['b_cubed_singleton_men_conll']['f1'])
                coref_bcubed_singleton_men_conll_pr.append(curr_exp_result['execution_results']['coref']
                                                           ['b_cubed_singleton_men_conll']['pr'])
                coref_bcubed_singleton_men_conll_re.append(curr_exp_result['execution_results']['coref']
                                                           ['b_cubed_singleton_men_conll']['re'])

                avg_f1 = (curr_exp_result['execution_results']['coref']['b_cubed_singleton_men_conll']['f1'] +
                          curr_exp_result['execution_results']['coref']['muc']['f1'] +
                          curr_exp_result['execution_results']['coref']['ceafe_singleton']['f1']) / 3
                avg_re = (curr_exp_result['execution_results']['coref']['b_cubed_singleton_men_conll']['re'] +
                          curr_exp_result['execution_results']['coref']['muc']['re'] +
                          curr_exp_result['execution_results']['coref']['ceafe_singleton']['re']) / 3
                avg_pr = (curr_exp_result['execution_results']['coref']['b_cubed_singleton_men_conll']['pr'] +
                          curr_exp_result['execution_results']['coref']['muc']['pr'] +
                          curr_exp_result['execution_results']['coref']['ceafe_singleton']['pr']) / 3

                coref_avg_f1.append(avg_f1)
                coref_avg_re.append(avg_re)
                coref_avg_pr.append(avg_pr)

                links_all_f1.append(curr_exp_result['execution_results']['links']['links-all']['f1'])
                links_all_re.append(curr_exp_result['execution_results']['links']['links-all']['re'])
                links_all_pr.append(curr_exp_result['execution_results']['links']['links-all']['pr'])

                links_links_f1.append(curr_exp_result['execution_results']['links']['links-links']['f1'])
                links_links_re.append(curr_exp_result['execution_results']['links']['links-links']['re'])
                links_links_pr.append(curr_exp_result['execution_results']['links']['links-links']['pr'])

                links_nill_f1.append(curr_exp_result['execution_results']['links']['links-nill']['f1'])
                links_nill_re.append(curr_exp_result['execution_results']['links']['links-nill']['re'])
                links_nill_pr.append(curr_exp_result['execution_results']['links']['links-nill']['pr'])

                links_acc_cand.append(curr_exp_result['execution_results']['links']['links-accuracy']['acc_candidates'])
                links_acc_no_cand.append(
                    curr_exp_result['execution_results']['links']['links-accuracy']['acc_no_candidates'])

                links_all_from_ent_f1.append(curr_exp_result['execution_results']['links']['links-all-from-ents']['f1'])
                links_all_from_ent_re.append(curr_exp_result['execution_results']['links']['links-all-from-ents']['re'])
                links_all_from_ent_pr.append(curr_exp_result['execution_results']['links']['links-all-from-ents']['pr'])

                links_links_from_ent_f1.append(
                    curr_exp_result['execution_results']['links']['links-links-from-ents']['f1'])
                links_links_from_ent_re.append(
                    curr_exp_result['execution_results']['links']['links-links-from-ents']['re'])
                links_links_from_ent_pr.append(
                    curr_exp_result['execution_results']['links']['links-links-from-ents']['pr'])

                links_nill_from_ent_f1.append(
                    curr_exp_result['execution_results']['links']['links-nill-from-ents']['f1'])
                links_nill_from_ent_re.append(
                    curr_exp_result['execution_results']['links']['links-nill-from-ents']['re'])
                links_nill_from_ent_pr.append(
                    curr_exp_result['execution_results']['links']['links-nill-from-ents']['pr'])

                links_acc_no_cand_from_ent.append(
                    curr_exp_result['execution_results']['links']['links-accuracy-from-ents']['acc_no_candidates'])

                links_links_soft_f1.append(curr_exp_result['execution_results']['links']['links-links-soft']['f1'])
                links_links_soft_pr.append(curr_exp_result['execution_results']['links']['links-links-soft']['pr'])
                links_links_soft_re.append(curr_exp_result['execution_results']['links']['links-links-soft']['re'])

                links_links_mentionsoft_f1.append(curr_exp_result['execution_results']['links']
                                                  ['links-links-mentionsoft']['f1'])
                links_links_mentionsoft_pr.append(curr_exp_result['execution_results']['links']
                                                  ['links-links-mentionsoft']['pr'])
                links_links_mentionsoft_re.append(curr_exp_result['execution_results']['links']
                                                  ['links-links-mentionsoft']['re'])

                links_links_hard_f1.append(curr_exp_result['execution_results']['links']['links-links-hard']['f1'])
                links_links_hard_pr.append(curr_exp_result['execution_results']['links']['links-links-hard']['pr'])
                links_links_hard_re.append(curr_exp_result['execution_results']['links']['links-links-hard']['re'])

                links_nill_hard_f1.append(curr_exp_result['execution_results']['links']['links-nill-hard']['f1'])
                links_nill_hard_pr.append(curr_exp_result['execution_results']['links']['links-nill-hard']['pr'])
                links_nill_hard_re.append(curr_exp_result['execution_results']['links']['links-nill-hard']['re'])

            avg_results = dict()

            if len(rels_mention_f1) < 1:  # TODO: here see how to do for experiment with mention-based relations!
                logger.warning('!!!!!!!!!!!!!!!!!!!IGNORING EXPERIMENT %s with len(rels_mention_f1) of %s '
                               '!!!!!!!!!!!!!!!!' % (curr_experiment, len(rels_mention_f1)))

                continue

            avg_results['nr_runs'] = len(rels_mention_f1)
            avg_results['relations'] = {'rels_mention': {'f1': (mean(rels_mention_f1), my_stdev(rels_mention_f1)),
                                                         'pr': (mean(rels_mention_pr), my_stdev(rels_mention_pr)),
                                                         're': (mean(rels_mention_re), my_stdev(rels_mention_re))},
                                        'rels_mention_expanded': {
                                            'f1': (mean(rels_mention_expanded_f1), my_stdev(rels_mention_expanded_f1)),
                                            'pr': (mean(rels_mention_expanded_pr), my_stdev(rels_mention_expanded_pr)),
                                            're': (mean(rels_mention_expanded_re), my_stdev(rels_mention_expanded_re))},
                                        'rels_soft': {'f1': (mean(rels_soft_f1), my_stdev(rels_soft_f1)),
                                                      'pr': (mean(rels_soft_pr), my_stdev(rels_soft_pr)),
                                                      're': (mean(rels_soft_re), my_stdev(rels_soft_re))},
                                        'rels_hard': {'f1': (mean(rels_hard_f1), my_stdev(rels_hard_f1)),
                                                      'pr': (mean(rels_hard_pr), my_stdev(rels_hard_pr)),
                                                      're': (mean(rels_hard_re), my_stdev(rels_hard_re))},
                                        }

            avg_results['tags'] = {'tags_mention': {'f1': (mean(tags_mention_f1), my_stdev(tags_mention_f1)),
                                                    'pr': (mean(tags_mention_pr), my_stdev(tags_mention_pr)),
                                                    're': (mean(tags_mention_re), my_stdev(tags_mention_re))},
                                   'tags_mention_expanded': {'f1': (mean(tags_mention_expanded_f1),
                                                                    my_stdev(tags_mention_expanded_f1)),
                                                             'pr': (mean(tags_mention_expanded_pr),
                                                                    my_stdev(tags_mention_expanded_pr)),
                                                             're': (mean(tags_mention_expanded_re),
                                                                    my_stdev(tags_mention_expanded_re))},
                                   'tags_soft': {'f1': (mean(tags_soft_f1), my_stdev(tags_soft_f1)),
                                                 'pr': (mean(tags_soft_pr), my_stdev(tags_soft_pr)),
                                                 're': (mean(tags_soft_re), my_stdev(tags_soft_re))},
                                   'tags_hard': {'f1': (mean(tags_hard_f1), my_stdev(tags_hard_f1)),
                                                 'pr': (mean(tags_hard_pr), my_stdev(tags_hard_pr)),
                                                 're': (mean(tags_hard_re), my_stdev(tags_hard_re))},
                                   }

            avg_results['coref'] = {
                'ceafe_not_singleton': {
                    'f1': (mean(coref_ceafe_not_singleton_f1), my_stdev(coref_ceafe_not_singleton_f1)),
                    'pr': (mean(coref_ceafe_not_singleton_pr), my_stdev(coref_ceafe_not_singleton_pr)),
                    're': (
                        mean(coref_ceafe_not_singleton_re), my_stdev(coref_ceafe_not_singleton_re))},
                'ceafe_singleton': {
                    'f1': (mean(coref_ceafe_singleton_f1), my_stdev(coref_ceafe_singleton_f1)),
                    'pr': (mean(coref_ceafe_singleton_pr), my_stdev(coref_ceafe_singleton_pr)),
                    're': (mean(coref_ceafe_singleton_re), my_stdev(coref_ceafe_singleton_re))},
                'ceafm_singleton': {
                    'f1': (mean(coref_ceafm_singleton_f1), my_stdev(coref_ceafm_singleton_f1)),
                    'pr': (mean(coref_ceafm_singleton_pr), my_stdev(coref_ceafm_singleton_pr)),
                    're': (mean(coref_ceafm_singleton_re), my_stdev(coref_ceafm_singleton_re))},
                'b_cubed_not_singleton': {
                    'f1': (mean(coref_bcubed_not_singleton_f1), my_stdev(coref_bcubed_not_singleton_f1)),
                    'pr': (mean(coref_bcubed_not_singleton_pr), my_stdev(coref_bcubed_not_singleton_pr)),
                    're': (mean(coref_bcubed_not_singleton_re), my_stdev(coref_bcubed_not_singleton_re))},
                'b_cubed_singleton_ent': {
                    'f1': (mean(coref_bcubed_singleton_ent_f1), my_stdev(coref_bcubed_singleton_ent_f1)),
                    'pr': (mean(coref_bcubed_singleton_ent_pr), my_stdev(coref_bcubed_singleton_ent_pr)),
                    're': (mean(coref_bcubed_singleton_ent_re), my_stdev(coref_bcubed_singleton_ent_re))},
                'b_cubed_singleton_men_conll': {
                    'f1': (mean(coref_bcubed_singleton_men_conll_f1), my_stdev(coref_bcubed_singleton_men_conll_f1)),
                    'pr': (mean(coref_bcubed_singleton_men_conll_pr), my_stdev(coref_bcubed_singleton_men_conll_pr)),
                    're': (mean(coref_bcubed_singleton_men_conll_re), my_stdev(coref_bcubed_singleton_men_conll_re))
                },
                'muc': {
                    'f1': (mean(coref_muc_f1), my_stdev(coref_muc_f1)),
                    'pr': (mean(coref_muc_pr), my_stdev(coref_muc_pr)),
                    're': (mean(coref_muc_re), my_stdev(coref_muc_re))
                },
                'coref_avg': {
                    'f1': (mean(coref_avg_f1), my_stdev(coref_avg_f1)),
                    'pr': (mean(coref_avg_pr), my_stdev(coref_avg_pr)),
                    're': (mean(coref_avg_re), my_stdev(coref_avg_re))
                }
            }

            avg_results['links'] = {'links-all': {'f1': (mean(links_all_f1), my_stdev(links_all_f1)),
                                                  'pr': (mean(links_all_pr), my_stdev(links_all_pr)),
                                                  're': (mean(links_all_re), my_stdev(links_all_re))},
                                    'links-links': {'f1': (mean(links_links_f1), my_stdev(links_links_f1)),
                                                    'pr': (mean(links_links_pr), my_stdev(links_links_pr)),
                                                    're': (mean(links_links_re), my_stdev(links_links_re))},
                                    'links-nill': {'f1': (mean(links_nill_f1), my_stdev(links_nill_f1)),
                                                   'pr': (mean(links_nill_pr), my_stdev(links_nill_pr)),
                                                   're': (mean(links_nill_re), my_stdev(links_nill_re))},
                                    'links-accuracy': {
                                        'acc_candidates': (mean(links_acc_cand), my_stdev(links_acc_cand)),
                                        'acc_no_candidates': (mean(links_acc_no_cand), my_stdev(links_acc_no_cand))},
                                    'links-all-from-ent': {
                                        'f1': (mean(links_all_from_ent_f1), my_stdev(links_all_from_ent_f1)),
                                        'pr': (mean(links_all_from_ent_pr), my_stdev(links_all_from_ent_pr)),
                                        're': (mean(links_all_from_ent_re), my_stdev(links_all_from_ent_re))},
                                    'links-links-from-ent': {
                                        'f1': (mean(links_links_from_ent_f1), my_stdev(links_links_from_ent_f1)),
                                        'pr': (mean(links_links_from_ent_pr), my_stdev(links_links_from_ent_pr)),
                                        're': (mean(links_links_from_ent_re), my_stdev(links_links_from_ent_re))},
                                    'links-nill-from-ent': {
                                        'f1': (mean(links_nill_from_ent_f1), my_stdev(links_nill_from_ent_f1)),
                                        'pr': (mean(links_nill_from_ent_pr), my_stdev(links_nill_from_ent_pr)),
                                        're': (mean(links_nill_from_ent_re), my_stdev(links_nill_from_ent_re))},
                                    'links-accuracy-from-ent': {
                                        'acc_no_candidates': (
                                            mean(links_acc_no_cand_from_ent), my_stdev(links_acc_no_cand_from_ent))},
                                    'links-links-soft': {
                                        'f1': (mean(links_links_soft_f1), my_stdev(links_links_soft_f1)),
                                        'pr': (mean(links_links_soft_pr), my_stdev(links_links_soft_pr)),
                                        're': (mean(links_links_soft_re), my_stdev(links_links_soft_re))
                                    },
                                    'links-links-mentionsoft': {
                                        'f1': (mean(links_links_mentionsoft_f1), my_stdev(links_links_mentionsoft_f1)),
                                        'pr': (mean(links_links_mentionsoft_pr), my_stdev(links_links_mentionsoft_pr)),
                                        're': (mean(links_links_mentionsoft_re), my_stdev(links_links_mentionsoft_re))
                                    },
                                    'links-links-hard': {
                                        'f1': (mean(links_links_hard_f1), my_stdev(links_links_hard_f1)),
                                        'pr': (mean(links_links_hard_pr), my_stdev(links_links_hard_pr)),
                                        're': (mean(links_links_hard_re), my_stdev(links_links_hard_re))
                                    },
                                    'links-nill-hard': {
                                        'f1': (mean(links_nill_hard_f1), my_stdev(links_nill_hard_f1)),
                                        'pr': (mean(links_nill_hard_pr), my_stdev(links_nill_hard_pr)),
                                        're': (mean(links_nill_hard_re), my_stdev(links_nill_hard_re))
                                    }
                                    }

            res_values['experiments'][curr_experiment]['avg_result'] = avg_results

            curr_experiment_config = res_values['config']
            alias = res_values['alias']
            sort_order = res_values['sort_order']
            curr_csv_line = get_csv_line_v2(experiment_config=curr_experiment_config,
                                            experiment_results=avg_results,
                                            experiment_id=curr_experiment)
            csv_writer.writerow(curr_csv_line)

            to_print_results_latex.append(
                {'config': curr_experiment_config, 'results': avg_results, 'experiment_id': curr_experiment,
                 'alias': alias, 'sort_order': sort_order})

    print_latex_table_coreflinker(to_print_results_latex)

    logger.info('==========================AND NOW PRINTING FOR NILL==========================')
    print_latex_table_coreflinker_nill(to_print_results_latex)

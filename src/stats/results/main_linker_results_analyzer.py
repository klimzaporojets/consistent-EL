import copy
import json
import os
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import nltk
import pandas as pd
from matplotlib.ticker import MaxNLocator


def complete_predictions_cluster_level(models_path='models/', experiment_name='20201007-coreflinker-scores',
                                       experiment_type='coreflinker', input_structure=None, ground_truth_type=None,
                                       gold_structure=None):
    nr_parallels = 0
    dirs_processed = set()
    for (dirpath, dirnames, filenames) in os.walk(models_path, followlinks=True):
        if experiment_name in dirpath and dirpath not in dirs_processed:
            dirs_processed.add(dirpath)
            pred_file_name = '{}.jsonl'.format(ground_truth_type)
            if pred_file_name in filenames:
                file_process = os.path.join(dirpath, pred_file_name)
                print('CLUSTER LEVEL, PROCESSING FOLLOWING DIR: ', file_process)
                nr_parallels += 1
                for curr_json_line in open(file_process):
                    loaded_pred_json = json.loads(curr_json_line)
                    doc_orig = loaded_pred_json['id']
                    doc_id = loaded_pred_json['id']
                    assert doc_id in gold_structure
                    doc_id = '{}-{}'.format(doc_id, nr_parallels)

                    # for each of the runs, add a different document
                    if doc_id not in input_structure:
                        input_structure[doc_id] = copy.deepcopy(gold_structure[doc_orig])

                    # initializes all to false and only puts to true if coincide both the cluster and the link
                    for spans, cluster_data in input_structure[doc_id].items():
                        cluster_data['{}_correct'.format(experiment_type)] = False

                    pred_cluster_to_mentions = dict()

                    for curr_mention in loaded_pred_json['mentions']:
                        concept_id = curr_mention['concept']
                        if concept_id not in pred_cluster_to_mentions:
                            pred_cluster_to_mentions[concept_id] = list()

                        pred_cluster_to_mentions[concept_id].append(curr_mention)

                    for curr_cluster in loaded_pred_json['concepts']:
                        if curr_cluster['concept'] not in pred_cluster_to_mentions:
                            continue
                        if 'link_pred' in curr_cluster and curr_cluster['link_pred'] is not None and \
                                curr_cluster['link_pred'] != 'NILL':
                            link_pred = curr_cluster['link_pred']
                            cluster_mentions = pred_cluster_to_mentions[curr_cluster['concept']]

                            cluster_spans = [(cm['begin'], cm['end']) for cm in cluster_mentions]
                            cluster_spans = tuple(sorted(cluster_spans, key=lambda x: (x[0], x[1])))

                            if cluster_spans in input_structure[doc_id] and \
                                    input_structure[doc_id][cluster_spans]['link_gold'] == link_pred:
                                input_structure[doc_id][cluster_spans]['{}_correct'.format(experiment_type)] = True
    return input_structure


def complete_predictions(models_path='models/', experiment_name='20201007-coreflinker-scores',
                         experiment_type='coreflinker', input_structure=None, ground_truth_type=None,
                         gold_structure=None):
    """

    :param experiment_name:
    :param experiment_type: 'coreflinker' or 'baseline_linker'
    :return:
    """

    nr_parallels = 0
    dirs_processed = set()
    for (dirpath, dirnames, filenames) in os.walk(models_path, followlinks=True):
        if experiment_name in dirpath and dirpath not in dirs_processed:
            dirs_processed.add(dirpath)
            pred_file_name = '{}.jsonl'.format(ground_truth_type)
            if pred_file_name in filenames:
                nr_parallels += 1
                curr_file_path = os.path.join(dirpath, pred_file_name)
                print('MENTION LEVEL PROCESSING: ', curr_file_path)
                for curr_json_line in open(curr_file_path):
                    loaded_pred_json = json.loads(curr_json_line)
                    doc_id = loaded_pred_json['id']
                    assert doc_id in gold_structure
                    doc_orig = loaded_pred_json['id']
                    doc_id = '{}-{}'.format(doc_id, nr_parallels)

                    # for each of the runs, add a different document
                    if doc_id not in input_structure:
                        input_structure[doc_id] = copy.deepcopy(gold_structure[doc_orig])

                    for curr_mention in loaded_pred_json['mentions']:
                        pred_mention = dict()
                        mention_pos = (curr_mention['begin'], curr_mention['end'])
                        if mention_pos not in input_structure[doc_id] and nr_parallels > 1:
                            pred_mention['{}_variable_result'.format(experiment_type)] = True

                        if mention_pos in input_structure[doc_id]:
                            pred_mention = input_structure[doc_id][mention_pos]
                        else:
                            pred_mention['gold_mention'] = False

                        link_pred = None
                        if 'link_pred' in curr_mention:
                            link_pred = curr_mention['link_pred']

                        if '{}_link'.format(experiment_type) in pred_mention:
                            if pred_mention['{}_link'.format(experiment_type)] != link_pred and nr_parallels > 1:
                                pred_mention['{}_variable_result'.format(experiment_type)] = True
                        else:
                            if nr_parallels > 1:
                                pred_mention['{}_variable_result'.format(experiment_type)] = True

                        link_gold = None

                        ground_truth_mention = dict()
                        if mention_pos in input_structure[doc_id]:
                            ground_truth_mention = input_structure[doc_id][mention_pos]

                        # kzaporoj 20210304 - 'gold_link' before to 'link_gold'
                        if 'link_gold' in ground_truth_mention:
                            # kzaporoj 20210304 - 'gold_link' before to 'link_gold'
                            link_gold = ground_truth_mention['link_gold']

                        pred_type = ''  # true positive (tp), false positive (fp), false negative (fn)
                        if (link_pred is None or link_pred == 'NILL') and (
                                link_gold is not None and link_gold != 'NILL'):
                            pred_type = 'fn'
                        elif (link_pred is None or link_pred == 'NILL') and (
                                link_gold is None or link_gold == 'NILL'):
                            pred_type = 'tn'

                        if link_pred is not None and link_pred != 'NILL':
                            assigned_here = False
                            if link_gold != link_pred:
                                pred_type += 'fp'
                                assigned_here = True
                                if link_gold is not None and link_gold != 'NILL':
                                    pred_type += 'fn'

                            if link_gold is not None and link_gold != 'NILL':
                                if link_gold == link_pred:
                                    pred_type += 'tp'
                                    if assigned_here:
                                        print('WARN!!!: already seem have assigned fp, but has to be tp: '
                                              '{} (pred) vs {} (gold)'.format(link_pred, link_gold))

                                    assigned_here = True
                            if not assigned_here:
                                print('WARN!!!: SOMETHING WRONG WITH ASSIGNMENT!: {} (pred) vs {} (gold)'.format(
                                    link_pred, link_gold))
                                raise Exception('WARN!!!: SOMETHING WRONG WITH ASSIGNMENT!: {} (pred) vs {} '
                                                '(gold)'.format(link_pred, link_gold))

                        if len(pred_type) > 4:
                            print('!!!THIS SHOULD NOT HAPPEN FOR pred_type: ', pred_type)

                        assert pred_type != ''  # there should be always a pred_type

                        pred_mention['{}_mention'.format(experiment_type)] = True
                        pred_mention['{}_pred_type'.format(experiment_type)] = pred_type
                        pred_mention['{}_link'.format(experiment_type)] = link_pred
                        pred_mention['{}_concept_id'.format(experiment_type)] = curr_mention['concept']

                        if 'coref_scores' in curr_mention:
                            coref_scores = curr_mention['coref_scores']
                            max_score_span_idx = -1
                            max_score = -999999.9
                            max_score_span = None
                            for idx_coref_span, curr_coref_span in enumerate(coref_scores):
                                if curr_coref_span['score'] > max_score:
                                    max_score = curr_coref_span['score']
                                    max_score_span_idx = idx_coref_span
                                    max_score_span = curr_coref_span['span']

                            pred_mention['{}_coref_span_start'.format(experiment_type)] = max_score_span[0]
                            pred_mention['{}_coref_span_end'.format(experiment_type)] = max_score_span[1]
                            pred_mention['{}_coref_score'.format(experiment_type)] = max_score
                            pred_mention['{}_coref_scores'.format(experiment_type)] = coref_scores
                            if max_score_span_idx + 1 == len(coref_scores):
                                pred_mention['{}_coref_span_type'.format(experiment_type)] = 'self'
                            else:
                                pred_mention['{}_coref_span_type'.format(experiment_type)] = 'other'
                        else:
                            pred_mention['{}_coref_span_type'.format(experiment_type)] = 'self'

                        if 'candidates' in curr_mention:
                            pred_mention['{}_candidates'.format(experiment_type)] = curr_mention['candidates'][
                                                                                    :nr_candidates]
                        else:
                            pred_mention['{}_candidates'.format(experiment_type)] = None

                        if 'scores' in curr_mention:
                            pred_mention['{}_link_scores'.format(experiment_type)] = curr_mention['scores']
                        else:
                            pred_mention['{}_link_scores'.format(experiment_type)] = None

                        pred_mention['link_gold'] = link_gold
                        if 'linkable' in ground_truth_mention:
                            pred_mention['linkable'] = ground_truth_mention['linkable']
                        else:
                            pred_mention['linkable'] = False

                        if pred_mention['linkable'] and 'coreflinker_solvable' in ground_truth_mention:
                            pred_mention['coreflinker_solvable'] = ground_truth_mention['coreflinker_solvable']

                        if pred_mention['linkable']:
                            pred_mention['correct_link_seen'] = ground_truth_mention['correct_link_seen']
                            pred_mention['has_correct_link'] = ground_truth_mention['has_correct_link']
                            # if actually there is a corrrect candidate (i.e., from independent alias table: ex: cpn-alias-table.json)
                            if 'candidates' in curr_mention and link_gold in curr_mention['candidates'][:nr_candidates]:
                                pred_mention['has_correct_link'] = True
                            else:
                                pred_mention['has_correct_link'] = False

                            pred_mention['correct_link_in_cluster'] = ground_truth_mention['correct_link_in_cluster']

                        if 'coref_connection_type' in curr_mention:
                            pred_mention['{}_connection_type'.format(experiment_type)] = curr_mention[
                                'coref_connection_type']
                        else:
                            # check this good, I think it should not happen!
                            pred_mention['{}_connection_type'.format(experiment_type)] = '--NOT DEFINED--'
                        pred_mention['text'] = curr_mention['text']
                        input_structure[doc_id][mention_pos] = pred_mention
    return input_structure


def merge_predictions(input_structure: Dict = None, experiment_types: List = None):
    """
    The goal is to add ground truth spans that are both not detected by coreflinker and baseline; indicating in
    the input structure which mention spans were detected by each of the models in experiment_types
    :param input_structure:
    :param experiment_types:
    :return:
    """

    for curr_exp_type in experiment_types:
        for curr_doc, curr_spans in input_structure.items():
            for curr_span, curr_span_details in curr_spans.items():
                curr_exp_mention = '{}_mention'.format(curr_exp_type)
                if curr_exp_mention not in curr_span_details:
                    curr_span_details[curr_exp_mention] = False

                curr_span_pred_type = ''
                if curr_span_details['gold_mention'] and curr_span_details[curr_exp_mention]:
                    curr_span_pred_type = 'tp'
                elif curr_span_details['gold_mention'] and not curr_span_details[curr_exp_mention]:
                    curr_span_pred_type = 'fn'
                elif not curr_span_details['gold_mention'] and not curr_span_details[curr_exp_mention]:
                    curr_span_pred_type = 'tn'
                elif not curr_span_details['gold_mention'] and curr_span_details[curr_exp_mention]:
                    curr_span_pred_type = 'fp'

                assert curr_span_pred_type in {'tp', 'fp', 'fn', 'tn'}

                curr_span_details['{}_span_pred_type'.format(curr_exp_type)] = curr_span_pred_type

                if 'link_gold' not in curr_span_details:
                    curr_span_details['link_gold'] = None

                if not curr_span_details[curr_exp_mention] and \
                        (curr_span_details['link_gold'] is None or curr_span_details['link_gold'] == 'NILL'):
                    curr_span_details['{}_pred_type'.format(curr_exp_type)] = 'tn'
                elif not curr_span_details[curr_exp_mention] and \
                        not (curr_span_details['link_gold'] is None or curr_span_details['link_gold'] == 'NILL'):
                    curr_span_details['{}_pred_type'.format(curr_exp_type)] = 'fn'

                assert '{}_pred_type'.format(curr_exp_type) in curr_span_details
    return input_structure


def show_result_statistics_clusters_acc(predictors: List, predictions_cluster, ranges_cluster_data):
    rows_cl_size = []
    for curr_cl_size in ranges_cluster_data:
        if curr_cl_size == 1:
            filtered_curr_cl_size = predictions_cluster[predictions_cluster['cluster_size'] == curr_cl_size]
        else:
            filtered_curr_cl_size = predictions_cluster[predictions_cluster['cluster_size'] >= curr_cl_size]
        nr_gold_clusters = len(filtered_curr_cl_size.index)
        print('-----')
        print('gold cluster nr: ', nr_gold_clusters)
        for curr_predictor in predictors:
            curr_pred_field = '{}_correct'.format(curr_predictor)
            curr_pred_accuracy = filtered_curr_cl_size[curr_pred_field].sum()
            curr_pred_accuracy = curr_pred_accuracy / nr_gold_clusters
            print('curr accuracy for {} for {} cluster size: '.format(curr_predictor, curr_cl_size),
                  curr_pred_accuracy)
            rows_cl_size.append({'setup': curr_predictor, 'cluster_size': curr_cl_size, 'accuracy': curr_pred_accuracy})

    df_cl_size_stats = pd.DataFrame(rows_cl_size)
    df_piv = df_cl_size_stats.pivot(index='cluster_size', columns='setup', values='accuracy').reset_index()

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    df_piv.plot.line(ax=ax, x='cluster_size',
                     title='Accuracy cluster size')

    plt.xticks(ranges_cluster_data)
    ax.set_xlabel(">= Mentions in cluster")
    ax.set_ylabel("Accuracy")

    plt.show()


def get_result_statistics(predictor: str, predictions: pd.DataFrame):
    df_stats = predictions.groupby(['{}_pred_type'.format(predictor)])['text'].count().reset_index()
    tp = 0
    filtered_tp = df_stats[df_stats['{}_pred_type'.format(predictor)] == 'tp']
    if len(filtered_tp.index) == 1:
        tp = filtered_tp.iloc[0]['text']

    filtered_fn = df_stats[df_stats['{}_pred_type'.format(predictor)] == 'fn']
    fn = 0
    if len(filtered_fn.index) == 1:
        fn = filtered_fn.iloc[0]['text']

    filtered_fp = df_stats[df_stats['{}_pred_type'.format(predictor)] == 'fp']
    fp = 0
    if len(filtered_fp.index) == 1:
        fp = filtered_fp.iloc[0]['text']

    filtered_fpfn = df_stats[df_stats['{}_pred_type'.format(predictor)] == 'fpfn']
    fpfn = 0
    if len(filtered_fpfn.index) == 1:
        fpfn = filtered_fpfn.iloc[0]['text']

    print('{:20}{:6}{:6}{:6}{:6}{:6}{:6}{:6}{:6}'.format('MODEL', 'TP', 'FP/FN', 'FP', 'FN', 'Pr', 'Re', 'F1', 'Acc'))
    pr = tp / (tp + fpfn + fp)
    re = tp / (tp + fpfn + fn)
    f1 = (2 * pr * re) / (re + pr)
    print('{:20}{:<6}{:<6}{:<6}{:<6}{:<6.3f}{:<6.3f}{:<6.3f}{:<6.3f}'.format(predictor, tp, fpfn, fp, fn, pr, re, f1,
                                                                             re))


def get_avg_cluster_size_distribution(df_data: pd.DataFrame):
    pass


def get_link_coref_distribution(df_datas: List[Tuple[str, pd.DataFrame]]):
    """
    Percentage (and #) of links predicted using link prediction vs percentage (and #) predicted using reference to
    other mention (coref)

    :param df_data:
    :return:
    """

    print('{:<20}{:>7}{:>7}{:>7}{:>7}'.format('description', 'men', '% men', 'link', '% link'))

    for df_data_tuple in df_datas:
        df_data = df_data_tuple[1]
        description = df_data_tuple[0]

        df_grouped = df_data.groupby(['coreflinker_connection_type'])['text'].count().reset_index()
        link = 0
        filtered = df_grouped[df_grouped['coreflinker_connection_type'] == 'link']
        if len(filtered.index) == 1:
            link = filtered.iloc[0]['text']

        mention_other = 0
        filtered = df_grouped[df_grouped['coreflinker_connection_type'] == 'mention_other']
        if len(filtered.index) == 1:
            mention_other = filtered.iloc[0]['text']

        print('{:<20}{:>7}{:>7.2f}%{:>7}{:>7.2f}%'.format(description, mention_other,
                                                          (mention_other / (mention_other + link)) * 100,
                                                          link, (link / (mention_other + link)) * 100))


def get_most_common_cases(df_data: pd.DataFrame):
    pass


def get_top_entity_types(df_data: pd.DataFrame):
    pass


def get_nr_intersected_v2(df_data: pd.DataFrame, model_types: List, pred_types: List):
    """
    An improved version from get_nr_intersected, that autimatically builds the matrix based on predictors and
    pred types passed as parameters
    :param df_data:
    :return:
    """
    matrix_res = dict()
    for curr_pred_type_m1 in pred_types:
        nr_entry_type_m1 = df_data[(df_data['{}_pred_type'.format(model_types[0])] == curr_pred_type_m1)]
        nr_entry_type_m1 = len(nr_entry_type_m1.index)
        nr_entry_type_m2 = df_data[(df_data['{}_pred_type'.format(model_types[1])] == curr_pred_type_m1)]
        nr_entry_type_m2 = len(nr_entry_type_m2.index)

        matrix_res[(curr_pred_type_m1, '')] = nr_entry_type_m1
        matrix_res[('', curr_pred_type_m1)] = nr_entry_type_m2

        for curr_pred_type_m2 in pred_types:
            model_type1 = model_types[0]
            model_type2 = model_types[1]
            entry_type = (curr_pred_type_m1, curr_pred_type_m2)
            nr_entry_type = df_data[(df_data['{}_pred_type'.format(model_type1)] == curr_pred_type_m1) &
                                    (df_data['{}_pred_type'.format(model_type2)] == curr_pred_type_m2)]
            nr_entry_type = len(nr_entry_type.index)
            matrix_res[entry_type] = nr_entry_type

    print('{:^42s}'.format(model_types[1]))
    row_format = '{:<15}{:>6}|' + '{:<6}' * (len(pred_types) + 1)

    # prints the title
    title_args = ['', ''] + pred_types + ['Tot.']
    title = row_format.format(*title_args)
    print(title)
    sum_margins_m1 = 0
    for curr_pred_type_m1 in pred_types:
        curr_line_args = [model_types[0], curr_pred_type_m1]
        for curr_pred_type_m2 in pred_types:
            curr_line_args.append(matrix_res[(curr_pred_type_m1, curr_pred_type_m2)])
        curr_line_args.append(matrix_res[(curr_pred_type_m1, '')])
        sum_margins_m1 += matrix_res[(curr_pred_type_m1, '')]
        curr_line = row_format.format(*curr_line_args)
        print(curr_line)

    sum_m2_lst = [matrix_res[('', cptm2)] for cptm2 in pred_types]
    sum_margins_m2 = sum(sum_m2_lst)
    title_args = ['', 'Tot.'] + sum_m2_lst
    title_args += ['{}\{}'.format(sum_margins_m2, sum_margins_m1)]
    title = row_format.format(*title_args)
    print(title)


def get_nr_intersected(df_data: pd.DataFrame):
    """

    :param df_data:
    :param pred_type:
    :return: how many mentions are fpfn for both coreflinker and baseline
    """
    df_data_coreflinker_fpfn = df_data[df_data['coreflinker_pred_type'] == 'fpfn']
    df_data_coreflinker_fn = df_data[df_data['coreflinker_pred_type'] == 'fn']
    df_data_coreflinker_fp = df_data[df_data['coreflinker_pred_type'] == 'fp']
    df_data_coreflinker_tp = df_data[df_data['coreflinker_pred_type'] == 'tp']

    df_data_base_fpfn = df_data[df_data['baseline_linker_pred_type'] == 'fpfn']
    df_data_base_fn = df_data[df_data['baseline_linker_pred_type'] == 'fn']
    df_data_base_fp = df_data[df_data['baseline_linker_pred_type'] == 'fp']
    df_data_base_tp = df_data[df_data['baseline_linker_pred_type'] == 'tp']

    df_data_tp_tp = df_data[(df_data['baseline_linker_pred_type'] == 'tp') &
                            (df_data['coreflinker_pred_type'] == 'tp')]
    df_data_tp_fpfn = df_data[(df_data['baseline_linker_pred_type'] == 'tp') &
                              (df_data['coreflinker_pred_type'] == 'fpfn')]

    df_data_tp_fn = df_data[(df_data['baseline_linker_pred_type'] == 'tp') &
                            (df_data['coreflinker_pred_type'] == 'fn')]

    df_data_tp_fp = df_data[(df_data['baseline_linker_pred_type'] == 'tp') &
                            (df_data['coreflinker_pred_type'] == 'fp')]

    df_data_fpfn_tp = df_data[(df_data['baseline_linker_pred_type'] == 'fpfn') &
                              (df_data['coreflinker_pred_type'] == 'tp')]

    df_data_fpfn_fpfn = df_data[(df_data['baseline_linker_pred_type'] == 'fpfn') &
                                (df_data['coreflinker_pred_type'] == 'fpfn')]

    df_data_fpfn_fp = df_data[(df_data['baseline_linker_pred_type'] == 'fpfn') &
                              (df_data['coreflinker_pred_type'] == 'fp')]

    df_data_fpfn_fn = df_data[(df_data['baseline_linker_pred_type'] == 'fpfn') &
                              (df_data['coreflinker_pred_type'] == 'fn')]

    df_data_fp_tp = df_data[(df_data['baseline_linker_pred_type'] == 'fp') &
                            (df_data['coreflinker_pred_type'] == 'tp')]

    df_data_fp_fpfn = df_data[(df_data['baseline_linker_pred_type'] == 'fp') &
                              (df_data['coreflinker_pred_type'] == 'fpfn')]

    df_data_fp_fp = df_data[(df_data['baseline_linker_pred_type'] == 'fp') &
                            (df_data['coreflinker_pred_type'] == 'fp')]

    df_data_fp_fn = df_data[(df_data['baseline_linker_pred_type'] == 'fp') &
                            (df_data['coreflinker_pred_type'] == 'fn')]

    df_data_fn_tp = df_data[(df_data['baseline_linker_pred_type'] == 'fn') &
                            (df_data['coreflinker_pred_type'] == 'tp')]

    df_data_fn_fpfn = df_data[(df_data['baseline_linker_pred_type'] == 'fn') &
                              (df_data['coreflinker_pred_type'] == 'fpfn')]

    df_data_fn_fp = df_data[(df_data['baseline_linker_pred_type'] == 'fn') &
                            (df_data['coreflinker_pred_type'] == 'fp')]

    df_data_fn_fn = df_data[(df_data['baseline_linker_pred_type'] == 'fn') &
                            (df_data['coreflinker_pred_type'] == 'fn')]

    print('{:^42s}'.format('coreflinker'))
    print('{:<5}{:>6}|{:<6}{:<6}{:<6}{:<6}{:<6}'.format('', '', 'TP', 'FP/FN', 'FP', 'FN', 'Tot.'))
    print('{:<5}{:>6}|{:<6}{:<6}{:<6}{:<6}{:<6}'.format('base', 'TP',
                                                        len(df_data_tp_tp.index),
                                                        len(df_data_tp_fpfn.index),
                                                        len(df_data_tp_fp.index),
                                                        len(df_data_tp_fn.index),
                                                        len(df_data_base_tp.index)))
    print('{:<5}{:>6}|{:<6}{:<6}{:<6}{:<6}{:<6}'.format('base', 'FP/FN',
                                                        len(df_data_fpfn_tp.index),
                                                        len(df_data_fpfn_fpfn.index),
                                                        len(df_data_fpfn_fp.index),
                                                        len(df_data_fpfn_fn.index),
                                                        len(df_data_base_fpfn.index)))
    print('{:<5}{:>6}|{:<6}{:<6}{:<6}{:<6}{:<6}'.format('base', 'FP',
                                                        len(df_data_fp_tp.index),
                                                        len(df_data_fp_fpfn.index),
                                                        len(df_data_fp_fp.index),
                                                        len(df_data_fp_fn.index),
                                                        len(df_data_base_fp.index)))
    print('{:<5}{:>6}|{:<6}{:<6}{:<6}{:<6}{:<6}'.format('base', 'FN',
                                                        len(df_data_fn_tp.index),
                                                        len(df_data_fn_fpfn.index),
                                                        len(df_data_fn_fp.index),
                                                        len(df_data_fn_fn.index),
                                                        len(df_data_base_fn.index)))
    print('-------------------------------------------------')
    print('{:<5}{:>6}|{:<6}{:<6}{:<6}{:<6}{:<6}'.format('', 'Tot.',
                                                        len(df_data_coreflinker_tp.index),
                                                        len(df_data_coreflinker_fpfn.index),
                                                        len(df_data_coreflinker_fp.index),
                                                        len(df_data_coreflinker_fn.index),
                                                        ''))

    get_link_coref_distribution([('coreflinker tp', df_data_coreflinker_tp),
                                 ('coreflinker fpfn', df_data_coreflinker_fpfn),
                                 ('coreflinker fn', df_data_coreflinker_fn),
                                 ('base-cl tp tp', df_data_tp_tp),
                                 ('base-cl tp fpfn', df_data_tp_fpfn),
                                 ('base-cl tp fn', df_data_tp_fn),
                                 ('base-cl fpfn tp', df_data_fpfn_tp),
                                 ('base-cl fpfn fpfn', df_data_fpfn_fpfn),
                                 ('base-cl fpfn fn', df_data_fpfn_fn),
                                 ('base-cl fn tp', df_data_fn_tp),
                                 ('base-cl fn fpfn', df_data_fn_fpfn),
                                 ('base-cl fn fn', df_data_fn_fn),
                                 ('base tp', df_data_base_tp),
                                 ('base fpfn', df_data_base_fpfn),
                                 ('base fn', df_data_base_fn)
                                 ])

    print('---')
    print('LINK/Mention other distribution for coreflinker tp: ')

    print('here to see what intersecter does')


def stat001_connection_type_effect_on_others():
    """
    Stat on the effect the different connection_types (ex: 'link', 'mention_other') have in other modules.
    For example, we may expect that for mention where coreflinker or coreflinker mtt predicted links through
    coreference (connection_type in 'mention_other'), the baseline can have worse result.
    :return:
    """
    return

    # df_tp_pred_type_baseline = df_tp_coreflinker.groupby(['baseline_linker_pred_type'])['baseline_linker_link'].count()
    df_tp_link_pred_type_baseline = \
        df_tp_coreflinker[df_tp_coreflinker['coreflinker_connection_type'] == 'link'].groupby(
            ['baseline_linker_pred_type'])['baseline_linker_link', 'text'].count()

    print('distribution of pred_types for baseline for coreflinker connected with "link": ')
    print(df_tp_link_pred_type_baseline)
    df_tp_mention_pred_type_baseline = \
        df_tp_coreflinker[df_tp_coreflinker['coreflinker_connection_type'] == 'mention_other'].groupby(
            ['baseline_linker_pred_type'])['baseline_linker_link', 'text'].count()

    print('distribution of pred_types for baseline for coreflinker connected with "mention_other": ')
    print(df_tp_mention_pred_type_baseline)


def stat002_edit_distance_comparison(df_tp_per_model_type: Dict, model_types: List):
    """
    This stat takes all the true positives and compares the edit distance mention vs correct link for each of the
    connection_types ; for the baseline (I think) there is a single connection_type which is '--NOT DEFINED--'
    (see in this module)

    :return:
    """
    print('==================================BEGIN: stat002_edit_distance_comparison=================================')
    for curr_model_type in model_types:
        model_name = curr_model_type['alias']
        df_cnt_per_connection_type = df_tp_per_model_type[model_name] \
            .groupby(['text', 'link_gold', '{}_connection_type'.format(model_name)])['{}_mention'.format(model_name)] \
            .count().reset_index().rename(columns={'{}_mention'.format(model_name): 'count'})

        df_cnt_per_connection_type.sort_values(by=['count'], ascending=False, inplace=True)

        df_cnt_per_connection_type_nf = \
            df_tp_per_model_type[model_name].groupby(['text', 'link_gold',
                                                      '{}_connection_type_not_first'.format(model_name)])[
                '{}_mention'.format(model_name)].count().reset_index().rename(
                columns={'{}_mention'.format(model_name): 'count'})

        df_cnt_per_connection_type_nf.sort_values(by=['count'], ascending=False, inplace=True)

        df_cnt_conn_type_mention_other = \
            df_cnt_per_connection_type[df_cnt_per_connection_type['{}_connection_type'.format(model_name)]
                                       == 'mention_other']

        df_cnt_conn_type_mention_link = \
            df_cnt_per_connection_type[df_cnt_per_connection_type['{}_connection_type'.format(model_name)] == 'link']

        df_cnt_conn_type_not_defined = \
            df_cnt_per_connection_type[df_cnt_per_connection_type['{}_connection_type'.format(model_name)] ==
                                       '--NOT DEFINED--']

        print('----------FOR MODEL {}----------'.format(model_name))
        print('TOP 20 connected with mention_other (including first mentions in cluster): ')
        print('{:20}{:30}{:10}'.format('Mention', 'Link', 'Count'))
        for index, row_men_other in df_cnt_conn_type_mention_other.head(10).iterrows():
            print('{:.<20}{:.<30}{:<10}'.format(row_men_other['text'], row_men_other['link_gold'],
                                                row_men_other['count']))
        print()
        print('TOP 20 connected with link (including first mentions in cluster): ')
        print('{:20}{:30}{:10}'.format('Mention', 'Link', 'Count'))
        for index, row_men_other in df_cnt_conn_type_mention_link.head(10).iterrows():
            print('{:.<20}{:.<30}{:<10}'.format(row_men_other['text'], row_men_other['link_gold'],
                                                row_men_other['count']))
        print()
        print('TOP 20 connected with not defined (including first mentions in cluster): ')
        print('{:20}{:30}{:10}'.format('Mention', 'Link', 'Count'))
        for index, row_men_other in df_cnt_conn_type_not_defined.head(10).iterrows():
            print('{:.<20}{:.<30}{:<10}'.format(row_men_other['text'], row_men_other['link_gold'],
                                                row_men_other['count']))

        df_cnt_conn_type_mention_other_nf = \
            df_cnt_per_connection_type_nf[
                df_cnt_per_connection_type_nf['{}_connection_type_not_first'.format(model_name)] == 'mention_other']

        df_cnt_conn_type_mention_link_nf = \
            df_cnt_per_connection_type_nf[
                df_cnt_per_connection_type_nf['{}_connection_type_not_first'.format(model_name)] == 'link']

        df_cnt_conn_type_not_defined_nf = \
            df_cnt_per_connection_type_nf[
                df_cnt_per_connection_type_nf['{}_connection_type_not_first'.format(model_name)] == '--NOT DEFINED--']

        print()
        print('----')
        print('TOP 20 connected with mention_other (excluding first mentions in cluster): ')
        for index, row_men_other in df_cnt_conn_type_mention_other_nf.head(10).iterrows():
            print('{:.<20}{:.<30}{:<10}'.format(row_men_other['text'], row_men_other['link_gold'],
                                                row_men_other['count']))
        print()
        print('TOP 20 connected with link (excluding first mentions in cluster): ')
        for index, row_men_other in df_cnt_conn_type_mention_link_nf.head(10).iterrows():
            print('{:.<20}{:.<30}{:<10}'.format(row_men_other['text'], row_men_other['link_gold'],
                                                row_men_other['count']))

        print()
        print('TOP 20 connected with not defined (excluding first mentions in cluster): ')
        for index, row_men_other in df_cnt_conn_type_not_defined_nf.head(10).iterrows():
            print('{:.<20}{:.<30}{:<10}'.format(row_men_other['text'], row_men_other['link_gold'],
                                                row_men_other['count']))

        if len(edit_distances_mention_other[model_name]) > 0:
            print('# and avg edit distances mention_other (including first mentions in cluster): ',
                  len(edit_distances_mention_other[model_name]), ' ----- ',
                  sum(edit_distances_mention_other[model_name]) / len(edit_distances_mention_other[model_name]))

        if len(edit_distances_link[model_name]) > 0:
            print('# and avg edit distances link (including first mentions in cluster): ',
                  len(edit_distances_link[model_name]), ' ----- ',
                  sum(edit_distances_link[model_name]) / len(edit_distances_link[model_name]))

        if len(edit_distances_mention_other_nf[model_name]):
            print('# and avg edit distances mention_other (excluding first mentions in cluster): ',
                  len(edit_distances_mention_other_nf[model_name]), ' ----- ',
                  sum(edit_distances_mention_other_nf[model_name]) / len(edit_distances_mention_other_nf[model_name]))

        if len(edit_distances_link_nf[model_name]):
            print('# and avg edit distances link (excluding first mentions in cluster): ',
                  len(edit_distances_link_nf[model_name]), ' ----- ',
                  sum(edit_distances_link_nf[model_name]) / len(edit_distances_link_nf[model_name]))

        print('most common mention prev (mention-link-nr times): ')
        print('most common link (mention-link-nr times): ')
        print('total number of true positives: ', len(df_tp_per_model_type[model_name].index))
    print('===================================END: stat002_edit_distance_comparison===================================')


def plot_data(df_frame_to_plot: pd.DataFrame, title_plot: str, fields_to_plot: list, label_fields: list,
              color_fields: list, x_field: str, x_axis: str = '', y_axis: str = '',
              top_rows=-1, is_head=False, is_tail=False, type_plot='bar', x_ticks=None, y_ticks=None,
              x_scale=None):
    line_markers = ['o', 'x', '2', '*', 'p', '.']
    if is_head:
        df_frame_to_plot = df_frame_to_plot.head(top_rows)
    elif is_tail:
        df_frame_to_plot = df_frame_to_plot.tail(top_rows)

    fig, ax = plt.subplots(1, 1, figsize=(20, 6))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    max_field_value = 0
    for field in fields_to_plot:
        curr_field = df_frame_to_plot.iloc[0][field]
        if curr_field > max_field_value:
            max_field_value = curr_field

    for idx, field in enumerate(fields_to_plot):
        label = label_fields[idx]
        color = color_fields[idx]
        if type_plot == 'bar':
            bars = df_frame_to_plot.plot.bar(ax=ax, x=x_field, y=field,
                                             color=color, label=label,
                                             title=title_plot)
            for bar in bars.patches:
                yval = bar.get_height()
                plt.text(bar.get_x() - len(df_frame_to_plot.index) * 0.002,
                         yval + (max_field_value) * 0.005, yval)
        elif type_plot == 'line':
            bars = df_frame_to_plot.plot(ax=ax, x=x_field, y=field,
                                         color=color, label=label,
                                         title=title_plot, marker=line_markers[idx])
            x_field_vals = list(df_frame_to_plot[x_field])
            y_field_vals = list(df_frame_to_plot[field])
            if x_ticks is not None:
                while len(x_field_vals) > len(x_ticks):
                    x_field_vals = x_field_vals[::2]
                    y_field_vals = y_field_vals[::2]

            for idx, (x_point, y_point) in enumerate(zip(*[x_field_vals, y_field_vals])):
                ax.text(x_point, y_point + 0.01, '{:.2f}'.format(y_point))
            for bar in bars.patches:
                yval = bar.get_height()
                plt.text(bar.get_x() - len(df_frame_to_plot.index) * 0.002,
                         yval + (max_field_value) * 0.005, yval)
    ax.set_ylabel(y_axis)
    ax.set_xlabel(x_axis)
    if x_scale is not None:
        ax.set_xscale(x_scale)

    if x_ticks is not None:
        ax.set_xticks(x_ticks)
    if y_ticks is not None:
        ax.set_yticks(y_ticks)
    fig.subplots_adjust(bottom=0.3)
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.xticks(rotation=90)
    plt.show()


if __name__ == "__main__":

    ###################### OLD COMMENTS BEFORE 20210303###########################
    # models_path = 'models_studied/'
    # experiment_coreflinker_name = '20201007-coreflinker-scores'
    # experiment_baseline_linker_name = '20201007-base-coref-scores'

    # experiment_coreflinker_name = '20201011-coreflinker-ap0'

    # trying the end-to-end one
    # experiment_coreflinker_name = '20201025-coreflinker_e2e-ap0-1'
    # experiment_baseline_linker_name = '20201011-base-linker-coref-ap0-1'
    ###################### END OLD COMMENTS BEFORE 20210303###########################

    # experiment_coreflinker_name = '20201207-coreflinker_e2e-ap0-2'
    # experiment_coreflinker_name = '20201208-cl_e2e_singl_matr_snnet_ignore_chains-ap0'
    # experiment_coreflinker_name = '20210222b-cl_e2e_mtt_float32_arsinh-ap0-1'
    # experiment_coreflinker_name = '20210222b-cl_e2e_mtt_float32_arsinh-ap0'

    # experiment_baseline_linker_name = '20201207-coreflinker_e2e-ap0-1'
    # experiment_baseline_linker_name = '20201214-linker_e2e_pruner_coref-ap0-2'
    # experiment_baseline_linker_name = '20201214-linker_e2e_pruner_coref-ap0'
    # experiment_baseline_linker_name = '20201207-linker_e2e_pruner-ap0-1'

    # the names of the models can be suffixed by the id of a specific run (ex: "20201214-linker_e2e_pruner_coref-ap0-1")
    # or not (ex: "20201214-linker_e2e_pruner_coref-ap0")

    # models_to_compare = [{'alias': 'CLinker+MTT', 'model': '20210222b-cl_e2e_mtt_float32_arsinh-ap0-1'},
    #                      {'alias': 'CLinker', 'model': '20201207-coreflinker_e2e-ap0-1'},
    #                      {'alias': 'Base', 'model': '20201214-linker_e2e_pruner_coref-ap0-1'}]
    # models_to_compare = [{'alias': 'CLinker+MTT', 'model': '20210428b-mtt_root_init_std_clip_norm10-2'},
    #                      {'alias': 'CLinker', 'model': '20210428a-cl-clip_norm10-1'},
    #                      {'alias': 'Base', 'model': '20210428a-l-1'}]
    # models_to_compare = [{'alias': 'CLinker+MTT', 'model': '20210428b-mtt_aida_init_std_clip_norm10-1'},
    #                      {'alias': 'CLinker', 'model': '20210428a-aida-cl_clip_norm10-2'},
    #                      {'alias': 'Base', 'model': '20210428a-aida_l_clip_norm10-2'}]

    ####### BEGIN AIDA
    # models_path = 'models/selected_aida/'
    # models_to_compare = [
    #     {'alias': 'CLinker+MTT', 'model': '20210513a-mtt_aida_0_to_l_tr_rnd_root_singletons_kolitsas-3'},
    #     {'alias': 'CLinker', 'model': '20210513a-aida-cl_kolitsas_emb_singletons_50eps-3'},
    #     {'alias': 'Base', 'model': '20210508a-aida_only_linker_kolitsas_emb-3'}]
    # # models_to_compare = [
    # #     {'alias': 'CLinker+MTT', 'model': '20210513a-mtt_aida_0_to_l_tr_rnd_root_singletons_kolitsas'},
    # #     {'alias': 'CLinker', 'model': '20210513a-aida-cl_kolitsas_emb_singletons_50eps'},
    # #     {'alias': 'Base', 'model': '20210508a-aida_only_linker_kolitsas_emb'}]
    # ground_truth_dir = 'data/aida/aida_reannotated/aida-20210402/transformed/aida_tokenization_adaptation'
    # # ground_truth_type = 'testa'  # can be also train if we want to compare on train set
    # ground_truth_type = 'testb'  # can be also train if we want to compare on train set
    # nr_candidates = 999
    ####### END AIDA

    ###### BEGIN DWIE
    models_path = 'models/selected'
    models_to_compare = [{'alias': 'CLinker+MTT', 'model': '20210513a-mtt_0_to_links_trainable_rnd_root_50_eps'},
                         {'alias': 'CLinker', 'model': '20210513a-cl_singletons-50eps'},
                         {'alias': 'Base', 'model': '20210508a-only_linking_50eps'}]
    ground_truth_dir = 'data/data-20200921'
    ground_truth_type = 'test'  # can be also train if we want to compare on train set
    nr_candidates = 999
    ###### END DWIE

    #  so for gold need to have:
    #    doc_id -> mention span pos -> solvable/unsolvable ; linkable/unlinkable (if NILL as well) ; valid link

    ground_truth_structure_cluster_based = dict()

    ground_truth_structure_mention_based = dict()
    ground_truth_text = dict()
    top_load = -1

    # loads ground truth driven by the proposed above structure
    nr_processed_files = 0
    for (dirpath, dirnames, filenames) in os.walk(ground_truth_dir):
        for curr_file in filenames:
            if '.json' in curr_file and 'main' not in curr_file:

                already_seen_coreflinker_clusters = set()
                # clusters where a mention with correct link in its candidate list has already been seen
                clusters_with_seen_correct_mentions = set()
                unsolvable_clusters_coreflinker = set()

                loaded_json = json.load(open(os.path.join(dirpath, curr_file)))
                if ground_truth_type not in loaded_json['tags']:
                    continue

                if -1 < top_load > nr_processed_files:
                    continue

                nr_processed_files += 1

                doc_id = loaded_json['id']
                print('processing the following doc_id: ', doc_id)
                ground_truth_structure_mention_based[doc_id] = dict()

                ground_truth_text[doc_id] = loaded_json['content']

                concept_to_mentions = dict()
                for curr_mention in loaded_json['mentions']:
                    if curr_mention['concept'] not in concept_to_mentions:
                        concept_to_mentions[curr_mention['concept']] = list()
                    concept_to_mentions[curr_mention['concept']].append(curr_mention)

                ground_truth_structure_cluster_based[doc_id] = dict()
                for curr_cluster in loaded_json['concepts']:
                    concept_id = curr_cluster['concept']
                    if concept_id not in concept_to_mentions:
                        continue
                    is_cluster_linkable = 'link' in curr_cluster and curr_cluster['link'] is not None and \
                                          curr_cluster['link'] != 'NILL'
                    if not is_cluster_linkable:
                        continue

                    cluster_mentions = concept_to_mentions[concept_id]
                    cluster_spans = sorted([(cm['begin'], cm['end']) for cm in cluster_mentions],
                                           key=lambda x: (x[0], x[1]))
                    cluster_spans = tuple(cluster_spans)
                    ground_truth_structure_cluster_based[doc_id][cluster_spans] = dict()
                    ground_truth_structure_cluster_based[doc_id][cluster_spans]['cluster_spans'] = cluster_spans
                    ground_truth_structure_cluster_based[doc_id][cluster_spans]['link_gold'] = curr_cluster['link']
                    cluster_size = len(cluster_mentions)
                    ground_truth_structure_cluster_based[doc_id][cluster_spans]['cluster_size'] = cluster_size
                    cluster_texts = [men['text'] for men in cluster_mentions]
                    ground_truth_structure_cluster_based[doc_id][cluster_spans]['cluster_size_distinct'] = len(
                        set(cluster_texts))


                for curr_mention in loaded_json['mentions']:
                    mention_to_add = dict()
                    # we do not do Null or 'NILL' for now
                    mention_to_add['linkable'] = 'link' in loaded_json['concepts'][curr_mention['concept']] and \
                                                 loaded_json['concepts'][curr_mention['concept']][
                                                     'link'] is not None and \
                                                 loaded_json['concepts'][curr_mention['concept']]['link'] != 'NILL'

                    gold_link = None

                    mention_pos = (curr_mention['begin'], curr_mention['end'])
                    if mention_to_add['linkable']:
                        gold_link = loaded_json['concepts'][curr_mention['concept']]['link']
                        # kzaporoj 20210304 - 'gold_link' before to 'link_gold'
                        mention_to_add['link_gold'] = gold_link

                    # it is a first mention in the cluster

                    if gold_link is not None:
                        # - correct_link_seen --> already some mention (or the mention itself) in cluster has been
                        # detected to have correct link (ordered by span appearance)
                        # - has_correct_link --> the mention itself contains correct link
                        # - correct_link_in_cluster --> whether the correct link is detected in any of the mentions in
                        # cluster
                        mentions_concept = concept_to_mentions[curr_mention['concept']]
                        if curr_mention['concept'] not in clusters_with_seen_correct_mentions:
                            mention_to_add['correct_link_seen'] = False
                        else:
                            mention_to_add['correct_link_seen'] = True

                        if 'candidates' not in curr_mention or gold_link not in curr_mention['candidates'][
                                                                                :nr_candidates]:
                            mention_to_add['has_correct_link'] = False
                            mention_to_add['correct_link_in_cluster'] = False
                            cluster_has_correct_candidate = len(
                                [men for men in mentions_concept if 'candidates' in men and
                                 gold_link in men['candidates'][:nr_candidates]]) > 0
                            mention_to_add['correct_link_in_mention'] = False
                            if cluster_has_correct_candidate:
                                mention_to_add['correct_link_in_cluster'] = True
                                if curr_mention['concept'] not in already_seen_coreflinker_clusters:
                                    unsolvable_clusters_coreflinker.add(curr_mention['concept'])
                            else:
                                mention_to_add['correct_link_in_cluster'] = False
                        elif 'candidates' in curr_mention and gold_link in curr_mention['candidates'][:nr_candidates]:
                            clusters_with_seen_correct_mentions.add(curr_mention['concept'])
                            mention_to_add['correct_link_seen'] = True
                            mention_to_add['has_correct_link'] = True
                            mention_to_add['correct_link_in_cluster'] = True
                        already_seen_coreflinker_clusters.add(curr_mention['concept'])

                    if mention_to_add['linkable']:
                        if curr_mention['concept'] in unsolvable_clusters_coreflinker:
                            mention_to_add['coreflinker_solvable'] = False
                        else:
                            mention_to_add['coreflinker_solvable'] = True

                    mention_to_add['concept'] = loaded_json['concepts'][curr_mention['concept']]
                    if 'candidates' in curr_mention:
                        mention_to_add['candidates'] = curr_mention['candidates'][:nr_candidates]

                    mention_to_add['gold_mention'] = True
                    ground_truth_structure_mention_based[doc_id][
                        mention_pos] = mention_to_add  # TODO: here how to add text?

    print('ground_truth_structure loaded')
    pred_structure_clusters = dict()
    for curr_model_to_compare in models_to_compare:
        pred_structure_clusters = complete_predictions_cluster_level(models_path=models_path,
                                                                     experiment_name=curr_model_to_compare['model'],
                                                                     experiment_type=curr_model_to_compare['alias'],
                                                                     input_structure=pred_structure_clusters,
                                                                     ground_truth_type=ground_truth_type,
                                                                     gold_structure=ground_truth_structure_cluster_based)

    pred_structure = dict()

    for curr_model_to_compare in models_to_compare:
        pred_structure = complete_predictions(models_path=models_path, experiment_name=curr_model_to_compare['model'],
                                              experiment_type=curr_model_to_compare['alias'],
                                              input_structure=pred_structure,
                                              ground_truth_type=ground_truth_type,
                                              gold_structure=ground_truth_structure_mention_based)

    experiment_types = [mc['alias'] for mc in models_to_compare]
    # for e2e setting we need to merge the pred_structure because some mentions are only predicted by coreflinker
    # and others only predicted by baseline_linker
    pred_structure = merge_predictions(input_structure=pred_structure, experiment_types=experiment_types)

    edit_distances_mention_other = dict()
    edit_distances_link = dict()
    edit_distances_not_defined = dict()

    # suffix nf comes from 'not first' and means that it is not the first mention for a particular cluster, the first
    # mention will always have "link", can not have "mention_other", right?
    edit_distances_mention_other_nf = dict()
    edit_distances_link_nf = dict()
    edit_distances_not_defined_nf = dict()

    total_tps = 0

    span_list_clusters = []
    for curr_doc_id, curr_clusters in pred_structure_clusters.items():
        for curr_cluster_spans, curr_cluster_details in curr_clusters.items():
            row_to_load_cl = dict()
            row_to_load_cl['doc_id'] = curr_doc_id
            row_to_load_cl['link_gold'] = curr_cluster_details['link_gold']
            row_to_load_cl['cluster_size'] = curr_cluster_details['cluster_size']
            row_to_load_cl['cluster_size_distinct'] = curr_cluster_details['cluster_size_distinct']
            row_to_load_cl['cluster_spans'] = curr_cluster_spans
            for curr_model_to_compare in models_to_compare:
                alias = curr_model_to_compare['alias']
                row_to_load_cl['{}_correct'.format(alias)] = curr_cluster_details['{}_correct'.format(alias)]
            span_list_clusters.append(row_to_load_cl)

    df_cluster_data = pd.DataFrame(span_list_clusters)

    ranges_df_cluster_data = [1, 2]
    predictors = [pr['alias'] for pr in models_to_compare]
    show_result_statistics_clusters_acc(predictors, df_cluster_data, ranges_df_cluster_data)

    span_list = []

    # for curr_model_to_compare in models_to_compare:
    for curr_doc, curr_spans in pred_structure.items():
        # (kzaporoj) 02/12/2020 - by sorting spans first we can add other stats such as whether a particular span is
        # the first span of the cluster
        span_preds = sorted(curr_spans.items(), key=lambda x: x[0][0])
        already_seen_coreflinker_clusters = dict()

        for curr_span, curr_pred in span_preds:
            row_to_load = dict()
            row_to_load['doc_id'] = curr_doc
            row_to_load['span_start'] = curr_span[0]
            row_to_load['span_end'] = curr_span[1]
            row_to_load['gold_mention'] = curr_pred['gold_mention']
            if curr_pred['gold_mention']:
                row_to_load['concept_gold'] = curr_pred['concept']['concept']
            else:
                row_to_load['concept_gold'] = None

            row_to_load['link_gold'] = curr_pred['link_gold']
            row_to_load['linkable'] = curr_pred['linkable']

            row_to_load['coreflinker_solvable'] = curr_pred.get('coreflinker_solvable')

            row_to_load['correct_link_seen'] = curr_pred.get('correct_link_seen')
            row_to_load['has_correct_link'] = curr_pred.get('has_correct_link')
            row_to_load['correct_link_in_cluster'] = curr_pred.get('correct_link_in_cluster')

            if 'text' in curr_pred and curr_pred['text'] != '':
                row_to_load['text'] = curr_pred['text']
            else:
                # ex: 1163-1
                if '-' in curr_doc:
                    row_to_load['text'] = ground_truth_text[curr_doc[:-2]][curr_span[0]:curr_span[1]]
                else:
                    row_to_load['text'] = ground_truth_text[curr_doc][curr_span[0]:curr_span[1]]


            for curr_model_to_compare in models_to_compare:

                row_to_load['{}_mention'.format(curr_model_to_compare['alias'])] = \
                    curr_pred['{}_mention'.format(curr_model_to_compare['alias'])]
                row_to_load['{}_connection_type'.format(curr_model_to_compare['alias'])] = \
                    curr_pred.get('{}_connection_type'.format(curr_model_to_compare['alias']))

                if curr_model_to_compare['alias'] not in already_seen_coreflinker_clusters:
                    already_seen_coreflinker_clusters[curr_model_to_compare['alias']] = set()
                if curr_pred['gold_mention'] and curr_pred['concept']['concept'] in \
                        already_seen_coreflinker_clusters[curr_model_to_compare['alias']] \
                        and row_to_load['{}_mention'.format(curr_model_to_compare['alias'])]:
                    row_to_load['{}_connection_type_not_first'.format(curr_model_to_compare['alias'])] = \
                        curr_pred['{}_connection_type'.format(curr_model_to_compare['alias'])]
                else:
                    row_to_load['{}_connection_type_not_first'.format(curr_model_to_compare['alias'])] = 'NONE'

                row_to_load['concept_{}'.format(curr_model_to_compare['alias'])] = \
                    curr_pred.get('{}_concept_id'.format(curr_model_to_compare['alias']))

                row_to_load['{}_pred_type'.format(curr_model_to_compare['alias'])] = \
                    curr_pred['{}_pred_type'.format(curr_model_to_compare['alias'])]

                row_to_load['{}_link'.format(curr_model_to_compare['alias'])] = \
                    curr_pred.get('{}_link'.format(curr_model_to_compare['alias']))

                row_to_load['{}_coref_span_type'.format(curr_model_to_compare['alias'])] = \
                    curr_pred.get('{}_coref_span_type'.format(curr_model_to_compare['alias']))

                row_to_load['{}_coref_span_start'.format(curr_model_to_compare['alias'])] = \
                    curr_pred.get('{}_coref_span_start'.format(curr_model_to_compare['alias']))
                row_to_load['{}_coref_span_end'.format(curr_model_to_compare['alias'])] = \
                    curr_pred.get('{}_coref_span_end'.format(curr_model_to_compare['alias']))

                row_to_load['{}_coref_score'.format(curr_model_to_compare['alias'])] = \
                    curr_pred.get('{}_coref_score'.format(curr_model_to_compare['alias']))

                row_to_load['{}_coref_scores'.format(curr_model_to_compare['alias'])] = \
                    curr_pred.get('{}_coref_scores'.format(curr_model_to_compare['alias']))

                row_to_load['{}_link_scores'.format(curr_model_to_compare['alias'])] = \
                    curr_pred.get('{}_link_scores'.format(curr_model_to_compare['alias']))

                row_to_load['{}_candidates'.format(curr_model_to_compare['alias'])] = \
                    curr_pred.get('{}_candidates'.format(curr_model_to_compare['alias']))

                if curr_model_to_compare['alias'] not in edit_distances_mention_other:
                    edit_distances_mention_other[curr_model_to_compare['alias']] = list()

                if curr_model_to_compare['alias'] not in edit_distances_link:
                    edit_distances_link[curr_model_to_compare['alias']] = list()

                if curr_model_to_compare['alias'] not in edit_distances_not_defined:
                    edit_distances_not_defined[curr_model_to_compare['alias']] = list()

                if curr_model_to_compare['alias'] not in edit_distances_not_defined_nf:
                    edit_distances_not_defined_nf[curr_model_to_compare['alias']] = list()

                if curr_model_to_compare['alias'] not in edit_distances_mention_other_nf:
                    edit_distances_mention_other_nf[curr_model_to_compare['alias']] = list()

                if curr_model_to_compare['alias'] not in edit_distances_link_nf:
                    edit_distances_link_nf[curr_model_to_compare['alias']] = list()

                if 'tp' in curr_pred['{}_pred_type'.format(curr_model_to_compare['alias'])]:
                    if curr_pred['{}_connection_type'.format(curr_model_to_compare['alias'])] == 'mention_other':
                        edit_distances_mention_other[curr_model_to_compare['alias']].append(
                            nltk.edit_distance(row_to_load['text'], curr_pred['link_gold']))
                    elif curr_pred['{}_connection_type'.format(curr_model_to_compare['alias'])] == 'link':
                        if 'link_gold' not in curr_pred:
                            print('something weird is happening...')
                        edit_distances_link[curr_model_to_compare['alias']] \
                            .append(nltk.edit_distance(row_to_load['text'], curr_pred['link_gold']))
                    elif curr_pred['{}_connection_type'.format(curr_model_to_compare['alias'])] == '--NOT DEFINED--':
                        edit_distances_not_defined[curr_model_to_compare['alias']] \
                            .append(nltk.edit_distance(row_to_load['text'], curr_pred['link_gold']))

                    if row_to_load['{}_connection_type_not_first'.format(curr_model_to_compare['alias'])] == \
                            'mention_other':
                        edit_distances_mention_other_nf[curr_model_to_compare['alias']] \
                            .append(nltk.edit_distance(row_to_load['text'], curr_pred['link_gold']))
                    elif row_to_load['{}_connection_type_not_first'.format(curr_model_to_compare['alias'])] == 'link':
                        edit_distances_link_nf[curr_model_to_compare['alias']] \
                            .append(nltk.edit_distance(row_to_load['text'], curr_pred['link_gold']))
                    elif row_to_load['{}_connection_type_not_first'.format(curr_model_to_compare['alias'])] == \
                            '--NOT DEFINED--':
                        edit_distances_not_defined_nf[curr_model_to_compare['alias']] \
                            .append(nltk.edit_distance(row_to_load['text'], curr_pred['link_gold']))

                if row_to_load['{}_mention'.format(curr_model_to_compare['alias'])] and row_to_load['gold_mention']:
                    already_seen_coreflinker_clusters[curr_model_to_compare['alias']] \
                        .add(curr_pred['concept']['concept'])

            span_list.append(row_to_load)

    df_data = pd.DataFrame(span_list)

    df_tp_per_model_type = dict()

    for curr_model_to_compare in models_to_compare:
        df_tp_per_model_type[curr_model_to_compare['alias']] = \
            df_data[df_data['{}_pred_type'.format(curr_model_to_compare['alias']).format()] == 'tp']

    stat001_connection_type_effect_on_others()

    ########## BEGIN stat002 on edit distance comparison
    stat002_edit_distance_comparison(df_tp_per_model_type=df_tp_per_model_type, model_types=models_to_compare)
    ########## END stat002 on edit distance comparison

    print('df data shape: ', df_data.shape)
    df_linkable = df_data[df_data['linkable'] == True]

    print('df linkable shape: ', df_linkable.shape)
    print('df data shape: ', df_data.shape)
    df_coref_linker_solvable = df_linkable[df_linkable['coreflinker_solvable'] == True]
    df_coref_linker_not_solvable = df_linkable[df_linkable['coreflinker_solvable'] == False]

    print('df solvable shape: ', df_coref_linker_solvable.shape)
    print('df not solvable shape: ', df_coref_linker_not_solvable.shape)

    print(
        '=======COMPARISON ON HOW MODELS PERFORM (MENTION-BASED) on the coreflinker not solvable '
        '(1st mention in cluster without correct link in candidate list): ')
    for curr_model_to_compare in models_to_compare:
        # print('pred types for not solvable for ', curr_model_to_compare['alias'], ': ')
        get_result_statistics(predictor=curr_model_to_compare['alias'], predictions=df_coref_linker_not_solvable)

    for curr_model_to_compare in models_to_compare:
        curr_model_name = curr_model_to_compare['alias']
        df_curr_pred_types = df_data[df_data['linkable'] == False].groupby(['{}_pred_type'.format(curr_model_name)])[
            '{}_link'.format(curr_model_name), 'text'].count()
        print('pred types for not linkable df_data {}: ')
        print(df_curr_pred_types)

    print(
        '=======COMPARISON ON HOW MODELS PERFORM (MENTION-BASED) on the mentions without correct candidate but with '
        'correct candidate somewhere in cluster ')

    df_mentions_not_solvable_cluster_solvable = df_linkable[(df_linkable['correct_link_in_cluster'] == True) &
                                                            (df_linkable['has_correct_link'] == False)]
    for curr_model_to_compare in models_to_compare:
        get_result_statistics(predictor=curr_model_to_compare['alias'],
                              predictions=df_mentions_not_solvable_cluster_solvable)

    print(
        '=======COMPARISON ON HOW MODELS PERFORM (MENTION-BASED) on the mentions without correct candidate but with '
        'correct candidate somewhere in cluster and still no correct link seen in the mentions')

    df_cluster_solvable_link_not_seen = df_linkable[(df_linkable['correct_link_in_cluster'] == True) &
                                                    (df_linkable['correct_link_seen'] == False)]
    for curr_model_to_compare in models_to_compare:
        get_result_statistics(predictor=curr_model_to_compare['alias'],
                              predictions=df_cluster_solvable_link_not_seen)

    print(
        '=======COMPARISON ON HOW MODELS PERFORM (MENTION-BASED) on the mentions without correct candidate')
    df_mentions_not_solvable = df_linkable[df_linkable['has_correct_link'] == False]
    for curr_model_to_compare in models_to_compare:
        get_result_statistics(predictor=curr_model_to_compare['alias'], predictions=df_mentions_not_solvable)

    ## WIP: Johannes idea: count the number of links assigned to gold clusters by baseline linker and coreflinker
    cluster_sizes_link = {'cluster_size': [], 'nr_links': [], 'method': [], 'fraction': []}
    for nr_mentions_in_cluster in range(1, 11):
        data = {'links_per_cluster': []}
        df_links_per_cluster = pd.DataFrame(data)
        model_names = []

        for curr_model_to_compare in models_to_compare:
            curr_model_name = curr_model_to_compare['alias']
            model_names.append(curr_model_name)
            print('processing model: ', curr_model_name)
            print('len of df_linkable with all nulls for curr model: ', df_linkable.shape)
            df_curr_model_linkable = df_linkable[df_linkable['{}_link'.format(curr_model_name)].notnull()]
            print('len of df_linkable without  nulls for curr model: ', df_curr_model_linkable.shape)

            cntd_mentions = df_curr_model_linkable[['doc_id', 'link_gold', 'span_start']].groupby(
                ['doc_id', 'link_gold']).count()

            cntd_mentions = cntd_mentions.reset_index()

            cntd_mentions_2 = cntd_mentions[cntd_mentions['span_start'] >= nr_mentions_in_cluster]

            df_curr_model_linkable_j = pd.merge(cntd_mentions_2, df_curr_model_linkable,
                                                how='inner', left_on=['doc_id', 'link_gold'],
                                                right_on=['doc_id', 'link_gold'])
            print('after filtering by mention number (2): ', df_curr_model_linkable_j.shape)
            gp_by_gold_link = df_curr_model_linkable_j.groupby(['doc_id', 'link_gold'])[
                '{}_link'.format(curr_model_name)] \
                .nunique(dropna=False)

            fracts = gp_by_gold_link.reset_index()['{}_link'.format(curr_model_name)].value_counts(normalize=True)
            tmp_curr_df = pd.DataFrame(fracts).reset_index().rename(
                columns={'{}_link'.format(curr_model_name): curr_model_name, 'index': 'links_per_cluster'})
            df_links_per_cluster = pd.merge(df_links_per_cluster, tmp_curr_df, how='right',
                                            left_on=['links_per_cluster'], right_on=['links_per_cluster'])

            for nr_links_per_cluster in range(1, 5):
                if tmp_curr_df[tmp_curr_df['links_per_cluster'] == nr_links_per_cluster].shape[0] > 0:
                    frac_item = tmp_curr_df[tmp_curr_df['links_per_cluster'] == nr_links_per_cluster][
                        curr_model_name].item()
                    cluster_sizes_link['cluster_size'].append(nr_mentions_in_cluster)
                    cluster_sizes_link['nr_links'].append(nr_links_per_cluster)
                    cluster_sizes_link['method'].append(curr_model_name)
                    cluster_sizes_link['fraction'].append(frac_item)
                else:
                    cluster_sizes_link['cluster_size'].append(nr_mentions_in_cluster)
                    cluster_sizes_link['nr_links'].append(nr_links_per_cluster)
                    cluster_sizes_link['method'].append(curr_model_name)
                    cluster_sizes_link['fraction'].append(0.0)

            print('average of nr of links: ', gp_by_gold_link.mean())
            print('max of nr of links: ', gp_by_gold_link.max())
            print('std of nr of links: ', gp_by_gold_link.std())

        df_links_per_cluster = df_links_per_cluster.fillna(0.0)

    exit()
    df_cluster_sizes_link = pd.DataFrame(cluster_sizes_link)

    df_cluster_sizes_link1 = df_cluster_sizes_link[df_cluster_sizes_link['nr_links'] == 1]
    df_cluster_sizes_link1 = df_cluster_sizes_link1[['method', 'cluster_size', 'fraction']]
    df_piv = df_cluster_sizes_link1.pivot(index='cluster_size', columns='method', values='fraction').reset_index()
    print('df_cluster_sizes_link: ', df_cluster_sizes_link)
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    bars = df_piv.plot(ax=ax, x='cluster_size',
                       # color=color, label='',
                       title='Ratio of 1 Links per Gold Cluster')
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    ax.set_xlabel('>= Mentions in cluster')
    ax.set_ylabel('Fraction of clusters with 4 assigned links')

    plt.show()

    exit()

    print('pred types for ALL df_data coreflinker: ')
    get_result_statistics(predictor='coreflinker', predictions=df_data)
    print('pred types for ALL df_data baseline: ')
    get_result_statistics(predictor='baseline_linker', predictions=df_data)

    # without accounting or NILL
    df_nr_links_per_cluster = df_data.groupby(['doc_id', 'concept_gold'])[
        'baseline_linker_link'].nunique(dropna=False).reset_index()

    print('total nr of clusters: ', df_nr_links_per_cluster.shape)
    df_clusters_with_multilinks = df_nr_links_per_cluster[df_nr_links_per_cluster['baseline_linker_link'] > 1]
    print('nr of multilink clusters for baseline (wrt gold clusters): ', df_clusters_with_multilinks.shape)

    print('nr of multilink clusters for coreflinker (wrt gold clusters): ', df_clusters_with_multilinks.shape)

    df_pred_joined = pd.merge(df_clusters_with_multilinks, df_data, how='inner', on=['doc_id', 'concept_gold'])
    print('====STATISTICS FOR CASES WHERE THERE ARE MORE THAN ONE LINK PREDICTED PER CLUSTER IN BASELINE=====')
    get_result_statistics(predictor='coreflinker', predictions=df_pred_joined)
    get_result_statistics(predictor='baseline_linker', predictions=df_pred_joined)

    # IGNORING NILS NILL
    df_nr_links_per_cluster = \
        df_data[(df_data['baseline_linker_link'] != 'NILL') & (df_data['baseline_linker_link'].notnull())] \
            .groupby(['doc_id', 'concept_gold'])['baseline_linker_link'].nunique(dropna=True).reset_index()

    print('total nr of clusters: ', df_nr_links_per_cluster.shape)
    df_clusters_with_multilinks = df_nr_links_per_cluster[df_nr_links_per_cluster['baseline_linker_link'] > 1]

    df_pred_joined = pd.merge(df_clusters_with_multilinks, df_data, how='inner', on=['doc_id', 'concept_gold'])
    print('====STATISTICS FOR CASES WHERE THERE ARE MORE THAN ONE LINK PREDICTED PER CLUSTER IN '
          'BASELINE (IGNORING NILLS)=====')
    get_result_statistics(predictor='coreflinker', predictions=df_pred_joined)
    get_result_statistics(predictor='baseline_linker', predictions=df_pred_joined)

    df_nr_links_per_cluster = df_data.groupby(['doc_id', 'concept_gold'])[
        'coreflinker_link'].nunique(dropna=False).reset_index()

    print('total nr of clusters: ', df_nr_links_per_cluster.shape)
    df_clusters_with_multilinks = df_nr_links_per_cluster[df_nr_links_per_cluster['coreflinker_link'] > 1]
    print('nr of multilink clusters for coreflinker (wrt gold clusters): ', df_clusters_with_multilinks.shape)
    df_pred_joined = pd.merge(df_clusters_with_multilinks, df_data, how='inner', on=['doc_id', 'concept_gold'])

    print('====STATISTICS FOR CASES WHERE THERE ARE MORE THAN ONE LINK PREDICTED PER CLUSTER IN '
          'COREFLINKER (NOT IGNORING NILLS)=====')
    get_result_statistics(predictor='coreflinker', predictions=df_pred_joined)
    get_result_statistics(predictor='baseline_linker', predictions=df_pred_joined)

    df_nr_links_per_cluster = df_data[df_data['coreflinker_link'] != 'NILL'].groupby(['doc_id', 'concept_gold'])[
        'coreflinker_link'].nunique(dropna=True).reset_index()

    print('total nr of clusters: ', df_nr_links_per_cluster.shape)
    df_clusters_with_multilinks = df_nr_links_per_cluster[df_nr_links_per_cluster['coreflinker_link'] > 1]
    print('nr of multilink clusters for coreflinker (wrt gold clusters): ', df_clusters_with_multilinks.shape)
    df_pred_joined = pd.merge(df_clusters_with_multilinks, df_data, how='inner', on=['doc_id', 'concept_gold'])

    print('====STATISTICS FOR CASES WHERE THERE ARE MORE THAN ONE LINK PREDICTED PER CLUSTER IN '
          'COREFLINKER (IGNORING NILLS)=====')
    get_result_statistics(predictor='coreflinker', predictions=df_pred_joined)
    get_result_statistics(predictor='baseline_linker', predictions=df_pred_joined)

    df_solvable_coreflinker_pred_types = df_coref_linker_solvable.groupby(['coreflinker_pred_type'])[
        'coreflinker_link', 'text'].count()
    df_solvable_baseline_pred_types = df_coref_linker_solvable.groupby(['baseline_linker_pred_type'])[
        'baseline_linker_link', 'text'].count()

    print('====pred types for solvable and linkable for coreflinker (coreflinker): ')
    get_result_statistics(predictor='coreflinker', predictions=df_coref_linker_solvable)
    print('====pred types for solvable and linkable for coreflinker (baseline): ')
    get_result_statistics(predictor='baseline_linker', predictions=df_coref_linker_solvable)

    print('============intersected on linkable and solvable==============')
    get_nr_intersected(df_coref_linker_solvable)

    print('============intersected on ALL the data==============')
    get_nr_intersected(df_data)

    print('============intersected on ALL the data (printed v2)==============')
    get_nr_intersected_v2(df_data, ['baseline_linker', 'coreflinker'], ['tp', 'tn', 'fpfn', 'fp', 'fn'])

    print('the end!')

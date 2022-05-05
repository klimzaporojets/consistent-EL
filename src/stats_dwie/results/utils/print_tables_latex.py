import math
from math import isclose
from typing import List, Dict


def is_bert(config_json):
    if 'bert_embedder' in config_json['model']['text_embedder']:
        return True
    return False


def print_comment_config(config_json, experiment_id, nr_runs):
    att = False
    att_prop = 0
    if 'spanprop' in config_json['model']:
        att = True
        att_prop = config_json['model']['spanprop']['att_prop']
    is_it_bert = False

    if 'bert_embedder' in config_json['model']['text_embedder']:
        is_it_bert = True
    # print('printing comment for ', experiment_id)
    to_ret = 'id: {}, runs:{}, att: {}, att_prop: {}, tag: {}, coref: {}, coref prop: {}, rels: {}, rels prop: {}, bert: {}' \
        .format(experiment_id, nr_runs, att, att_prop,
                config_json['model']['ner']['enabled'],
                config_json['model']['coref']['enabled'],
                (config_json['model']['corefprop']['coref_prop']
                 if 'coref_prop' in config_json['model']['corefprop'] else 'No'),
                config_json['model']['relations']['enabled'],
                (config_json['model']['relprop']['rel_prop']
                 if 'rel_prop' in config_json['model']['relprop'] else 'No'),
                is_it_bert)
    return to_ret


def print_comment_config_multiple(config_json, experiment_id, nr_runs):
    to_ret = ''
    if 'joint' in config_json:
        to_ret += '% Joint: ' + print_comment_config(config_json['joint'], experiment_id['joint'],
                                                     nr_runs['joint']) + '\n'
    if 'ner' in config_json:
        to_ret += '% NER: ' + print_comment_config(config_json['ner'], experiment_id['ner'], nr_runs['ner']) + '\n'
    if 'coref' in config_json:
        to_ret += '% Coref: ' + print_comment_config(config_json['coref'], experiment_id['coref'],
                                                     nr_runs['coref']) + '\n'
    if 'rels' in config_json:
        to_ret += '% Rels: ' + print_comment_config(config_json['rels'], experiment_id['rels'], nr_runs['rels']) + '\n'

    return to_ret


def get_res_or_nothing(result, task_activated=True, best_res=None, best_res_all=None,
                       color='default'):
    # to_ret = ''
    if result[1] > -1 and task_activated:
        if best_res is None:
            to_ret = '{:.1f}'.format(result[0] * 100)
        else:
            if isclose(result[0], best_res):
                to_ret = '\\textbf{{{:.1f}}}'.format(result[0] * 100)
                if best_res_all is not None and isclose(result[0], best_res_all):
                    to_ret = '\\underline{{{}}}'.format(to_ret)
            else:
                to_ret = '{:.1f}'.format(result[0] * 100)
    else:
        to_ret = '-'

    if color != 'default':
        to_ret = '\\leavevmode\\color{' + color + '}' + to_ret

    return to_ret


def get_res(result, best_res):
    if isclose(result[0], best_res):
        to_ret = '\\textbf{{{:.1f}}}${{\\scriptstyle \\pm\\textbf{{{:.1f}}}}}$' \
            .format(result[0] * 100, result[1] * 100)
    else:
        to_ret = '{:.1f}${{\\scriptstyle \\pm\\text{{{:.1f}}}}}$'.format(result[0] * 100, result[1] * 100)
    return to_ret


def print_latex_table(results_and_model_types: List, only_f1=True, metric_types=['mention', 'soft', 'hard']):
    """

    :param results_and_model_types:
    :param only_f1:
    :param metric_types:
    :return:

    """

    best_ner_mention_f1 = 0.0
    best_ner_soft_f1 = 0.0
    best_ner_hard_f1 = 0.0
    best_rel_mention_f1 = 0.0
    best_rel_soft_f1 = 0.0
    best_rel_hard_f1 = 0.0
    best_coref_muc_f1 = 0.0
    best_coref_bcubed_f1 = 0.0
    best_coref_ceafe_f1 = 0.0
    best_coref_avg_f1 = 0.0

    best_bert_ner_mention_f1 = 0.0
    best_bert_ner_soft_f1 = 0.0
    best_bert_ner_hard_f1 = 0.0
    best_bert_rel_mention_f1 = 0.0
    best_bert_rel_soft_f1 = 0.0
    best_bert_rel_hard_f1 = 0.0
    best_bert_coref_muc_f1 = 0.0
    best_bert_coref_bcubed_f1 = 0.0
    best_bert_coref_ceafe_f1 = 0.0
    best_bert_coref_avg_f1 = 0.0

    for curr_result in results_and_model_types:
        if curr_result['results'] is None:
            continue
        is_joint = 'joint' in curr_result['setup'].lower()
        if not is_bert(curr_result['results']['experiment_config']):
            if curr_result['results']['experiment_results']['relations']['rels_mention']['f1'][0] > best_rel_mention_f1:
                if not is_joint:
                    best_rel_mention_f1 = \
                        curr_result['results']['experiment_results']['relations']['rels_mention']['f1'][0]
                else:
                    # for now just takes the expanded if it is joint
                    best_rel_mention_f1 = \
                        curr_result['results']['experiment_results']['relations']['rels_mention_expanded']['f1'][0]

            if curr_result['results']['experiment_results']['relations']['rels_soft']['f1'][0] > best_rel_soft_f1:
                best_rel_soft_f1 = curr_result['results']['experiment_results']['relations']['rels_soft']['f1'][0]

            if curr_result['results']['experiment_results']['relations']['rels_hard']['f1'][0] > best_rel_hard_f1:
                best_rel_hard_f1 = curr_result['results']['experiment_results']['relations']['rels_hard']['f1'][0]

            if curr_result['results']['experiment_results']['coref']['ceafe_singleton']['f1'][0] > best_coref_ceafe_f1:
                best_coref_ceafe_f1 = curr_result['results']['experiment_results']['coref']['ceafe_singleton']['f1'][0]

            if curr_result['results']['experiment_results']['coref']['b_cubed_singleton_men_conll']['f1'][
                0] > best_coref_bcubed_f1:
                best_coref_bcubed_f1 = \
                    curr_result['results']['experiment_results']['coref']['b_cubed_singleton_men_conll']['f1'][0]

            if curr_result['results']['experiment_results']['coref']['muc']['f1'][0] > best_coref_muc_f1:
                best_coref_muc_f1 = curr_result['results']['experiment_results']['coref']['muc']['f1'][0]

            if curr_result['results']['experiment_results']['tags']['tags_mention']['f1'][0] > best_ner_mention_f1:
                if not is_joint:
                    best_ner_mention_f1 = curr_result['results']['experiment_results']['tags']['tags_mention']['f1'][0]
                else:
                    best_ner_mention_f1 = \
                        curr_result['results']['experiment_results']['tags']['tags_mention_expanded']['f1'][0]

            if curr_result['results']['experiment_results']['tags']['tags_soft']['f1'][0] > best_ner_soft_f1:
                best_ner_soft_f1 = curr_result['results']['experiment_results']['tags']['tags_soft']['f1'][0]

            if curr_result['results']['experiment_results']['tags']['tags_hard']['f1'][0] > best_ner_hard_f1:
                best_ner_hard_f1 = curr_result['results']['experiment_results']['tags']['tags_hard']['f1'][0]

            avg_coref = (curr_result['results']['experiment_results']['coref']['muc']['f1'][0] +
                         curr_result['results']['experiment_results']['coref']['b_cubed_singleton_men_conll']['f1'][0] +
                         curr_result['results']['experiment_results']['coref']['ceafe_singleton']['f1'][0]
                         ) / 3
            if avg_coref > best_coref_avg_f1:
                best_coref_avg_f1 = avg_coref
        else:
            if curr_result['results']['experiment_results']['relations']['rels_mention']['f1'][0] > \
                    best_bert_rel_mention_f1:
                if not is_joint:
                    best_bert_rel_mention_f1 = \
                        curr_result['results']['experiment_results']['relations']['rels_mention']['f1'][0]
                else:
                    best_bert_rel_mention_f1 = \
                        curr_result['results']['experiment_results']['relations']['rels_mention_expanded']['f1'][0]

            if curr_result['results']['experiment_results']['relations']['rels_soft']['f1'][0] > best_bert_rel_soft_f1:
                best_bert_rel_soft_f1 = curr_result['results']['experiment_results']['relations']['rels_soft']['f1'][0]

            if curr_result['results']['experiment_results']['relations']['rels_hard']['f1'][0] > best_bert_rel_hard_f1:
                best_bert_rel_hard_f1 = curr_result['results']['experiment_results']['relations']['rels_hard']['f1'][0]

            if curr_result['results']['experiment_results']['coref']['ceafe_singleton']['f1'][
                0] > best_bert_coref_ceafe_f1:
                best_bert_coref_ceafe_f1 = \
                    curr_result['results']['experiment_results']['coref']['ceafe_singleton']['f1'][0]

            if curr_result['results']['experiment_results']['coref']['b_cubed_singleton_men_conll']['f1'][
                0] > best_bert_coref_bcubed_f1:
                best_bert_coref_bcubed_f1 = \
                    curr_result['results']['experiment_results']['coref']['b_cubed_singleton_men_conll']['f1'][0]

            if curr_result['results']['experiment_results']['coref']['muc']['f1'][0] > best_bert_coref_muc_f1:
                best_bert_coref_muc_f1 = curr_result['results']['experiment_results']['coref']['muc']['f1'][0]

            if curr_result['results']['experiment_results']['tags']['tags_mention']['f1'][0] > best_bert_ner_mention_f1:
                if not is_joint:
                    best_bert_ner_mention_f1 = \
                        curr_result['results']['experiment_results']['tags']['tags_mention']['f1'][0]
                else:
                    best_bert_ner_mention_f1 = \
                        curr_result['results']['experiment_results']['tags']['tags_mention_expanded']['f1'][0]

            if curr_result['results']['experiment_results']['tags']['tags_soft']['f1'][0] > best_bert_ner_soft_f1:
                best_bert_ner_soft_f1 = curr_result['results']['experiment_results']['tags']['tags_soft']['f1'][0]

            if curr_result['results']['experiment_results']['tags']['tags_hard']['f1'][0] > best_bert_ner_hard_f1:
                best_bert_ner_hard_f1 = curr_result['results']['experiment_results']['tags']['tags_hard']['f1'][0]

            avg_coref = (curr_result['results']['experiment_results']['coref']['muc']['f1'][0] +
                         curr_result['results']['experiment_results']['coref']['b_cubed_singleton_men_conll']['f1'][0] +
                         curr_result['results']['experiment_results']['coref']['ceafe_singleton']['f1'][0]
                         ) / 3
            if avg_coref > best_bert_coref_avg_f1:
                best_bert_coref_avg_f1 = avg_coref

    latex_content = '\\begin{table}[!ht]\n'
    latex_content += '\t\\centering\n'
    latex_content += '\t\\setlength{\\tabcolsep}{4pt}\n'
    latex_content += '\t\\renewcommand{\\arraystretch}{1.0}\n'
    latex_content += '\t\\resizebox{1.0\\textwidth}{!}{\n'
    latex_content += '\t\\begin{tabular}{l cccc | ccc | ccc}\n'
    latex_content += '\t\\toprule\n'
    latex_content += '\t&'
    latex_content += '\t\multicolumn{4}{c}{\\textbf{Coreference F1}} & ' \
                     '\multicolumn{3}{c}{\\textbf{NER F1}} & ' \
                     '\multicolumn{3}{c}{\\textbf{Relation F1}} \\\\ \n'
    latex_content += '\cmidrule(lr){2-5} \cmidrule(lr){6-8} \cmidrule(lr){9-11}\n'
    latex_content += '\t \\textbf{Model Setup} & \n'
    latex_content += '\t \\makebox[2.6em]{\\textbf{MUC}} & \n'
    latex_content += '\t \\makebox[2.6em]{\\textbf{CEAF}$\mathbf{_{e}}$} & \n'
    latex_content += '\t \\makebox[2.6em]{\\textbf{B}$\mathbf{^3}$} & \n'
    latex_content += '\t \\makebox[2.6em]{\\textbf{AVG}} & \n'
    latex_content += '\t \\makebox[2.6em]{\\textbf{M-F}$\mathbf{_1}$} & \n'
    latex_content += '\t \\makebox[2.6em]{\\textbf{H-F}$\mathbf{_1}$} & \n'
    latex_content += '\t \\makebox[2.6em]{\\textbf{S-F}$\mathbf{_1}$} & \n'
    latex_content += '\t \\makebox[2.6em]{\\textbf{M-F}$\mathbf{_1}$} & \n'
    latex_content += '\t \\makebox[2.6em]{\\textbf{H-F}$\mathbf{_1}$} & \n'
    latex_content += '\t \\makebox[2.6em]{\\textbf{S-F}$\mathbf{_1}$} \\\\ \n'
    latex_content += '\t \\midrule \n'
    previous_bert = False

    for curr_result in results_and_model_types:

        if curr_result['results'] is None:
            latex_content += '\t{} & \\leavevmode\\color{{red}}? & ' \
                             '\\leavevmode\\color{{red}}? & ' \
                             '\\leavevmode\\color{{red}}? & ' \
                             '\\leavevmode\\color{{red}}? & ' \
                             '\\leavevmode\\color{{red}}? & ' \
                             '\\leavevmode\\color{{red}}? & ' \
                             '\\leavevmode\\color{{red}}? & ' \
                             '\\leavevmode\\color{{red}}? & ' \
                             '\\leavevmode\\color{{red}}? & ' \
                             '\\leavevmode\\color{{red}}? \\\\ \n' \
                .format(curr_result['setup'])
        else:
            curr_col_ner_mention_f1 = 'default'
            curr_col_ner_soft_f1 = 'default'
            curr_col_ner_hard_f1 = 'default'
            curr_col_rel_mention_f1 = 'default'
            curr_col_rel_soft_f1 = 'default'
            curr_col_rel_hard_f1 = 'default'
            curr_col_coref_muc_f1 = 'default'
            curr_col_coref_ceaf_f1 = 'default'
            curr_col_coref_bcuded_f1 = 'default'
            curr_col_coref_avg_f1 = 'default'
            if curr_result['setup'] == 'Coref':
                # TODO: assign color to coref scores in red
                curr_col_coref_avg_f1 = 'default'
                curr_col_coref_ceaf_f1 = 'default'
                curr_col_coref_muc_f1 = 'default'
                curr_col_coref_bcuded_f1 = 'default'
            elif curr_result['setup'] == 'Coref+NER':
                curr_col_coref_avg_f1 = 'red'
                curr_col_coref_ceaf_f1 = 'red'
                curr_col_coref_muc_f1 = 'red'
                curr_col_coref_bcuded_f1 = 'red'
                curr_col_ner_mention_f1 = 'red'
                curr_col_ner_soft_f1 = 'red'
                curr_col_ner_hard_f1 = 'red'
                curr_result['results']['experiment_config']['model']['ner']['enabled'] = True

            is_joint = False
            if 'joint' in curr_result['setup'].lower():
                # TODO: assign color to mention-based scores in orange
                # curr_col_ner_mention_f1 = 'orange'
                curr_col_rel_mention_f1 = 'orange'
                is_joint = True
                # pass

            is_it_bert = is_bert(curr_result['results']['experiment_config'])
            if is_it_bert and not previous_bert:
                previous_bert = is_it_bert
                latex_content += '\t \\cmidrule{2-11}\n'

            avg_coref = (curr_result['results']['experiment_results']['coref']['muc']['f1'][0] +
                         curr_result['results']['experiment_results']['coref']['b_cubed_singleton_men_conll']['f1'][0] +
                         curr_result['results']['experiment_results']['coref']['ceafe_singleton']['f1'][0]
                         ) / 3
            latex_content += '\t{} & {} & {} & {} & {} & {} & {} & {} & {} & {} & {} \\\\ \n % {} \n ' \
                .format(curr_result['setup'],
                        get_res_or_nothing(curr_result['results']['experiment_results']['coref']['muc']['f1'],
                                           curr_result['results']['experiment_config']['model']['coref']['enabled'],
                                           best_res=(best_bert_coref_muc_f1 if is_it_bert else best_coref_muc_f1),
                                           color=curr_col_coref_muc_f1),
                        get_res_or_nothing(
                            curr_result['results']['experiment_results']['coref']['b_cubed_singleton_men_conll']['f1'],
                            curr_result['results']['experiment_config']['model']['coref']['enabled'],
                            best_res=(best_bert_coref_bcubed_f1 if is_it_bert else best_coref_bcubed_f1),
                            color=curr_col_coref_bcuded_f1),
                        get_res_or_nothing(
                            curr_result['results']['experiment_results']['coref']['ceafe_singleton']['f1'],
                            curr_result['results']['experiment_config']['model']['coref']['enabled'],
                            best_res=(best_bert_coref_ceafe_f1 if is_it_bert else best_coref_ceafe_f1),
                            color=curr_col_coref_ceaf_f1),
                        get_res_or_nothing((avg_coref, 0.0),
                                           curr_result['results']['experiment_config']['model']['coref']['enabled'],
                                           best_res=(best_bert_coref_avg_f1 if is_it_bert else best_coref_avg_f1),
                                           color=curr_col_coref_avg_f1),

                        get_res_or_nothing(
                            curr_result['results']['experiment_results']['tags']['tags_mention_expanded']['f1']
                            if is_joint else
                            curr_result['results']['experiment_results']['tags']['tags_mention']['f1'],
                            curr_result['results']['experiment_config']['model']['ner']['enabled'],
                            best_res=(best_bert_ner_mention_f1 if is_it_bert else best_ner_mention_f1),
                            color=curr_col_ner_mention_f1),
                        get_res_or_nothing(curr_result['results']['experiment_results']['tags']['tags_hard']['f1'],
                                           curr_result['results']['experiment_config']['model']['ner']['enabled'],
                                           best_res=(best_bert_ner_hard_f1 if is_it_bert else best_ner_hard_f1),
                                           color=curr_col_ner_hard_f1),
                        get_res_or_nothing(curr_result['results']['experiment_results']['tags']['tags_soft']['f1'],
                                           curr_result['results']['experiment_config']['model']['ner']['enabled'],
                                           best_res=(best_bert_ner_soft_f1 if is_it_bert else best_ner_soft_f1),
                                           color=curr_col_ner_soft_f1),
                        get_res_or_nothing(
                            curr_result['results']['experiment_results']['relations']['rels_mention_expanded']['f1']
                            if is_joint else
                            curr_result['results']['experiment_results']['relations']['rels_mention']['f1'],
                            curr_result['results']['experiment_config']['model']['relations']['enabled'],
                            best_res=(best_bert_rel_mention_f1 if is_it_bert else best_rel_mention_f1),
                            color=curr_col_rel_mention_f1),
                        get_res_or_nothing(curr_result['results']['experiment_results']['relations']['rels_hard']['f1'],
                                           curr_result['results']['experiment_config']['model']['relations']['enabled'],
                                           best_res=(best_bert_rel_hard_f1 if is_it_bert else best_rel_hard_f1),
                                           color=curr_col_rel_hard_f1),
                        get_res_or_nothing(curr_result['results']['experiment_results']['relations']['rels_soft']['f1'],
                                           curr_result['results']['experiment_config']['model']['relations'][
                                               'enabled'],
                                           best_res=(best_bert_rel_soft_f1 if is_it_bert else best_rel_soft_f1),
                                           color=curr_col_rel_soft_f1),
                        print_comment_config(curr_result['results']['experiment_config'],
                                             curr_result['results']['experiment_id'],
                                             curr_result['results']['experiment_results']['nr_runs'])
                        )

    latex_content += '\t\\bottomrule \n'
    latex_content += '\t\\end{tabular}\n'
    latex_content += '\t}\n'
    latex_content += '\t\\caption{Main Results}\n'
    latex_content += '\\label{tab:main_results}\n'
    latex_content += '\\end{table}'
    # print('results and model types: ', results_and_model_types)
    print('latex final results table: ')
    print(latex_content)


def get_any_experiment_config(experiment_config: Dict):
    if 'joint' in experiment_config:
        return experiment_config['joint']
    elif 'ner' in experiment_config:
        return experiment_config['ner']
    elif 'coref' in experiment_config:
        return experiment_config['coref']
    elif 'rels' in experiment_config:
        return experiment_config['rels']
    return dict()


def print_analysis_delta_table_1(data_delta: Dict):
    latex_content = '\\begin{table}[!t]\n'
    latex_content += '\\centering\n'
    latex_content += '\\caption{Analysis.}\n'
    latex_content += '\\resizebox{0.6\\textwidth}{!}{\n'
    latex_content += '\\begin{tabular}{l cc c cc}\n'
    latex_content += '\\toprule\n'
    latex_content += '\\hspace*{2cm} & \n'
    latex_content += '\\multicolumn{2}{c}{\\textbf{Joint}} & & \n'
    latex_content += '\\multicolumn{2}{c}{\\textbf{Joint+BERT}} \\\\ \n'
    latex_content += '\\cmidrule(lr){2-3} \\cmidrule(lr){5-6} \n'
    latex_content += '& \n'
    latex_content += '\\makebox[3em]{\\textbf{NER}} & \n'
    latex_content += '\\makebox[3em]{\\textbf{Rels}} & \\makebox[0.2em]{} & \n'
    latex_content += '\\makebox[3em]{\\textbf{NER}} & \\makebox[3em]{\\textbf{Rels}} \\\\ \n'
    latex_content += '\\midrule \n'
    latex_content += '$\\Delta\\mathrm{{\\ }}$\\propformat{{AttProp}} & {:.2f} & {:.2f} &  & {:.2f} & {:.2f} \\\\ \n' \
        .format(data_delta['att_prop']['delta_joint_ner'],
                data_delta['att_prop']['delta_joint_rel'],
                data_delta['att_prop']['delta_bert_ner'],
                data_delta['att_prop']['delta_bert_rel'])
    latex_content += '$\\Delta\\mathrm{{\\ }}$\\propformat{{CorefProp}} & {:.2f} & {:.2f} &  & {:.2f} & {:.2f} \\\\ \n' \
        .format(data_delta['coref_prop']['delta_joint_ner'],
                data_delta['coref_prop']['delta_joint_rel'],
                data_delta['coref_prop']['delta_bert_ner'],
                data_delta['coref_prop']['delta_bert_rel'])
    latex_content += '$\\Delta\\mathrm{{\\ }}$\\propformat{{RelProp}} & {:.2f} & {:.2f} &  & {:.2f} & {:.2f} \\\\ \n' \
        .format(data_delta['rel_prop']['delta_joint_ner'],
                data_delta['rel_prop']['delta_joint_rel'],
                data_delta['rel_prop']['delta_bert_ner'],
                data_delta['rel_prop']['delta_bert_rel'])

    latex_content += '\\bottomrule\n'
    latex_content += '\\end{tabular}}\n'
    latex_content += '\\label{tab:analysis_deltas_props}\n'
    latex_content += '\\end{table}'
    print(latex_content)


def print_analysis_delta_table_2_ner(results_joint_base: Dict, results_joint_att_prop: Dict,
                                     results_joint_coref_prop: Dict,
                                     results_joint_rel_prop: Dict,
                                     results_joint_bert_base: Dict, results_joint_bert_att_prop: Dict,
                                     results_joint_bert_coref_prop: Dict, results_joint_bert_rel_prop: Dict):
    latex_content = '\\begin{table}[!t]\n'
    latex_content += '\\centering\n'
    latex_content += '\\caption{Analysis NER Deltas.}\n'
    latex_content += '\\resizebox{0.6\\textwidth}{!}{\n'
    latex_content += '\\begin{tabular}{l ccc c ccc}\n'
    latex_content += '\\toprule\n'
    latex_content += '\\hspace*{2cm} & \n'
    latex_content += '\\multicolumn{3}{c}{\\textbf{Joint}} & & \n'
    latex_content += '\\multicolumn{3}{c}{\\textbf{Joint+BERT}} \\\\ \n'
    latex_content += '\\cmidrule(lr){2-4} \\cmidrule(lr){6-8} \n'
    latex_content += '& \n'
    latex_content += '\\makebox[3em]{\\textbf{M-F1}} & \n'
    latex_content += '\\makebox[3em]{\\textbf{H-F1}} & ' \
                     '\\makebox[3em]{\\textbf{S-F1}} & ' \
                     '\\makebox[0.2em]{} & \n'
    latex_content += '\\makebox[3em]{\\textbf{M-F1}} & ' \
                     '\\makebox[3em]{\\textbf{H-F1}} & ' \
                     '\\makebox[3em]{\\textbf{S-F1}} \\\\ \n'
    latex_content += '\\midrule \n'
    latex_content += '$\\Delta\\mathrm{{\\ }}$\\propformat{{AttProp}} & {:.2f} & {:.2f} & {:.2f} &  & {:.2f} & {:.2f} & {:.2f} \\\\ \n' \
        .format(results_joint_att_prop['tags']['tags_mention']['f1'][0] * 100 -
                results_joint_base['tags']['tags_mention']['f1'][0] * 100,
                results_joint_att_prop['tags']['tags_hard']['f1'][0] * 100 -
                results_joint_base['tags']['tags_hard']['f1'][0] * 100,
                results_joint_att_prop['tags']['tags_soft']['f1'][0] * 100 -
                results_joint_base['tags']['tags_soft']['f1'][0] * 100,
                results_joint_bert_att_prop['tags']['tags_mention']['f1'][0] * 100 -
                results_joint_bert_base['tags']['tags_mention']['f1'][0] * 100,
                results_joint_bert_att_prop['tags']['tags_hard']['f1'][0] * 100 -
                results_joint_bert_base['tags']['tags_hard']['f1'][0] * 100,
                results_joint_bert_att_prop['tags']['tags_soft']['f1'][0] * 100 -
                results_joint_bert_base['tags']['tags_soft']['f1'][0] * 100
                )
    latex_content += '$\\Delta\\mathrm{{\\ }}$\\propformat{{CorefProp}} & {:.2f} & {:.2f} & {:.2f} &  & {:.2f} & {:.2f} & {:.2f} \\\\ \n' \
        .format(results_joint_coref_prop['tags']['tags_mention']['f1'][0] * 100 -
                results_joint_base['tags']['tags_mention']['f1'][0] * 100,
                results_joint_coref_prop['tags']['tags_hard']['f1'][0] * 100 -
                results_joint_base['tags']['tags_hard']['f1'][0] * 100,
                results_joint_coref_prop['tags']['tags_soft']['f1'][0] * 100 -
                results_joint_base['tags']['tags_soft']['f1'][0] * 100,
                results_joint_bert_coref_prop['tags']['tags_mention']['f1'][0] * 100 -
                results_joint_bert_base['tags']['tags_mention']['f1'][0] * 100,
                results_joint_bert_coref_prop['tags']['tags_hard']['f1'][0] * 100 -
                results_joint_bert_base['tags']['tags_hard']['f1'][0] * 100,
                results_joint_bert_coref_prop['tags']['tags_soft']['f1'][0] * 100 -
                results_joint_bert_base['tags']['tags_soft']['f1'][0] * 100
                )
    latex_content += '$\\Delta\\mathrm{{\\ }}$\\propformat{{RelProp}} & {:.2f} & {:.2f} & {:.2f} &  & {:.2f} & {:.2f} & {:.2f} \\\\ \n' \
        .format(results_joint_rel_prop['tags']['tags_mention']['f1'][0] * 100 -
                results_joint_base['tags']['tags_mention']['f1'][0] * 100,
                results_joint_rel_prop['tags']['tags_hard']['f1'][0] * 100 -
                results_joint_base['tags']['tags_hard']['f1'][0] * 100,
                results_joint_rel_prop['tags']['tags_soft']['f1'][0] * 100 -
                results_joint_base['tags']['tags_soft']['f1'][0] * 100,
                results_joint_bert_rel_prop['tags']['tags_mention']['f1'][0] * 100 -
                results_joint_bert_base['tags']['tags_mention']['f1'][0] * 100,
                results_joint_bert_rel_prop['tags']['tags_hard']['f1'][0] * 100 -
                results_joint_bert_base['tags']['tags_hard']['f1'][0] * 100,
                results_joint_bert_rel_prop['tags']['tags_soft']['f1'][0] * 100 -
                results_joint_bert_base['tags']['tags_soft']['f1'][0] * 100
                )

    latex_content += '\\bottomrule\n'
    latex_content += '\\end{tabular}}\n'
    latex_content += '\\label{tab:analysis_deltas_props_ner}\n'
    latex_content += '\\end{table}'
    print(latex_content)


def print_analysis_delta_table_2_rel(results_joint_base: Dict, results_joint_att_prop: Dict,
                                     results_joint_coref_prop: Dict,
                                     results_joint_rel_prop: Dict,
                                     results_joint_bert_base: Dict, results_joint_bert_att_prop: Dict,
                                     results_joint_bert_coref_prop: Dict, results_joint_bert_rel_prop: Dict):
    latex_content = '\\begin{table}[!t]\n'
    latex_content += '\\centering\n'
    latex_content += '\\caption{Analysis Rel Deltas.}\n'
    latex_content += '\\resizebox{0.6\\textwidth}{!}{\n'
    latex_content += '\\begin{tabular}{l ccc c ccc}\n'
    latex_content += '\\toprule\n'
    latex_content += '\\hspace*{2cm} & \n'
    latex_content += '\\multicolumn{3}{c}{\\textbf{Joint}} & & \n'
    latex_content += '\\multicolumn{3}{c}{\\textbf{Joint+BERT}} \\\\ \n'
    latex_content += '\\cmidrule(lr){2-4} \\cmidrule(lr){6-8} \n'
    latex_content += '& \n'
    latex_content += '\\makebox[3em]{\\textbf{M-F1}} & \n'
    latex_content += '\\makebox[3em]{\\textbf{H-F1}} & ' \
                     '\\makebox[3em]{\\textbf{S-F1}} & ' \
                     '\\makebox[0.2em]{} & \n'
    latex_content += '\\makebox[3em]{\\textbf{M-F1}} & ' \
                     '\\makebox[3em]{\\textbf{H-F1}} & ' \
                     '\\makebox[3em]{\\textbf{S-F1}} \\\\ \n'
    latex_content += '\\midrule \n'
    latex_content += '$\\Delta\\mathrm{{\\ }}$\\propformat{{AttProp}} & {:.2f} & {:.2f} & {:.2f} &  & {:.2f} & {:.2f} & {:.2f} \\\\ \n' \
        .format(results_joint_att_prop['relations']['rels_mention']['f1'][0] * 100 -
                results_joint_base['relations']['rels_mention']['f1'][0] * 100,
                results_joint_att_prop['relations']['rels_hard']['f1'][0] * 100 -
                results_joint_base['relations']['rels_hard']['f1'][0] * 100,
                results_joint_att_prop['relations']['rels_soft']['f1'][0] * 100 -
                results_joint_base['relations']['rels_soft']['f1'][0] * 100,
                results_joint_bert_att_prop['relations']['rels_mention']['f1'][0] * 100 -
                results_joint_bert_base['relations']['rels_mention']['f1'][0] * 100,
                results_joint_bert_att_prop['relations']['rels_hard']['f1'][0] * 100 -
                results_joint_bert_base['relations']['rels_hard']['f1'][0] * 100,
                results_joint_bert_att_prop['relations']['rels_soft']['f1'][0] * 100 -
                results_joint_bert_base['relations']['rels_soft']['f1'][0] * 100
                )
    latex_content += '$\\Delta\\mathrm{{\\ }}$\\propformat{{CorefProp}} & {:.2f} & {:.2f} & {:.2f} &  & {:.2f} & {:.2f} & {:.2f} \\\\ \n' \
        .format(results_joint_coref_prop['relations']['rels_mention']['f1'][0] * 100 -
                results_joint_base['relations']['rels_mention']['f1'][0] * 100,
                results_joint_coref_prop['relations']['rels_hard']['f1'][0] * 100 -
                results_joint_base['relations']['rels_hard']['f1'][0] * 100,
                results_joint_coref_prop['relations']['rels_soft']['f1'][0] * 100 -
                results_joint_base['relations']['rels_soft']['f1'][0] * 100,
                results_joint_bert_coref_prop['relations']['rels_mention']['f1'][0] * 100 -
                results_joint_bert_base['relations']['rels_mention']['f1'][0] * 100,
                results_joint_bert_coref_prop['relations']['rels_hard']['f1'][0] * 100 -
                results_joint_bert_base['relations']['rels_hard']['f1'][0] * 100,
                results_joint_bert_coref_prop['relations']['rels_soft']['f1'][0] * 100 -
                results_joint_bert_base['relations']['rels_soft']['f1'][0] * 100
                )
    latex_content += '$\\Delta\\mathrm{{\\ }}$\\propformat{{RelProp}} & {:.2f} & {:.2f} & {:.2f} &  & {:.2f} & {:.2f} & {:.2f} \\\\ \n' \
        .format(results_joint_rel_prop['relations']['rels_mention']['f1'][0] * 100 -
                results_joint_base['relations']['rels_mention']['f1'][0] * 100,
                results_joint_rel_prop['relations']['rels_hard']['f1'][0] * 100 -
                results_joint_base['relations']['rels_hard']['f1'][0] * 100,
                results_joint_rel_prop['relations']['rels_soft']['f1'][0] * 100 -
                results_joint_base['relations']['rels_soft']['f1'][0] * 100,
                results_joint_bert_rel_prop['relations']['rels_mention']['f1'][0] * 100 -
                results_joint_bert_base['relations']['rels_mention']['f1'][0] * 100,
                results_joint_bert_rel_prop['relations']['rels_hard']['f1'][0] * 100 -
                results_joint_bert_base['relations']['rels_hard']['f1'][0] * 100,
                results_joint_bert_rel_prop['relations']['rels_soft']['f1'][0] * 100 -
                results_joint_bert_base['relations']['rels_soft']['f1'][0] * 100
                )

    latex_content += '\\bottomrule\n'
    latex_content += '\\end{tabular}}\n'
    latex_content += '\\label{tab:analysis_deltas_props_rel}\n'
    latex_content += '\\end{table}'
    print(latex_content)


def print_analysis_delta_table_3_both(results_joint_base: Dict, results_joint_att_prop: Dict,
                                      results_joint_coref_prop: Dict,
                                      results_joint_rel_prop: Dict,
                                      results_joint_bert_base: Dict, results_joint_bert_att_prop: Dict,
                                      results_joint_bert_coref_prop: Dict, results_joint_bert_rel_prop: Dict):
    # table based on Chris' improvement in merging two delta tables into one containing deltas for NER and RE in
    # a single table
    """

    :param esults_joint_base:
    :param results_joint_att_prop:
    :param results_joint_coref_prop:
    :param results_joint_rel_prop:
    :param results_joint_bert_base:
    :param results_joint_bert_att_prop:
    :param results_joint_bert_coref_prop:
    :param results_joint_bert_rel_prop:
    :return:
    """

    latex_content = '\\begin{table}\n'
    latex_content += '\\centering\n'
    latex_content += '\\caption{Deltas of improvement in performance for each of the graph propagation methods ' \
                     '(\\propformat{AttProp}, \\propformat{CorefProp}, \\propformat{RelProp}) in $\\mathrm{F_1}$ ' \
                     'scores for \\textbf{(a)}~NER and \\textbf{(b)}~relation extraction tasks.}\n'
    latex_content += '\\label{tab:analysis_deltas_props}\n'
    latex_content += '\\begin{tabular}{c l ccc c ccc}\n'
    latex_content += '\\toprule\n'
    latex_content += '& & \n'
    latex_content += '\\multicolumn{3}{c}{\\textbf{Joint}} & & \n'
    latex_content += '\\multicolumn{3}{c}{\\textbf{Joint+BERT}} \\\\ \n'
    latex_content += '\\cmidrule(lr){3-5} \\cmidrule(lr){7-9} \n'
    latex_content += '& & \n'
    latex_content += '$\\mathbf{F_{1,m}}$ & \n'
    latex_content += '$\\mathbf{F_{1,h}}$ & \n'
    latex_content += '$\\mathbf{F_{1,s}}$ & \n'
    latex_content += '& \n'
    latex_content += '$\\mathbf{F_{1,m}}$ & \n'
    latex_content += '$\\mathbf{F_{1,h}}$ & \n'
    latex_content += '$\\mathbf{F_{1,s}}$ \\\\ \n'
    latex_content += '\\midrule \n'
    latex_content += '\\multirow{3}{*}{\\textbf{(a)~NER}} & \n'
    latex_content += '$\\Delta\\mathrm{{\\ }}$\\propformat{{AttProp}} & {:.2f} & {:.2f} & {:.2f} &  & ' \
                     '{:.2f} & {:.2f} & {:.2f} \\\\ \n' \
        .format(results_joint_att_prop['tags']['tags_mention']['f1'][0] * 100 -
                results_joint_base['tags']['tags_mention']['f1'][0] * 100,
                results_joint_att_prop['tags']['tags_hard']['f1'][0] * 100 -
                results_joint_base['tags']['tags_hard']['f1'][0] * 100,
                results_joint_att_prop['tags']['tags_soft']['f1'][0] * 100 -
                results_joint_base['tags']['tags_soft']['f1'][0] * 100,
                results_joint_bert_att_prop['tags']['tags_mention']['f1'][0] * 100 -
                results_joint_bert_base['tags']['tags_mention']['f1'][0] * 100,
                results_joint_bert_att_prop['tags']['tags_hard']['f1'][0] * 100 -
                results_joint_bert_base['tags']['tags_hard']['f1'][0] * 100,
                results_joint_bert_att_prop['tags']['tags_soft']['f1'][0] * 100 -
                results_joint_bert_base['tags']['tags_soft']['f1'][0] * 100
                )
    latex_content += '& $\\Delta\\mathrm{{\\ }}$\\propformat{{CorefProp}} & {:.2f} & {:.2f} & {:.2f} &  & ' \
                     '{:.2f} & {:.2f} & {:.2f} \\\\ \n' \
        .format(results_joint_coref_prop['tags']['tags_mention']['f1'][0] * 100 -
                results_joint_base['tags']['tags_mention']['f1'][0] * 100,
                results_joint_coref_prop['tags']['tags_hard']['f1'][0] * 100 -
                results_joint_base['tags']['tags_hard']['f1'][0] * 100,
                results_joint_coref_prop['tags']['tags_soft']['f1'][0] * 100 -
                results_joint_base['tags']['tags_soft']['f1'][0] * 100,
                results_joint_bert_coref_prop['tags']['tags_mention']['f1'][0] * 100 -
                results_joint_bert_base['tags']['tags_mention']['f1'][0] * 100,
                results_joint_bert_coref_prop['tags']['tags_hard']['f1'][0] * 100 -
                results_joint_bert_base['tags']['tags_hard']['f1'][0] * 100,
                results_joint_bert_coref_prop['tags']['tags_soft']['f1'][0] * 100 -
                results_joint_bert_base['tags']['tags_soft']['f1'][0] * 100
                )
    latex_content += '& $\\Delta\\mathrm{{\\ }}$\\propformat{{RelProp}} & {:.2f} & {:.2f} & {:.2f} &  & ' \
                     '{:.2f} & {:.2f} & {:.2f} \\\\ \n' \
        .format(results_joint_rel_prop['tags']['tags_mention']['f1'][0] * 100 -
                results_joint_base['tags']['tags_mention']['f1'][0] * 100,
                results_joint_rel_prop['tags']['tags_hard']['f1'][0] * 100 -
                results_joint_base['tags']['tags_hard']['f1'][0] * 100,
                results_joint_rel_prop['tags']['tags_soft']['f1'][0] * 100 -
                results_joint_base['tags']['tags_soft']['f1'][0] * 100,
                results_joint_bert_rel_prop['tags']['tags_mention']['f1'][0] * 100 -
                results_joint_bert_base['tags']['tags_mention']['f1'][0] * 100,
                results_joint_bert_rel_prop['tags']['tags_hard']['f1'][0] * 100 -
                results_joint_bert_base['tags']['tags_hard']['f1'][0] * 100,
                results_joint_bert_rel_prop['tags']['tags_soft']['f1'][0] * 100 -
                results_joint_bert_base['tags']['tags_soft']['f1'][0] * 100
                )
    latex_content += '\\midrule \n'
    latex_content += '\\multirow{3}{*}{\\textbf{(a)~NER}} & \n'
    latex_content += '$\\Delta\\mathrm{{\\ }}$\\propformat{{AttProp}} & {:.2f} & {:.2f} & {:.2f} &  & ' \
                     '{:.2f} & {:.2f} & {:.2f} \\\\ \n' \
        .format(results_joint_att_prop['relations']['rels_mention']['f1'][0] * 100 -
                results_joint_base['relations']['rels_mention']['f1'][0] * 100,
                results_joint_att_prop['relations']['rels_hard']['f1'][0] * 100 -
                results_joint_base['relations']['rels_hard']['f1'][0] * 100,
                results_joint_att_prop['relations']['rels_soft']['f1'][0] * 100 -
                results_joint_base['relations']['rels_soft']['f1'][0] * 100,
                results_joint_bert_att_prop['relations']['rels_mention']['f1'][0] * 100 -
                results_joint_bert_base['relations']['rels_mention']['f1'][0] * 100,
                results_joint_bert_att_prop['relations']['rels_hard']['f1'][0] * 100 -
                results_joint_bert_base['relations']['rels_hard']['f1'][0] * 100,
                results_joint_bert_att_prop['relations']['rels_soft']['f1'][0] * 100 -
                results_joint_bert_base['relations']['rels_soft']['f1'][0] * 100)
    latex_content += '& $\\Delta\\mathrm{{\\ }}$\\propformat{{CorefProp}} & {:.2f} & {:.2f} & {:.2f} &  & ' \
                     '{:.2f} & {:.2f} & {:.2f} \\\\ \n' \
        .format(results_joint_coref_prop['relations']['rels_mention']['f1'][0] * 100 -
                results_joint_base['relations']['rels_mention']['f1'][0] * 100,
                results_joint_coref_prop['relations']['rels_hard']['f1'][0] * 100 -
                results_joint_base['relations']['rels_hard']['f1'][0] * 100,
                results_joint_coref_prop['relations']['rels_soft']['f1'][0] * 100 -
                results_joint_base['relations']['rels_soft']['f1'][0] * 100,
                results_joint_bert_coref_prop['relations']['rels_mention']['f1'][0] * 100 -
                results_joint_bert_base['relations']['rels_mention']['f1'][0] * 100,
                results_joint_bert_coref_prop['relations']['rels_hard']['f1'][0] * 100 -
                results_joint_bert_base['relations']['rels_hard']['f1'][0] * 100,
                results_joint_bert_coref_prop['relations']['rels_soft']['f1'][0] * 100 -
                results_joint_bert_base['relations']['rels_soft']['f1'][0] * 100
                )
    latex_content += '& $\\Delta\\mathrm{{\\ }}$\\propformat{{RelProp}} & {:.2f} & {:.2f} & {:.2f} &  & ' \
                     '{:.2f} & {:.2f} & {:.2f} \\\\ \n' \
        .format(results_joint_rel_prop['relations']['rels_mention']['f1'][0] * 100 -
                results_joint_base['relations']['rels_mention']['f1'][0] * 100,
                results_joint_rel_prop['relations']['rels_hard']['f1'][0] * 100 -
                results_joint_base['relations']['rels_hard']['f1'][0] * 100,
                results_joint_rel_prop['relations']['rels_soft']['f1'][0] * 100 -
                results_joint_base['relations']['rels_soft']['f1'][0] * 100,
                results_joint_bert_rel_prop['relations']['rels_mention']['f1'][0] * 100 -
                results_joint_bert_base['relations']['rels_mention']['f1'][0] * 100,
                results_joint_bert_rel_prop['relations']['rels_hard']['f1'][0] * 100 -
                results_joint_bert_base['relations']['rels_hard']['f1'][0] * 100,
                results_joint_bert_rel_prop['relations']['rels_soft']['f1'][0] * 100 -
                results_joint_bert_base['relations']['rels_soft']['f1'][0] * 100
                )
    latex_content += '\\bottomrule \n'
    latex_content += '\\end{tabular} \n'
    latex_content += '\\end{table}'

    print(latex_content)


def print_latex_table_layout2(results_and_model_types: List, is_ner_expanded=False, is_rels_expanded=False):
    """

    :param results_and_model_types:
    :param only_f1:
    :param metric_types:
    :return:

    """
    rels_mention_metric = 'rels_mention' if not is_rels_expanded else 'rels_mention_expanded'
    ner_mention_metric = 'tags_mention' if not is_ner_expanded else 'tags_mention_expanded'

    best_single_ner_mention_f1 = 0.0
    best_single_ner_soft_f1 = 0.0
    best_single_ner_hard_f1 = 0.0
    best_single_rel_mention_f1 = 0.0
    best_single_rel_soft_f1 = 0.0
    best_single_rel_hard_f1 = 0.0
    best_single_coref_muc_f1 = 0.0
    best_single_coref_bcubed_f1 = 0.0
    best_single_coref_ceafe_f1 = 0.0
    best_single_coref_avg_f1 = 0.0

    best_joint_ner_mention_f1 = 0.0
    best_joint_ner_soft_f1 = 0.0
    best_joint_ner_hard_f1 = 0.0
    best_joint_rel_mention_f1 = 0.0
    best_joint_rel_soft_f1 = 0.0
    best_joint_rel_hard_f1 = 0.0
    best_joint_coref_muc_f1 = 0.0
    best_joint_coref_bcubed_f1 = 0.0
    best_joint_coref_ceafe_f1 = 0.0
    best_joint_coref_avg_f1 = 0.0

    best_joint_bert_ner_mention_f1 = 0.0
    best_joint_bert_ner_soft_f1 = 0.0
    best_joint_bert_ner_hard_f1 = 0.0
    best_joint_bert_rel_mention_f1 = 0.0
    best_joint_bert_rel_soft_f1 = 0.0
    best_joint_bert_rel_hard_f1 = 0.0
    best_joint_bert_coref_muc_f1 = 0.0
    best_joint_bert_coref_bcubed_f1 = 0.0
    best_joint_bert_coref_ceafe_f1 = 0.0
    best_joint_bert_coref_avg_f1 = 0.0

    # best_all_ner_mention_f1 = 0.0
    # best_all_ner_soft_f1 = 0.0
    # best_all_ner_hard_f1 = 0.0
    # best_all_rel_mention_f1 = 0.0
    # best_all_rel_soft_f1 = 0.0
    # best_all_rel_hard_f1 = 0.0
    # best_all_coref_muc_f1 = 0.0
    # best_all_coref_bcubed_f1 = 0.0
    # best_all_coref_ceafe_f1 = 0.0
    # best_all_coref_avg_f1 = 0.0

    for curr_result in results_and_model_types:
        experiment_config = dict()
        if curr_result['type'] == 'single':
            if 'results_ner' in curr_result:
                res_ner = curr_result['results_ner']['experiment_results']['tags']
                experiment_config['ner'] = curr_result['results_ner']['experiment_config']
            else:
                res_ner = None
            if 'results_coref' in curr_result:
                res_coref = curr_result['results_coref']['experiment_results']['coref']
                experiment_config['coref'] = curr_result['results_coref']['experiment_config']
            else:
                res_coref = None

            if 'results_rel' in curr_result:
                res_relations = curr_result['results_rel']['experiment_results']['relations']
                experiment_config['rels'] = curr_result['results_rel']['experiment_config']
            else:
                res_relations = None

        else:
            res_ner = curr_result['results']['experiment_results']['tags']
            res_coref = curr_result['results']['experiment_results']['coref']
            res_relations = curr_result['results']['experiment_results']['relations']
            experiment_config['joint'] = curr_result['results']['experiment_config']

        is_joint = 'joint' in curr_result['type'].lower()
        any_exp_config = get_any_experiment_config(experiment_config)
        if not is_bert(any_exp_config):
            if res_relations is not None:
                if not is_joint:
                    best_single_rel_mention_f1 = max(best_single_rel_mention_f1,
                                                     res_relations[rels_mention_metric]['f1'][0])
                    best_single_rel_soft_f1 = max(best_single_rel_soft_f1, res_relations['rels_soft']['f1'][0])
                    best_single_rel_hard_f1 = max(best_single_rel_hard_f1, res_relations['rels_hard']['f1'][0])
                else:
                    best_joint_rel_mention_f1 = max(best_joint_rel_mention_f1,
                                                    res_relations[rels_mention_metric]['f1'][0])
                    best_joint_rel_soft_f1 = max(best_joint_rel_soft_f1, res_relations['rels_soft']['f1'][0])
                    best_joint_rel_hard_f1 = max(best_joint_rel_hard_f1, res_relations['rels_hard']['f1'][0])

            if res_coref is not None:
                if not is_joint:
                    best_single_coref_ceafe_f1 = max(res_coref['ceafe_singleton']['f1'][0], best_single_coref_ceafe_f1)
                    best_single_coref_bcubed_f1 = max(res_coref['b_cubed_singleton_men_conll']['f1'][0],
                                                      best_single_coref_bcubed_f1)
                    best_single_coref_muc_f1 = max(best_single_coref_muc_f1, res_coref['muc']['f1'][0])
                else:
                    best_joint_coref_ceafe_f1 = max(res_coref['ceafe_singleton']['f1'][0], best_joint_coref_ceafe_f1)
                    best_joint_coref_bcubed_f1 = max(res_coref['b_cubed_singleton_men_conll']['f1'][0],
                                                     best_joint_coref_bcubed_f1)
                    best_joint_coref_muc_f1 = max(best_joint_coref_muc_f1, res_coref['muc']['f1'][0])

            if res_ner is not None:
                if not is_joint:
                    best_single_ner_mention_f1 = max(best_single_ner_mention_f1, res_ner[ner_mention_metric]['f1'][0])
                    best_single_ner_soft_f1 = max(best_single_ner_soft_f1, res_ner['tags_soft']['f1'][0])
                    best_single_ner_hard_f1 = max(best_single_ner_hard_f1, res_ner['tags_hard']['f1'][0])
                else:
                    best_joint_ner_mention_f1 = max(best_joint_ner_mention_f1, res_ner[ner_mention_metric]['f1'][0])
                    best_joint_ner_soft_f1 = max(best_joint_ner_soft_f1, res_ner['tags_soft']['f1'][0])
                    best_joint_ner_hard_f1 = max(best_joint_ner_hard_f1, res_ner['tags_hard']['f1'][0])

            if res_coref is not None:
                avg_coref = (res_coref['muc']['f1'][0] + res_coref['b_cubed_singleton_men_conll']['f1'][0] +
                             res_coref['ceafe_singleton']['f1'][0]) / 3
                if not is_joint:
                    best_single_coref_avg_f1 = max(best_single_coref_avg_f1, avg_coref)
                else:
                    best_joint_coref_avg_f1 = max(best_joint_coref_avg_f1, avg_coref)
        else:
            if res_relations is not None:
                best_joint_bert_rel_mention_f1 = max(best_joint_bert_rel_mention_f1,
                                                     res_relations[rels_mention_metric]['f1'][0])
                best_joint_bert_rel_soft_f1 = max(best_joint_bert_rel_soft_f1, res_relations['rels_soft']['f1'][0])
                best_joint_bert_rel_hard_f1 = max(best_joint_bert_rel_hard_f1, res_relations['rels_hard']['f1'][0])

            if res_coref is not None:
                best_joint_bert_coref_ceafe_f1 = max(res_coref['ceafe_singleton']['f1'][0],
                                                     best_joint_bert_coref_ceafe_f1)
                best_joint_bert_coref_bcubed_f1 = max(res_coref['b_cubed_singleton_men_conll']['f1'][0],
                                                      best_joint_bert_coref_bcubed_f1)
                best_joint_bert_coref_muc_f1 = max(best_joint_bert_coref_muc_f1, res_coref['muc']['f1'][0])

            if res_ner is not None:
                best_joint_bert_ner_mention_f1 = max(best_joint_bert_ner_mention_f1,
                                                     res_ner[ner_mention_metric]['f1'][0])
                best_joint_bert_ner_soft_f1 = max(best_joint_bert_ner_soft_f1, res_ner['tags_soft']['f1'][0])
                best_joint_bert_ner_hard_f1 = max(best_joint_bert_ner_hard_f1, res_ner['tags_hard']['f1'][0])

            if res_coref is not None:
                avg_coref = (res_coref['muc']['f1'][0] + res_coref['b_cubed_singleton_men_conll']['f1'][0] +
                             res_coref['ceafe_singleton']['f1'][0]) / 3
                best_joint_bert_coref_avg_f1 = max(best_joint_bert_coref_avg_f1, avg_coref)

    best_all_coref_avg_f1 = max(best_joint_bert_coref_avg_f1, best_joint_coref_avg_f1, best_single_coref_avg_f1)
    best_all_coref_ceafe_f1 = max(best_joint_bert_coref_ceafe_f1, best_joint_coref_ceafe_f1, best_single_coref_ceafe_f1)
    best_all_coref_muc_f1 = max(best_joint_bert_coref_muc_f1, best_joint_coref_muc_f1, best_single_coref_muc_f1)
    best_all_coref_bcubed_f1 = max(best_joint_bert_coref_bcubed_f1, best_joint_coref_bcubed_f1,
                                   best_single_coref_bcubed_f1)

    best_all_rel_hard_f1 = max(best_joint_bert_rel_hard_f1, best_joint_rel_hard_f1, best_single_rel_hard_f1)
    best_all_rel_soft_f1 = max(best_joint_bert_rel_soft_f1, best_joint_rel_soft_f1, best_single_rel_soft_f1)
    best_all_rel_mention_f1 = max(best_joint_bert_rel_mention_f1, best_joint_rel_mention_f1, best_single_rel_mention_f1)

    best_all_ner_hard_f1 = max(best_joint_bert_ner_hard_f1, best_joint_ner_hard_f1, best_single_ner_hard_f1)
    best_all_ner_soft_f1 = max(best_joint_bert_ner_soft_f1, best_joint_ner_soft_f1, best_single_ner_soft_f1)
    best_all_ner_mention_f1 = max(best_joint_bert_ner_mention_f1, best_joint_ner_mention_f1, best_single_ner_mention_f1)

    # latex_content = '\\begin{table}[!ht]\n'
    latex_content = '\\begin{table}\n'
    latex_content += '\t\\centering\n'
    latex_content += '\\caption[test]{\\klimtext{Main results of the experiments grouped in three model setups: ' \
                     '\\begin{enumerate*}[(i)]\n'
    latex_content += '\t\\item \\textit{Single} models trained individually,\n'
    latex_content += '\t\\item \\textit{Joint} model trained using as input GloVe and character embeddings, and \n'
    latex_content += '\t\\item \\textit{Joint+BERT} model trained on BERT$_{\\mathrm{BASE}}$ embeddings.\n'
    latex_content += '\t\\end{enumerate*}. \n' \
                     '\t To report the results, we use MUC, CEAF$_\\text{e}$, B$^\\text{3}$ as well as the average ' \
                     '(Avg.) of these three metrics for \\textit{coreference resolution}. For NER and RE we use \n' \
                     '\t mention-level (F$_\\text{1,m}$), hard entity-level (F$_\\text{1,h}$), and soft entity-level ' \
                     '(F$_\\text{1,s}$) metrics described in \\secref{sec:metrics}. In bold we mark the best results ' \
                     'for each model setup, the best overall results are underlined.}} \n'
    latex_content += '\t\\label{tab:main_results}\n'
    latex_content += '\t\\setlength{\\tabcolsep}{4pt}\n'
    latex_content += '\t\\renewcommand{\\arraystretch}{1.0}\n'
    # latex_content += '\t\\resizebox{1.0\\textwidth}{!}{\n'
    # latex_content += '\t\\begin{tabular}{l cccc | ccc | ccc}\n'
    latex_content += '\t\\begin{tabular}{l cccc c ccc c ccc}\n'
    latex_content += '\t\\toprule\n'
    latex_content += '\t&'
    latex_content += '\t\multicolumn{4}{c}{\\textbf{Coreference} $\\mathbf{F_1}$} && ' \
                     '\multicolumn{3}{c}{\\textbf{NER} $\\mathbf{F_1}$} && ' \
                     '\multicolumn{3}{c}{\\textbf{Relation} $\\mathbf{F_1}$} \\\\ \n'
    # latex_content += '\\cmidrule(lr){2-5} \\cmidrule(lr){6-8} \\cmidrule(lr){9-11}\n'
    latex_content += '\\cmidrule(lr){2-5} \\cmidrule(lr){7-9} \\cmidrule(lr){11-13}\n'
    latex_content += '\t \\textbf{Model Setup} & \n'
    latex_content += '\t \\textbf{MUC} & \n'
    latex_content += '\t \\textbf{CEAF}$\\mathbf{_{e}}$ & \n'
    latex_content += '\t $\\mathbf{B^3}$ & \n'
    latex_content += '\t \\textbf{Avg.} & \n'
    latex_content += '\t \\hspace{.5em} & \n'
    latex_content += '\t $\\mathbf{F_{1,m}}$ & \n'
    latex_content += '\t $\\mathbf{F_{1,h}}$ & \n'
    latex_content += '\t $\\mathbf{F_{1,s}}$ & \n'
    latex_content += '\t \\hspace{.5em} & \n'
    latex_content += '\t $\\mathbf{F_{1,m}}$ & \n'
    latex_content += '\t $\\mathbf{F_{1,h}}$ & \n'
    latex_content += '\t $\\mathbf{F_{1,s}}$ \\\\ \n'
    # latex_content += '\t \\makebox[2.6em]{\\textbf{MUC}} & \n'
    # latex_content += '\t \\makebox[2.6em]{\\textbf{CEAF}$\mathbf{_{e}}$} & \n'
    # latex_content += '\t \\makebox[2.6em]{\\textbf{B}$\mathbf{^3}$} & \n'
    # latex_content += '\t \\makebox[2.6em]{\\textbf{Avg.}} & \n'
    # latex_content += '\t \\makebox[2.6em]{\\textbf{M-F}$\mathbf{_1}$} & \n'
    # latex_content += '\t \\makebox[2.6em]{\\textbf{H-F}$\mathbf{_1}$} & \n'
    # latex_content += '\t \\makebox[2.6em]{\\textbf{S-F}$\mathbf{_1}$} & \n'
    # latex_content += '\t \\makebox[2.6em]{\\textbf{M-F}$\mathbf{_1}$} & \n'
    # latex_content += '\t \\makebox[2.6em]{\\textbf{H-F}$\mathbf{_1}$} & \n'
    # latex_content += '\t \\makebox[2.6em]{\\textbf{S-F}$\mathbf{_1}$} \\\\ \n'
    latex_content += '\t \\midrule \n'
    previous_bert = False
    previous_joint = False

    for curr_result in results_and_model_types:

        curr_col_ner_mention_f1 = 'default'
        curr_col_ner_soft_f1 = 'default'
        curr_col_ner_hard_f1 = 'default'
        curr_col_rel_mention_f1 = 'default'
        curr_col_rel_soft_f1 = 'default'
        curr_col_rel_hard_f1 = 'default'
        curr_col_coref_muc_f1 = 'default'
        curr_col_coref_ceaf_f1 = 'default'
        curr_col_coref_bcuded_f1 = 'default'
        curr_col_coref_avg_f1 = 'default'

        nr_runs = dict()
        experiment_id = dict()
        experiment_config = dict()

        best_res_ner_mention_f1 = 0.0
        best_res_ner_soft_f1 = 0.0
        best_res_ner_hard_f1 = 0.0
        best_res_rel_mention_f1 = 0.0
        best_res_rel_soft_f1 = 0.0
        best_res_rel_hard_f1 = 0.0
        best_res_coref_muc_f1 = 0.0
        best_res_coref_bcubed_f1 = 0.0
        best_res_coref_ceafe_f1 = 0.0
        best_res_coref_avg_f1 = 0.0

        if curr_result['type'] == 'single':
            if 'results_ner' in curr_result:
                res_ner = curr_result['results_ner']['experiment_results']['tags']
                experiment_config['ner'] = curr_result['results_ner']['experiment_config']
                nr_runs['ner'] = curr_result['results_ner']['experiment_results']['nr_runs']
                experiment_id['ner'] = curr_result['results_ner']['experiment_id']
            else:
                res_ner = None
            if 'results_coref' in curr_result:
                res_coref = curr_result['results_coref']['experiment_results']['coref']
                experiment_config['coref'] = curr_result['results_coref']['experiment_config']
                nr_runs['coref'] = curr_result['results_coref']['experiment_results']['nr_runs']
                experiment_id['coref'] = curr_result['results_coref']['experiment_id']
            else:
                res_coref = None

            if 'results_rel' in curr_result:
                res_relations = curr_result['results_rel']['experiment_results']['relations']
                experiment_config['rels'] = curr_result['results_rel']['experiment_config']
                nr_runs['rels'] = curr_result['results_rel']['experiment_results']['nr_runs']
                experiment_id['rels'] = curr_result['results_rel']['experiment_id']
            else:
                res_relations = None

        else:
            res_ner = curr_result['results']['experiment_results']['tags']
            res_coref = curr_result['results']['experiment_results']['coref']
            res_relations = curr_result['results']['experiment_results']['relations']
            experiment_config['joint'] = curr_result['results']['experiment_config']
            nr_runs['joint'] = curr_result['results']['experiment_results']['nr_runs']
            experiment_id['joint'] = curr_result['results']['experiment_id']

        any_exp_config = get_any_experiment_config(experiment_config)
        is_it_bert = is_bert(any_exp_config)

        is_joint = False
        if curr_result['type'] == 'joint':
            is_joint = True
        if not is_joint:
            best_res_coref_avg_f1 = best_single_coref_avg_f1
            best_res_ner_mention_f1 = best_single_ner_mention_f1
            best_res_ner_soft_f1 = best_single_ner_soft_f1
            best_res_ner_hard_f1 = best_single_ner_hard_f1
            best_res_rel_mention_f1 = best_single_rel_mention_f1
            best_res_rel_soft_f1 = best_single_rel_soft_f1
            best_res_rel_hard_f1 = best_single_rel_hard_f1
            best_res_coref_muc_f1 = best_single_coref_muc_f1
            best_res_coref_bcubed_f1 = best_single_coref_bcubed_f1
            best_res_coref_ceafe_f1 = best_single_coref_ceafe_f1
        else:
            if is_it_bert:
                best_res_coref_avg_f1 = best_joint_bert_coref_avg_f1
                best_res_ner_mention_f1 = best_joint_bert_ner_mention_f1
                best_res_ner_soft_f1 = best_joint_bert_ner_soft_f1
                best_res_ner_hard_f1 = best_joint_bert_ner_hard_f1
                best_res_rel_mention_f1 = best_joint_bert_rel_mention_f1
                best_res_rel_soft_f1 = best_joint_bert_rel_soft_f1
                best_res_rel_hard_f1 = best_joint_bert_rel_hard_f1
                best_res_coref_muc_f1 = best_joint_bert_coref_muc_f1
                best_res_coref_bcubed_f1 = best_joint_bert_coref_bcubed_f1
                best_res_coref_ceafe_f1 = best_joint_bert_coref_ceafe_f1
            else:
                best_res_coref_avg_f1 = best_joint_coref_avg_f1
                best_res_ner_mention_f1 = best_joint_ner_mention_f1
                best_res_ner_soft_f1 = best_joint_ner_soft_f1
                best_res_ner_hard_f1 = best_joint_ner_hard_f1
                best_res_rel_mention_f1 = best_joint_rel_mention_f1
                best_res_rel_soft_f1 = best_joint_rel_soft_f1
                best_res_rel_hard_f1 = best_joint_rel_hard_f1
                best_res_coref_muc_f1 = best_joint_coref_muc_f1
                best_res_coref_bcubed_f1 = best_joint_coref_bcubed_f1
                best_res_coref_ceafe_f1 = best_joint_coref_ceafe_f1

        if is_joint and not previous_joint:
            previous_joint = is_joint
            # latex_content += '\t \\cmidrule{2-11}\n'
            latex_content += '\t \\midrule\n'

        if is_it_bert and not previous_bert:
            previous_bert = is_it_bert
            # latex_content += '\t \\cmidrule{2-11}\n'
            latex_content += '\t \\midrule\n'

        if res_coref is not None:
            avg_coref = (res_coref['muc']['f1'][0] + res_coref['b_cubed_singleton_men_conll']['f1'][0] +
                         res_coref['ceafe_singleton']['f1'][0]) / 3
        else:
            avg_coref = -1

        if res_ner is None:
            res_ner_mentions_f1 = [-1, -1]
            res_ner_soft_f1 = [-1, -1]
            res_ner_hard_f1 = [-1, -1]
        else:
            res_ner_mentions_f1 = res_ner[ner_mention_metric]['f1']
            res_ner_soft_f1 = res_ner['tags_soft']['f1']
            res_ner_hard_f1 = res_ner['tags_hard']['f1']

        if res_relations is None:
            res_rel_mentions_f1 = [-1, -1]
            res_rel_soft_f1 = [-1, -1]
            res_rel_hard_f1 = [-1, -1]
        else:
            res_rel_mentions_f1 = res_relations[rels_mention_metric]['f1']
            res_rel_soft_f1 = res_relations['rels_soft']['f1']
            res_rel_hard_f1 = res_relations['rels_hard']['f1']

        latex_content += '\t{} & {} & {} & {} & {} && {} & {} & {} && {} & {} & {} \\\\ \n {} \n ' \
            .format(curr_result['setup'],
                    get_res_or_nothing(res_coref['muc']['f1'] if res_coref is not None else [-1, -1],
                                       res_coref is not None,
                                       best_res=best_res_coref_muc_f1,
                                       best_res_all=best_all_coref_muc_f1,
                                       color=curr_col_coref_muc_f1),
                    get_res_or_nothing(
                        res_coref['b_cubed_singleton_men_conll']['f1'] if res_coref is not None else [-1, -1],
                        res_coref is not None,
                        best_res=best_res_coref_bcubed_f1,
                        best_res_all=best_all_coref_bcubed_f1,
                        color=curr_col_coref_bcuded_f1),
                    get_res_or_nothing(
                        res_coref['ceafe_singleton']['f1'] if res_coref is not None else [-1, -1],
                        res_coref is not None,
                        best_res=best_res_coref_ceafe_f1,
                        best_res_all=best_all_coref_ceafe_f1,
                        color=curr_col_coref_ceaf_f1),
                    get_res_or_nothing((avg_coref, 0.0),
                                       res_coref is not None,
                                       best_res=best_res_coref_avg_f1,
                                       best_res_all=best_all_coref_avg_f1,
                                       color=curr_col_coref_avg_f1),

                    get_res_or_nothing(
                        res_ner_mentions_f1,
                        res_ner is not None,
                        best_res=best_res_ner_mention_f1,
                        best_res_all=best_all_ner_mention_f1,
                        color=curr_col_ner_mention_f1),
                    get_res_or_nothing(res_ner_hard_f1,
                                       res_ner is not None,
                                       best_res=best_res_ner_hard_f1,
                                       best_res_all=best_all_ner_hard_f1,
                                       color=curr_col_ner_hard_f1) if is_joint else '-',
                    get_res_or_nothing(res_ner_soft_f1,
                                       res_ner is not None,
                                       best_res=best_res_ner_soft_f1,
                                       best_res_all=best_all_ner_soft_f1,
                                       color=curr_col_ner_soft_f1) if is_joint else '-',
                    get_res_or_nothing(
                        res_rel_mentions_f1,
                        res_relations is not None,
                        best_res=best_res_rel_mention_f1,
                        best_res_all=best_all_rel_mention_f1,
                        color=curr_col_rel_mention_f1),
                    get_res_or_nothing(res_rel_hard_f1,
                                       res_relations is not None,
                                       best_res=best_res_rel_hard_f1,
                                       best_res_all=best_all_rel_hard_f1,
                                       color=curr_col_rel_hard_f1) if is_joint else '-',
                    get_res_or_nothing(res_rel_soft_f1,
                                       res_relations is not None,
                                       best_res=best_res_rel_soft_f1,
                                       best_res_all=best_all_rel_soft_f1,
                                       color=curr_col_rel_soft_f1) if is_joint else '-',
                    print_comment_config_multiple(experiment_config,
                                                  experiment_id,
                                                  nr_runs)
                    )

    latex_content += '\t\\bottomrule \n'
    latex_content += '\t\\end{tabular}\n'
    # latex_content += '\t}\n'
    latex_content += '\\end{table}'
    print('latex final results table: ')
    print(latex_content)


def print_latex_table_coreflinker_nill(results_and_model_info: List, is_ner_expanded=False, is_rels_expanded=False):
    # to address one of the comments of the reviewer of also reporting the scores of NIL, do a separate function to not
    # mess up with the original print_latex_table_coreflinker , will use this function to ONLY report nills
    # the important thing/challenge here (for dwie) is to report the nill only for mentions that are "linkable" like
    # people names, but not for other mentions that we just don't link like the roles (ex: "President"); but this is
    # probably something that will have to be addressed in cpn_eval.py .

    best_link_nill_mention_re = 0.0
    best_link_nill_mention_pr = 0.0
    best_link_nill_mention_f1 = 0.0

    best_link_nill_hard_re = 0.0
    best_link_nill_hard_pr = 0.0
    best_link_nill_hard_f1 = 0.0

    results_and_model_info = sorted(results_and_model_info, key=lambda x: x['sort_order'])
    # latex_content = '\\begin{table}[!ht]\n'
    latex_content = '\\begin{table*}\n'
    latex_content += '\\centering\n'
    latex_content += '\\resizebox{1.0\\textwidth}{!}{\\begin{tabular}{c ccc c ccc c ccc c c}\n'
    latex_content += '\\toprule\n'
    latex_content += '& \\multicolumn{3}{c}{Linking Mention} && \\multicolumn{3}{c}{Linking Soft} && ' \
                     '\\multicolumn{3}{c}{Linking Hard} && \\multicolumn{1}{c}{Coref} \\\\ \n'
    latex_content += '\\cmidrule(lr){2-4}\\cmidrule(lr){6-8}\\cmidrule(lr){10-12}\\cmidrule(lr){14-14} \n'
    latex_content += '\\cmidrule(lr){2-4}\\cmidrule(lr){6-8}\\cmidrule(lr){10-12}\\cmidrule(lr){14-14} \n'
    latex_content += 'Setup & \\multicolumn{1}{c}{\\textbf{Pr}} & \\multicolumn{1}{c}{\\textbf{Re}} & ' \
                     '\\multicolumn{1}{c}{$\\mathbf{F_1}$} && \\multicolumn{1}{c}{\\textbf{Pr} } & ' \
                     '\\multicolumn{1}{c}{\\textbf{Re}} & \\multicolumn{1}{c}{$\\mathbf{F_1}$} && ' \
                     '\\multicolumn{1}{c}{\\textbf{Pr} } & \\multicolumn{1}{c}{\\textbf{Re}} & ' \
                     '\\multicolumn{1}{c}{$\\mathbf{F_1}$} && '
    latex_content += '\\multicolumn{1}{c}{\\textbf{Avg. }$\\mathbf{F_1}$} \\\\ \n'
    latex_content += '\\toprule \n'

    previous_bert = False
    previous_joint = False

    # this for pass is needed to detect the maximum values
    for curr_result in results_and_model_info:
        best_link_nill_mention_pr = max(curr_result['results']['links']['links-nill-from-ent']['pr'][0],
                                        best_link_nill_mention_pr)
        best_link_nill_mention_re = max(curr_result['results']['links']['links-nill-from-ent']['re'][0],
                                        best_link_nill_mention_re)
        best_link_nill_mention_f1 = max(curr_result['results']['links']['links-nill-from-ent']['f1'][0],
                                        best_link_nill_mention_f1)

        # best_link_soft_pr = max(curr_result['results']['links']['links-links-soft']['pr'][0], best_link_soft_pr)
        # best_link_soft_re = max(curr_result['results']['links']['links-links-soft']['re'][0], best_link_soft_re)
        # best_link_soft_f1 = max(curr_result['results']['links']['links-links-soft']['f1'][0], best_link_soft_f1)

        best_link_nill_hard_pr = max(curr_result['results']['links']['links-nill-hard']['pr'][0],
                                     best_link_nill_hard_pr)
        best_link_nill_hard_re = max(curr_result['results']['links']['links-nill-hard']['re'][0],
                                     best_link_nill_hard_re)
        best_link_nill_hard_f1 = max(curr_result['results']['links']['links-nill-hard']['f1'][0],
                                     best_link_nill_hard_f1)

        # best_coref_f1 = max(curr_result['results']['coref']['coref_avg']['f1'][0], best_coref_f1)

    for curr_result in results_and_model_info:
        latex_content += '\t{} & {} & ' \
                         '{} & ' \
                         '{} && ' \
                         '{} & ' \
                         '{} & ' \
                         '{} &&' \
                         '{} & ' \
                         '{} & ' \
                         '{} && ' \
                         '{}  \\\\ \n % {} \n ' \
            .format(curr_result['alias'],
                    get_res(curr_result['results']['links']['links-nill-from-ent']['pr'], best_link_nill_mention_pr),
                    get_res(curr_result['results']['links']['links-nill-from-ent']['re'], best_link_nill_mention_re),
                    get_res(curr_result['results']['links']['links-nill-from-ent']['f1'], best_link_nill_mention_f1),

                    # no softs
                    '-',
                    '-',
                    '-',

                    get_res(curr_result['results']['links']['links-nill-hard']['pr'], best_link_nill_hard_pr),
                    get_res(curr_result['results']['links']['links-nill-hard']['re'], best_link_nill_hard_re),
                    get_res(curr_result['results']['links']['links-nill-hard']['f1'], best_link_nill_hard_f1),

                    # no coref
                    '-',
                    curr_result['experiment_id']
                    )

    latex_content += '\t\\bottomrule \n'
    latex_content += '\t\\end{tabular}}\n'
    latex_content += '\t\\caption{Results on NIL mentions.}\n'
    latex_content += '\t\\label{tab:overview_results}\n'
    latex_content += '\\end{table*}'
    print('latex final results table: ')
    print(latex_content)


def print_latex_table_coreflinker(results_and_model_info: List, is_ner_expanded=False, is_rels_expanded=False):
    """

    :param results_and_model_info:
    :param only_f1:
    :param metric_types:

    """
    # rels_mention_metric = 'rels_mention' if not is_rels_expanded else 'rels_mention_expanded'
    # ner_mention_metric = 'tags_mention' if not is_ner_expanded else 'tags_mention_expanded'

    best_link_mention_re = 0.0
    best_link_mention_pr = 0.0
    best_link_mention_f1 = 0.0
    best_link_soft_re = 0.0
    best_link_soft_pr = 0.0
    best_link_soft_f1 = 0.0
    best_link_hard_re = 0.0
    best_link_hard_pr = 0.0
    best_link_hard_f1 = 0.0
    best_coref_f1 = 0.0

    results_and_model_info = sorted(results_and_model_info, key=lambda x: x['sort_order'])
    # latex_content = '\\begin{table}[!ht]\n'
    latex_content = '\\begin{table*}\n'
    latex_content += '\\centering\n'
    latex_content += '\\resizebox{1.0\\textwidth}{!}{\\begin{tabular}{c ccc c ccc c ccc c c}\n'
    latex_content += '\\toprule\n'
    latex_content += '& \\multicolumn{3}{c}{Linking Mention} && \\multicolumn{3}{c}{Linking Soft} && ' \
                     '\\multicolumn{3}{c}{Linking Hard} && \\multicolumn{1}{c}{Coref} \\\\ \n'
    latex_content += '\\cmidrule(lr){2-4}\\cmidrule(lr){6-8}\\cmidrule(lr){10-12}\\cmidrule(lr){14-14} \n'
    latex_content += '\\cmidrule(lr){2-4}\\cmidrule(lr){6-8}\\cmidrule(lr){10-12}\\cmidrule(lr){14-14} \n'
    latex_content += 'Setup & \\multicolumn{1}{c}{\\textbf{Pr}} & \\multicolumn{1}{c}{\\textbf{Re}} & ' \
                     '\\multicolumn{1}{c}{$\\mathbf{F_1}$} && \\multicolumn{1}{c}{\\textbf{Pr} } & ' \
                     '\\multicolumn{1}{c}{\\textbf{Re}} & \\multicolumn{1}{c}{$\\mathbf{F_1}$} && ' \
                     '\\multicolumn{1}{c}{\\textbf{Pr} } & \\multicolumn{1}{c}{\\textbf{Re}} & ' \
                     '\\multicolumn{1}{c}{$\\mathbf{F_1}$} && '
    latex_content += '\\multicolumn{1}{c}{\\textbf{Avg. }$\\mathbf{F_1}$} \\\\ \n'
    latex_content += '\\toprule \n'

    previous_bert = False
    previous_joint = False

    # this for pass is needed to detect the maximum values
    for curr_result in results_and_model_info:
        best_link_mention_pr = max(curr_result['results']['links']['links-links-from-ent']['pr'][0],
                                   best_link_mention_pr)
        best_link_mention_re = max(curr_result['results']['links']['links-links-from-ent']['re'][0],
                                   best_link_mention_re)
        best_link_mention_f1 = max(curr_result['results']['links']['links-links-from-ent']['f1'][0],
                                   best_link_mention_f1)

        best_link_soft_pr = max(curr_result['results']['links']['links-links-soft']['pr'][0], best_link_soft_pr)
        best_link_soft_re = max(curr_result['results']['links']['links-links-soft']['re'][0], best_link_soft_re)
        best_link_soft_f1 = max(curr_result['results']['links']['links-links-soft']['f1'][0], best_link_soft_f1)

        best_link_hard_pr = max(curr_result['results']['links']['links-links-hard']['pr'][0], best_link_hard_pr)
        best_link_hard_re = max(curr_result['results']['links']['links-links-hard']['re'][0], best_link_hard_re)
        best_link_hard_f1 = max(curr_result['results']['links']['links-links-hard']['f1'][0], best_link_hard_f1)

        best_coref_f1 = max(curr_result['results']['coref']['coref_avg']['f1'][0], best_coref_f1)

    for curr_result in results_and_model_info:
        latex_content += '\t{} & {} & ' \
                         '{} & ' \
                         '{} && ' \
                         '{} & ' \
                         '{} & ' \
                         '{} &&' \
                         '{} & ' \
                         '{} & ' \
                         '{} && ' \
                         '{}  \\\\ \n % {} \n ' \
            .format(curr_result['alias'],
                    get_res(curr_result['results']['links']['links-links-from-ent']['pr'], best_link_mention_pr),
                    get_res(curr_result['results']['links']['links-links-from-ent']['re'], best_link_mention_re),
                    get_res(curr_result['results']['links']['links-links-from-ent']['f1'], best_link_mention_f1),

                    get_res(curr_result['results']['links']['links-links-soft']['pr'], best_link_soft_pr),
                    get_res(curr_result['results']['links']['links-links-soft']['re'], best_link_soft_re),
                    get_res(curr_result['results']['links']['links-links-soft']['f1'], best_link_soft_f1),

                    get_res(curr_result['results']['links']['links-links-hard']['pr'], best_link_hard_pr),
                    get_res(curr_result['results']['links']['links-links-hard']['re'], best_link_hard_re),
                    get_res(curr_result['results']['links']['links-links-hard']['f1'], best_link_hard_f1),

                    get_res(curr_result['results']['coref']['coref_avg']['f1'], best_coref_f1),
                    curr_result['experiment_id']
                    )

    latex_content += '\t\\bottomrule \n'
    latex_content += '\t\\end{tabular}}\n'
    latex_content += '\t\\caption{Main results of the experiments using Baseline, CorefLinker, etc on DWIE.}\n'
    latex_content += '\t\\label{tab:overview_results}\n'
    latex_content += '\\end{table*}'
    print('latex final results table: ')
    print(latex_content)


def print_latex_table_ner_prop(results_and_model_types: List, only_f1=True, metric_types=['mention', 'soft', 'hard']):
    """

    :param results_and_model_types:
    :param only_f1:
    :param metric_types:
    :return:

    """
    best_ner_mention_pr = 0.0
    best_ner_mention_re = 0.0
    best_ner_mention_f1 = 0.0
    best_ner_soft_pr = 0.0
    best_ner_soft_re = 0.0
    best_ner_soft_f1 = 0.0
    best_ner_hard_pr = 0.0
    best_ner_hard_re = 0.0
    best_ner_hard_f1 = 0.0
    for curr_result in results_and_model_types:
        # print('curr_result: ', curr_result)
        if curr_result['results'] is None:
            continue
        if curr_result['results']['experiment_results']['tags']['tags_mention']['pr'][0] > best_ner_mention_pr:
            best_ner_mention_pr = curr_result['results']['experiment_results']['tags']['tags_mention']['pr'][0]

        if curr_result['results']['experiment_results']['tags']['tags_mention']['re'][0] > best_ner_mention_re:
            best_ner_mention_re = curr_result['results']['experiment_results']['tags']['tags_mention']['re'][0]

        if curr_result['results']['experiment_results']['tags']['tags_mention']['f1'][0] > best_ner_mention_f1:
            best_ner_mention_f1 = curr_result['results']['experiment_results']['tags']['tags_mention']['f1'][0]

        if curr_result['results']['experiment_results']['tags']['tags_soft']['pr'][0] > best_ner_soft_pr:
            best_ner_soft_pr = curr_result['results']['experiment_results']['tags']['tags_soft']['pr'][0]

        if curr_result['results']['experiment_results']['tags']['tags_soft']['re'][0] > best_ner_soft_re:
            best_ner_soft_re = curr_result['results']['experiment_results']['tags']['tags_soft']['re'][0]

        if curr_result['results']['experiment_results']['tags']['tags_soft']['f1'][0] > best_ner_soft_f1:
            best_ner_soft_f1 = curr_result['results']['experiment_results']['tags']['tags_soft']['f1'][0]

        if curr_result['results']['experiment_results']['tags']['tags_hard']['pr'][0] > best_ner_hard_pr:
            best_ner_hard_pr = curr_result['results']['experiment_results']['tags']['tags_hard']['pr'][0]

        if curr_result['results']['experiment_results']['tags']['tags_hard']['re'][0] > best_ner_hard_re:
            best_ner_hard_re = curr_result['results']['experiment_results']['tags']['tags_hard']['re'][0]

        if curr_result['results']['experiment_results']['tags']['tags_hard']['f1'][0] > best_ner_hard_f1:
            best_ner_hard_f1 = curr_result['results']['experiment_results']['tags']['tags_hard']['f1'][0]

    latex_content = '\\begin{table}[!ht]\n'
    latex_content += '\t\\centering\n'
    latex_content += '\t\\setlength{\\tabcolsep}{4pt}\n'
    latex_content += '\t\\begin{tabular}{l ccc }\n'
    latex_content += '\t\\toprule\n'
    latex_content += '\t\\multirow{2}{*}{\\textbf{Model Setup}} &   \\multicolumn{3}{c}{\\textbf{NER}}\n'
    latex_content += '\t\\\\ \n'
    latex_content += '\t\\cmidrule{2-4}\n'
    latex_content += '\t& \multicolumn{1}{c}{\\textbf{S-Pr}} & \\textbf{S-Re} ' \
                     '& \\textbf{S-F$\mathbf{_1}$} \\\\ \n'
    latex_content += '\\midrule \n'
    for curr_result in results_and_model_types:
        if curr_result['results'] is None:
            latex_content += '\t{} & ? & ? & ? \\\\ \n' \
                .format(curr_result['setup'])
        else:
            latex_content += '\t{} & {} & {} & {} \\\\ \n' \
                .format(curr_result['setup'],
                        get_res_or_nothing(curr_result['results']['experiment_results']['tags']['tags_soft']['pr'],
                                           curr_result['results']['experiment_config']['model']['ner']['enabled'],
                                           best_res=best_ner_soft_pr),
                        get_res_or_nothing(curr_result['results']['experiment_results']['tags']['tags_soft']['re'],
                                           curr_result['results']['experiment_config']['model']['ner']['enabled'],
                                           best_res=best_ner_soft_re),
                        get_res_or_nothing(curr_result['results']['experiment_results']['tags']['tags_soft']['f1'],
                                           curr_result['results']['experiment_config']['model']['ner']['enabled'],
                                           best_res=best_ner_soft_f1),
                        )
    latex_content += '\\bottomrule\n'
    latex_content += '\\end{tabular}\n'
    latex_content += '\\caption{Impact of CorefProp and RelProp on NER in our joint setup. The missing results ' \
                     '(\\textit{?} marks) will be completed in the final version of the manuscript.}\n'

    latex_content += '\\label{tab:props_on_ner_results}'
    latex_content += '\\end{table}'
    print('================BEGIN TABLE NER PROP SOFT METRICS=======================')
    print(latex_content)
    print('================END TABLE NER PROP SOFT METRICS=======================')


def print_latex_table_ner_rel_prop(results_and_model_types: List, only_f1=True,
                                   metric_types=['mention', 'soft', 'hard']):
    """

    :param results_and_model_types:
    :param only_f1:
    :param metric_types:
    :return:

    """
    best_ner_soft_pr = 0.0
    best_ner_soft_re = 0.0
    best_ner_soft_f1 = 0.0
    best_rel_soft_pr = 0.0
    best_rel_soft_re = 0.0
    best_rel_soft_f1 = 0.0

    best_bert_ner_soft_pr = 0.0
    best_bert_ner_soft_re = 0.0
    best_bert_ner_soft_f1 = 0.0
    best_bert_rel_soft_pr = 0.0
    best_bert_rel_soft_re = 0.0
    best_bert_rel_soft_f1 = 0.0

    for curr_result in results_and_model_types:
        # print('curr_result: ', curr_result)
        if curr_result['results'] is None:
            continue

        if not is_bert(curr_result['results']['experiment_config']):
            if curr_result['results']['experiment_results']['tags']['tags_soft']['pr'][0] > best_ner_soft_pr:
                best_ner_soft_pr = curr_result['results']['experiment_results']['tags']['tags_soft']['pr'][0]

            if curr_result['results']['experiment_results']['tags']['tags_soft']['re'][0] > best_ner_soft_re:
                best_ner_soft_re = curr_result['results']['experiment_results']['tags']['tags_soft']['re'][0]

            if curr_result['results']['experiment_results']['tags']['tags_soft']['f1'][0] > best_ner_soft_f1:
                best_ner_soft_f1 = curr_result['results']['experiment_results']['tags']['tags_soft']['f1'][0]

            if curr_result['results']['experiment_results']['relations']['rels_soft']['pr'][0] > best_rel_soft_pr:
                best_rel_soft_pr = curr_result['results']['experiment_results']['relations']['rels_soft']['pr'][0]

            if curr_result['results']['experiment_results']['relations']['rels_soft']['re'][0] > best_rel_soft_re:
                best_rel_soft_re = curr_result['results']['experiment_results']['relations']['rels_soft']['re'][0]

            if curr_result['results']['experiment_results']['relations']['rels_soft']['f1'][0] > best_rel_soft_f1:
                best_rel_soft_f1 = curr_result['results']['experiment_results']['relations']['rels_soft']['f1'][0]
        else:
            if curr_result['results']['experiment_results']['tags']['tags_soft']['pr'][0] > best_bert_ner_soft_pr:
                best_bert_ner_soft_pr = curr_result['results']['experiment_results']['tags']['tags_soft']['pr'][0]

            if curr_result['results']['experiment_results']['tags']['tags_soft']['re'][0] > best_bert_ner_soft_re:
                best_bert_ner_soft_re = curr_result['results']['experiment_results']['tags']['tags_soft']['re'][0]

            if curr_result['results']['experiment_results']['tags']['tags_soft']['f1'][0] > best_bert_ner_soft_f1:
                best_bert_ner_soft_f1 = curr_result['results']['experiment_results']['tags']['tags_soft']['f1'][0]

            if curr_result['results']['experiment_results']['relations']['rels_soft']['pr'][0] > best_bert_rel_soft_pr:
                best_bert_rel_soft_pr = curr_result['results']['experiment_results']['relations']['rels_soft']['pr'][0]

            if curr_result['results']['experiment_results']['relations']['rels_soft']['re'][0] > best_bert_rel_soft_re:
                best_bert_rel_soft_re = curr_result['results']['experiment_results']['relations']['rels_soft']['re'][0]

            if curr_result['results']['experiment_results']['relations']['rels_soft']['f1'][0] > best_bert_rel_soft_f1:
                best_bert_rel_soft_f1 = curr_result['results']['experiment_results']['relations']['rels_soft']['f1'][0]

    latex_content = '\\begin{table}[!ht] \n'
    latex_content += '\t\\centering \n'
    latex_content += '\t\\resizebox{0.9\\textwidth}{!}{ \n'
    latex_content += '\t\\begin{tabular}{ l ccc c ccc } \n'
    latex_content += '\t\\toprule \n'
    latex_content += '\t\\hspace*{2cm} & \n'
    latex_content += '\t\\multicolumn{3}{c}{\\textbf{NER}} & & \n'
    latex_content += '\t\\multicolumn{3}{c}{\\textbf{Relations}} \\\\ \n'
    latex_content += '\t\\cmidrule(lr){2-4} \\cmidrule(lr){6-8}  \n'
    latex_content += '\t\\textbf{Model Setup} & \n'
    latex_content += '\t\\makebox[3em]{\\textbf{S-Pr}} & \n'
    latex_content += '\t\\makebox[3em]{\\textbf{S-Re}} & \n'
    latex_content += '\t\\makebox[3em]{\\textbf{S-F$\mathbf{_1}$}} & \n'
    latex_content += '\t\\makebox[0.2em]{} & \n'
    latex_content += '\t\\makebox[3em]{\\textbf{S-Pr}} & \n'
    latex_content += '\t\\makebox[3em]{\\textbf{S-Re}} & \n'
    latex_content += '\t\\makebox[3em]{\\textbf{S-F$\mathbf{_1}$}} \\\\ \n '
    latex_content += '\t\\midrule \n'
    previous_bert = False
    for curr_result in results_and_model_types:

        if curr_result['results'] is None:
            latex_content += '\t{} & ' \
                             '\\leavevmode\\color{{red}}? & ' \
                             '\\leavevmode\\color{{red}}? & ' \
                             '\\leavevmode\\color{{red}}? & & ' \
                             '\\leavevmode\\color{{red}}? & ' \
                             '\\leavevmode\\color{{red}}? & ' \
                             '\\leavevmode\\color{{red}}? \\\\ \n' \
                .format(curr_result['setup'])
        else:
            is_it_bert = is_bert(curr_result['results']['experiment_config'])

            if is_it_bert and not previous_bert:
                previous_bert = is_it_bert
                latex_content += '\t \\cmidrule{2-8}\n'

            latex_content += '\t{} & {} & {} & {} & & {} & {} & {} \\\\ \n % {} \n' \
                .format(curr_result['setup'],
                        get_res_or_nothing(curr_result['results']['experiment_results']['tags']['tags_soft']['pr'],
                                           curr_result['results']['experiment_config']['model']['ner']['enabled'],
                                           best_res=(best_bert_ner_soft_pr if is_it_bert else best_ner_soft_pr)),
                        get_res_or_nothing(curr_result['results']['experiment_results']['tags']['tags_soft']['re'],
                                           curr_result['results']['experiment_config']['model']['ner']['enabled'],
                                           best_res=(best_bert_ner_soft_re if is_it_bert else best_ner_soft_re)),
                        get_res_or_nothing(curr_result['results']['experiment_results']['tags']['tags_soft']['f1'],
                                           curr_result['results']['experiment_config']['model']['ner']['enabled'],
                                           best_res=(best_bert_ner_soft_f1 if is_it_bert else best_ner_soft_f1)),
                        get_res_or_nothing(curr_result['results']['experiment_results']['relations']['rels_soft']['pr'],
                                           curr_result['results']['experiment_config']['model']['relations']['enabled'],
                                           best_res=(best_bert_rel_soft_pr if is_it_bert else best_rel_soft_pr)),
                        get_res_or_nothing(curr_result['results']['experiment_results']['relations']['rels_soft']['re'],
                                           curr_result['results']['experiment_config']['model']['relations']['enabled'],
                                           best_res=(best_bert_rel_soft_re if is_it_bert else best_rel_soft_re)),
                        get_res_or_nothing(curr_result['results']['experiment_results']['relations']['rels_soft']['f1'],
                                           curr_result['results']['experiment_config']['model']['relations']['enabled'],
                                           best_res=(best_bert_rel_soft_f1 if is_it_bert else best_rel_soft_f1)),
                        print_comment_config(curr_result['results']['experiment_config'],
                                             curr_result['results']['experiment_id'],
                                             curr_result['results']['experiment_results']['nr_runs'])
                        )
    latex_content += '\\bottomrule\n'
    latex_content += '\\end{tabular}}\n'
    latex_content += '\\caption{Impact of CorefProp and RelProp on NER and relations in our joint setup. The missing results ' \
                     '(\\textit{?} marks) will be completed in the final version of the manuscript.}\n'

    latex_content += '\\label{tab:props_on_ner_results}\n'
    latex_content += '\\end{table}'
    print('===============BEGIN TABLE NER AND REL PROP SOFT METRICS======================')
    print(latex_content)
    print('================END TABLE NER AND REL PROP SOFT METRICS=======================')


def print_latex_table_ner_prop_all_metrics(results_and_model_types: List, only_f1=True,
                                           metric_types=['mention', 'soft', 'hard']):
    """

    :param results_and_model_types:
    :param only_f1:
    :param metric_types:
    :return:

    """
    # latex_content += '\t\\cmidrule{2-5} \cmidrule{6-8} \cmidrule{9-11}\n'

    best_ner_mention_pr = 0.0
    best_ner_mention_re = 0.0
    best_ner_mention_f1 = 0.0
    best_ner_soft_pr = 0.0
    best_ner_soft_re = 0.0
    best_ner_soft_f1 = 0.0
    best_ner_hard_pr = 0.0
    best_ner_hard_re = 0.0
    best_ner_hard_f1 = 0.0
    for curr_result in results_and_model_types:
        # print('curr_result: ', curr_result)
        if curr_result['results'] is None:
            continue
        if curr_result['results']['experiment_results']['tags']['tags_mention']['pr'][0] > best_ner_mention_pr:
            best_ner_mention_pr = curr_result['results']['experiment_results']['tags']['tags_mention']['pr'][0]

        if curr_result['results']['experiment_results']['tags']['tags_mention']['re'][0] > best_ner_mention_re:
            best_ner_mention_re = curr_result['results']['experiment_results']['tags']['tags_mention']['re'][0]

        if curr_result['results']['experiment_results']['tags']['tags_mention']['f1'][0] > best_ner_mention_f1:
            best_ner_mention_f1 = curr_result['results']['experiment_results']['tags']['tags_mention']['f1'][0]

        if curr_result['results']['experiment_results']['tags']['tags_soft']['pr'][0] > best_ner_soft_pr:
            best_ner_soft_pr = curr_result['results']['experiment_results']['tags']['tags_soft']['pr'][0]

        if curr_result['results']['experiment_results']['tags']['tags_soft']['re'][0] > best_ner_soft_re:
            best_ner_soft_re = curr_result['results']['experiment_results']['tags']['tags_soft']['re'][0]

        if curr_result['results']['experiment_results']['tags']['tags_soft']['f1'][0] > best_ner_soft_f1:
            best_ner_soft_f1 = curr_result['results']['experiment_results']['tags']['tags_soft']['f1'][0]

        if curr_result['results']['experiment_results']['tags']['tags_hard']['pr'][0] > best_ner_hard_pr:
            best_ner_hard_pr = curr_result['results']['experiment_results']['tags']['tags_hard']['pr'][0]

        if curr_result['results']['experiment_results']['tags']['tags_hard']['re'][0] > best_ner_hard_re:
            best_ner_hard_re = curr_result['results']['experiment_results']['tags']['tags_hard']['re'][0]

        if curr_result['results']['experiment_results']['tags']['tags_hard']['f1'][0] > best_ner_hard_f1:
            best_ner_hard_f1 = curr_result['results']['experiment_results']['tags']['tags_hard']['f1'][0]

    latex_content = '\\begin{table}[!ht]\n'
    latex_content += '\t\\centering\n'
    latex_content += '\t\\setlength{\\tabcolsep}{4pt}\n'
    latex_content += '\t\\begin{tabular}{l ccc ccc ccc}\n'
    latex_content += '\t\\toprule\n'
    latex_content += '\t\\multirow{2}{*}{\\textbf{Model Setup}} &   \multicolumn{9}{c}{\\textbf{NER}}\n'
    latex_content += '\t\\\\ \n'
    latex_content += '\t\\cmidrule{2-4} \cmidrule{5-7} \cmidrule{8-10} \n'
    latex_content += '\t& \multicolumn{1}{c}{\\textbf{M-Pr}} & \\textbf{M-Re} ' \
                     '& \\textbf{M-F$\mathbf{_1}$} '
    latex_content += '& \multicolumn{1}{c}{\\textbf{S-Pr}} & \\textbf{S-Re} ' \
                     '& \\textbf{S-F$\mathbf{_1}$} '
    latex_content += '& \multicolumn{1}{c}{\\textbf{E-Pr}} & \\textbf{E-Re} ' \
                     '& \\textbf{E-F$\mathbf{_1}$} \\\\ \n'
    latex_content += '\\midrule \n'
    for curr_result in results_and_model_types:
        if curr_result['results'] is None:
            latex_content += '\t{} & ? & ? & ? & ? & ? & ? & ? & ? & ? \\\\ \n' \
                .format(curr_result['setup'])
        else:
            latex_content += '\t{} & {} & {} & {} & {} & {} & {} & {} & {} & {} \\\\ \n' \
                .format(curr_result['setup'],
                        get_res_or_nothing(
                            curr_result['results']['experiment_results']['tags']['tags_mention']['pr'],
                            curr_result['results']['experiment_config']['model']['ner']['enabled'],
                            best_res=best_ner_mention_pr),
                        get_res_or_nothing(
                            curr_result['results']['experiment_results']['tags']['tags_mention']['re'],
                            curr_result['results']['experiment_config']['model']['ner']['enabled'],
                            best_res=best_ner_mention_re),
                        get_res_or_nothing(
                            curr_result['results']['experiment_results']['tags']['tags_mention']['f1'],
                            curr_result['results']['experiment_config']['model']['ner']['enabled'],
                            best_res=best_ner_mention_f1),
                        get_res_or_nothing(
                            curr_result['results']['experiment_results']['tags']['tags_soft']['pr'],
                            curr_result['results']['experiment_config']['model']['ner']['enabled'],
                            best_res=best_ner_soft_pr),
                        get_res_or_nothing(
                            curr_result['results']['experiment_results']['tags']['tags_soft']['re'],
                            curr_result['results']['experiment_config']['model']['ner']['enabled'],
                            best_res=best_ner_soft_re),
                        get_res_or_nothing(
                            curr_result['results']['experiment_results']['tags']['tags_soft']['f1'],
                            curr_result['results']['experiment_config']['model']['ner']['enabled'],
                            best_res=best_ner_soft_f1),
                        get_res_or_nothing(
                            curr_result['results']['experiment_results']['tags']['tags_hard']['pr'],
                            curr_result['results']['experiment_config']['model']['ner']['enabled'],
                            best_res=best_ner_hard_pr),
                        get_res_or_nothing(
                            curr_result['results']['experiment_results']['tags']['tags_hard']['re'],
                            curr_result['results']['experiment_config']['model']['ner']['enabled'],
                            best_res=best_ner_hard_re),
                        get_res_or_nothing(
                            curr_result['results']['experiment_results']['tags']['tags_hard']['f1'],
                            curr_result['results']['experiment_config']['model']['ner']['enabled'],
                            best_res=best_ner_hard_f1),
                        )
    latex_content += '\\end{tabular}\n'
    latex_content += '\\caption{Impact of CorefProp and RelProp on NER in our joint setup. The missing results ' \
                     '(\\textit{?} marks) will be completed in the final version of the manuscript.}\n'

    latex_content += '\\label{tab:props_on_ner_results_all}'
    latex_content += '\\end{table}'
    print('================BEGIN TABLE NER PROP ALL METRICS=======================')
    print(latex_content)
    print('================END TABLE NER PROP ALL METRICS=======================')


def print_latex_table_rel_prop(results_and_model_types: List, only_f1=True, metric_types=['mention', 'soft', 'hard']):
    """

    :param results_and_model_types:
    :param only_f1:
    :param metric_types:
    :return:

    """

    # marks the best results
    best_rel_mention_pr = 0.0
    best_rel_mention_re = 0.0
    best_rel_mention_f1 = 0.0
    best_rel_soft_pr = 0.0
    best_rel_soft_re = 0.0
    best_rel_soft_f1 = 0.0
    best_rel_hard_pr = 0.0
    best_rel_hard_re = 0.0
    best_rel_hard_f1 = 0.0
    for curr_result in results_and_model_types:
        # print('curr_result: ', curr_result)
        if curr_result['results'] is None:
            continue
        if curr_result['results']['experiment_results']['relations']['rels_mention']['pr'][0] > best_rel_mention_pr:
            best_rel_mention_pr = curr_result['results']['experiment_results']['relations']['rels_mention']['pr'][0]

        if curr_result['results']['experiment_results']['relations']['rels_mention']['re'][0] > best_rel_mention_re:
            best_rel_mention_re = curr_result['results']['experiment_results']['relations']['rels_mention']['re'][0]

        if curr_result['results']['experiment_results']['relations']['rels_mention']['f1'][0] > best_rel_mention_f1:
            best_rel_mention_f1 = curr_result['results']['experiment_results']['relations']['rels_mention']['f1'][0]

        if curr_result['results']['experiment_results']['relations']['rels_soft']['pr'][0] > best_rel_soft_pr:
            best_rel_soft_pr = curr_result['results']['experiment_results']['relations']['rels_soft']['pr'][0]

        if curr_result['results']['experiment_results']['relations']['rels_soft']['re'][0] > best_rel_soft_re:
            best_rel_soft_re = curr_result['results']['experiment_results']['relations']['rels_soft']['re'][0]

        if curr_result['results']['experiment_results']['relations']['rels_soft']['f1'][0] > best_rel_soft_f1:
            best_rel_soft_f1 = curr_result['results']['experiment_results']['relations']['rels_soft']['f1'][0]

        if curr_result['results']['experiment_results']['relations']['rels_hard']['pr'][0] > best_rel_hard_pr:
            best_rel_hard_pr = curr_result['results']['experiment_results']['relations']['rels_hard']['pr'][0]

        if curr_result['results']['experiment_results']['relations']['rels_hard']['re'][0] > best_rel_hard_re:
            best_rel_hard_re = curr_result['results']['experiment_results']['relations']['rels_hard']['re'][0]

        if curr_result['results']['experiment_results']['relations']['rels_hard']['f1'][0] > best_rel_hard_f1:
            best_rel_hard_f1 = curr_result['results']['experiment_results']['relations']['rels_hard']['f1'][0]

    latex_content = '\\begin{table}[!ht]\n'
    latex_content += '\t\\centering\n'
    latex_content += '\t\\setlength{\\tabcolsep}{4pt}\n'
    latex_content += '\t\\begin{tabular}{l ccc }\n'
    latex_content += '\t\\toprule\n'
    latex_content += '\t\\multirow{2}{*}{\\textbf{Model Setup}} &   \multicolumn{3}{c}{\\textbf{Relations}}\n'
    latex_content += '\t\\\\ \n'
    latex_content += '\t\\cmidrule{2-4}\n'
    latex_content += '\t& \multicolumn{1}{c}{\\textbf{S-Pr}} & \\textbf{S-Re} ' \
                     '& \\textbf{S-F$\mathbf{_1}$} \\\\ \n'
    latex_content += '\\midrule \n'
    for curr_result in results_and_model_types:
        if curr_result['results'] is None:
            latex_content += '\t{} & ? & ? & ? \\\\ \n' \
                .format(curr_result['setup'])
        else:
            latex_content += '\t{} & {} & {} & {} \\\\ \n' \
                .format(curr_result['setup'],
                        get_res_or_nothing(
                            curr_result['results']['experiment_results']['relations']['rels_soft']['pr'],
                            curr_result['results']['experiment_config']['model']['relations']['enabled'],
                            best_res=best_rel_mention_pr),
                        get_res_or_nothing(
                            curr_result['results']['experiment_results']['relations']['rels_soft']['re'],
                            curr_result['results']['experiment_config']['model']['relations']['enabled'],
                            best_res=best_rel_mention_re),
                        get_res_or_nothing(
                            curr_result['results']['experiment_results']['relations']['rels_soft']['f1'],
                            curr_result['results']['experiment_config']['model']['relations']['enabled'],
                            best_res=best_rel_mention_f1),
                        )
    latex_content += '\\end{tabular}\n'
    latex_content += '\\caption{Impact of CorefProp and RelProp on Relation in our joint setup. The missing results ' \
                     '(\\textit{?} marks) will be completed in the final version of the manuscript.}\n'

    latex_content += '\\label{tab:props_on_rel_results}'
    latex_content += '\\end{table}'
    print('================BEGIN TABLE RELATION PROP SOFT METRIC=======================')
    print(latex_content)
    print('================END TABLE RELATION PROP SOFT METRIC=======================')


def print_latex_table_rel_prop_all_metrics(results_and_model_types: List, only_f1=True,
                                           metric_types=['mention', 'soft', 'hard']):
    """

    :param results_and_model_types:
    :param only_f1:
    :param metric_types:
    :return:

    """

    # marks the best results
    best_rel_mention_pr = 0.0
    best_rel_mention_re = 0.0
    best_rel_mention_f1 = 0.0
    best_rel_soft_pr = 0.0
    best_rel_soft_re = 0.0
    best_rel_soft_f1 = 0.0
    best_rel_hard_pr = 0.0
    best_rel_hard_re = 0.0
    best_rel_hard_f1 = 0.0
    for curr_result in results_and_model_types:
        # print('curr_result: ', curr_result)
        if curr_result['results'] is None:
            continue
        if curr_result['results']['experiment_results']['relations']['rels_mention']['pr'][0] > best_rel_mention_pr:
            best_rel_mention_pr = curr_result['results']['experiment_results']['relations']['rels_mention']['pr'][0]

        if curr_result['results']['experiment_results']['relations']['rels_mention']['re'][0] > best_rel_mention_re:
            best_rel_mention_re = curr_result['results']['experiment_results']['relations']['rels_mention']['re'][0]

        if curr_result['results']['experiment_results']['relations']['rels_mention']['f1'][0] > best_rel_mention_f1:
            best_rel_mention_f1 = curr_result['results']['experiment_results']['relations']['rels_mention']['f1'][0]

        if curr_result['results']['experiment_results']['relations']['rels_soft']['pr'][0] > best_rel_soft_pr:
            best_rel_soft_pr = curr_result['results']['experiment_results']['relations']['rels_soft']['pr'][0]

        if curr_result['results']['experiment_results']['relations']['rels_soft']['re'][0] > best_rel_soft_re:
            best_rel_soft_re = curr_result['results']['experiment_results']['relations']['rels_soft']['re'][0]

        if curr_result['results']['experiment_results']['relations']['rels_soft']['f1'][0] > best_rel_soft_f1:
            best_rel_soft_f1 = curr_result['results']['experiment_results']['relations']['rels_soft']['f1'][0]

        if curr_result['results']['experiment_results']['relations']['rels_hard']['pr'][0] > best_rel_hard_pr:
            best_rel_hard_pr = curr_result['results']['experiment_results']['relations']['rels_hard']['pr'][0]

        if curr_result['results']['experiment_results']['relations']['rels_hard']['re'][0] > best_rel_hard_re:
            best_rel_hard_re = curr_result['results']['experiment_results']['relations']['rels_hard']['re'][0]

        if curr_result['results']['experiment_results']['relations']['rels_hard']['f1'][0] > best_rel_hard_f1:
            best_rel_hard_f1 = curr_result['results']['experiment_results']['relations']['rels_hard']['f1'][0]

    latex_content = '\\begin{table}[!ht]\n'
    latex_content += '\t\\centering\n'
    latex_content += '\t\\setlength{\\tabcolsep}{4pt}\n'
    latex_content += '\t\\begin{tabular}{l ccc ccc ccc}\n'
    latex_content += '\t\\toprule\n'
    latex_content += '\t\\multirow{2}{*}{\\textbf{Model Setup}} &   \multicolumn{9}{c}{\\textbf{Relations}}\n'
    latex_content += '\t\\\\ \n'
    latex_content += '\t\\cmidrule{2-4} \cmidrule{5-7} \cmidrule{8-10} \n'
    latex_content += '\t& \multicolumn{1}{c}{\\textbf{M-Pr}} & \\textbf{M-Re} ' \
                     '& \\textbf{M-F$\mathbf{_1}$} '
    latex_content += '& \multicolumn{1}{c}{\\textbf{S-Pr}} & \\textbf{S-Re} ' \
                     '& \\textbf{S-F$\mathbf{_1}$} '
    latex_content += '& \multicolumn{1}{c}{\\textbf{E-Pr}} & \\textbf{E-Re} ' \
                     '& \\textbf{E-F$\mathbf{_1}$} \\\\ \n'
    latex_content += '\\midrule \n'
    for curr_result in results_and_model_types:
        if curr_result['results'] is None:
            latex_content += '\t{} & ? & ? & ? & ? & ? & ? & ? & ? & ? \\\\ \n' \
                .format(curr_result['setup'])
        else:
            latex_content += '\t{} & {} & {} & {} & {} & {} & {} & {} & {} & {} \\\\ \n' \
                .format(curr_result['setup'],
                        get_res_or_nothing(
                            curr_result['results']['experiment_results']['relations']['rels_mention']['pr'],
                            curr_result['results']['experiment_config']['model']['relations']['enabled'],
                            best_res=best_rel_mention_pr),
                        get_res_or_nothing(
                            curr_result['results']['experiment_results']['relations']['rels_mention']['re'],
                            curr_result['results']['experiment_config']['model']['relations']['enabled'],
                            best_res=best_rel_mention_re),
                        get_res_or_nothing(
                            curr_result['results']['experiment_results']['relations']['rels_mention']['f1'],
                            curr_result['results']['experiment_config']['model']['relations']['enabled'],
                            best_res=best_rel_mention_f1),
                        get_res_or_nothing(
                            curr_result['results']['experiment_results']['relations']['rels_soft']['pr'],
                            curr_result['results']['experiment_config']['model']['relations']['enabled'],
                            best_res=best_rel_soft_pr),
                        get_res_or_nothing(
                            curr_result['results']['experiment_results']['relations']['rels_soft']['re'],
                            curr_result['results']['experiment_config']['model']['relations']['enabled'],
                            best_res=best_rel_soft_re),
                        get_res_or_nothing(
                            curr_result['results']['experiment_results']['relations']['rels_soft']['f1'],
                            curr_result['results']['experiment_config']['model']['relations']['enabled'],
                            best_res=best_rel_soft_f1),
                        get_res_or_nothing(
                            curr_result['results']['experiment_results']['relations']['rels_hard']['pr'],
                            curr_result['results']['experiment_config']['model']['relations']['enabled'],
                            best_res=best_rel_hard_pr),
                        get_res_or_nothing(
                            curr_result['results']['experiment_results']['relations']['rels_hard']['re'],
                            curr_result['results']['experiment_config']['model']['relations']['enabled'],
                            best_res=best_rel_hard_re),
                        get_res_or_nothing(
                            curr_result['results']['experiment_results']['relations']['rels_hard']['f1'],
                            curr_result['results']['experiment_config']['model']['relations']['enabled'],
                            best_res=best_rel_hard_f1),
                        )
    latex_content += '\\end{tabular}\n'
    latex_content += '\\caption{Impact of CorefProp and RelProp on Relation in our joint setup. The missing results ' \
                     '(\\textit{?} marks) will be completed in the final version of the manuscript.}\n'

    latex_content += '\\label{tab:props_on_rel_results_all}'
    latex_content += '\\end{table}'
    print('================BEGIN TABLE RELATION PROP ALL METRICS=======================')
    print(latex_content)
    print('================END TABLE RELATION PROP ALL METRICS=======================')


def get_graph_plots_single(tasks_to_plot=['ner', 'coref', 'rel'], prop_types=['coref_prop', 'att_prop'],
                           detail_plot: Dict = None,
                           caption='Impact of the Coref and Relation Propagation Iterations different tasks.',
                           max_coref_props=4, max_att_props=4, max_rel_props=4, same_scaling=True,
                           label='fig:res_prop_single'):
    """
    :return:

    """
    prop_types_to_prop_types = {'coref_prop': '\\propformat{CorefProp}',
                                'rel_prop': '\\propformat{RelProp}',
                                'att_prop': '\\propformat{AttProp}'}

    print('to plot graphs: ', tasks_to_plot)
    print('types graphs: ', prop_types)
    print('=========BEGIN GRAPH PLOTS=========')

    latex_content = '\\begin{subfigure}[b]{1.0\\textwidth}\n'
    latex_content += '\\centering\n'
    latex_content += '\\begin{tikzpicture}\n'

    f1_bounds_per_task = dict()
    for idx, curr_task_to_plot in enumerate(tasks_to_plot):
        curr_min_f1 = 999.9
        curr_max_f1 = 0.0
        for curr_prop_type in prop_types:
            max_nr_props = 4
            if curr_prop_type == 'coref_prop':
                max_nr_props = max_coref_props
            elif curr_prop_type == 'att_prop':
                max_nr_props = max_att_props
            elif curr_prop_type == 'rel_prop':
                max_nr_props = max_rel_props
            for curr_prop in range(max_nr_props):
                curr_f1 = None
                curr_stdev = None
                # if curr_prop not in detail_plot[curr_prop_type] or \
                #         detail_plot[curr_prop_type][curr_prop] is None:
                #     print('continuing for graph with ', curr_prop, ' for ', curr_task_to_plot, ' for prop type ',
                #           curr_prop_type)
                #     continue
                if curr_task_to_plot not in detail_plot[curr_prop_type] or \
                        curr_prop not in detail_plot[curr_prop_type][curr_task_to_plot]:
                    print('continuing single for graph with ', curr_prop, ' for ', curr_task_to_plot, ' for prop type ',
                          curr_prop_type)
                    continue

                if curr_task_to_plot == 'rel':
                    curr_f1 = \
                        detail_plot[curr_prop_type][curr_task_to_plot][curr_prop]['experiment_results']['relations'][
                            'rels_mention']['f1'][0]
                    curr_stdev = \
                        detail_plot[curr_prop_type][curr_task_to_plot][curr_prop]['experiment_results']['relations'][
                            'rels_mention']['f1'][1]
                elif curr_task_to_plot == 'ner':
                    curr_f1 = detail_plot[curr_prop_type][curr_task_to_plot][curr_prop]['experiment_results']['tags'][
                        'tags_mention']['f1'][0]
                    curr_stdev = \
                        detail_plot[curr_prop_type][curr_task_to_plot][curr_prop]['experiment_results']['tags'][
                            'tags_mention']['f1'][1]
                elif curr_task_to_plot == 'coref':
                    curr_f1 = detail_plot[curr_prop_type][curr_task_to_plot][curr_prop]['experiment_results']['coref'][
                        'coref_avg']['f1'][
                        0]
                    curr_stdev = \
                        detail_plot[curr_prop_type][curr_task_to_plot][curr_prop]['experiment_results']['coref'][
                            'coref_avg']['f1'][1]

                if curr_f1 is not None:
                    curr_f1 = curr_f1 * 100
                    curr_stdev = curr_stdev * 100
                    if curr_f1 + curr_stdev > curr_max_f1:
                        curr_max_f1 = curr_f1 + curr_stdev

                    if curr_f1 - curr_stdev < curr_min_f1:
                        curr_min_f1 = curr_f1 - curr_stdev
        f1_bounds_per_task[curr_task_to_plot] = {'lower': math.trunc(curr_min_f1), 'upper': math.trunc(curr_max_f1)}

    if same_scaling:
        max_distance = max([v['upper'] - v['lower'] for v in f1_bounds_per_task.values()])
        for curr_task, curr_limits in f1_bounds_per_task.items():
            curr_lower = False
            while curr_limits['upper'] - curr_limits['lower'] < max_distance:
                if curr_lower:
                    curr_limits['lower'] -= 1
                else:
                    curr_limits['upper'] += 1
                curr_lower = not curr_lower

    for idx, curr_task_to_plot in enumerate(tasks_to_plot):
        latex_content += '\\begin{axis}[%\n'
        latex_content += 'name=plot_{},\n'.format(idx)
        latex_content += 'height=5cm,width=5cm,\n'
        if curr_task_to_plot != 'coref':
            latex_content += 'ylabel={M-F$_1$},\n'
        else:
            latex_content += 'ylabel={AVG-F$_1$},\n'

        latex_content += 'ylabel shift=-2.0pt,\n'
        curr_name = 'NO NAME???'
        if curr_task_to_plot == 'rel':
            curr_name = 'Relations'
        elif curr_task_to_plot == 'ner':
            curr_name = 'NER'
        elif curr_task_to_plot == 'coref':
            curr_name = 'Coreference'

        latex_content += 'xlabel={{{}}},\n'.format(curr_name)
        latex_content += 'xmin=-0.5, xmax=3.5,\n'

        latex_content += 'ymin={:d}, ymax={:d},\n'.format(f1_bounds_per_task[curr_task_to_plot]['lower'],
                                                          f1_bounds_per_task[curr_task_to_plot]['upper'] + 1)

        latex_content += 'ymajorgrids=true,\n'
        latex_content += 'xmajorgrids=true,\n'
        latex_content += 'grid style=dashed,\n'
        latex_content += 'mark size=1.0pt,\n'
        latex_content += 'line width=1.0pt,\n'
        if idx > 0:
            latex_content += 'at={{($(plot_{}.east)+(1.3cm, 0)$)}},\n'.format(idx - 1)
            latex_content += 'anchor=west,\n'
        if idx == len(tasks_to_plot) - 1:
            if len(tasks_to_plot) < 3:
                latex_content += 'legend style={at={(-0.2,1.1)},anchor=south}, \n'
            else:
                latex_content += 'legend style={at={(-0.9,1.1)},anchor=south}, \n'

            latex_content += 'legend columns=-1, \n'
            latex_content += 'legend style={/tikz/every even column/.append style={column sep=0.4cm}}, \n'
        latex_content += 'mark=*]\n'

        for curr_prop_type in prop_types:
            latex_content += '\\addplot+[ % curr prop type: {} \n '.format(curr_prop_type)
            latex_content += '\t error bars/.cd,\n'
            latex_content += '\t y dir=both, y explicit,\n'
            latex_content += '\t error bar style={line width=1pt,solid},\n'
            latex_content += '\t error mark options={line width=0.5pt,mark size=3pt,rotate=90}\n'
            latex_content += '\t]\n'
            latex_content += '\t coordinates {\n'
            max_nr_props = 0
            if curr_prop_type == 'coref_prop':
                max_nr_props = max_coref_props
            elif curr_prop_type == 'att_prop':
                max_nr_props = max_att_props
            elif curr_prop_type == 'rel_prop':
                max_nr_props = max_rel_props

            for curr_prop in range(max_nr_props):
                curr_f1 = None
                curr_stdev = None
                if curr_task_to_plot not in detail_plot[curr_prop_type] or \
                        curr_prop not in detail_plot[curr_prop_type][curr_task_to_plot]:
                    print('continuing single for graph (2) with ', curr_prop, ' for ', curr_task_to_plot,
                          ' for prop type ',
                          curr_prop_type)
                    latex_content += '\t (0,0) +- (0.0, 0) % no data \n'
                    continue

                experiment_id = detail_plot[curr_prop_type][curr_task_to_plot][curr_prop]['experiment_id']
                if curr_task_to_plot == 'rel':
                    curr_f1 = \
                        detail_plot[curr_prop_type][curr_task_to_plot][curr_prop]['experiment_results']['relations'][
                            'rels_mention']['f1'][0]
                    curr_stdev = \
                        detail_plot[curr_prop_type][curr_task_to_plot][curr_prop]['experiment_results']['relations'][
                            'rels_mention']['f1'][1]
                elif curr_task_to_plot == 'ner':
                    curr_f1 = detail_plot[curr_prop_type][curr_task_to_plot][curr_prop]['experiment_results']['tags'][
                        'tags_mention']['f1'][0]
                    curr_stdev = \
                        detail_plot[curr_prop_type][curr_task_to_plot][curr_prop]['experiment_results']['tags'][
                            'tags_mention']['f1'][1]
                elif curr_task_to_plot == 'coref':
                    curr_f1 = detail_plot[curr_prop_type][curr_task_to_plot][curr_prop]['experiment_results']['coref'][
                        'coref_avg']['f1'][
                        0]
                    curr_stdev = \
                        detail_plot[curr_prop_type][curr_task_to_plot][curr_prop]['experiment_results']['coref'][
                            'coref_avg']['f1'][1]

                if curr_f1 is not None:
                    latex_content += '\t ({},{}) +- (0.0, {})    % {} \n'.format(curr_prop, curr_f1 * 100,
                                                                                 curr_stdev * 100,
                                                                                 experiment_id)
                else:
                    latex_content += '\t (0,0) +- (0.0, 0) % no data \n'

            latex_content += '\t }; \n'

        prop_types_adapted = [prop_types_to_prop_types[prop_type] for prop_type in prop_types]
        if idx == len(tasks_to_plot) - 1:
            # latex_content += '\\legend{{{}}} \n'.format(', '.join(prop_types).replace('_', '\\_'))
            latex_content += '\\legend{{{}}} \n'.format(', '.join(prop_types_adapted).replace('_', '\\_'))

        latex_content += '\\end{axis} \n'

    latex_content += '\\end{tikzpicture} \n'
    latex_content += '\\caption{} \n'
    # latex_content += '\\caption{{{}}} \n'.format(caption)
    latex_content += '\\label{{{}}} \n'.format(label)
    # latex_content += '\\label{} \n'
    latex_content += '\\end{subfigure}\n'

    # print(latex_content)
    print('=========END GRAPH PLOTS=========')
    return latex_content


def get_csv_graph_plots(setup: str, detail_data: Dict,
                        max_coref_props=4, max_att_props=4, max_rel_props=4):
    """

    :param max_rel_props:
    :param max_coref_props:
    :param max_att_props:
    :param setup: can be 'single', 'joint' or 'joint+bert'
    :param detail_data:
    :param csv_list_add_to:
    :return:
    """
    tasks_to_output = ['ner', 'coref', 'rel']
    prop_types = ['coref_prop', 'rel_prop', 'att_prop']
    curr_csv_list = []
    for idx, curr_task_to_output in enumerate(tasks_to_output):
        for curr_prop_type in prop_types:
            max_nr_props = 4
            if curr_prop_type == 'coref_prop':
                max_nr_props = max_coref_props
            elif curr_prop_type == 'att_prop':
                max_nr_props = max_att_props
            elif curr_prop_type == 'rel_prop':
                max_nr_props = max_rel_props
            for curr_prop in range(max_nr_props):
                print('curr prop: ', curr_prop)
                if setup == 'single':
                    if curr_task_to_output not in detail_data[curr_prop_type] or \
                            curr_prop not in detail_data[curr_prop_type][curr_task_to_output] or \
                            detail_data[curr_prop_type][curr_task_to_output][curr_prop] is None:
                        print('continuing for SINGLE csv with ', curr_prop, ' for ', curr_task_to_output,
                              ' for prop type ',
                              curr_prop_type)
                        continue
                else:
                    if curr_prop not in detail_data[curr_prop_type] or \
                            detail_data[curr_prop_type][curr_prop] is None:
                        print('continuing for csv with ', curr_prop, ' for ', curr_task_to_output, ' for prop type ',
                              curr_prop_type)
                        continue
                curr_metric = ''
                if curr_task_to_output == 'rel':
                    if setup == 'single':
                        curr_f1 = \
                            detail_data[curr_prop_type]['rel'][curr_prop]['experiment_results']['relations'][
                                'rels_mention'][
                                'f1'][0] * 100
                        curr_stdev = \
                            detail_data[curr_prop_type]['rel'][curr_prop]['experiment_results']['relations'][
                                'rels_mention'][
                                'f1'][1] * 100
                        curr_metric = 'F_{1,m}'
                    else:
                        curr_f1 = \
                            detail_data[curr_prop_type][curr_prop]['experiment_results']['relations']['rels_soft'][
                                'f1'][0] * 100
                        curr_stdev = \
                            detail_data[curr_prop_type][curr_prop]['experiment_results']['relations']['rels_soft'][
                                'f1'][1] * 100
                        curr_metric = 'F_{1,s}'

                elif curr_task_to_output == 'ner':
                    if setup == 'single':
                        curr_f1 = \
                            detail_data[curr_prop_type]['ner'][curr_prop]['experiment_results']['tags']['tags_mention'][
                                'f1'][
                                0] * 100
                        curr_stdev = \
                            detail_data[curr_prop_type]['ner'][curr_prop]['experiment_results']['tags']['tags_mention'][
                                'f1'][
                                1] * 100
                        curr_metric = 'F_{1,m}'
                    else:
                        curr_f1 = \
                            detail_data[curr_prop_type][curr_prop]['experiment_results']['tags']['tags_soft']['f1'][
                                0] * 100
                        curr_stdev = \
                            detail_data[curr_prop_type][curr_prop]['experiment_results']['tags']['tags_soft']['f1'][
                                1] * 100
                        curr_metric = 'F_{1,s}'
                elif curr_task_to_output == 'coref':
                    curr_metric = 'Avg. F_1'
                    if setup == 'single':
                        curr_f1 = \
                            detail_data[curr_prop_type]['coref'][curr_prop]['experiment_results']['coref']['coref_avg'][
                                'f1'][0] * 100
                        curr_stdev = \
                            detail_data[curr_prop_type]['coref'][curr_prop]['experiment_results']['coref']['coref_avg'][
                                'f1'][1] * 100
                    else:
                        curr_f1 = \
                            detail_data[curr_prop_type][curr_prop]['experiment_results']['coref']['coref_avg']['f1'][
                                0] * 100
                        curr_stdev = \
                            detail_data[curr_prop_type][curr_prop]['experiment_results']['coref']['coref_avg']['f1'][
                                1] * 100
                curr_csv_list.append({'setup': setup, 'task': curr_task_to_output, 'prop_type': curr_prop_type,
                                      'prop_nr': curr_prop, 'metric': curr_metric, 'result': curr_f1,
                                      'stdev': curr_stdev})

    return curr_csv_list


def get_graph_plots(tasks_to_plot=['ner', 'coref', 'rel'], prop_types=['coref_prop', 'att_prop'],
                    detail_plot: Dict = None,
                    caption='Impact of the Coref and Relation Propagation Iterations different tasks.',
                    max_coref_props=4, max_att_props=4, max_rel_props=4,
                    same_scaling=True,
                    label='fig:res_prop_joint'):
    """
    :return:

    """

    prop_types_to_prop_types = {'coref_prop': '\\propformat{CorefProp}',
                                'rel_prop': '\\propformat{RelProp}',
                                'att_prop': '\\propformat{AttProp}'}
    # prop_types_to_prop_types = {'coref_prop': 'CorefProp', 'rel_prop': 'RelProp', 'att_prop': 'AttProp'}

    print('to plot graphs: ', tasks_to_plot)
    print('types graphs: ', prop_types)
    print('=========BEGIN GRAPH PLOTS=========')

    latex_content = '\\begin{subfigure}[b]{1.0\\textwidth}\n'
    latex_content += '\\centering\n'
    latex_content += '\\begin{tikzpicture}\n'

    f1_bounds_per_task = dict()

    for idx, curr_task_to_plot in enumerate(tasks_to_plot):
        curr_min_f1 = 999.9
        curr_max_f1 = 0.0

        for curr_prop_type in prop_types:
            max_nr_props = 4
            if curr_prop_type == 'coref_prop':
                max_nr_props = max_coref_props
            elif curr_prop_type == 'att_prop':
                max_nr_props = max_att_props
            elif curr_prop_type == 'rel_prop':
                max_nr_props = max_rel_props
            for curr_prop in range(max_nr_props):
                curr_f1 = None
                curr_stdev = None
                if curr_prop not in detail_plot[curr_prop_type] or \
                        detail_plot[curr_prop_type][curr_prop] is None:
                    print('continuing for graph with ', curr_prop, ' for ', curr_task_to_plot, ' for prop type ',
                          curr_prop_type)
                    continue
                if curr_task_to_plot == 'rel':
                    curr_f1 = \
                        detail_plot[curr_prop_type][curr_prop]['experiment_results']['relations']['rels_soft']['f1'][0]
                    curr_stdev = \
                        detail_plot[curr_prop_type][curr_prop]['experiment_results']['relations']['rels_soft']['f1'][1]
                elif curr_task_to_plot == 'ner':
                    curr_f1 = detail_plot[curr_prop_type][curr_prop]['experiment_results']['tags']['tags_soft']['f1'][0]
                    curr_stdev = \
                        detail_plot[curr_prop_type][curr_prop]['experiment_results']['tags']['tags_soft']['f1'][1]
                elif curr_task_to_plot == 'coref':
                    curr_f1 = detail_plot[curr_prop_type][curr_prop]['experiment_results']['coref']['coref_avg']['f1'][
                        0]
                    curr_stdev = \
                        detail_plot[curr_prop_type][curr_prop]['experiment_results']['coref']['coref_avg']['f1'][1]
                curr_f1 = curr_f1 * 100
                curr_stdev = curr_stdev * 100
                if curr_f1 + curr_stdev > curr_max_f1:
                    curr_max_f1 = curr_f1 + curr_stdev

                if curr_f1 - curr_stdev < curr_min_f1:
                    curr_min_f1 = curr_f1 - curr_stdev
        f1_bounds_per_task[curr_task_to_plot] = {'lower': math.trunc(curr_min_f1), 'upper': math.trunc(curr_max_f1)}

    if same_scaling:
        max_distance = max([v['upper'] - v['lower'] for v in f1_bounds_per_task.values()])
        for curr_task, curr_limits in f1_bounds_per_task.items():
            curr_lower = False
            while curr_limits['upper'] - curr_limits['lower'] < max_distance:
                if curr_lower:
                    curr_limits['lower'] -= 1
                else:
                    curr_limits['upper'] += 1
                curr_lower = not curr_lower

    for idx, curr_task_to_plot in enumerate(tasks_to_plot):
        latex_content += '\\begin{axis}[%\n'
        latex_content += 'name=plot_{},\n'.format(idx)
        latex_content += 'height=5cm,width=5cm,\n'
        if curr_task_to_plot != 'coref':
            latex_content += 'ylabel={S-F$_1$},\n'
        else:
            latex_content += 'ylabel={AVG-F$_1$},\n'

        latex_content += 'ylabel shift=-2.0pt,\n'
        curr_name = 'NO NAME???'
        if curr_task_to_plot == 'rel':
            curr_name = 'Relations'
        elif curr_task_to_plot == 'ner':
            curr_name = 'NER'
        elif curr_task_to_plot == 'coref':
            curr_name = 'Coreference'

        latex_content += 'xlabel={{{}}},\n'.format(curr_name)
        latex_content += 'xmin=-0.5, xmax=3.5,\n'
        # TODO: see how to create the legend, and also respective comments on which coordinates is from which propagation
        latex_content += 'ymin={:d}, ymax={:d},\n'.format(f1_bounds_per_task[curr_task_to_plot]['lower'],
                                                          f1_bounds_per_task[curr_task_to_plot]['upper'] + 1)

        latex_content += 'ymajorgrids=true,\n'
        latex_content += 'xmajorgrids=true,\n'
        latex_content += 'grid style=dashed,\n'
        latex_content += 'mark size=1.0pt,\n'
        latex_content += 'line width=1.0pt,\n'
        if idx > 0:
            latex_content += 'at={{($(plot_{}.east)+(1.3cm, 0)$)}},\n'.format(idx - 1)
            latex_content += 'anchor=west,\n'
        if idx == len(tasks_to_plot) - 1:
            if len(tasks_to_plot) < 3:
                latex_content += 'legend style={at={(-0.2,1.1)},anchor=south}, \n'
            else:
                latex_content += 'legend style={at={(-0.9,1.1)},anchor=south}, \n'

            latex_content += 'legend columns=-1, \n'
            latex_content += 'legend style={/tikz/every even column/.append style={column sep=0.4cm}}, \n'
        latex_content += 'mark=*]\n'

        for curr_prop_type in prop_types:
            latex_content += '\\addplot+[ % curr prop type: {} \n '.format(curr_prop_type)
            latex_content += '\t error bars/.cd,\n'
            latex_content += '\t y dir=both, y explicit,\n'
            latex_content += '\t error bar style={line width=1pt,solid},\n'
            latex_content += '\t error mark options={line width=0.5pt,mark size=3pt,rotate=90}\n'
            latex_content += '\t]\n'
            latex_content += '\t coordinates {\n'
            max_nr_props = 0
            if curr_prop_type == 'coref_prop':
                max_nr_props = max_coref_props
            elif curr_prop_type == 'att_prop':
                max_nr_props = max_att_props
            elif curr_prop_type == 'rel_prop':
                max_nr_props = max_rel_props
            for curr_prop in range(max_nr_props):
                curr_f1 = None
                curr_stdev = None
                if curr_prop not in detail_plot[curr_prop_type] or \
                        detail_plot[curr_prop_type][curr_prop] is None:
                    print('continuing for graph with ', curr_prop, ' for ', curr_task_to_plot, ' for prop type ',
                          curr_prop_type)
                    continue
                experiment_id = detail_plot[curr_prop_type][curr_prop]['experiment_id']
                if curr_task_to_plot == 'rel':
                    curr_f1 = \
                        detail_plot[curr_prop_type][curr_prop]['experiment_results']['relations']['rels_soft']['f1'][0]
                    curr_stdev = \
                        detail_plot[curr_prop_type][curr_prop]['experiment_results']['relations']['rels_soft']['f1'][1]
                elif curr_task_to_plot == 'ner':
                    curr_f1 = detail_plot[curr_prop_type][curr_prop]['experiment_results']['tags']['tags_soft']['f1'][0]
                    curr_stdev = \
                        detail_plot[curr_prop_type][curr_prop]['experiment_results']['tags']['tags_soft']['f1'][1]
                elif curr_task_to_plot == 'coref':
                    curr_f1 = detail_plot[curr_prop_type][curr_prop]['experiment_results']['coref']['coref_avg']['f1'][
                        0]
                    curr_stdev = \
                        detail_plot[curr_prop_type][curr_prop]['experiment_results']['coref']['coref_avg']['f1'][1]

                if curr_f1 is not None:
                    latex_content += '\t ({},{}) +- (0.0, {})    % {} \n'.format(curr_prop, curr_f1 * 100,
                                                                                 curr_stdev * 100,
                                                                                 experiment_id)
            latex_content += '\t }; \n'
        # latex_content += '\t }; \n'
        prop_types_adapted = [prop_types_to_prop_types[prop_type] for prop_type in prop_types]
        if idx == len(tasks_to_plot) - 1:
            # latex_content += '\\legend{{{}}} \n'.format(', '.join(prop_types).replace('_', '\\_'))
            latex_content += '\\legend{{{}}} \n'.format(', '.join(prop_types_adapted).replace('_', '\\_'))

        latex_content += '\\end{axis} \n'

    latex_content += '\\end{tikzpicture} \n'
    latex_content += '\\caption{} \n'
    # latex_content += '\\caption{{{}}} \n'.format(caption)
    latex_content += '\\label{{{}}} \n'.format(label)
    # latex_content += '\\label{} \n'
    latex_content += '\\end{subfigure}\n'

    # print(latex_content)
    print('=========END GRAPH PLOTS=========')
    return latex_content


def get_csv_line(experiment_config: Dict, experiment_results: Dict, experiment_id: str,
                 only_f1=False, only_title=False, tasks_to_report_on={'Coref', 'Linking'}):
    """ Model file	att enabled	att props	coref enabled	coref props	rel enabled	rel props	Tag M-f1 """

    if not only_title:
        nr_runs = experiment_results['nr_runs']

        model_type = ''
        nilin = ''
        if experiment_config['model']['linker']['enabled']:
            model_type = 'baseline'
        elif experiment_config['model']['linkercoref']['enabled']:
            model_type = 'coref+link-' + experiment_config['model']['linkercoref']['model_type']
            nilin = not experiment_config['model']['linkercoref']['no_nil_in_targets']

        to_ret = [experiment_id, nr_runs, nilin, model_type]

        # to_ret = [experiment_id, nr_runs, field_is_bert, field_att_prop, field_ner_enabled,
        #           field_coref_enabled, field_coref_prop, field_rel_enabled, field_rel_prop]
    else:
        # to_ret = ['Experiment ID', 'Nr Runs', 'NILinM', 'Att Props', 'Tag', 'Coref', 'Coref props',
        #           'Rels', 'Rel props']
        to_ret = ['Experiment ID', 'Nr Runs', 'NILin', 'Model Type']

    if 'Tag' in tasks_to_report_on:
        if not only_title:
            if not only_f1:
                to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['tags']['tags_mention']['pr'][0],
                                                        experiment_results['tags']['tags_mention']['pr'][1]))
                to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['tags']['tags_mention']['re'][0],
                                                        experiment_results['tags']['tags_mention']['re'][1]))

            to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['tags']['tags_mention']['f1'][0],
                                                    experiment_results['tags']['tags_mention']['f1'][1]))
        else:
            if not only_f1:
                to_ret.append('Tag M-pr')
                to_ret.append('Tag M-re')
            to_ret.append('Tag M-f1')

        if not only_title:
            if not only_f1:
                to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['tags']['tags_soft']['pr'][0],
                                                        experiment_results['tags']['tags_soft']['pr'][1]))
                to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['tags']['tags_soft']['re'][0],
                                                        experiment_results['tags']['tags_soft']['re'][1]))

            to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['tags']['tags_soft']['f1'][0],
                                                    experiment_results['tags']['tags_soft']['f1'][1]))
        else:
            if not only_f1:
                to_ret.append('Tag S-pr')
                to_ret.append('Tag S-re')
            to_ret.append('Tag S-f1')

        if not only_title:
            if not only_f1:
                to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['tags']['tags_hard']['pr'][0],
                                                        experiment_results['tags']['tags_hard']['pr'][1]))
                to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['tags']['tags_hard']['re'][0],
                                                        experiment_results['tags']['tags_hard']['re'][1]))

            to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['tags']['tags_hard']['f1'][0],
                                                    experiment_results['tags']['tags_hard']['f1'][1]))
        else:
            if not only_f1:
                to_ret.append('Tag H-pr')
                to_ret.append('Tag H-re')
            to_ret.append('Tag H-f1')

    if 'Rels' in tasks_to_report_on:
        if not only_title:
            if not only_f1:
                to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['relations']['rels_mention']['pr'][0],
                                                        experiment_results['relations']['rels_mention']['pr'][1]))
                to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['relations']['rels_mention']['re'][0],
                                                        experiment_results['relations']['rels_mention']['re'][1]))

            to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['relations']['rels_mention']['f1'][0],
                                                    experiment_results['relations']['rels_mention']['f1'][1]))
        else:
            if not only_f1:
                to_ret.append('Rels M-pr')
                to_ret.append('Rels M-re')
            to_ret.append('Rels M-f1')

        if not only_title:
            if not only_f1:
                to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['relations']['rels_soft']['pr'][0],
                                                        experiment_results['relations']['rels_soft']['pr'][1]))
                to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['relations']['rels_soft']['re'][0],
                                                        experiment_results['relations']['rels_soft']['re'][1]))

            to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['relations']['rels_soft']['f1'][0],
                                                    experiment_results['relations']['rels_soft']['f1'][1]))
        else:
            if not only_f1:
                to_ret.append('Rels S-pr')
                to_ret.append('Rels S-re')
            to_ret.append('Rels S-f1')

        if not only_title:
            if not only_f1:
                to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['relations']['rels_hard']['pr'][0],
                                                        experiment_results['relations']['rels_hard']['pr'][1]))
                to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['relations']['rels_hard']['re'][0],
                                                        experiment_results['relations']['rels_hard']['re'][1]))

            to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['relations']['rels_hard']['f1'][0],
                                                    experiment_results['relations']['rels_hard']['f1'][1]))
        else:
            if not only_f1:
                to_ret.append('Rels H-pr')
                to_ret.append('Rels H-re')
            to_ret.append('Rels H-f1')

    if 'Linking' in tasks_to_report_on:
        if not only_title:
            if not only_f1:
                to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['links']['links-all']['pr'][0],
                                                        experiment_results['links']['links-all']['pr'][1]))
                to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['links']['links-all']['re'][0],
                                                        experiment_results['links']['links-all']['re'][1]))

            to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['links']['links-all']['f1'][0],
                                                    experiment_results['links']['links-all']['f1'][1]))
        else:
            if not only_f1:
                to_ret.append('Pr Links-all')
                to_ret.append('Re Links-all')
            to_ret.append('F1 Links-all')

        if not only_title:
            if not only_f1:
                to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['links']['links-all-from-ent']['pr'][0],
                                                        experiment_results['links']['links-all-from-ent']['pr'][1]))
                to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['links']['links-all-from-ent']['re'][0],
                                                        experiment_results['links']['links-all-from-ent']['re'][1]))

            to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['links']['links-all-from-ent']['f1'][0],
                                                    experiment_results['links']['links-all-from-ent']['f1'][1]))
        else:
            if not only_f1:
                to_ret.append('Pr Links-all-from-ent')
                to_ret.append('Re Links-all-from-ent')
            to_ret.append('F1 Links-all-from-ent')

        if not only_title:
            if not only_f1:
                to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['links']['links-links']['pr'][0],
                                                        experiment_results['links']['links-links']['pr'][1]))
                to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['links']['links-links']['re'][0],
                                                        experiment_results['links']['links-links']['re'][1]))

            to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['links']['links-links']['f1'][0],
                                                    experiment_results['links']['links-links']['f1'][1]))
        else:
            if not only_f1:
                to_ret.append('Pr Links-links')
                to_ret.append('Re Links-links')
            to_ret.append('F1 Links-links')

        if not only_title:
            if not only_f1:
                to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['links']['links-links-from-ent']['pr'][0],
                                                        experiment_results['links']['links-links-from-ent']['pr'][1]))
                to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['links']['links-links-from-ent']['re'][0],
                                                        experiment_results['links']['links-links-from-ent']['re'][1]))

            to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['links']['links-links-from-ent']['f1'][0],
                                                    experiment_results['links']['links-links-from-ent']['f1'][1]))
        else:
            if not only_f1:
                to_ret.append('Pr Links-links-from-ent')
                to_ret.append('Re Links-links-from-ent')
            to_ret.append('F1 Links-links-from-ent')

        if not only_title:
            if not only_f1:
                to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['links']['links-nill']['pr'][0],
                                                        experiment_results['links']['links-nill']['pr'][1]))
                to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['links']['links-nill']['re'][0],
                                                        experiment_results['links']['links-nill']['re'][1]))

            to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['links']['links-nill']['f1'][0],
                                                    experiment_results['links']['links-nill']['f1'][1]))
        else:
            if not only_f1:
                to_ret.append('Pr Links-nill')
                to_ret.append('Re Links-nill')
            to_ret.append('F1 Links-nill')

        if not only_title:
            if not only_f1:
                to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['links']['links-nill-from-ent']['pr'][0],
                                                        experiment_results['links']['links-nill-from-ent']['pr'][1]))
                to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['links']['links-nill-from-ent']['re'][0],
                                                        experiment_results['links']['links-nill-from-ent']['re'][1]))

            to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['links']['links-nill-from-ent']['f1'][0],
                                                    experiment_results['links']['links-nill-from-ent']['f1'][1]))
        else:
            if not only_f1:
                to_ret.append('Pr Links-nill-from-ent')
                to_ret.append('Re Links-nill-from-ent')
            to_ret.append('F1 Links-nill-from-ent')

        if not only_title:
            to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['links']['links-accuracy']['acc_candidates'][0],
                                                    experiment_results['links']['links-accuracy']['acc_candidates'][1]))
        else:
            to_ret.append('Acc-link (candidates)')

        if not only_title:
            to_ret.append(
                '{:0.4f}+-{:0.4f}'.format(experiment_results['links']['links-accuracy']['acc_no_candidates'][0],
                                          experiment_results['links']['links-accuracy']['acc_no_candidates'][1]))
        else:
            to_ret.append('Acc-link (NO candidates)')

        if not only_title:
            to_ret.append(
                '{:0.4f}+-{:0.4f}'.format(
                    experiment_results['links']['links-accuracy-from-ent']['acc_no_candidates'][0],
                    experiment_results['links']['links-accuracy-from-ent']['acc_no_candidates'][1]))
        else:
            to_ret.append('Acc-link-from-ent (NO candidates)')

    if 'Coref' in tasks_to_report_on:
        if not only_title:
            if not only_f1:
                to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['coref']['coref_avg']['pr'][0],
                                                        experiment_results['coref']['coref_avg']['pr'][1]))
                to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['coref']['coref_avg']['re'][0],
                                                        experiment_results['coref']['coref_avg']['re'][1]))

            to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['coref']['coref_avg']['f1'][0],
                                                    experiment_results['coref']['coref_avg']['f1'][1]))
        else:
            if not only_f1:
                to_ret.append('AVG pr (coref)')
                to_ret.append('AVG re (coref)')
            to_ret.append('AVG f1 (coref)')

        if not only_title:
            if not only_f1:
                to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['coref']['muc']['pr'][0],
                                                        experiment_results['coref']['muc']['pr'][1]))
                to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['coref']['muc']['re'][0],
                                                        experiment_results['coref']['muc']['re'][1]))

            to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['coref']['muc']['f1'][0],
                                                    experiment_results['coref']['muc']['f1'][1]))
        else:
            if not only_f1:
                to_ret.append('MUC pr')
                to_ret.append('MUC re')
            to_ret.append('MUC f1')

        if not only_title:
            if not only_f1:
                to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['coref']['ceafm_singleton']['pr'][0],
                                                        experiment_results['coref']['ceafm_singleton']['pr'][1]))
                to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['coref']['ceafm_singleton']['re'][0],
                                                        experiment_results['coref']['ceafm_singleton']['re'][1]))

            to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['coref']['ceafm_singleton']['f1'][0],
                                                    experiment_results['coref']['ceafm_singleton']['f1'][1]))
        else:
            if not only_f1:
                to_ret.append('CEAFm pr')
                to_ret.append('CEAFm re')
            to_ret.append('CEAFm f1')

        if not only_title:
            if not only_f1:
                to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['coref']['ceafe_singleton']['pr'][0],
                                                        experiment_results['coref']['ceafe_singleton']['pr'][1]))
                to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['coref']['ceafe_singleton']['re'][0],
                                                        experiment_results['coref']['ceafe_singleton']['re'][1]))

            to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['coref']['ceafe_singleton']['f1'][0],
                                                    experiment_results['coref']['ceafe_singleton']['f1'][1]))
        else:
            if not only_f1:
                to_ret.append('CEAFe pr')
                to_ret.append('CEAFe re')
            to_ret.append('CEAFe f1')

        if not only_title:
            if not only_f1:
                to_ret.append(
                    '{:0.4f}+-{:0.4f}'.format(experiment_results['coref']['b_cubed_singleton_men_conll']['pr'][0],
                                              experiment_results['coref']['b_cubed_singleton_men_conll']['pr'][
                                                  1]))
                to_ret.append(
                    '{:0.4f}+-{:0.4f}'.format(experiment_results['coref']['b_cubed_singleton_men_conll']['re'][0],
                                              experiment_results['coref']['b_cubed_singleton_men_conll']['re'][
                                                  1]))

            to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['coref']['b_cubed_singleton_men_conll']['f1'][0],
                                                    experiment_results['coref']['b_cubed_singleton_men_conll']['f1'][
                                                        1]))
        else:
            if not only_f1:
                to_ret.append('B-3 men conll pr')
                to_ret.append('B-3 men conll re')
            to_ret.append('B-3 men conll f1')

        if not only_title:
            if not only_f1:
                to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['coref']['b_cubed_singleton_ent']['pr'][0],
                                                        experiment_results['coref']['b_cubed_singleton_ent']['pr'][1]))
                to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['coref']['b_cubed_singleton_ent']['re'][0],
                                                        experiment_results['coref']['b_cubed_singleton_ent']['re'][1]))

            to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['coref']['b_cubed_singleton_ent']['f1'][0],
                                                    experiment_results['coref']['b_cubed_singleton_ent']['f1'][1]))
        else:
            if not only_f1:
                to_ret.append('B-3 ent pr')
                to_ret.append('B-3 ent re')
            to_ret.append('B-3 ent f1')

        if not only_title:
            if not only_f1:
                to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['coref']['ceafe_not_singleton']['pr'][0],
                                                        experiment_results['coref']['ceafe_not_singleton']['pr'][1]))
                to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['coref']['ceafe_not_singleton']['re'][0],
                                                        experiment_results['coref']['ceafe_not_singleton']['re'][1]))

            to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['coref']['ceafe_not_singleton']['f1'][0],
                                                    experiment_results['coref']['ceafe_not_singleton']['f1'][1]))
        else:
            if not only_f1:
                to_ret.append('CEAFe not singleton pr')
                to_ret.append('CEAFe not singleton re')
            to_ret.append('CEAFe not singleton f1')

        if not only_title:
            if not only_f1:
                to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['coref']['b_cubed_not_singleton']['pr'][0],
                                                        experiment_results['coref']['b_cubed_not_singleton']['pr'][1]))
                to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['coref']['b_cubed_not_singleton']['re'][0],
                                                        experiment_results['coref']['b_cubed_not_singleton']['re'][1]))

            to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['coref']['b_cubed_not_singleton']['f1'][0],
                                                    experiment_results['coref']['b_cubed_not_singleton']['f1'][1]))
        else:
            if not only_f1:
                to_ret.append('B-3 not singleton pr')
                to_ret.append('B-3 not singleton re')
            to_ret.append('B-3 not singleton f1')

    return to_ret


def get_csv_line_v2(experiment_config: Dict, experiment_results: Dict, experiment_id: str,
                    only_f1=False, only_title=False, tasks_to_report_on={'Coref', 'Linking'}):
    """

    The difference with get_csv_line is that in this get_csv_lin_v2 version some of the metrics such as entity linking
    with nils are left out, since this is not the current focus. Some other metrics, such as hard and soft linking are
    added. The idea is that the resulting .csv file is less confusing and more focused on the metrics that we are
    interested in at this particular point in time. This function is added Dec 4, 2020.

    :param experiment_config:
    :param experiment_results:
    :param experiment_id:
    :param only_f1:
    :param only_title:
    :param tasks_to_report_on:
    :return:
    """

    if not only_title:
        nr_runs = experiment_results['nr_runs']

        model_type = ''
        nilin = ''
        if 'linker' in experiment_config['model'] and experiment_config['model']['linker']['enabled']:
            model_type = 'baseline'
        elif 'linker' in experiment_config['model'] and 'linkercoref' in experiment_config['model'] and \
                experiment_config['model']['linkercoref']['enabled']:
            model_type = 'coref+link-' + experiment_config['model']['linkercoref']['model_type']
            nilin = not experiment_config['model']['linkercoref']['no_nil_in_targets']
        elif 'linker' in experiment_config['model'] and 'coreflinker' in experiment_config['model'] and \
                experiment_config['model']['coreflinker']['enabled']:
            nilin = not experiment_config['model']['coreflinker']['no_nil_in_targets']
            model_type = 'coref+link-' + experiment_config['model']['coreflinker']['type']
        else:
            model_type = 'other'
            nilin = False

        to_ret = [experiment_id, nr_runs, nilin, model_type]

        # to_ret = [experiment_id, nr_runs, field_is_bert, field_att_prop, field_ner_enabled,
        #           field_coref_enabled, field_coref_prop, field_rel_enabled, field_rel_prop]
    else:
        # to_ret = ['Experiment ID', 'Nr Runs', 'NILinM', 'Att Props', 'Tag', 'Coref', 'Coref props',
        #           'Rels', 'Rel props']
        to_ret = ['Experiment ID', 'Nr Runs', 'NILin', 'Model Type']

    if 'Tag' in tasks_to_report_on:
        if not only_title:
            if not only_f1:
                to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['tags']['tags_mention']['pr'][0],
                                                        experiment_results['tags']['tags_mention']['pr'][1]))
                to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['tags']['tags_mention']['re'][0],
                                                        experiment_results['tags']['tags_mention']['re'][1]))

            to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['tags']['tags_mention']['f1'][0],
                                                    experiment_results['tags']['tags_mention']['f1'][1]))
        else:
            if not only_f1:
                to_ret.append('Tag M-pr')
                to_ret.append('Tag M-re')
            to_ret.append('Tag M-f1')

        if not only_title:
            if not only_f1:
                to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['tags']['tags_soft']['pr'][0],
                                                        experiment_results['tags']['tags_soft']['pr'][1]))
                to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['tags']['tags_soft']['re'][0],
                                                        experiment_results['tags']['tags_soft']['re'][1]))

            to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['tags']['tags_soft']['f1'][0],
                                                    experiment_results['tags']['tags_soft']['f1'][1]))
        else:
            if not only_f1:
                to_ret.append('Tag S-pr')
                to_ret.append('Tag S-re')
            to_ret.append('Tag S-f1')

        if not only_title:
            if not only_f1:
                to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['tags']['tags_hard']['pr'][0],
                                                        experiment_results['tags']['tags_hard']['pr'][1]))
                to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['tags']['tags_hard']['re'][0],
                                                        experiment_results['tags']['tags_hard']['re'][1]))

            to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['tags']['tags_hard']['f1'][0],
                                                    experiment_results['tags']['tags_hard']['f1'][1]))
        else:
            if not only_f1:
                to_ret.append('Tag H-pr')
                to_ret.append('Tag H-re')
            to_ret.append('Tag H-f1')

    if 'Rels' in tasks_to_report_on:
        if not only_title:
            if not only_f1:
                to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['relations']['rels_mention']['pr'][0],
                                                        experiment_results['relations']['rels_mention']['pr'][1]))
                to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['relations']['rels_mention']['re'][0],
                                                        experiment_results['relations']['rels_mention']['re'][1]))

            to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['relations']['rels_mention']['f1'][0],
                                                    experiment_results['relations']['rels_mention']['f1'][1]))
        else:
            if not only_f1:
                to_ret.append('Rels M-pr')
                to_ret.append('Rels M-re')
            to_ret.append('Rels M-f1')

        if not only_title:
            if not only_f1:
                to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['relations']['rels_soft']['pr'][0],
                                                        experiment_results['relations']['rels_soft']['pr'][1]))
                to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['relations']['rels_soft']['re'][0],
                                                        experiment_results['relations']['rels_soft']['re'][1]))

            to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['relations']['rels_soft']['f1'][0],
                                                    experiment_results['relations']['rels_soft']['f1'][1]))
        else:
            if not only_f1:
                to_ret.append('Rels S-pr')
                to_ret.append('Rels S-re')
            to_ret.append('Rels S-f1')

        if not only_title:
            if not only_f1:
                to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['relations']['rels_hard']['pr'][0],
                                                        experiment_results['relations']['rels_hard']['pr'][1]))
                to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['relations']['rels_hard']['re'][0],
                                                        experiment_results['relations']['rels_hard']['re'][1]))

            to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['relations']['rels_hard']['f1'][0],
                                                    experiment_results['relations']['rels_hard']['f1'][1]))
        else:
            if not only_f1:
                to_ret.append('Rels H-pr')
                to_ret.append('Rels H-re')
            to_ret.append('Rels H-f1')

    if 'Linking' in tasks_to_report_on:

        # if not only_title:
        #     if not only_f1:
        #         to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['links']['links-links']['pr'][0],
        #                                                 experiment_results['links']['links-links']['pr'][1]))
        #         to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['links']['links-links']['re'][0],
        #                                                 experiment_results['links']['links-links']['re'][1]))
        #
        #     to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['links']['links-links']['f1'][0],
        #                                             experiment_results['links']['links-links']['f1'][1]))
        # else:
        #     if not only_f1:
        #         to_ret.append('Pr Links-links-from-mention')
        #         to_ret.append('Re Links-links-from-mention')
        #     to_ret.append('F1 Links-links-from-mention')

        if not only_title:
            if not only_f1:
                to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['links']['links-links-from-ent']['pr'][0],
                                                        experiment_results['links']['links-links-from-ent']['pr'][1]))
                to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['links']['links-links-from-ent']['re'][0],
                                                        experiment_results['links']['links-links-from-ent']['re'][1]))

            to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['links']['links-links-from-ent']['f1'][0],
                                                    experiment_results['links']['links-links-from-ent']['f1'][1]))
        else:
            if not only_f1:
                to_ret.append('Pr Links-links-from-ent')
                to_ret.append('Re Links-links-from-ent')
            to_ret.append('F1 Links-links-from-ent')

        if not only_title:
            if not only_f1:
                to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['links']['links-links-hard']['pr'][0],
                                                        experiment_results['links']['links-links-hard']['pr'][1]))
                to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['links']['links-links-hard']['re'][0],
                                                        experiment_results['links']['links-links-hard']['re'][1]))

            to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['links']['links-links-hard']['f1'][0],
                                                    experiment_results['links']['links-links-hard']['f1'][1]))

        else:
            if not only_f1:
                to_ret.append('Pr Links-links-hard')
                to_ret.append('Re Links-links-hard')
            to_ret.append('F1 Links-links-hard')

        if not only_title:
            if not only_f1:
                to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['links']['links-links-soft']['pr'][0],
                                                        experiment_results['links']['links-links-soft']['pr'][1]))
                to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['links']['links-links-soft']['re'][0],
                                                        experiment_results['links']['links-links-soft']['re'][1]))

            to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['links']['links-links-soft']['f1'][0],
                                                    experiment_results['links']['links-links-soft']['f1'][1]))

        else:
            if not only_f1:
                to_ret.append('Pr Links-links-soft')
                to_ret.append('Re Links-links-soft')
            to_ret.append('F1 Links-links-soft')

        if not only_title:
            if not only_f1:
                to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['links']['links-links-mentionsoft']['pr'][0],
                                                        experiment_results['links']['links-links-mentionsoft']['pr'][
                                                            1]))
                to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['links']['links-links-mentionsoft']['re'][0],
                                                        experiment_results['links']['links-links-mentionsoft']['re'][
                                                            1]))

            to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['links']['links-links-mentionsoft']['f1'][0],
                                                    experiment_results['links']['links-links-mentionsoft']['f1'][1]))

        else:
            if not only_f1:
                to_ret.append('Pr Links-links-mentionsoft')
                to_ret.append('Re Links-links-mentionsoft')
            to_ret.append('F1 Links-links-mentionsoft')

        # if not only_title:
        #     to_ret.append(
        #         '{:0.4f}+-{:0.4f}'.format(experiment_results['links']['links-accuracy']['acc_no_candidates'][0],
        #                                   experiment_results['links']['links-accuracy']['acc_no_candidates'][1]))
        # else:
        #     to_ret.append('Acc-link (NO candidates)')

        if not only_title:
            to_ret.append(
                '{:0.4f}+-{:0.4f}'.format(
                    experiment_results['links']['links-accuracy-from-ent']['acc_no_candidates'][0],
                    experiment_results['links']['links-accuracy-from-ent']['acc_no_candidates'][1]))
        else:
            to_ret.append('Acc-link-from-ent (NO candidates)')

    if 'Coref' in tasks_to_report_on:
        if not only_title:
            if not only_f1:
                to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['coref']['coref_avg']['pr'][0],
                                                        experiment_results['coref']['coref_avg']['pr'][1]))
                to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['coref']['coref_avg']['re'][0],
                                                        experiment_results['coref']['coref_avg']['re'][1]))

            to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['coref']['coref_avg']['f1'][0],
                                                    experiment_results['coref']['coref_avg']['f1'][1]))
        else:
            if not only_f1:
                to_ret.append('AVG pr (coref)')
                to_ret.append('AVG re (coref)')
            to_ret.append('AVG f1 (coref)')

        if not only_title:
            if not only_f1:
                to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['coref']['muc']['pr'][0],
                                                        experiment_results['coref']['muc']['pr'][1]))
                to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['coref']['muc']['re'][0],
                                                        experiment_results['coref']['muc']['re'][1]))

            to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['coref']['muc']['f1'][0],
                                                    experiment_results['coref']['muc']['f1'][1]))
        else:
            if not only_f1:
                to_ret.append('MUC pr')
                to_ret.append('MUC re')
            to_ret.append('MUC f1')

        # if not only_title:
        #     if not only_f1:
        #         to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['coref']['ceafm_singleton']['pr'][0],
        #                                                 experiment_results['coref']['ceafm_singleton']['pr'][1]))
        #         to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['coref']['ceafm_singleton']['re'][0],
        #                                                 experiment_results['coref']['ceafm_singleton']['re'][1]))
        #
        #     to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['coref']['ceafm_singleton']['f1'][0],
        #                                             experiment_results['coref']['ceafm_singleton']['f1'][1]))
        # else:
        #     if not only_f1:
        #         to_ret.append('CEAFm pr')
        #         to_ret.append('CEAFm re')
        #     to_ret.append('CEAFm f1')

        if not only_title:
            if not only_f1:
                to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['coref']['ceafe_singleton']['pr'][0],
                                                        experiment_results['coref']['ceafe_singleton']['pr'][1]))
                to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['coref']['ceafe_singleton']['re'][0],
                                                        experiment_results['coref']['ceafe_singleton']['re'][1]))

            to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['coref']['ceafe_singleton']['f1'][0],
                                                    experiment_results['coref']['ceafe_singleton']['f1'][1]))
        else:
            if not only_f1:
                to_ret.append('CEAFe pr')
                to_ret.append('CEAFe re')
            to_ret.append('CEAFe f1')

        if not only_title:
            if not only_f1:
                to_ret.append(
                    '{:0.4f}+-{:0.4f}'.format(experiment_results['coref']['b_cubed_singleton_men_conll']['pr'][0],
                                              experiment_results['coref']['b_cubed_singleton_men_conll']['pr'][
                                                  1]))
                to_ret.append(
                    '{:0.4f}+-{:0.4f}'.format(experiment_results['coref']['b_cubed_singleton_men_conll']['re'][0],
                                              experiment_results['coref']['b_cubed_singleton_men_conll']['re'][
                                                  1]))

            to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['coref']['b_cubed_singleton_men_conll']['f1'][0],
                                                    experiment_results['coref']['b_cubed_singleton_men_conll']['f1'][
                                                        1]))
        else:
            if not only_f1:
                to_ret.append('B-3 men conll pr')
                to_ret.append('B-3 men conll re')
            to_ret.append('B-3 men conll f1')

        # if not only_title:
        #     if not only_f1:
        #         to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['coref']['b_cubed_singleton_ent']['pr'][0],
        #                                                 experiment_results['coref']['b_cubed_singleton_ent']['pr'][1]))
        #         to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['coref']['b_cubed_singleton_ent']['re'][0],
        #                                                 experiment_results['coref']['b_cubed_singleton_ent']['re'][1]))
        #
        #     to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['coref']['b_cubed_singleton_ent']['f1'][0],
        #                                             experiment_results['coref']['b_cubed_singleton_ent']['f1'][1]))
        # else:
        #     if not only_f1:
        #         to_ret.append('B-3 ent pr')
        #         to_ret.append('B-3 ent re')
        #     to_ret.append('B-3 ent f1')

        # if not only_title:
        #     if not only_f1:
        #         to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['coref']['ceafe_not_singleton']['pr'][0],
        #                                                 experiment_results['coref']['ceafe_not_singleton']['pr'][1]))
        #         to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['coref']['ceafe_not_singleton']['re'][0],
        #                                                 experiment_results['coref']['ceafe_not_singleton']['re'][1]))
        #
        #     to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['coref']['ceafe_not_singleton']['f1'][0],
        #                                             experiment_results['coref']['ceafe_not_singleton']['f1'][1]))
        # else:
        #     if not only_f1:
        #         to_ret.append('CEAFe not singleton pr')
        #         to_ret.append('CEAFe not singleton re')
        #     to_ret.append('CEAFe not singleton f1')
        #
        # if not only_title:
        #     if not only_f1:
        #         to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['coref']['b_cubed_not_singleton']['pr'][0],
        #                                                 experiment_results['coref']['b_cubed_not_singleton']['pr'][1]))
        #         to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['coref']['b_cubed_not_singleton']['re'][0],
        #                                                 experiment_results['coref']['b_cubed_not_singleton']['re'][1]))
        #
        #     to_ret.append('{:0.4f}+-{:0.4f}'.format(experiment_results['coref']['b_cubed_not_singleton']['f1'][0],
        #                                             experiment_results['coref']['b_cubed_not_singleton']['f1'][1]))
        # else:
        #     if not only_f1:
        #         to_ret.append('B-3 not singleton pr')
        #         to_ret.append('B-3 not singleton re')
        #     to_ret.append('B-3 not singleton f1')

    return to_ret

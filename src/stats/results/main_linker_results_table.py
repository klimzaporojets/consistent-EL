import argparse
import json
import logging
import os
from statistics import stdev, mean

from tqdm import tqdm

from misc.cpn_eval import load_jsonl, EvaluatorCPN, load_json

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger()


def get_coref_metrics(cpn_evaluator: EvaluatorCPN):
    to_ret_coref_metrics = {
        'ceafe_not_singleton': {'f1': cpn_evaluator.coref_ceafe.get_f1(),
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
            're': cpn_evaluator.coref_bcubed_singleton_men.get_re()},
        'muc': {
            'f1': cpn_evaluator.coref_muc.get_f1(),
            'pr': cpn_evaluator.coref_muc.get_pr(),
            're': cpn_evaluator.coref_muc.get_re()
        }
    }

    to_ret_coref_metrics['avg'] = dict()
    avg_f1 = (to_ret_coref_metrics['b_cubed_singleton_men_conll']['f1'] +
              to_ret_coref_metrics['muc']['f1'] +
              to_ret_coref_metrics['ceafe_singleton']['f1']) / 3
    avg_re = (to_ret_coref_metrics['b_cubed_singleton_men_conll']['re'] +
              to_ret_coref_metrics['muc']['re'] +
              to_ret_coref_metrics['ceafe_singleton']['re']) / 3
    avg_pr = (to_ret_coref_metrics['b_cubed_singleton_men_conll']['pr'] +
              to_ret_coref_metrics['muc']['pr'] +
              to_ret_coref_metrics['ceafe_singleton']['pr']) / 3

    to_ret_coref_metrics['avg']['f1'] = avg_f1
    to_ret_coref_metrics['avg']['re'] = avg_re
    to_ret_coref_metrics['avg']['pr'] = avg_pr

    return to_ret_coref_metrics


def get_linking_metrics(cpn_evaluator: EvaluatorCPN):
    to_ret_linker_metrics = {
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

    return to_ret_linker_metrics


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file'
                        , type=str
                        , default='experiments/evaluate_config.json'
                        , help='configuration file')
    args = parser.parse_args()

    config_file_path = args.config_file

    config_file = json.load(open(config_file_path, 'rt'))
    base_path = config_file['base_path']
    dataset_to_details = dict()
    # dataset -> setup -> metric name -> result
    dataset_to_setup_to_metrics = dict()

    dataset_to_loaded_ground_truth = dict()
    dataset_names = list()
    for curr_dataset in config_file['datasets']:
        loaded_gold_test = dict()
        dataset_names.append(curr_dataset['dataset'])
        dataset_to_details[curr_dataset['dataset']] = curr_dataset
        ground_truth_path = curr_dataset['ground_truth_path']
        logger.info('loading dataset %s' % curr_dataset['dataset'])
        for curr_file in tqdm(os.listdir(ground_truth_path)):
            if 'main' not in curr_file:
                curr_file_path = os.path.join(ground_truth_path, curr_file)
                loaded_json = load_json(curr_file_path, tag=None)
                loaded_gold_test = dict(loaded_gold_test, **loaded_json)

        dataset_to_loaded_ground_truth[curr_dataset['dataset']] = loaded_gold_test
        dataset_to_setup_to_metrics[curr_dataset['dataset']] = dict()

    metrics_list = ['ELm', 'ELh', 'Coref']
    metrics_per_task = {'entity-linking': ['ELm', 'ELh'], 'coreference': ['Coref']}
    setups_list = ['Standalone', 'Local', 'Global']
    for curr_setup in config_file['setups']:
        curr_setup_alias = curr_setup['setup_alias']
        for curr_setup_task in curr_setup['setup_tasks']:
            for curr_prediction in curr_setup['predictions']:
                curr_pred_file_path = curr_prediction['path']
                curr_prediction_dataset = curr_prediction['dataset']
                if curr_setup_alias not in dataset_to_setup_to_metrics[curr_prediction_dataset]:
                    dataset_to_setup_to_metrics[curr_prediction_dataset][curr_setup_alias] = dict()
                for curr_task_metric in metrics_per_task[curr_setup_task]:
                    if curr_task_metric not in dataset_to_setup_to_metrics[curr_prediction_dataset][curr_setup_alias]:
                        dataset_to_setup_to_metrics[curr_prediction_dataset][curr_setup_alias][curr_task_metric] = \
                            {'list': list(), 'avg': None, 'std': None, 'is_best': False}

                subset_tag = curr_prediction['subset_tag']
                for curr_run_path in curr_prediction['runs']:
                    prediction_path = os.path.join(base_path, curr_run_path, curr_pred_file_path)
                    if os.path.exists(prediction_path):
                        logger.info('loading predictions from %s' % prediction_path)
                        loaded_predicted_test = load_jsonl(prediction_path, subset_tag)
                        ####
                        loaded_gold_test = dataset_to_loaded_ground_truth[curr_prediction_dataset]
                        s1 = set(loaded_gold_test)
                        s2 = set(loaded_predicted_test)
                        res = s1 & s2
                        assert len(res) == len(loaded_predicted_test)
                        loaded_gold_test = {s: loaded_gold_test[s] for s in res}
                        assert len(loaded_gold_test) == len(loaded_predicted_test)
                        logger.info('adding to evaluator predictions from %s' % prediction_path)
                        cpn_evaluator = EvaluatorCPN()

                        for idx, identifier in enumerate(loaded_gold_test.keys()):
                            cpn_evaluator.add(loaded_predicted_test[identifier], loaded_gold_test[identifier])

                        curr_metrics = None
                        if curr_setup_task == 'entity-linking':
                            curr_metrics = get_linking_metrics(cpn_evaluator)
                        elif curr_setup_task == 'coreference':
                            curr_metrics = get_coref_metrics(cpn_evaluator)
                        else:
                            raise RuntimeError('setup task not recognized: %s' % curr_setup_task)
                        for curr_task_metric in metrics_per_task[curr_setup_task]:
                            if curr_task_metric == 'ELm':
                                curr_calculated_metric = curr_metrics['links-links-from-ents']['f1'] * 100
                            elif curr_task_metric == 'ELh':
                                curr_calculated_metric = curr_metrics['links-links-hard']['f1'] * 100
                            elif curr_task_metric == 'Coref':
                                curr_calculated_metric = curr_metrics['avg']['f1'] * 100
                            else:
                                raise RuntimeError('No such metric %s' % curr_task_metric)
                            dataset_to_setup_to_metrics[curr_prediction_dataset][curr_setup_alias] \
                                [curr_task_metric]['list'].append(curr_calculated_metric)
                    else:
                        logger.warning('the prediction path does not exist: %s' % prediction_path)

    # calculates intermediate results such as the average and standard deviation (std)
    for curr_setup_name in setups_list:
        for curr_dataset_name in dataset_names:
            for curr_metric_name in metrics_list:
                m_results = dataset_to_setup_to_metrics[curr_dataset_name][curr_setup_name][curr_metric_name]
                avg = mean(m_results['list'])
                if len(m_results['list']) > 1:
                    std = stdev(m_results['list'])
                else:
                    std = 0.0
                m_results['avg'] = avg
                m_results['std'] = std

    # gets the best values (by setting 'is_best' to true) for each of the metrics in a particular dataset
    # TODO

    # prints the table
    setup_name_to_replace = dict()
    title_row_datasets = '{:^10}|'.format('')
    title_row_metrics = '{:^10}|'.format('Setup')
    results_row_generic = '{:^10}|'
    title_to_replace = []
    for idx_setup, curr_setup_name in enumerate(setups_list):
        setup_name_to_replace[curr_setup_name] = list()
        setup_name_to_replace[curr_setup_name].append(curr_setup_name)
        for curr_dataset_name in dataset_names:
            if idx_setup == 0:
                title_row_datasets += '{:^24}|'.format(curr_dataset_name)

            for idx_metric, curr_metric_name in enumerate(metrics_list):
                m_results = dataset_to_setup_to_metrics[curr_dataset_name][curr_setup_name][curr_metric_name]
                replace = '{:.2f}'.format(m_results['avg'])

                setup_name_to_replace[curr_setup_name].append(replace)
                if idx_setup == 0:
                    results_row_generic += '{:^8}'
                    title_row_metrics += '{:^8}'.format(curr_metric_name)
                    if idx_metric == len(metrics_list) - 1:
                        title_row_metrics += '|'
                        results_row_generic += '|'
    print(title_row_datasets)
    print(title_row_metrics)
    titles_separator = '-' * len(title_row_metrics)
    print(titles_separator)
    for curr_setup_name in setups_list:
        print(results_row_generic.format(*setup_name_to_replace[curr_setup_name]))

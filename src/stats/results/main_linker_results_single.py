import argparse
import logging
import os

from misc.cpn_eval import load_jsonl, EvaluatorCPN, load_json

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger()


def print_single_result(task_to_metrics: Dict):
    """

    :param task_to_metrics:
    :return:

    Metrics to print for entity linking:
        1- links-links-hard (F1)
        2- links-links-from-ents (F1)

    Metrics to print for coreference resolution:
        1- Avg. F1
    """
    print('{:<10}{:<10}{:<20}'.format('ELm (F1)', 'ELh (F1)', 'Coref (Avg. F1)'))
    print('{:<10}{:<10}{:<20}'.format(
        '{:.3f}'.format(task_to_metrics['links']['links-links-from-ents']['f1']),
        '{:.3f}'.format(task_to_metrics['links']['links-links-hard']['f1']),
        '{:.3f}'.format(task_to_metrics['coref']['avg']['f1'])))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--predicted_path'
                        , type=str
                        , default='experiments/dwie-standalone-linking/e1/predictions/test/test.jsonl'
                        , help='configuration file')

    parser.add_argument('--ground_truth_path'
                        , type=str
                        # , default='data/aida+/plain/testa/'
                        , default='data/dwie/plain_format/data/annos_with_content/'
                        , help='configuration file')

    args = parser.parse_args()

    predicted_path = args.predicted_path
    ground_truth_path = args.ground_truth_path

    loaded_predicted_test = load_jsonl(predicted_path, None)

    cpn_evaluator = EvaluatorCPN()

    loaded_gold_test = dict()

    for curr_file in os.listdir(ground_truth_path):
        if 'main' not in curr_file:
            curr_file_path = os.path.join(ground_truth_path, curr_file)
            loaded_json = load_json(curr_file_path, tag=None)
            loaded_gold_test = dict(loaded_gold_test, **loaded_json)

    # only leaves in the gold the ones with the same id as predicted
    s1 = set(loaded_gold_test)
    s2 = set(loaded_predicted_test)
    res = s1 & s2
    assert len(res) == len(loaded_predicted_test)
    loaded_gold_test = {s: loaded_gold_test[s] for s in res}
    assert len(loaded_gold_test) == len(loaded_predicted_test)

    for idx, identifier in enumerate(loaded_gold_test.keys()):
        cpn_evaluator.add(loaded_predicted_test[identifier], loaded_gold_test[identifier])

    task_to_metrics = dict()

    task_to_metrics['coref'] = {'ceafe_not_singleton': {'f1': cpn_evaluator.coref_ceafe.get_f1(),
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
    task_to_metrics['coref']['avg'] = dict()
    avg_f1 = (task_to_metrics['coref']['b_cubed_singleton_men_conll']['f1'] +
              task_to_metrics['coref']['muc']['f1'] +
              task_to_metrics['coref']['ceafe_singleton']['f1']) / 3
    avg_re = (task_to_metrics['coref']['b_cubed_singleton_men_conll']['re'] +
              task_to_metrics['coref']['muc']['re'] +
              task_to_metrics['coref']['ceafe_singleton']['re']) / 3
    avg_pr = (task_to_metrics['coref']['b_cubed_singleton_men_conll']['pr'] +
              task_to_metrics['coref']['muc']['pr'] +
              task_to_metrics['coref']['ceafe_singleton']['pr']) / 3

    task_to_metrics['coref']['avg']['f1'] = avg_f1
    task_to_metrics['coref']['avg']['re'] = avg_re
    task_to_metrics['coref']['avg']['pr'] = avg_pr

    task_to_metrics['links'] = {
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

    print_single_result(task_to_metrics)

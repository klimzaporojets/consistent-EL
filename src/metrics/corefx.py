import logging

# from misc.python_cpn_eval import MetricCoref
from misc.cpn_eval import MetricCoref

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger()


# version modified by Klim to include singletons
class MetricCorefExternal:

    def __init__(self, task, verbose=False):
        self.task = task
        self.debug = False
        self.iter = 0
        self.clear()

    def clear(self):
        self.coref_muc = MetricCoref('muc', MetricCoref.muc)
        self.coref_bcubed_singleton_men = MetricCoref('bcubed-m', MetricCoref.b_cubed_singleton_mentions)
        self.coref_ceafe_singleton_ent = MetricCoref('ceaf-e', MetricCoref.ceafe_singleton_entities)

    def step(self):
        self.clear()
        self.iter += 1

    def update2(self, output_dict, metadata):
        for idx, (pred, gold) in enumerate(zip(output_dict['pred'], output_dict['gold'])):
            # print("pred:", len(pred), pred)
            # print("gold:", len(gold), gold)
            self.coref_muc.add(pred, gold)
            # self.coref_bcubed.add(pred, gold)
            self.coref_bcubed_singleton_men.add(pred, gold)
            # self.coref_bcubed_singleton_ent.add(pred, gold)
            # self.coref_ceafe.add(pred, gold)
            # self.coref_ceafe_singleton_men.add(pred, gold)
            self.coref_ceafe_singleton_ent.add(pred, gold)

            if self.debug:
                logger.debug('ID %s' % metadata['identifiers'][idx])
                logger.debug('pred: %s', pred)
                logger.debug('gold: %s', gold)
                tokens = metadata['tokens'][idx]
                logger.debug('pred: %s' %
                             ([[' '.join(tokens[begin:(end + 1)]) for begin, end in cluster] for cluster in pred]))
                logger.debug('gold: %s' %
                             ([[' '.join(tokens[begin:(end + 1)]) for begin, end in cluster] for cluster in gold]))

    def print(self, dataset_name, details=False):
        logger.info('EVAL-COREF\t{}-{}\tcurr-iter: {}\t{}-f1: {}'
                    .format(dataset_name, self.task, self.iter, 'muc-ext', self.coref_muc.get_f1()))
        logger.info('EVAL-COREF\t{}-{}\tcurr-iter: {}\t{}-f1: {}'
                    .format(dataset_name, self.task, self.iter, 'bcubed-m-ext',
                            self.coref_bcubed_singleton_men.get_f1()))
        logger.info('EVAL-COREF\t{}-{}\tcurr-iter: {}\t{}-f1: {}'
                    .format(dataset_name, self.task, self.iter, 'ceaf-e-ext', self.coref_ceafe_singleton_ent.get_f1()))
        tmp = (self.coref_muc.get_f1() + self.coref_bcubed_singleton_men.get_f1() +
               self.coref_ceafe_singleton_ent.get_f1()) / 3
        logger.info('EVAL-COREF\t{}-{}\tcurr-iter: {}\t{}-f1: {}'
                    .format(dataset_name, self.task, self.iter, 'avg-ext', tmp))

    def log(self, tb_logger, dataset_name):
        # (kzaporoj) - log to tensorboard
        tb_logger.log_value('metrics-coref-ext/{}-f1'.format('muc-ext'), self.coref_muc.get_f1(), self.iter)
        tb_logger.log_value('metrics-coref-ext/{}-f1'.format('bcubed-m-ext'),
                            self.coref_bcubed_singleton_men.get_f1(), self.iter)
        tb_logger.log_value('metrics-coref-ext/{}-f1'.format('ceaf-e-ext'), self.coref_ceafe_singleton_ent.get_f1(),
                            self.iter)
        tmp = (self.coref_muc.get_f1() + self.coref_bcubed_singleton_men.get_f1() +
               self.coref_ceafe_singleton_ent.get_f1()) / 3
        tb_logger.log_value('metrics-coref-ext/{}-f1'.format('avg-ext'), tmp, self.iter)

        # precision

        tb_logger.log_value('metrics-coref-ext/{}-pr'.format('muc-ext'), self.coref_muc.get_pr(), self.iter)
        tb_logger.log_value('metrics-coref-ext/{}-pr'.format('bcubed-m-ext'),
                            self.coref_bcubed_singleton_men.get_pr(), self.iter)
        tb_logger.log_value('metrics-coref-ext/{}-pr'.format('ceaf-e-ext'), self.coref_ceafe_singleton_ent.get_pr(),
                            self.iter)
        tmp = (self.coref_muc.get_pr() + self.coref_bcubed_singleton_men.get_pr() +
               self.coref_ceafe_singleton_ent.get_pr()) / 3
        tb_logger.log_value('metrics-coref-ext/{}-pr'.format('avg-ext'), tmp, self.iter)
        # recall
        tb_logger.log_value('metrics-coref-ext/{}-re'.format('avg-ext'), tmp, self.iter)
        tb_logger.log_value('metrics-coref-ext/{}-re'.format('muc-ext'), self.coref_muc.get_re(), self.iter)
        tb_logger.log_value('metrics-coref-ext/{}-re'.format('bcubed-m-ext'),
                            self.coref_bcubed_singleton_men.get_re(), self.iter)
        tb_logger.log_value('metrics-coref-ext/{}-re'.format('ceaf-e-ext'), self.coref_ceafe_singleton_ent.get_re(),
                            self.iter)
        tmp = (self.coref_muc.get_re() + self.coref_bcubed_singleton_men.get_re() +
               self.coref_ceafe_singleton_ent.get_re()) / 3
        tb_logger.log_value('metrics-coref-ext/{}-re'.format('avg-ext'), tmp, self.iter)

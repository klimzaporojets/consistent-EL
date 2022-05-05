from external.python_cpn_eval import MetricCoref


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
                print("ID", metadata['identifiers'][idx])
                print("pred:", pred)
                print("gold:", gold)
                tokens = metadata['tokens'][idx]
                print("pred:", [[' '.join(tokens[begin:(end + 1)]) for begin, end in cluster] for cluster in pred])
                print("gold:", [[' '.join(tokens[begin:(end + 1)]) for begin, end in cluster] for cluster in gold])
                print()

    def print(self, dataset_name, details=False):
        print('EVAL-COREF\t{}-{}\tcurr-iter: {}\t{}-f1: {}'.format(dataset_name, self.task, self.iter, "muc-ext",
                                                                   self.coref_muc.get_f1()))
        print('EVAL-COREF\t{}-{}\tcurr-iter: {}\t{}-f1: {}'.format(dataset_name, self.task, self.iter, "bcubed-m-ext",
                                                                   self.coref_bcubed_singleton_men.get_f1()))
        print('EVAL-COREF\t{}-{}\tcurr-iter: {}\t{}-f1: {}'.format(dataset_name, self.task, self.iter, "ceaf-e-ext",
                                                                   self.coref_ceafe_singleton_ent.get_f1()))
        tmp = (
                      self.coref_muc.get_f1() + self.coref_bcubed_singleton_men.get_f1() + self.coref_ceafe_singleton_ent.get_f1()) / 3
        print('EVAL-COREF\t{}-{}\tcurr-iter: {}\t{}-f1: {}'.format(dataset_name, self.task, self.iter, "avg-ext", tmp))

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

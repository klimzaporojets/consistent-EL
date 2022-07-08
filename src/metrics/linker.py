import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger()


def exclude_links(links):
    return [link for link in links if link[2] == 'NILL']


def exclude_nills(links):
    return [link for link in links if link[2] != 'NILL']


class MetricLinkerImproved:

    def __init__(self, task, mode='default'):
        self.task = task
        self.mode = mode
        self.epoch = 0
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.best_f1 = 0
        self.best_epoch = 0

        if mode == 'nills':
            self.filter = exclude_links
        elif mode == 'links':
            self.filter = exclude_nills
        else:
            self.filter = lambda x: x

    def step(self):
        self.epoch += 1
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def update2(self, args):
        for pred, gold in zip(args['pred'], args['gold']):
            if pred is None and gold is None:
                continue
            pred = self.filter(pred)
            gold = self.filter(gold)

            P = set(pred)
            G = set(gold)

            self.tp += len(P & G)
            self.fp += len(P - G)
            self.fn += len(G - P)

    def get_pr(self):
        return self.tp / (self.tp + self.fp) if self.tp != 0 else 0.0

    def get_re(self):
        return self.tp / (self.tp + self.fn) if self.tp != 0 else 0.0

    def get_f1(self):
        return 2.0 * self.tp / (2.0 * self.tp + self.fp + self.fn) if self.tp != 0 else 0.0

    def print(self, dataset_name):
        if self.get_f1() > self.best_f1:
            self.best_f1 = self.get_f1()
            self.best_epoch = self.epoch

        stall = self.epoch - self.best_epoch

        logger.info(
            'EVAL-LINKER\t{}-{}-{}\ttp: {:7}   fp: {:7}   fn: {:7}     pr: {:7.6f}   re: {:7.6f}   '
            'f1: {:7.6f}    stall: {:2}   best-f1: {:7.6f}'
                .format(dataset_name, self.task, self.mode, self.tp, self.fp, self.fn, self.get_pr(), self.get_re(),
                        self.get_f1(), stall, self.best_f1))

    def log(self, tb_logger):
        tb_logger.log_value('metrics-{}/{}'.format(self.task, self.mode + '-f1'), self.get_f1(), self.epoch)
        tb_logger.log_value('metrics-{}/{}'.format(self.task, self.mode + '-re'), self.get_re(), self.epoch)
        tb_logger.log_value('metrics-{}/{}'.format(self.task, self.mode + '-pr'), self.get_pr(), self.epoch)


# Look for top scoring link, and ignore nills
class MetricLinkAccuracy:

    def __init__(self, task):
        self.task = task
        self.epoch = 0
        self.numer = 0
        self.denom = 0
        self.best_linkacc = 0
        self.best_epoch = 0

    def step(self):
        self.epoch += 1
        self.numer = 0
        self.denom = 0

    def update2(self, args):
        for pred, gold in zip(args['scores'], args['gold']):
            if pred is None and (gold is None or len(gold) == 0):
                continue
            P = []
            G = exclude_nills(gold)

            for (begin, end), candidates, scores in pred:
                best_score = - float("inf")
                best_candidate = None
                for candidate, score in zip(candidates, scores):
                    if candidate == 'NILL':
                        continue
                    elif score > best_score:
                        best_score = score
                        best_candidate = candidate
                if best_candidate is not None:
                    P.append((begin, end, best_candidate))

            P = set(P)
            G = set(G)

            self.numer += len(P & G)
            self.denom += len(G)

    def print(self, dataset_name):
        acc = self.numer / self.denom if self.numer != 0 else 0.0
        if acc > self.best_linkacc:
            self.best_linkacc = acc
            self.best_epoch = self.epoch

        stall = self.epoch - self.best_epoch

        logger.info('EVAL-LINKER\t{}-{}\tlink-acc: {:7} / {:7} = {:7.6f}    stall: {:2}   best: {:7.6f}'
                    .format(dataset_name, self.task, self.numer, self.denom, acc, stall, self.best_linkacc))

    def log(self, tb_logger):
        acc = self.numer / self.denom if self.numer != 0 else 0.0
        tb_logger.log_value('metrics-{}/{}'.format(self.task, 'acc'), acc, self.epoch)


# (kzaporoj) - accuracy without any candidates, based on predictions only, ignores the NILLs
class MetricLinkAccuracyNoCandidates:

    def __init__(self, task):
        self.task = task
        self.epoch = 0
        self.numer = 0
        self.denom = 0
        self.best_linkacc = 0
        self.best_epoch = 0

    def step(self):
        self.epoch += 1
        self.numer = 0
        self.denom = 0

    def update2(self, args):
        for pred, gold in zip(args['pred'], args['gold']):
            if pred is None and (gold is None or len(gold) == 0):
                continue
            P = exclude_nills(pred)
            G = exclude_nills(gold)

            P = set(P)
            G = set(G)

            self.numer += len(P & G)
            self.denom += len(G)

    def print(self, dataset_name):
        acc = self.numer / self.denom if self.numer != 0 else 0.0
        if acc > self.best_linkacc:
            self.best_linkacc = acc
            self.best_epoch = self.epoch

        stall = self.epoch - self.best_epoch

        logger.info(
            'EVAL-LINKER\t{}-{}\tlink-acc (no candidates): {:7} / {:7} = {:7.6f}    stall: {:2}   best: {:7.6f}'
                .format(dataset_name, self.task, self.numer, self.denom, acc, stall, self.best_linkacc))

    def log(self, tb_logger):
        acc = self.numer / self.denom if self.numer != 0 else 0.0
        tb_logger.log_value('metrics-{}/{}'.format(self.task, 'acc-no-candidates'), acc, self.epoch)

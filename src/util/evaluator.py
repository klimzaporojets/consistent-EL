from util.linker import update_linker, update_linker2, evaluate_linker
from util.ner import ner_to_list, update_ner, evaluate_ner


class EvaluatorList:

    def __init__(self):
        self.evals = {}

    def add(self, name, evaluator):
        self.evals[name] = evaluator

    def setup(self, progress, epoch):
        for name, evaluator in self.evals.items():
            if name not in progress:
                progress[name] = {'name': '{}-{}'.format(progress['name'], name)}
            evaluator.setup(progress[name], epoch)

    def evaluate(self, tb_logger=None):
        for eval in self.evals.values():
            eval.evaluate(tb_logger)


class EvaluatorNER:

    def __init__(self, name, labels):
        self.name = name
        self.labels = labels
        self.stats = None
        self.progress = None

    def setup(self, progress, epoch):
        self.stats = {'name': progress['name'], 'epoch': epoch}
        self.progress = progress

    def update(self, pred, gold, sequence_lengths, obj):
        gold = ner_to_list(gold, sequence_lengths)
        update_ner(self.stats, pred, gold, self.labels, obj)

    def evaluate(self, tb_logger=None):
        evaluate_ner(self.progress, self.stats, tb_logger)


class EvaluatorLinker:

    def __init__(self):
        self.stats = None
        self.progress = None

    def setup(self, progress, epoch):
        self.stats = {'name': progress['name'], 'epoch': epoch}
        self.progress = progress

    def update(self, linker_scores, linker_targets, table, linker_obj):
        update_linker(self.stats, linker_scores, linker_targets, table, linker_obj)

    def update2(self, linker_scores, linker_targets, table, linker_obj):
        update_linker2(self.stats, linker_scores, linker_targets, table, linker_obj)

    def evaluate(self, tb_logger=None):
        evaluate_linker(self.progress, self.stats)

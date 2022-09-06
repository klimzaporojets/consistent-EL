import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger()


class MetricObjective:

    def __init__(self, task):
        self.task = task
        self.clear()

    def clear(self):
        self.total = 0
        self.iter = 0

    def step(self):
        self.total = 0
        self.iter += 1

    # def update(self, logits, targets, args, metadata={}):
    #     self.total += args['obj']

    def update2(self, args, metadata=None):
        self.total += args['loss']

    def print(self, dataset_name):
        logger.info('EVAL-OBJ\t{}-{}\tcurr-iter: {}\tobj: {}'.format(dataset_name, self.task, self.iter, self.total))

    def log(self, tb_logger):
        tb_logger.log_value('metrics/{}/obj'.format(self.task), self.total, self.iter)

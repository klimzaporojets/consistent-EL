import argparse
import json
import logging
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from tensorboard_logger import Logger as TBLogger
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import AdamW

from data_processing.dataset import create_datasets
from data_processing.dictionary import Dictionary
from misc import settings
from models import CoreflinkerSpanBertHoi
from models.misc.linker import create_dictionaries, create_model

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger()


def do_evaluate(model, dataset, metrics, batch_size, filename=None, tb_logger=None, epoch=-1,
                config=None):
    device = torch.device(settings.device)
    collate_fn = model.collate_func()

    loader = DataLoader(dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=False)
    name = dataset.name

    model.eval()

    for m in metrics:
        m.step()

    if hasattr(model, 'begin_epoch'):
        model.begin_epoch()

    if filename is None:
        total_obj = 0
        for i, minibatch in enumerate(loader):
            obj, outputs = model.forward(**minibatch, metrics=metrics)
            if isinstance(obj, torch.Tensor):
                total_obj += obj.item()
            else:
                logging.info('SKIP MINIBATCH')
    else:
        logging.info('Writing predictions to %s' % filename)
        with open(filename, 'w', encoding='utf-8') as file:
            total_obj = 0
            for i, minibatch in enumerate(loader):
                # the config , how they are passed also a hack, TODO: make this cleaner
                obj, predictions = model.predict(**minibatch, metrics=metrics, output_config=config['output_config'])
                if isinstance(obj, torch.Tensor):
                    total_obj += obj.item()
                else:
                    logging.info('SKIP MINIBATCH')

                for pred in predictions:
                    json.dump(pred, file, ensure_ascii=False)
                    file.write('\n')

    tb_logger.log_value('loss/', total_obj, epoch)

    for m in metrics:
        m.print(name)
        m.log(tb_logger)

    logging.info('%s-avg-loss: %s' % (name, total_obj / (len(loader) + 1e-7)))

    if hasattr(model, 'log_stats'):
        if name == 'train':
            name = 'train-ep'  # if not will get mixed with log_stats when training
        model.log_stats(name, tb_logger, epoch)


def do_evaluate_only_loss(model, dataset, batch_size, tb_logger=None, epoch=-1):
    collate_fn = model.collate_func()

    loader = DataLoader(dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=False)
    name = dataset.name

    model.eval()

    if hasattr(model, 'begin_epoch'):
        model.begin_epoch()

    total_obj = 0
    for i, minibatch in enumerate(loader):
        obj, outputs = model.forward(**minibatch, metrics=[], only_loss=True)
        if isinstance(obj, torch.Tensor):
            total_obj += obj.item()
        else:
            logging.info('SKIP MINIBATCH')

    tb_logger.log_value('loss/', total_obj, epoch)

    logging.info('%s-avg-loss (only loss, no metrics): %s' % (name, total_obj / (len(loader) + 1e-7)))

    if hasattr(model, 'log_stats'):
        if name == 'train':
            name = 'train-ep'  # if not will get mixed with log_stats when training
        model.log_stats(name, tb_logger, epoch)


class Runner:
    def __init__(self, config):
        self.config = config
        self.bert_train_steps = config['lr-scheduler']['nr_iters_bert_training']

    def get_optimizer(self, model):
        no_decay = ['bias', 'LayerNorm.weight']
        bert_param, task_param = model.get_params(named=True)

        logger.debug('===bert_params in DECAY: ' +
                     str([n for n, p in bert_param if not any(nd in n for nd in no_decay)]))

        logger.debug('===bert_params in NO DECAY: ' +
                     str([n for n, p in bert_param if any(nd in n for nd in no_decay)]))

        logger.debug('=== task params: ' + str([n for n, p in task_param]))

        grouped_bert_param = [
            {
                'params': [p for n, p in bert_param if not any(nd in n for nd in no_decay)],
                'lr': self.config['lr-scheduler']['bert_learning_rate_start'],
                'weight_decay': self.config['optimizer']['adam_weight_decay']
            }, {
                'params': [p for n, p in bert_param if any(nd in n for nd in no_decay)],
                'lr': self.config['lr-scheduler']['bert_learning_rate_start'],
                'weight_decay': 0.0
            }
        ]
        optimizers = [
            AdamW(grouped_bert_param, lr=self.config['lr-scheduler']['bert_learning_rate_start'],
                  eps=self.config['optimizer']['adam_eps']),
            Adam(model.get_params()[1], lr=self.config['lr-scheduler']['task_learning_rate_start'],
                 eps=self.config['optimizer']['adam_eps'], weight_decay=0)
        ]
        return optimizers

    def get_scheduler_v2(self, optimizers, bert_start_step, bert_end_step, task_start_step, task_end_step,
                         min_lambda_bert, min_lambda_tasks):
        """
        The version _v2 intends to incorporate the nr of update steps for which bert and the tasks will be updated.
        """

        # Only warm up bert lr
        total_update_steps_bert = bert_end_step - bert_start_step
        total_update_steps_tasks = task_end_step - task_start_step
        warmup_steps = int(total_update_steps_bert * self.config['lr-scheduler']['bert_warmup_ratio'])

        ratio_increase_per_task_step = (min_lambda_tasks - 1.0) / total_update_steps_tasks
        ratio_increase_per_bert_step = (min_lambda_bert - 1.0) / (total_update_steps_bert - warmup_steps)

        def lr_lambda_bert(current_step):
            if current_step < bert_start_step:
                return 1.0  # no changes to learning rate

            if (current_step - bert_start_step) < warmup_steps and current_step >= bert_start_step:
                to_ret = float(current_step - bert_start_step + 1) / float(max(1, warmup_steps))
                return to_ret
            to_ret = max(min_lambda_bert,
                         ratio_increase_per_bert_step * (current_step - bert_start_step - warmup_steps + 1) + 1.0)
            return to_ret

        def lr_lambda_task(current_step):
            if current_step < task_start_step:
                return 1.0  # no changes to learning rate

            to_ret = max(min_lambda_tasks, ratio_increase_per_task_step * (current_step - task_start_step) + 1.0)
            return to_ret

        schedulers = [
            LambdaLR(optimizers[0], lr_lambda_bert),
            LambdaLR(optimizers[1], lr_lambda_task)
        ]
        return schedulers

    def train_spanbert(self, model: CoreflinkerSpanBertHoi, datasets):
        logging.info('CURRENT settings.device VALUE IS: %s' % settings.device)
        logging.info('TRAINING, PLEASE WAIT...')
        conf = self.config
        epochs = conf['optimizer']['iters']

        grad_accum = conf['optimizer']['gradient_accumulation_steps']
        batch_size = conf['optimizer']['batch_size']

        device_name = settings.device
        device = torch.device(device_name)
        model = model.to(device)
        collate_fn = model.collate_func()
        train = DataLoader(datasets[conf['trainer']['train']], collate_fn=collate_fn, batch_size=batch_size,
                           shuffle=True)

        examples_train = datasets['train']

        optimizers = self.get_optimizer(model)

        bert_start_epoch = config['lr-scheduler']['bert_start_epoch']
        bert_end_epoch = config['lr-scheduler']['bert_end_epoch']
        task_start_epoch = config['lr-scheduler']['task_start_epoch']
        task_end_epoch = config['lr-scheduler']['task_end_epoch']

        bert_start_step = len(examples_train) * bert_start_epoch // grad_accum
        bert_end_step = len(examples_train) * bert_end_epoch // grad_accum
        task_start_step = len(examples_train) * task_start_epoch // grad_accum
        task_end_step = len(examples_train) * task_end_epoch // grad_accum

        bert_learning_rate_start = config['lr-scheduler']['bert_learning_rate_start']
        bert_learning_rate_end = config['lr-scheduler']['bert_learning_rate_end']

        task_learning_rate_start = config['lr-scheduler']['task_learning_rate_start']
        task_learning_rate_end = config['lr-scheduler']['task_learning_rate_end']

        min_lambda_bert = bert_learning_rate_end / bert_learning_rate_start
        min_lambda_task = task_learning_rate_end / task_learning_rate_start

        schedulers = self.get_scheduler_v2(optimizers, bert_start_step, bert_end_step, task_start_step, task_end_step,
                                           min_lambda_bert, min_lambda_task)

        bert_param, task_param = model.get_params()

        tb_loggers = dict()
        for data_tag in datasets.keys():
            tb_logger_path = os.path.join(conf['path'], 'tensorboard_logs', data_tag)
            os.makedirs(tb_logger_path, exist_ok=True)
            tb_loggers[data_tag] = TBLogger(tb_logger_path, flush_secs=10)

        loss_during_accum = []  # To compute effective loss at each update
        loss_during_report = 0.0  # Effective loss during logging step
        loss_history = []  # Full history of effective loss; length equals total update steps

        metrics = {name: model.create_metrics() for name in conf['trainer']['evaluate']}

        start_time = time.time()
        model.zero_grad()
        optimizer_steps = 0
        train_tag = conf['trainer']['train']
        for epo in range(epochs):
            settings.epoch = epo
            max_norm_bert = 0
            max_norm_tasks = 0

            logger.info('EPOCH: ' + str(epo))

            for i, minibatch in enumerate(train):
                model.train()

                loss, _ = model.forward(**minibatch)

                if loss is None or isinstance(loss, int) or isinstance(loss, float):
                    logger.warning('SKIP EMPTY MINIBATCH')
                    continue

                if grad_accum > 1:
                    loss /= grad_accum
                loss.backward()

                if 'clip-norm' in conf['optimizer']:
                    norm_bert = torch.nn.utils.clip_grad_norm_(bert_param, conf['optimizer']['clip-norm'])
                    norm_tasks = torch.nn.utils.clip_grad_norm_(task_param, conf['optimizer']['clip-norm'])

                    if norm_bert > max_norm_bert:
                        max_norm_bert = norm_bert

                    if norm_tasks > max_norm_tasks:
                        max_norm_tasks = norm_tasks

                loss_during_accum.append(loss.item())

                # Update
                if len(loss_during_accum) % grad_accum == 0:
                    for optimizer in optimizers:
                        optimizer.step()
                    model.zero_grad()
                    for scheduler in schedulers:
                        scheduler.step()

                    # Compute effective loss
                    effective_loss = np.sum(loss_during_accum).item()
                    loss_during_accum = []
                    loss_during_report += effective_loss
                    loss_history.append(effective_loss)
                    optimizer_steps += 1

                    # Report
                    if optimizer_steps % conf['optimizer']['report_frequency'] == 0:
                        # Show avg loss during last report interval
                        avg_loss = loss_during_report / conf['optimizer']['report_frequency']
                        loss_during_report = 0.0
                        end_time = time.time()
                        logger.info('\tStep %d: avg loss %.2f; steps/sec %.2f' %
                                    (optimizer_steps, avg_loss, conf['optimizer']['report_frequency']
                                     / (end_time - start_time)))
                        start_time = end_time

                        tb_loggers[train_tag].log_value('loss/', avg_loss, optimizer_steps)
                        tb_loggers[train_tag].log_value('max-norm-bert/', max_norm_bert, optimizer_steps)
                        max_norm_bert = 0
                        tb_loggers[train_tag].log_value('max-norm-tasks/', max_norm_tasks, optimizer_steps)
                        max_norm_tasks = 0

                        tb_loggers[train_tag].log_value('learning-rate-bert/', schedulers[0].get_last_lr()[0],
                                                        len(loss_history))
                        tb_loggers[train_tag].log_value('learning-rate-task/', schedulers[1].get_last_lr()[-1],
                                                        len(loss_history))

                        model.log_stats(train_tag, tb_loggers[train_tag], optimizer_steps)

                    if self.bert_train_steps > 0 and self.bert_train_steps == optimizer_steps:
                        logging.info('freezing bert parameters after %s optimizer steps ' % optimizer_steps)
                        for param in model.embedder.spanbert_embedder.spanbert_model.parameters():
                            param.requires_grad = False

            with torch.no_grad():

                for name in conf['trainer']['evaluate']:
                    evaluate_only_loss = False
                    evaluate_all = False
                    if conf['trainer']['loss_frequency'][name] >= 1:
                        if epo < epochs - 1 and (epo + 1) % conf['trainer']['loss_frequency'][name] != 0:
                            evaluate_only_loss = False
                        else:
                            # TODO: evaluate only loss
                            evaluate_only_loss = True

                    if epo == epochs - 1:
                        evaluate_all = True
                    elif (epo + 1) % conf['trainer']['evaluation_frequency'][name] == 0 and \
                            conf['trainer']['evaluation_frequency'][name] > 0:
                        evaluate_all = True

                    if epo < epochs - 1:
                        predict_file = None
                        if evaluate_all and conf['trainer']['write-predictions']:
                            file_name = '{}_ep{:03d}.jsonl'.format(name, epo)
                            base_dir = conf['path']
                            subdir = 'predictions/{}/'.format(name)
                            predict_file = os.path.join(base_dir, subdir, file_name)
                            dirname = os.path.dirname(predict_file)
                            os.makedirs(dirname, exist_ok=True)
                    elif conf['trainer']['write-predictions']:
                        file_name = '{}.jsonl'.format(name)
                        base_dir = conf['path']
                        subdir = 'predictions/{}/'.format(name)
                        predict_file = os.path.join(base_dir, subdir, file_name)
                        dirname = os.path.dirname(predict_file)
                        os.makedirs(dirname, exist_ok=True)

                    if evaluate_all:
                        tb_logger_evaluate = tb_loggers[name]
                        do_evaluate(model, datasets[name], metrics[name], batch_size, predict_file, tb_logger_evaluate,
                                    epo, config=conf)
                    elif evaluate_only_loss:
                        tb_logger_evaluate = tb_loggers[name]
                        do_evaluate_only_loss(model, datasets[name], batch_size, tb_logger_evaluate, epo)

        while True:
            try:
                if config['optimizer']['write-last-model']:
                    logger.info('writing last.model to %s' % config['path'])
                    output_dir = os.path.join(config['path'], 'stored_models')
                    os.makedirs(output_dir, exist_ok=True)
                    model.write_model(os.path.join(output_dir, config['optimizer']['model']))
                    break
            except:
                logging.error('ERROR: failed to write model to disk')
                time.sleep(60)


def load_model(config, training=False, load_datasets_from_config=True):
    """
    creates the model from config, it can be either used to train it later, or to load the parameters of a saved model
    using state_dict() (https://pytorch.org/tutorials/beginner/saving_loading_models.html) .
    when using state_dict() to load saved model, the static embeddings are not loaded, only the trainable parameters are
    loaded, so it is necessary to use this method before the state_dict().

    :param config:
    :return:
    """

    dictionaries = create_dictionaries(config, training)
    datasets = None

    if load_datasets_from_config:
        datasets, data, evaluate = create_datasets(config, dictionaries)
    model, parameters = create_model(config, dictionaries)

    to_ret = {
        'dictionaries': dictionaries,
        'datasets': datasets,
        'model': model,
        'parameters': parameters
    }
    return to_ret


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', help='configuration file')
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--device', dest='device', type=str, default='cuda')
    args = parser.parse_args()

    commit_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip()
    commit_hash = commit_hash.decode('utf-8')

    dump_path = args.output_path
    json.dump({'commit_id': commit_hash}, open(os.path.join(dump_path, 'commit_info.json'), 'w'))

    settings.device = args.device

    with open(args.config_file) as f:
        config = json.load(f)

    if args.output_path is not None:
        logging.info('Setting path to %s' % args.output_path)
        config['path'] = args.output_path
        os.makedirs(config['path'], exist_ok=True)
    elif 'path' not in config:
        logging.info('set path=%s' % Path(args.config_file).parent)
        config['path'] = Path(args.config_file).parent

    config['dictionaries_path'] = os.path.join(config['path'], 'dictionaries')

    loaded_model_dict = load_model(config, True, load_datasets_from_config=True)
    dictionaries = loaded_model_dict['dictionaries']
    datasets = loaded_model_dict['datasets']
    model = loaded_model_dict['model']
    parameters = loaded_model_dict['parameters']

    logging.info('Dictionaries:')
    for name, dictionary in dictionaries.items():
        if isinstance(dictionary, Dictionary):  # sometimes can be BertTokenizer for example
            logging.info('- %s: %s (oov=%s)' % (name, dictionary.size, dictionary.out_of_voc))

            filename = os.path.join(config['dictionaries_path'], '{}.json'.format(name))
            filedir = os.path.dirname(filename)
            os.makedirs(filedir, exist_ok=True)
            logging.info('  write %s' % filename)
            dictionary.write(filename)

    if config['trainer']['version'] == 'spanbert':
        runner = Runner(config=config)
        runner.train_spanbert(model, datasets)
    else:
        raise RuntimeError('no implementation for the following trainer version: ' +
                           config['trainer']['version'])

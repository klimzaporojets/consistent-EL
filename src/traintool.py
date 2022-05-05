import argparse
import json
import logging
import os
import subprocess
import time
from pathlib import Path

# import git
import numpy as np
import torch
from tensorboard_logger import Logger as TBLogger
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW

import settings
from datass.dataset import create_datasets
from datass.dictionary import Dictionary
from linker import create_dictionaries, create_model, create_linking_candidates
from lrschedule import create_lr_scheduler
from models import CoreflinkerSpanBertHoi

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger()


def do_evaluate(model, dataset, metrics, batch_size, filename=None, tb_logger=None, epoch=-1,
                config=None):
    # device = torch.device("cuda")
    device = torch.device(settings.device)
    # print('device in do_evaluate is: ', device)
    collate_fn = model.collate_func(datasets, device)

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
                print("SKIP MINIBATCH")
    else:
        print("Writing predictions to {}".format(filename))
        with open(filename, 'w', encoding='utf-8') as file:
            total_obj = 0
            for i, minibatch in enumerate(loader):
                # the config , how they are passed also "atadura con alambres", need to change in final version
                obj, predictions = model.predict(**minibatch, metrics=metrics, output_config=config['output_config'])
                if isinstance(obj, torch.Tensor):
                    total_obj += obj.item()
                else:
                    print("SKIP MINIBATCH")

                for pred in predictions:
                    # json.dump(pred, file)
                    json.dump(pred, file, ensure_ascii=False)
                    file.write('\n')

    tb_logger.log_value('loss/', total_obj, epoch)

    for m in metrics:
        m.print(name, True)
        m.log(tb_logger, name)

    print("{}-avg-loss: {}".format(name, total_obj / (len(loader) + 1e-7)))

    # model.log_stats(name)
    # if hasattr(model, 'end_epoch'):
    #     model.end_epoch(name)
    if hasattr(model, 'log_stats'):
        if name == 'train':
            name = 'train-ep'  # if not will get mixed with log_stats when training
        model.log_stats(name, tb_logger, epoch)


def do_evaluate_only_loss(model, dataset, metrics, batch_size, filename=None, tb_logger=None, epoch=-1, config=None):
    # device = torch.device("cuda")
    device = torch.device(settings.device)
    # print('device in do_evaluate is: ', device)
    collate_fn = model.collate_func(datasets, device)

    loader = DataLoader(dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=False)
    name = dataset.name

    model.eval()

    # for m in metrics:
    #     m.step()

    if hasattr(model, 'begin_epoch'):
        model.begin_epoch()

    total_obj = 0
    for i, minibatch in enumerate(loader):
        obj, outputs = model.forward(**minibatch, metrics=[], only_loss=True)
        if isinstance(obj, torch.Tensor):
            total_obj += obj.item()
        else:
            print("SKIP MINIBATCH")

    tb_logger.log_value('loss/', total_obj, epoch)

    # for m in metrics:
    #     m.print(name, True)
    #     m.log(tb_logger, name)

    print("{}-avg-loss (only loss, no metrics): {}".format(name, total_obj / (len(loader) + 1e-7)))

    if hasattr(model, 'log_stats'):
        if name == 'train':
            name = 'train-ep'  # if not will get mixed with log_stats when training
        model.log_stats(name, tb_logger, epoch)
    # if hasattr(model, 'end_epoch'):
    #     model.end_epoch(name)


class Runner:
    def __init__(self, config, seed=None):
        self.config = config
        self.bert_train_steps = config['lr-scheduler']['nr_iters_bert_training']

    def get_optimizer(self, model):
        no_decay = ['bias', 'LayerNorm.weight']
        bert_param, task_param = model.get_params(named=True)

        logger.info('===bert_params in DECAY: ' +
                    str([n for n, p in bert_param if not any(nd in n for nd in no_decay)]))

        logger.info('===bert_params in NO DECAY: ' +
                    str([n for n, p in bert_param if any(nd in n for nd in no_decay)]))

        logger.info('=== task params: ' + str([n for n, p in task_param]))

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

    def get_scheduler(self, optimizers, total_update_steps):
        # Only warm up bert lr
        warmup_steps = int(total_update_steps * self.config['optimizer']['warmup_ratio'])

        def lr_lambda_bert(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(
                0.0, float(total_update_steps - current_step) / float(max(1, total_update_steps - warmup_steps))
            )

        def lr_lambda_task(current_step):
            return max(0.0, float(total_update_steps - current_step) / float(max(1, total_update_steps)))

        schedulers = [
            LambdaLR(optimizers[0], lr_lambda_bert),
            LambdaLR(optimizers[1], lr_lambda_task)
        ]
        return schedulers
        # return LambdaLR(optimizer, [lr_lambda_bert, lr_lambda_bert, lr_lambda_task, lr_lambda_task])

    def get_scheduler_v2(self, optimizers, bert_start_step, bert_end_step, task_start_step, task_end_step,
                         min_lambda_bert, min_lambda_tasks):
        """
        The version _v2 intends to incorporate the nr of update steps for which bert and the tasks will be updated.
        """

        # Only warm up bert lr
        total_update_steps_bert = bert_end_step - bert_start_step
        total_update_steps_tasks = task_end_step - task_start_step
        warmup_steps = int(total_update_steps_bert * self.config['lr-scheduler']['bert_warmup_ratio'])

        # total_proj_steps_task = total_update_steps_tasks/(1.0 - min_lambda_tasks)

        ratio_increase_per_task_step = (min_lambda_tasks - 1.0) / total_update_steps_tasks
        ratio_increase_per_bert_step = (min_lambda_bert - 1.0) / (total_update_steps_bert - warmup_steps)

        def lr_lambda_bert(current_step):
            if current_step < bert_start_step:
                return 1.0  # no changes to learning rate

            if (current_step - bert_start_step) < warmup_steps and current_step >= bert_start_step:
                to_ret = float(current_step - bert_start_step + 1) / float(max(1, warmup_steps))
                return to_ret
            # return max(
            #     min_lambda_bert, float(total_update_steps_bert - current_step) /
            #                      float(max(1, total_update_steps_bert - warmup_steps))
            # )
            to_ret = max(min_lambda_bert,
                         ratio_increase_per_bert_step * (current_step - bert_start_step - warmup_steps + 1) + 1.0)
            return to_ret

        def lr_lambda_task(current_step):
            # if current_step > total_update_steps_tasks:
            #     current_step = total_update_steps_tasks
            if current_step < task_start_step:
                return 1.0  # no changes to learning rate

            to_ret = max(min_lambda_tasks, ratio_increase_per_task_step * (current_step - task_start_step) + 1.0)
            return to_ret
            # if ratio_increase_per_task_step > 0.0:
            #     return (ratio_increase_per_task_step * current_step)
            # else:
            #     return max(min_lambda_tasks, float(total_update_steps_tasks - current_step) /
            #                float(max(1, total_update_steps_tasks)))

        schedulers = [
            LambdaLR(optimizers[0], lr_lambda_bert),
            LambdaLR(optimizers[1], lr_lambda_task)
        ]
        return schedulers
        # return LambdaLR(optimizer, [lr_lambda_bert, lr_lambda_bert, lr_lambda_task, lr_lambda_task])

    def train_spanbert(self, model: CoreflinkerSpanBertHoi, datasets, parameters):
        print('CURRENT settings.device VALUE IS: ', settings.device)
        conf = self.config
        epochs = conf['optimizer']['iters']

        grad_accum = conf['optimizer']['gradient_accumulation_steps']
        # max_epochs = conf['optimizer']['iters']
        batch_size = conf['optimizer']['batch_size']

        device_name = settings.device
        device = torch.device(device_name)
        model = model.to(device)
        collate_fn = model.collate_func(datasets, device)
        train = DataLoader(datasets[conf['trainer']['train']], collate_fn=collate_fn, batch_size=batch_size,
                           shuffle=True)

        examples_train = datasets['train']
        # total_update_steps = len(examples_train) * epochs // grad_accum

        optimizers = self.get_optimizer(model)
        # schedulers = self.get_scheduler(optimizers, total_update_steps)

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
            tb_logger_path = '{}/{}'.format(conf['path'], data_tag)
            # tb_logger_train_path = '{}/{}'.format(conf['path'], 'train')
            os.makedirs(tb_logger_path, exist_ok=True)
            # os.makedirs(tb_logger_train_path, exist_ok=True)
            # tb_logger_train = TBLogger(tb_logger_train_path, flush_secs=10)
            # tb_logger_test = TBLogger(tb_logger_test_path, flush_secs=10)
            tb_loggers[data_tag] = TBLogger(tb_logger_path, flush_secs=10)

        loss_during_accum = []  # To compute effective loss at each update
        loss_during_report = 0.0  # Effective loss during logging step
        loss_history = []  # Full history of effective loss; length equals total update steps

        metrics = {name: model.create_metrics() for name in conf['trainer']['evaluate']}

        start_time = time.time()
        model.zero_grad()
        optimizer_steps = 0
        # train_tag = datasets[conf['trainer']['train']]
        train_tag = conf['trainer']['train']
        # todo: delete from prod this set_detect_anomaly!!!
        # torch.autograd.set_detect_anomaly(True)
        for epo in range(epochs):
            settings.epoch = epo
            max_norm_bert = 0
            max_norm_tasks = 0

            logger.info('\n EPOCH: ' + str(epo))

            # if hasattr(model, 'begin_epoch'):
            #     model.begin_epoch()

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
                    # norm = torch.nn.utils.clip_grad_norm_(model.parameters(), conf['optimizer']['clip-norm'])
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
                    # if len(loss_history) % conf['report_frequency'] == 0:
                    if optimizer_steps % conf['optimizer']['report_frequency'] == 0:
                        # Show avg loss during last report interval
                        avg_loss = loss_during_report / conf['optimizer']['report_frequency']
                        loss_during_report = 0.0
                        end_time = time.time()
                        logger.info('\n Step %d: avg loss %.2f; steps/sec %.2f' %
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
                        print('freezing bert parameters after ', optimizer_steps, ' optimizer steps')
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

                    # if conf['trainer']['evaluation_frequency'][name] == -1 and epo < epochs - 1:
                    #     continue
                    # elif conf['trainer']['evaluation_frequency'][name] > 1:
                    #     if epo < epochs - 1 and epo % conf['trainer']['evaluation_frequency'][name] != 0:
                    #         continue
                    if epo == epochs - 1:
                        evaluate_all = True
                    elif (epo + 1) % conf['trainer']['evaluation_frequency'][name] == 0 and \
                            conf['trainer']['evaluation_frequency'][name] > 0:
                        evaluate_all = True

                    if epo < epochs - 1:
                        predict_file = None
                        if evaluate_all:
                            # pass
                            predict_file = '{}/{}_ep{:03d}.jsonl'.format(conf['path'], name, epo) \
                                if conf['trainer']['write-predictions'] else None
                    else:
                        predict_file = '{}/{}.jsonl'.format(conf['path'], name) if conf['trainer'][
                            'write-predictions'] else None

                    # tb_logger_evaluate = tb_logger_train if name == 'train' else tb_logger_test
                    if evaluate_all:
                        tb_logger_evaluate = tb_loggers[name]
                        do_evaluate(model, datasets[name], metrics[name], batch_size, predict_file, tb_logger_evaluate,
                                    epo, config=conf)
                    elif evaluate_only_loss:
                        tb_logger_evaluate = tb_loggers[name]
                        # TODO: we are working on this evaluation of only loss without actual predictions
                        do_evaluate_only_loss(model, datasets[name], metrics[name], batch_size, predict_file,
                                              tb_logger_evaluate, epo)

        #


def train(model, datasets, config, parameters):
    print('CURRENT settings.device VALUE IS: ', settings.device)
    device_name = settings.device
    # device = torch.device("cuda")
    device = torch.device(device_name)
    model = model.to(device)

    collate_fn = model.collate_func(datasets, device)
    max_epochs = config['optimizer']['iters']
    batch_size = config['optimizer']['batch_size']
    # lrate0 = config['optimizer']['lrate0']

    filename = config['optimizer'].get('init-model', None)
    if filename is not None:
        print("Initializing model {}".format(filename))
        model.load_model(filename, config['model'])

    train = DataLoader(datasets[config['trainer']['train']], collate_fn=collate_fn, batch_size=batch_size, shuffle=True)

    opt_type = config['optimizer'].get('optimizer', 'adam')
    swa = False
    if opt_type == 'adam':
        weight_decay = config['optimizer'].get('weight_decay', 0.0)
        print("ADAM: weight_decay={}".format(weight_decay))
        optimizer = torch.optim.Adam(parameters, weight_decay=weight_decay)  # , lr=lrate0
    elif opt_type == 'asgd':
        t0 = config['optimizer']['t0'] * len(train)
        print("asgd t0:", t0)
        optimizer = torch.optim.ASGD(parameters, t0=t0)
    elif opt_type == 'adam-swa':
        import torchcontrib.optim
        optimizer = torch.optim.Adam(parameters, lr=1e-3)
        optimizer = torchcontrib.optim.SWA(optimizer)
        swa_start = config['optimizer']['swa-start']
        swa_freq = config['optimizer']['swa-freq']
        swa = True
    # lrate=lrate0

    # print('CURRENT settings.device VALUE IS: ', settings.device)

    # setup logger
    tb_logger_all_path = '{}/{}'.format(config['path'], 'params')
    tb_logger_test_path = '{}/{}'.format(config['path'], 'test')
    tb_logger_train_path = '{}/{}'.format(config['path'], 'train')
    os.makedirs(tb_logger_all_path, exist_ok=True)
    os.makedirs(tb_logger_test_path, exist_ok=True)
    os.makedirs(tb_logger_train_path, exist_ok=True)
    tb_logger_all = TBLogger(tb_logger_all_path)
    tb_logger_train = TBLogger(tb_logger_train_path)
    tb_logger_test = TBLogger(tb_logger_test_path)

    print("Start optimization for {} iterations with batch_size={}".format(max_epochs, batch_size))
    name2progress = {name: {'name': name} for name in config['trainer']['evaluate']}

    scheduler = create_lr_scheduler(optimizer, config, max_epochs, len(train))

    # TODO: move this to lrschedule.py
    monitor = None
    factor = 1.0
    patience = 0
    if 'lrate-schedule' in config['trainer']:
        # try factor = 0.9
        monitor = name2progress[config['trainer']['lrate-schedule']['monitor']]
        factor = config['trainer']['lrate-schedule']['factor']
        module = config['trainer']['lrate-schedule']['module']
        metric = config['trainer']['lrate-schedule']['metric']
        patience = config['trainer']['lrate-schedule']['patience']

    metrics = {name: model.create_metrics() for name in config['trainer']['evaluate']}

    if hasattr(model, 'init'):
        model.init(train)

    # (kzaporoj) - this is O(n), but its ok since it is done only once
    # train_metrics = []

    # (kzaporoj) - this was not a good idea since the .eval() first has to be invoked, which makes functions
    # such as dropout() behave differently from training mode.
    # if train.dataset.name in config['trainer']['evaluate']:
    #     # remove it because do not want to run an evaluation on train itself when the metrics can be gather during the
    #     # training process
    #     config['trainer']['evaluate'].remove(train.dataset.name)
    #     train_metrics = metrics[train.dataset.name]

    # model.tb_logger = tb_logger

    for epoch in range(max_epochs):
        # for m in train_metrics:
        #     m.step()

        # print('optimizer', optimizer.state)
        # train_metrics = []
        # if train.dataset.name in metrics:
        #     train_metrics = model.create_metrics()

        # TODO: move this to lrschedule.py
        if 'lrate-schedule' in config['trainer']:
            print(monitor)
            if module in monitor:
                if (monitor[module][metric]['stall'] + 1) % patience == 0:
                    print("decrease lrate")
                    lrate *= factor
                    for group in optimizer.param_groups:
                        group['lr'] = lrate
            else:
                print("WARNING: no such module:", module)

        model.train()
        total_obj = 0
        tic = time.time()
        max_norm = 0

        if hasattr(model, 'begin_epoch'):
            model.begin_epoch()

        # train
        for i, minibatch in enumerate(tqdm(train)):
            # print("BEGIN\t", torch.cuda.memory_allocated() // (1024*1024))

            lrate = scheduler.step()
            # obj, _ = model.forward(**minibatch, metrics=train_metrics)
            obj, _ = model.forward(**minibatch)
            # print('type:', type(obj))

            # (kzaporoj) - there can be that for some reason (ex: no mentions with candidates in linker), the loss
            # is 0.0 (it is not a tensor, but a float) ; in this case we do not do a grad propagation
            if obj is None or isinstance(obj, int) or isinstance(obj, float):
                print("SKIP EMPTY MINIBATCH")
                continue

            total_obj += obj.item()
            optimizer.zero_grad()
            obj.backward()

            if 'clip-norm' in config['optimizer']:
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['optimizer']['clip-norm'])
                if norm > max_norm:
                    max_norm = norm

            # for name, param in model.named_parameters():
            #     print("{} norm={} grad={}".format(name, param.data.norm().item(), param.grad.norm().item() if param.grad is not None else -1))

            optimizer.step()

            # print("END\t", torch.cuda.memory_allocated() // (1024*1024))
            # break

        if hasattr(model, 'end_epoch'):
            model.end_epoch('train')

        if swa and epoch >= swa_start and epoch % swa_freq == 0:
            print("UPDATE-SWA")
            optimizer.update_swa()

        print("{}\tobj: {}   time: {}    lrate: {}".format(epoch, total_obj, time.time() - tic, lrate))

        tb_logger_train.log_value('loss/', total_obj, epoch)
        tb_logger_train.log_value('max-norm/', max_norm, epoch)
        tb_logger_train.log_value('optimizer/lrate', lrate, epoch)

        for name, param in model.named_parameters():
            tb_logger_all.log_value('parameters/mean/{}'.format(name), param.mean().item(), epoch)
            tb_logger_all.log_value('parameters/stdv/{}'.format(name), param.std().item(), epoch)
            tb_logger_all.log_value('parameters/norm/{}'.format(name), param.norm().item(), epoch)
            # print('calculating the max of ', param)
            # print('calculating the max of shape', param.shape)
            if min(list(param.shape)) == 0:  # (kzaporoj) - sometimes empty is coming
                continue
            else:
                tb_logger_all.log_value('parameters/max/{}'.format(name), param.max().item(), epoch)

        try:
            if config['optimizer']['write-iter-model']:
                model.write_model('{}/{}.model'.format(config['path'], epoch))
        except:
            print("ERROR: failed to write model to disk")

        if swa and epoch >= swa_start:
            optimizer.swap_swa_sgd()

        # if some metrics that were gathered during train process, then log them to tb_logger_train
        # for m in train_metrics:
        #     m.print(train.dataset.name, True)
        #     m.log(tb_logger_train, train.dataset.name)

        # evaluate
        with torch.no_grad():
            for name in config['trainer']['evaluate']:
                if config['trainer']['evaluation_frequency'][name] == -1 and epoch < max_epochs - 1:
                    continue
                elif config['trainer']['evaluation_frequency'][name] > 1:
                    if epoch < max_epochs - 1 and epoch % config['trainer']['evaluation_frequency'][name] != 0:
                        continue

                if epoch < max_epochs - 1:
                    # predict_file = '{}/{}_temp.jsonl'.format(config['path'], name) if config['trainer'][
                    #     'write-predictions'] else None
                    # (kzaporoj) - added the log of results only for the last epoch to not log too much:
                    # for some reason the jobs tend to fail when writing to _temp.jsonl
                    predict_file = None
                else:
                    predict_file = '{}/{}.jsonl'.format(config['path'], name) if config['trainer'][
                        'write-predictions'] else None

                tb_logger_evaluate = tb_logger_train if name == 'train' else tb_logger_test
                do_evaluate(model, datasets[name], metrics[name], batch_size, predict_file, tb_logger_evaluate, epoch,
                            config=config)
                # do_evaluate(model, datasets[name], metrics[name], 1, predict_file, tb_logger, epoch)

        if swa and epoch >= swa_start:
            optimizer.swap_swa_sgd()

    while True:
        try:
            if config['optimizer']['write-last-model']:
                model.write_model('{}/last.model'.format(config['path']))
                break
        except:
            print("ERROR: failed to write model to disk")
            time.sleep(60)


def test(model, datasets, config):
    device = torch.device(settings.device)
    model = model.to(device)

    collate_fn = model.collate_func(datasets, device)
    max_epochs = config['optimizer']['iters']
    batch_size = config['optimizer']['batch_size']

    filename = config['optimizer'].get('model', None)
    if filename is not None:
        print("Initializing model {}".format(filename))
        model.load_model(config['path'] + '/' + filename, config['model'])

    # train = DataLoader(datasets[config['trainer']['train']], collate_fn=collate_fn, batch_size=batch_size, shuffle=True)
    #
    # name2progress = {name: {'name': name} for name in config['trainer']['evaluate']}

    metrics = {name: model.create_metrics() for name in config['trainer']['evaluate']}

    # evaluate
    for name in config['trainer']['evaluate']:
        print("Start evaluating", name)

        loader = DataLoader(datasets[name], collate_fn=collate_fn, batch_size=batch_size, shuffle=False)

        model.eval()

        for m in metrics[name]:
            m.step()

        total_obj = 0
        for i, minibatch in enumerate(tqdm(loader)):
            obj, outputs = model.forward(**minibatch, metrics=metrics[name])
            total_obj += obj.item()

        for m in metrics[name]:
            m.print(name, True)


def predict_dwie_linker(model, datasets, config, output_path):
    """
    Originally, this function is just a copy-paste of predict(model,datasets,config) function (see below), the idea
    is to specifically test is with exported conll aida dataset in end-to-end setting.
    :param model:
    :param datasets:
    :param config:
    :return:
    """
    device = torch.device(settings.device)
    model = model.to(device)
    print('device to be assigned to collate_func: ', device)
    collate_fn = model.collate_func(datasets, device)
    batch_size = config['optimizer']['batch_size']

    filename = config['optimizer'].get('model', None)
    if filename is not None:
        print("Initializing model {}".format(filename))
        if settings.device == 'cpu':
            # model.load_model('{}/{}'.format(config['path'], filename), config['model'], load_word_embeddings=False)
            model.load_model('{}/{}'.format(config['path'], filename), to_cpu=True, load_word_embeddings=False)
        else:
            model.load_model('{}/{}'.format(config['path'], filename), load_word_embeddings=False)

    model = model.to(device)
    output_config = {
        "output_content": True,
        "_output_content": "Whether the 'content' is added to prediction json file.",
        "output_tokens": True,
        "_output_tokens": "Whether the 'tokens' are added to prediction json file. "
    }

    # evaluate
    for name in config['trainer']['evaluate']:
        print("Start evaluating", name)

        loader = DataLoader(datasets[name], collate_fn=collate_fn, batch_size=batch_size, shuffle=False)

        model.eval()

        if hasattr(model, 'begin_epoch'):
            model.begin_epoch()

        with open(os.path.join(output_path, '{}.jsonl').format(name), 'w') as file:
            for i, minibatch in enumerate(tqdm(loader)):
                _, predictions = model.predict(**minibatch, output_config=output_config)
                for pred in predictions:
                    json.dump(pred, file)
                    file.write('\n')

        if hasattr(model, 'end_epoch'):
            model.end_epoch(name)


def predict(model, datasets, config):
    device = torch.device(settings.device)
    model = model.to(device)

    collate_fn = model.collate_func(datasets, device)
    batch_size = config['optimizer']['batch_size']

    filename = config['optimizer'].get('model', None)
    if filename is not None:
        print("Initializing model {}".format(filename))
        model.load_model('{}/{}'.format(config['path'], filename), config['model'])

    # evaluate
    for name in config['trainer']['evaluate']:
        print("Start evaluating", name)

        loader = DataLoader(datasets[name], collate_fn=collate_fn, batch_size=batch_size, shuffle=False)

        model.eval()

        if hasattr(model, 'begin_epoch'):
            model.begin_epoch()

        with open('predictions.json', 'w') as file:
            for i, minibatch in enumerate(tqdm(loader)):
                _, predictions = model.predict(**minibatch)
                for pred in predictions:
                    json.dump(pred, file)
                    file.write('\n')

        if hasattr(model, 'end_epoch'):
            model.end_epoch(name)


def load_model(config, training=False, load_datasets_from_config=True):
    """
    creates the model from config, it can be either used to train it later, or to load the parameters of a saved model
    using state_dict() (https://pytorch.org/tutorials/beginner/saving_loading_models.html) .
    when using state_dict() to load saved model, the static embeddings are not loaded, only the trainable parameters are
    loaded, so it is necessary to use this method before the state_dict().

    :param config:
    :return:
    """
    # TODO, 26/11/2020

    # dictionaries = create_dictionaries(config, args.mode == 'train')
    dictionaries = create_dictionaries(config, training)
    linking_candidates = None
    datasets = None
    if 'linking_candidates' in config:
        # this also adds respective entries to the dictionary
        linking_candidates = create_linking_candidates(config['linking_candidates'], dictionaries['entities'])
        # create_linking_candidates(config['linking_candidates'], dictionaries['entities'])
    if load_datasets_from_config:
        datasets, data, evaluate = create_datasets(config, dictionaries, args.fold, linking_candidates)
    model, parameters = create_model(config, dictionaries)

    # dictionaries = loaded_model_dict['dictionaries']
    # datasets = loaded_model_dict['datasets']
    # model = loaded_model_dict['model']

    # return model, parameters, linking_candidates, dictionaries
    to_ret = {
        'dictionaries': dictionaries,
        'datasets': datasets,
        'model': model,
        'parameters': parameters,
        'linking_candidates': linking_candidates
    }
    return to_ret


if __name__ == "__main__":
    print("Start")

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "test", "predict"], help="mode")
    parser.add_argument("config_file", help="configuration file")
    parser.add_argument("--fold", dest="fold", type=int, default=-1)
    parser.add_argument("--debugging", dest="debugging", type=int, default=0)
    parser.add_argument("--path", dest="path", type=str, default=None)
    parser.add_argument("--device", dest="device", type=str, default="cuda")
    args = parser.parse_args()

    commit_hash = subprocess.check_output(["git", "rev-parse", '--short', 'HEAD']).strip()
    commit_hash = commit_hash.decode('utf-8')

    # repo = git.Repo(search_parent_directories=True)
    # commit_hash = repo.head.object.hexsha

    dump_path = os.path.dirname(args.config_file)
    json.dump({'commit_id': commit_hash}, open(os.path.join(dump_path, 'commit_info.json'), 'w'))
    if args.debugging == 1:
        # the seed is set for debugging purposes, only when debugging
        torch.manual_seed(1234)
        np.random.seed(1234)

    settings.device = args.device

    with open(args.config_file) as f:
        config = json.load(f)

    if 'path' not in config:
        print('set path=', Path(args.config_file).parent)
        config['path'] = Path(args.config_file).parent

    if args.path is not None:
        print("WARNING: setting path to {}".format(args.path))
        config['path'] = args.path
    settings.path = config['path']
    settings.debugging_path = os.path.join(config['path'], 'debugging_logs')
    os.makedirs(settings.debugging_path, exist_ok=True)

    # model, parameters, linking_candidates, dictionaries = load_model(config, args.mode == 'train',
    #                                                                  load_datasets_from_config=True)
    loaded_model_dict = load_model(config, args.mode == 'train', load_datasets_from_config=True)
    dictionaries = loaded_model_dict['dictionaries']
    datasets = loaded_model_dict['datasets']
    model = loaded_model_dict['model']
    parameters = loaded_model_dict['parameters']

    # dictionaries = create_dictionaries(config, args.mode == 'train')
    # linking_candidates = None
    # if 'linking_candidates' in config:
    #     linking_candidates = create_linking_candidates(config['linking_candidates'], dictionaries['entities'])
    # datasets, data, evaluate = create_datasets(config, dictionaries, args.fold, linking_candidates)
    # model, parameters = create_model(config, dictionaries)

    # (kzaporoj) - the default torch device (cuda/cpu) is set to the current device
    #   TODO

    if args.mode == 'train':
        print("Dictionaries:")
        for name, dictionary in dictionaries.items():
            if isinstance(dictionary, Dictionary):  # sometimes can be BertTokenizer for example
                print("- {}: {} (oov={})".format(name, dictionary.size, dictionary.out_of_voc))
                filename = '{}/{}.json'.format(config['path'], name)
                print('  write', filename)
                dictionary.write(filename)
        print()
        if 'version' in config['trainer']:
            if config['trainer']['version'] == 'old_dwie':
                train(model, datasets, config, parameters)
            elif config['trainer']['version'] == 'spanbert':
                runner = Runner(config=config)
                runner.train_spanbert(model, datasets, config)
            else:
                raise RuntimeError('no implementation for the following trainer version: ' +
                                   config['trainer']['version'])
        else:
            train(model, datasets, config, parameters)
    elif args.mode == 'test':
        test(model, datasets, config)
    elif args.mode == 'predict':
        # predict(model, datasets, config)
        output_dir = os.path.dirname(args.config_file)
        predict_dwie_linker(model, datasets, config, output_dir)

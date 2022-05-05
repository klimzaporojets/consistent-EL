import json
import random

from torch.utils.data import ConcatDataset

from cpn.data_reader import DatasetCPN
from cpn.data_reader_bert import DatasetDWIESpanBert
from cpn.data_reader_bert_hoi import DatasetDWIESpanBertHoi
from datass.stream2 import SubsetDataset
from datass.transform import create_data_transformer2


def create_datasets(config, dictionaries, default_fold, linking_candidates=None):
    # kzaporoj - commented out this part
    # if config['dataloader']['type'] == 'stream2':
    #     datasets = {name: Stream2Dataset(name, value, config['dataloader'], dictionaries) for name, value in config['datasets'].items()}
    # elif config['dataloader']['type'] == 'binary':
    #     datasets = {name: StreamBinary(value, {'tokens': dictionaries['words'], 'targets': dictionaries['entities']}) for name, value in config['datasets'].items()}
    # elif config['dataloader']['type'] == 'text':
    #     datasets = {name: TextDataset(value, config['dataloader'], dictionaries) for name, value in config['datasets'].items()}
    # elif config['dataloader']['type'] == 'indexed-text':
    #     datasets = {name: IndexedTextDataset(value, config['dataloader'], dictionaries) for name, value in config['datasets'].items()}
    # elif config['dataloader']['type'] == 'tcm':
    #     datasets = {name: TCMDataset(value, config['dataloader'], dictionaries) for name, value in config['datasets'].items()}

    if config['dataloader']['type'] == 'cpn':
        datasets = {name: DatasetCPN(name, {'dataset': value,
                                            'model': config['model'],
                                            'dataloader': config['dataloader']
                                            }, dictionaries, linking_candidates)
                    for name, value in config['datasets'].items()}
    elif config['dataloader']['type'] == 'dwie_spanbert':
        datasets = {name: DatasetDWIESpanBert(name, {'dataset': value,
                                                     'model': config['model'],
                                                     'dataloader': config['dataloader']
                                                     }, dictionaries, linking_candidates)
                    for name, value in config['datasets'].items()}
    elif config['dataloader']['type'] == 'dwie_spanbert_hoi':
        datasets = {name: DatasetDWIESpanBertHoi(name, {'dataset': value,
                                                     'model': config['model'],
                                                     'dataloader': config['dataloader'],
                                                     'output_config':config['output_config']
                                                     }, dictionaries, linking_candidates)
                    for name, value in config['datasets'].items()}
    else:
        raise BaseException("no such data loader:", config['dataloader'])

    if 'split' in config['trainer']:
        name = config['trainer']['split']['dataset']
        print("WARNING: splitting {} into train, test".format(name))
        all = datasets[name]
        indices = list(range(len(all)))
        mid = int(len(indices) * config['trainer']['split']['ratio'])
        random.seed(2018)
        random.shuffle(indices)
        datasets['train'] = SubsetDataset(all, indices[0:mid])
        datasets['test'] = SubsetDataset(all, indices[mid:])
    elif 'cv' in config['trainer']:
        all = datasets[config['trainer']['cv']['dataset']]
        del datasets[config['trainer']['cv']['dataset']]
        fold = int(config['trainer']['cv'].get('fold', default_fold))
        folds = int(config['trainer']['cv']['folds'])

        cv_train_name = 'cv-train'
        cv_test_name = 'cv-test'

        print("Fold {}/{}".format(fold, folds))
        if 'groupfile' in config['trainer']['cv']:
            with open(config['trainer']['cv']['groupfile']) as f:
                groups = [x.rstrip().split(',') for x in f.readlines() if len(x.rstrip()) > 0]
            random.seed(2018)
            random.shuffle(groups)
            groups_train = [groups[i] for i in range(len(groups)) if i % folds != fold]
            groups_test = [groups[i] for i in range(len(groups)) if i % folds == fold]
            # make it lists flat
            flatten = lambda l: [item for sublist in l for item in sublist]
            indices_train = [int(x) for x in flatten(groups_train)]
            indices_test = [int(x) for x in flatten(groups_test)]
            indices_train.sort()
            indices_test.sort()
            datasets[cv_train_name] = SubsetDataset(cv_train_name, all, indices_train)
            datasets[cv_test_name] = SubsetDataset(cv_test_name, all, indices_test)
        else:
            indices = list(range(len(all)))
            random.seed(2018)
            random.shuffle(indices)
            datasets[cv_train_name] = SubsetDataset(cv_train_name, all,
                                                    [indices[i] for i in range(len(indices)) if i % folds != fold])
            datasets[cv_test_name] = SubsetDataset(cv_test_name, all,
                                                   [indices[i] for i in range(len(indices)) if i % folds == fold])

        print("CV({}) {} -> {},{}".format(fold, len(all), len(datasets[cv_train_name]), len(datasets[cv_test_name])))

        with open('{}/traindocs.json'.format(config['path']), 'w') as file:
            json.dump(datasets[cv_train_name].indices, file)
        with open('{}/testdocs.json'.format(config['path']), 'w') as file:
            json.dump(datasets[cv_test_name].indices, file)

    if 'concat' in config['trainer']:
        for key, names in config['trainer']['concat'].items():
            print("CONCAT", key, names)
            datasets[key] = ConcatDataset([datasets[name] for name in names])

    if 'train' in config['trainer']:
        train = datasets[config['trainer']['train']]
        train.train = True
    else:
        train = None

    evaluate = config['trainer']['evaluate']

    transformers = [create_data_transformer2(cfg, dictionaries) for cfg in config['dataloader']['transformers-x']]

    for transformer in transformers:
        for stream in datasets.values():
            transformer.initialize(stream)

    for transformer in transformers:
        for stream in datasets.values():
            transformer.transform(stream)

    return datasets, train, evaluate

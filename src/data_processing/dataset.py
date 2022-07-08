import logging

from torch.utils.data import Dataset

from data_processing.data_reader_bert_hoi import DatasetDWIESpanBertHoi

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger()


class SubsetDataset(Dataset):
    def __init__(self, name, dataset, indices):
        self.name = name
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def create_datasets(config, dictionaries):
    if config['dataloader']['type'] == 'dwie_spanbert_hoi':
        datasets = {name: DatasetDWIESpanBertHoi(name, {'dataset': value,
                                                        'model': config['model'],
                                                        'dataloader': config['dataloader'],
                                                        'output_config': config['output_config']
                                                        }, dictionaries)
                    for name, value in config['datasets'].items()}
    else:
        raise BaseException('no such data loader: %s' % config['dataloader'])

    if 'train' in config['trainer']:
        train = datasets[config['trainer']['train']]
        train.train = True
    else:
        train = None

    evaluate = config['trainer']['evaluate']

    return datasets, train, evaluate

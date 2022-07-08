import argparse
import json
import logging
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_processing.dataset import create_datasets
from misc import settings
from models.misc.linker import create_dictionaries, create_model, create_linking_candidates

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger()


def predict(model, datasets, config):
    device = torch.device(settings.device)
    model = model.to(device)

    collate_fn = model.collate_func()
    batch_size = config['optimizer']['batch_size']

    model.load_model(config['model_path'])

    # evaluate
    for name in config['trainer']['evaluate']:
        logging.info('Start evaluating %s' % name)

        loader = DataLoader(datasets[name], collate_fn=collate_fn, batch_size=batch_size, shuffle=False)

        model.eval()

        if hasattr(model, 'begin_epoch'):
            model.begin_epoch()

        output_path = os.path.join(config['output_path'], name, ('%s.jsonl' % name))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as file:
            for i, minibatch in enumerate(tqdm(loader)):
                _, predictions = model.predict(**minibatch, output_config=config['output_config'])
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

    dictionaries = create_dictionaries(config, training)
    datasets = None

    # BEGIN - leaves in config only the dataset for which the evaluation is in true
    evaluate_datasets = set(config['trainer']['evaluate'])
    to_delete = list()
    for dataset_name, dataset in config['datasets'].items():
        if dataset_name not in evaluate_datasets:
            to_delete.append(dataset_name)
    for curr_del_name in to_delete:
        del config['datasets'][curr_del_name]

    if 'train' in config['trainer']:
        del config['trainer']['train']
    # END - leaves in config only the dataset for which the evaluation is in true

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
    parser.add_argument('--model_path', help='path to the model to be evaluated', type=str, default=None,
                        required=True)
    parser.add_argument('--output_path', help='path to where the output predicted files will be saved',
                        type=str, default=None, required=True)
    parser.add_argument('--device', dest='device', type=str, default='cuda')
    args = parser.parse_args()

    settings.device = args.device

    with open(args.config_file) as f:
        config = json.load(f)

    config['model_path'] = args.model_path
    config['output_path'] = args.output_path
    config['dictionaries_path'] = os.path.join(os.path.dirname(config['model_path']), 'dictionaries')

    os.makedirs(config['output_path'], exist_ok=True)
    loaded_model_dict = load_model(config, training=False, load_datasets_from_config=True)
    dictionaries = loaded_model_dict['dictionaries']
    datasets = loaded_model_dict['datasets']
    model = loaded_model_dict['model']
    parameters = loaded_model_dict['parameters']

    predict(model=model, datasets=datasets, config=config)

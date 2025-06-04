import os
import sys
import argparse
import logging
from datetime import datetime

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_path)

import torch
import yaml

from callbacks.collate import collate_waveforms_and_extract_spectrograms, collate_waveforms_and_extract_fbanks
from datasets.datasets import GPUSupportedDynamicAudioDataset
from training.loops import contrastive_training_loop, angular_training_loop

SEED = 42


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--config', required=True, help='Training config in json format.')

    return parser.parse_args()


if __name__ == '__main__':

    # parse training args
    args = parse_args()
    config_path = args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize logger
    logging.basicConfig(filename=os.path.join(project_path, 'logs',
                                              config['Model']['model_name'] + f'_{datetime.now().date()}.log'),
                        encoding='utf-8',
                        level=logging.INFO,
                        force=True,
                        filemode='w',
                        format='%(asctime)s %(message)s')

    logging.info(f'{10*"*"} Training configuration {10*"*"}\n')
    logging.info(f'Config:\n{config}\n')

    # Training args
    batch_size = config['Hyperparameters']['batch_size']
    lr = config['Hyperparameters']['lr'] * batch_size / 640
    optimizer = config['Hyperparameters']['optimizer']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    freq_cut_bool = config['Augmentations']['freq_cut_bool']

    # Collate fn
    assert config['Model']['feature'] in ["spectrogram", "fbanks"]
    collate_fn = collate_waveforms_and_extract_fbanks if config['Model'][
        'feature'] == 'fbanks' else collate_waveforms_and_extract_spectrograms

    # Get model
    model_str = config['Model']['model_str']
    backbone_weights = config['Model']['backbone_weights'] if 'backbone_weights' in config['Model'].keys() else None
    freeze_encoder = config['Model']['freeze_encoder'] if 'freeze_encoder' in config['Model'].keys() else False

    loss_function = config['Hyperparameters']['loss']
    assert loss_function in ["contrastive", "angular"]

    logging.info(f'Preparing training set...')

    train_set = GPUSupportedDynamicAudioDataset(data_path=config['Data']["data_path"],
                                                noise_path=os.path.join(project_path,
                                                                        config['Data']['background_noise_train']),
                                                ir_path=os.path.join(project_path,
                                                                     config['Data']['impulse_responses_train']),
                                                pickle_split=config['Data']['train_pickle'],
                                                freq_cut_bool=freq_cut_bool)

    val_set = GPUSupportedDynamicAudioDataset(data_path=config['Data']["data_path"],
                                              noise_path=os.path.join(project_path,
                                                                      config['Data']['background_noise_val']),
                                              ir_path=os.path.join(project_path,
                                                                   config['Data']['impulse_responses_val']),
                                              pickle_split=config['Data']['val_pickle'],
                                              freq_cut_bool=freq_cut_bool)

    logging.info(f'Train set size: {len(train_set)}')
    logging.info(f'Preparing val set...')

    logging.info(f'Validation set size: {len(val_set)}')

    logging.info(f'\n{10*"*"} Training starts {10*"*"}\n')

    if loss_function == "contrastive":
        logging.info(f'Loss function: {loss_function}')
        contrastive_training_loop(train_dset=train_set,
                                  val_dset=val_set,
                                  epochs=config['Hyperparameters']['epochs'],
                                  batch_size=batch_size,
                                  lr=lr,
                                  patience=config['Hyperparameters']['patience'],
                                  model_name=config['Model']['model_name'],
                                  output_path=config['Model']['output_path'],
                                  optim=optimizer,
                                  collate_fn=collate_fn,
                                  model_str=model_str,
                                  spec_aug_params=config['Augmentations']['spec_aug_params'],
                                  backbone_weights=backbone_weights,
                                  freeze_encoder=freeze_encoder,
                                  div_encoder_layer=config['Model'].get('div_encoder_layer', True))
    elif loss_function == "angular":
        logging.info(f'Loss function: {loss_function}')
        angular_training_loop(train_dset=train_set,
                              val_dset=val_set,
                              epochs=config['Hyperparameters']['epochs'],
                              batch_size=batch_size,
                              lr=lr,
                              patience=config['Hyperparameters']['patience'],
                              model_name=config['Model']['model_name'],
                              output_path=config['Model']['output_path'],
                              optim=optimizer,
                              collate_fn=collate_fn,
                              model_str=model_str,
                              spec_aug_params=config['Augmentations']['spec_aug_params'],
                              backbone_weights=backbone_weights,
                              freeze_encoder=freeze_encoder,
                              alpha=config['Hyperparameters'].get("alpha", 40),
                              miner_alpha=config['Hyperparameters'].get("miner_alpha", 20),
                              div_encoder_layer=config['Model'].get('div_encoder_layer', True))

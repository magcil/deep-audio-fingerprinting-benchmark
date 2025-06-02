import os
import sys
import json
import argparse
import random
import logging
from datetime import datetime
from typing import Callable, Optional

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_path)

from optim.lamb import Lamb
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import yaml

from callbacks.early_stopping import EarlyStopping
from callbacks.collate import collate_waveforms_and_extract_spectrograms, collate_waveforms_and_extract_fbanks
from utils.utils import extract_mel_spectrogram
from datasets.datasets import GPUSupportedDynamicAudioDataset
from utils.torch_utils import get_model, SpecAugMask
from loss.ntxent import NTxent_Loss_2

SEED = 42


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--config', required=True, help='Training config in json format.')

    return parser.parse_args()


def optimized_training_loop(train_dset,
                            val_dset,
                            epochs,
                            batch_size,
                            lr,
                            patience,
                            loss_fn,
                            model_name: str,
                            output_path: str,
                            optim="Adam",
                            model_str="fingerprinter",
                            collate_fn: Callable = collate_waveforms_and_extract_spectrograms,
                            spec_aug_params={
                                "value": -80.,
                                "H": 256,
                                "W": 32,
                                "H_prob": 0.5,
                                "W_prob": 0.1
                            },
                            backbone_weights: Optional[str] = None,
                            freeze_encoder: bool = False):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Current device: {device}")
    N = batch_size

    model = get_model(model_str=model_str).to(device=device)

    # Transfer Learning
    if backbone_weights:
        model.load_pretrained_encoder(backbone_weights)
        if freeze_encoder:
            model.freeze_encoder_weights()

    loss_fn = loss_fn.to(device)
    SpecAug = SpecAugMask(**spec_aug_params).to(device)

    num_workers = 8

    train_dloader = DataLoader(train_dset,
                               batch_size=N,
                               shuffle=True,
                               num_workers=num_workers,
                               drop_last=True,
                               collate_fn=collate_fn)

    val_dloader = DataLoader(val_dset,
                             batch_size=N,
                             shuffle=False,
                             num_workers=num_workers,
                             drop_last=True,
                             collate_fn=collate_fn)

    if optim == "Adam":
        optim = Adam(model.parameters(), lr=lr)
    elif optim == "Lamb":
        optim = Lamb(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Invalid optimizer specified: {optim}")

    lr_scheduler = CosineAnnealingLR(optimizer=optim, T_max=100, eta_min=1e-7)

    ear_stopping = EarlyStopping(patience=patience, verbose=True, path=os.path.join(output_path, model_name + ".pt"))

    train_loss, val_loss = 0.0, 0.0

    _padding = len(str(epochs + 1))

    for epoch in range(1, epochs + 1):

        model.train()
        i = 0

        with tqdm(train_dloader, unit="batch", leave=False, desc="Training set") as tbatch:
            for i, X in enumerate(tbatch, 1):
                # Forward pass
                X = X.to(device)

                # Apply SpecAug
                if random.random() <= 0.33:
                    X = SpecAug(X)

                # Inference
                X = model(X)

                # Split to calculate loss
                X_org, X_aug = torch.split(X, N, 0)

                loss = loss_fn(X_org, X_aug)
                train_loss += loss.item()

                # Backward pass
                optim.zero_grad()
                loss.backward()
                optim.step()

            train_loss /= len(train_dloader)
            lr_scheduler.step()

        model.eval()
        with torch.no_grad():
            with tqdm(val_dloader, unit="batch", leave=False, desc="Validation set") as vbatch:
                for X in vbatch:

                    X = X.to(device)

                    # Apply SpecAug
                    if random.random() <= 0.33:
                        X = SpecAug(X)

                    # Inference
                    X = model(X)

                    # Split to calculate loss
                    X_org, X_aug = torch.split(X, N, 0)

                    loss = loss_fn(X_org, X_aug)
                    val_loss += loss.item()
        val_loss /= len(val_dloader)

        logging.info(f"Epoch {epoch:<{_padding}}/{epochs}. Train Loss: {train_loss:.3f}. Val Loss: {val_loss:.3f}")

        ear_stopping(val_loss, model, epoch)
        if ear_stopping.early_stop:
            logging.info("Early Stopping.")
            break
        train_loss, val_loss = 0.0, 0.0


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

    # Initialize loss
    loss_fn = NTxent_Loss_2(n_org=batch_size, n_rep=batch_size, device=device).to(device)

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

    optimized_training_loop(train_dset=train_set,
                            val_dset=val_set,
                            epochs=config['Hyperparameters']['epochs'],
                            batch_size=batch_size,
                            lr=lr,
                            patience=config['Hyperparameters']['patience'],
                            loss_fn=loss_fn,
                            model_name=config['Model']['model_name'],
                            output_path=config['Model']['output_path'],
                            optim=optimizer,
                            collate_fn=collate_fn,
                            model_str=model_str,
                            spec_aug_params=config['Augmentations']['spec_aug_params'],
                            backbone_weights=backbone_weights,
                            freeze_encoder=freeze_encoder)

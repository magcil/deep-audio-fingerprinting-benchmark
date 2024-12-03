import os
import sys
import json
import argparse
import random

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_path)

from optim.lamb import Lamb
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch

from callbacks.collate import Collate_Fn
from callbacks.early_stopping import EarlyStopping
from datasets.datasets import DynamicAudioDataset, GPUSupportedDynamicAudioDataset
from utils.torch_utils import LogMelExtractor, SpecAugMask, BatchAugmentationChain
from loss.ntxent import NTxent_Loss_2
from models.neural_fingerprinter import Neural_Fingerprinter

SEED = 42


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--config', required=True, help='Training config in json format.')

    return parser.parse_args()


def training_loop(
    train_dset,
    val_dset,
    epochs,
    batch_size,
    lr,
    patience,
    loss_fn,
    model_name=None,
    output_path=None,
    optim="Adam",
):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Current device: {device}")
    N = batch_size // 2

    model = Neural_Fingerprinter().to(device)

    loss_fn = loss_fn.to(device)
    num_workers = 8
    train_dloader = DataLoader(train_dset,
                               batch_size=N,
                               shuffle=True,
                               collate_fn=Collate_Fn(rng=np.random.default_rng(SEED)),
                               num_workers=num_workers,
                               drop_last=True)
    val_dloader = DataLoader(val_dset,
                             batch_size=N,
                             shuffle=False,
                             collate_fn=Collate_Fn(rng=np.random.default_rng(SEED)),
                             num_workers=num_workers,
                             drop_last=True)
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
            for i, (x_org, x_aug) in enumerate(tbatch, 1):
                # Forward pass
                X = torch.cat((x_org, x_aug), dim=0).to(device)
                X = model(X)
                x_org, x_aug = torch.split(X, N, 0)
                loss = loss_fn(x_org, x_aug)
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
                for x_org, x_aug in vbatch:
                    # Forward pass
                    X = torch.cat((x_org, x_aug), dim=0).to(device)
                    X = model(X)
                    x_org, x_aug = torch.split(X, N, 0)
                    loss = loss_fn(x_org, x_aug)
                    val_loss += loss.item()
        val_loss /= len(val_dloader)

        print(f"Epoch {epoch:<{_padding}}/{epochs}. Train Loss: {train_loss:.3f}. Val Loss: {val_loss:.3f}")

        ear_stopping(val_loss, model, epoch)
        if ear_stopping.early_stop:
            print("Early Stopping.")
            break
        train_loss, val_loss = 0.0, 0.0


def optimized_training_loop(
    train_dset,
    val_dset,
    epochs,
    batch_size,
    lr,
    patience,
    loss_fn,
    train_noise_path,
    train_ir_path,
    val_noise_path,
    val_ir_path,
    model_name=None,
    output_path=None,
    optim="Adam",
):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Current device: {device}")
    N = batch_size

    model = Neural_Fingerprinter().to(device)

    loss_fn = loss_fn.to(device)
    MelExtractor = LogMelExtractor().to(device)
    SpecAug = SpecAugMask(value=-80., H=256, W=32, H_prob=0.5, W_prob=0.1).to(device)
    AugChainTrain = BatchAugmentationChain(noise_path=train_noise_path, ir_path=train_ir_path)
    AugChainVal = BatchAugmentationChain(noise_path=val_noise_path, ir_path=val_ir_path)

    num_workers = 8

    train_dloader = DataLoader(train_dset, batch_size=N, shuffle=True, num_workers=num_workers, drop_last=True)
    val_dloader = DataLoader(val_dset, batch_size=N, shuffle=False, num_workers=num_workers, drop_last=True)

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
                X_org = X['signal'].to(device)
                X_aug = X['shifted_signal'].to(device)

                # Apply augmentations
                X_aug = AugChainTrain(X_aug)

                # Concat to form Contrastive Batch
                X = torch.cat((X_org, X_aug), dim=0)

                # Get MelSpecs
                X = MelExtractor(X)

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

                    # Forward
                    X_org = X['signal'].to(device)
                    X_aug = X['shifted_signal'].to(device)

                    # Apply augmentations
                    X_aug = AugChainVal(X_org)

                    # Concat to form Contrastive Batch
                    X = torch.cat((X_org, X_aug), dim=0)

                    # Get MelSpecs
                    X = MelExtractor(X)

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

        print(f"Epoch {epoch:<{_padding}}/{epochs}. Train Loss: {train_loss:.3f}. Val Loss: {val_loss:.3f}")

        ear_stopping(val_loss, model, epoch)
        if ear_stopping.early_stop:
            print("Early Stopping.")
            break
        train_loss, val_loss = 0.0, 0.0


if __name__ == '__main__':

    # parse training args
    args = parse_args()
    config_path = args.config
    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f'{10*"*"} Training configuration {10*"*"}\n')
    print(f'Config:\n{config}\n')

    batch_size = config['batch_size']
    lr = config['lr'] * batch_size / 640
    impulse_train = os.path.join(project_path, config['impulse_responses_train'])
    impulse_val = os.path.join(project_path, config['impulse_responses_val'])
    data_path = config["data_path"]
    background_train = os.path.join(project_path, config['background_noise_train'])
    background_val = os.path.join(project_path, config['background_noise_val'])
    optimizer = config['optimizer']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    loss_fn = NTxent_Loss_2(n_org=batch_size, n_rep=batch_size, device=device).to(device)

    print(f'Preparing training set...')

    train_set = GPUSupportedDynamicAudioDataset(data_path=data_path, pickle_split=config['train_pickle'])
    val_set = GPUSupportedDynamicAudioDataset(data_path=data_path, pickle_split=config['val_pickle'])

    print(f'Train set size: {len(train_set)}')
    print(f'Preparing val set...')

    print(f'Validation set size: {len(val_set)}')

    print(f'\n{10*"*"} Training starts {10*"*"}\n')

    optimized_training_loop(train_dset=train_set,
                            val_dset=val_set,
                            epochs=config['epochs'],
                            batch_size=batch_size,
                            lr=lr,
                            patience=config['patience'],
                            loss_fn=loss_fn,
                            model_name=config['model_name'],
                            output_path=config['output_path'],
                            optim=optimizer,
                            train_noise_path=config['background_noise_train'],
                            train_ir_path=config['impulse_responses_train'],
                            val_noise_path=config['background_noise_val'],
                            val_ir_path=config['impulse_responses_val'])

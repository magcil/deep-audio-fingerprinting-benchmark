import os
import sys
import random
import logging
from typing import Callable, Optional

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_path)

from optim.lamb import Lamb
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

from callbacks.early_stopping import EarlyStopping
from callbacks.collate import collate_waveforms_and_extract_spectrograms
from utils.torch_utils import get_model, SpecAugMask
from loss.metric_learning_losses import NTxent_Loss_2
from pytorch_metric_learning import distances, losses, miners


def contrastive_training_loop(train_dset,
                              val_dset,
                              epochs,
                              batch_size,
                              lr,
                              patience,
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
                              freeze_encoder: bool = False,
                              div_encoder_layer: bool = True):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Current device: {device}")
    N = batch_size

    model = get_model(model_str=model_str, div_encoder_layer=div_encoder_layer).to(device=device)

    # Transfer Learning
    if backbone_weights:
        model.load_pretrained_encoder(backbone_weights)
        if freeze_encoder:
            model.freeze_encoder_weights()

    # Initialize loss
    loss_fn = NTxent_Loss_2(n_org=batch_size, n_rep=batch_size, device=device).to(device)
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


def angular_training_loop(train_dset,
                          val_dset,
                          epochs,
                          batch_size,
                          lr,
                          patience,
                          model_name: str,
                          output_path: str,
                          alpha: int = 40,
                          miner_alpha: int = 20,
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
                          freeze_encoder: bool = False,
                          div_encoder_layer: bool = True):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Current device: {device}")
    N = batch_size

    model = get_model(model_str=model_str, div_encoder_layer=div_encoder_layer).to(device=device)

    # Transfer Learning
    if backbone_weights:
        model.load_pretrained_encoder(backbone_weights)
        if freeze_encoder:
            model.freeze_encoder_weights()

    # Initialize Angular loss and miners
    distance = distances.LpDistance(normalize_embeddings=True)
    loss_fn = losses.AngularLoss(alpha=alpha, distance=distance).to(device)
    train_miner = miners.AngularMiner(angle=miner_alpha)
    valid_miner = miners.AngularMiner(angle=0)

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

                # Create pseudo-labels
                pseudo_labels = torch.cat([torch.arange(N), torch.arange(N)]).to(device)
                miner_triplets = train_miner(X, pseudo_labels)

                loss = loss_fn(X, pseudo_labels, miner_triplets)
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

                    # Create pseudo-labels
                    pseudo_labels = torch.cat([torch.arange(N), torch.arange(N)]).to(device)
                    miner_triplets = valid_miner(X, pseudo_labels)

                    loss = loss_fn(X, pseudo_labels, miner_triplets)

                    val_loss += loss.item()
        val_loss /= len(val_dloader)

        logging.info(f"Epoch {epoch:<{_padding}}/{epochs}. Train Loss: {train_loss:.3f}. Val Loss: {val_loss:.3f}")

        ear_stopping(val_loss, model, epoch)
        if ear_stopping.early_stop:
            logging.info("Early Stopping.")
            break
        train_loss, val_loss = 0.0, 0.0

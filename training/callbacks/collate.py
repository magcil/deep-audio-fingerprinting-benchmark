import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.utils import cutout_spec_augment_mask, extract_mel_spectrogram, extract_fbanks


class Collate_Fn():

    def __init__(self, rng: np.random.Generator, p: float = 0.33):
        self.rng = rng
        self.prob = p

    def __call__(self, batch):
        if self.rng.random() <= self.prob:
            mask = torch.from_numpy(cutout_spec_augment_mask(self.rng))
            x_orgs = [mask * sample[0] for sample in batch]
            x_augs = [mask * sample[1] for sample in batch]
            return torch.stack(x_orgs), torch.stack(x_augs)
        else:
            x_orgs, x_augs = list(zip(*batch))
            return torch.stack(x_orgs), torch.stack(x_augs)


def collate_waveforms_and_extract_spectrograms(batch):
    signals, shifted_signals = [], []
    for b in batch:
        signals.append(b['signal'])
        shifted_signals.append(b['shifted_signal'])
    # 2B x F x T
    new_batch = np.concatenate([np.vstack(signals), np.vstack(shifted_signals)], axis=0)
    # Extract spectrograms
    new_batch = extract_mel_spectrogram(new_batch)
    
    return torch.from_numpy(new_batch).unsqueeze(1)

def collate_waveforms_and_extract_fbanks(batch):
    signals, shifted_signals = [], []
    for b in batch:
        signals.append(b['signal'])
        shifted_signals.append(b['shifted_signal'])
    # 2B x F x T
    new_batch = np.concatenate([np.vstack(signals), np.vstack(shifted_signals)], axis=0)
    # Extract spectrograms
    new_batch = extract_fbanks(torch.from_numpy(new_batch))
    
    return new_batch.unsqueeze(1)

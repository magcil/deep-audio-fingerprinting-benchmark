import os
import sys
from typing import Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.utils import extract_fbanks, extract_mel_spectrogram

import torch.nn as nn
import torch
import numpy as np
import torchlibrosa as tl
from torch_audiomentations import Compose, AddBackgroundNoise, ApplyImpulseResponse, HighPassFilter, LowPassFilter
from models.attention import AttentionCNN
from models.neural_fingerprinter import Neural_Fingerprinter
from models.beats.beatswrapper import BEATsWrapper
from toolz import compose


class SpecAugMask(nn.Module):
    """
    Custom implementation of SpecAugMask.
    Applies a uniform Mask to all batch input spectrogram.
    """

    def __init__(self, value: float, H: int, W: int, H_prob: float, W_prob: float):
        """
        Args:
            value (float): The value corresponding to the masked points.
            H (int): Height of input spectrogram, i.e., Frequency bins.
            W (int): Width of input spectrogram, i.e., Time bins.
            H_prob (float): Max percentage of frequencies to mask.
            W_prob (float): Max percentage of time bins to mask.
        """
        super(SpecAugMask, self).__init__()

        self.value, self.W, self.H = value, W, H
        self.H_prob, self.W_prob = H_prob, W_prob
        self.rng = np.random.default_rng(seed=42)

        # Initialize mask
        self.mask = nn.Parameter(data=torch.ones((1, H, W), dtype=torch.float32))
        self.mask.requires_grad = False

    def forward(self, x):
        """
        Returns x multiplied by mask
        
        Args:
            x (torch.tensor): Shape (B, 1, H, W)
        Returns:
            (torch.tensor): Shape as input, multiplied by mask
        """
        H_max, W_max = int(self.H - self.H_prob * self.H), int(self.W - self.W_prob * self.W)
        H_start, dH = self.rng.integers(low=0, high=H_max), self.rng.integers(low=0, high=int(self.H_prob * self.H))
        W_start, dW = self.rng.integers(low=0, high=W_max), self.rng.integers(low=0, high=int(self.W_prob * self.W))
        # Create Mask
        self.mask[:, H_start:H_start + dH, W_start:W_start + dW] = 0.
        mask = self.mask.repeat(x.shape[0], 1, 1, 1).to(self.mask.device)

        # Reset mask
        self._reset_mask()

        with torch.no_grad():
            return x * mask + (1 - mask) * self.value

    def _reset_mask(self):
        """
        Reset mask, use before forward ends.
        """
        self.mask[self.mask != 1.] = 1.


class LogMelExtractor(nn.Module):

    def __init__(self,
                 sr: int = 8_000,
                 n_fft: int = 1024,
                 hop_length: int = 256,
                 n_mels: int = 256,
                 amin: float = 1e-10,
                 top_db: float = 80.):
        super(LogMelExtractor, self).__init__()
        self._mel = nn.Sequential(tl.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=1),
                                  tl.LogmelFilterBank(sr=sr, n_fft=n_fft, n_mels=n_mels, is_log=False))
        self.amin = amin
        self.top_db = top_db

    def __call__(self, x):
        """ Returns the log (dB) mel spectrogram of batch waveforms x
        Args:
            x (torch.float32): Input waveforms of shape (Batch size, Samples)
        Returns:
            (torch.float32): dB Mel spectrograms of size (Batch size, 1, F, T)
        """
        mels = torch.clamp(torch.transpose(self._mel(x), -2, -1), min=self.amin)
        max_amps = torch.amax(mels, dim=(-1, -2), keepdim=True)
        # Convert to dB scale and Return

        return torch.clamp(10.0 * torch.log(mels / max_amps), min=-self.top_db)


class BatchAugmentationChain(nn.Module):
    """
    Augmentation chain with GPU and Batch Support. Background Noise, Impulse Respone, Frequency cutouts.
    """

    def __init__(self, noise_path: str, ir_path: str, sr: int = 8_000, seed: int = 42):
        """
        Args
            noise_path (str): Path containing background noises
            ir_path (str): Path containing impulse responses
            sr (int): Sampling rate
        """
        super(BatchAugmentationChain, self).__init__()

        self.noise_path = noise_path
        self.ir_path = ir_path
        self.sr = sr

        self.rng = np.random.default_rng(seed=seed)

    def forward(self, x):
        """Applies a series of augmentations to a batch of waveforms.
        
        Args:
            x (torch.tensor): Batch of waveforms of shape (B, N_samples)
        
        Returns:
            (torch.tensor): Transformed waveforms of shape (B, N_samples)
        """
        # Add a pseudo dimension for audio-channel
        x = x.unsqueeze(1)

        # Background Noise & Impulse response
        snr_prob = self.rng.random()
        if snr_prob <= 0.5:
            snr = self.rng.uniform(low=0, high=5)
        elif 0.50 < snr_prob <= 0.80:
            snr = self.rng.uniform(low=5, high=10)
        else:
            snr = self.rng.uniform(low=10, high=15)

        augmentation_chain = Compose([
            AddBackgroundNoise(background_paths=self.noise_path,
                               min_snr_in_db=snr,
                               max_snr_in_db=snr,
                               p=0.8,
                               sample_rate=self.sr,
                               output_type="tensor"),
            ApplyImpulseResponse(ir_paths=self.ir_path, p=1., sample_rate=self.sr, output_type="tensor")
        ],
                                     output_type="tensor")

        cut_freq_prob = self.rng.random()
        if cut_freq_prob >= 0.60:
            freq_cuts = Compose([
                LowPassFilter(
                    min_cutoff_freq=2000, max_cutoff_freq=3000, p=1, sample_rate=self.sr, output_type="tensor"),
                HighPassFilter(
                    max_cutoff_freq=1000, min_cutoff_freq=500, p=1, sample_rate=self.sr, output_type="tensor")
            ],
                                output_type="tensor")
            augmentation_chain = Compose(augmentation_chain.transforms + freq_cuts.transforms, output_type="tensor")

        # Apply augmentations
        return torch.squeeze(augmentation_chain(x))


def get_model(model_str: str = "fingerprinter"):
    assert model_str in ["fingerprinter", "audsearch", "transformer"]
    if model_str == "fingerprinter":
        return Neural_Fingerprinter()
    elif model_str == "audsearch":
        return AttentionCNN()
    else:
        return BEATsWrapper()


class FeatureExtractor():

    def __init__(self, feature: str):
        assert feature in ["spectrogram", "fbanks"]
        self.feature_extractor: Callable[[np.ndarray], torch.Tensor] = compose(
            extract_fbanks, torch.from_numpy) if feature == "fbanks" else compose(torch.from_numpy,
                                                                                  extract_mel_spectrogram)

    def __call__(self, x: np.ndarray) -> torch.Tensor:
        return self.feature_extractor(x)

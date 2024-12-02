import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.nn as nn
import torch
import numpy as np


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
        mask = self.mask.repeat(x.shape[0], 1, 1, 1)

        # Reset mask
        self._reset_mask()

        with torch.no_grad():
            return x * mask + (1 - mask) * self.value

    def _reset_mask(self):
        """
        Reset mask, use before forward ends.
        """
        self.mask[self.mask != 1.] = 1.

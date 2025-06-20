import os
import wave
import random
from collections import Counter
from typing import Dict, Optional

import numpy as np
from audiomentations import Compose, AddBackgroundNoise, ApplyImpulseResponse, HighPassFilter, LowPassFilter
import librosa
import torch
import torchaudio.compliance.kaldi as ta_kaldi


def add_background_noise(y: np.ndarray, y_noise: np.ndarray, SNR: float) -> np.ndarray:
    """Apply the background noise y_noise to y with a given SNR
    
    Args:
        y (np.ndarray): The original signal
        y_noise (np.ndarray): The noisy signal
        SNR (float): Signal to Noise ratio (in dB)
        
    Returns:
        np.ndarray: The original signal with the noise added.
    """
    if y.size < y_noise.size:
        y_noise = y_noise[:y.size]
    else:
        y_noise = np.resize(y_noise, y.shape)
    snr = 10**(SNR / 10)
    E_y, E_n = np.sum(y**2), np.sum(y_noise**2)

    z = np.sqrt((E_n / E_y) * snr) * y + y_noise

    return z / z.max()


def crawl_directory(directory: str, extension: Optional[str] = None) -> list:
    """Crawling data directory
    Args:
        directory (str) : The directory to crawl
    Returns:
        tree (list)     : A list with all the filepaths
    """
    tree = []
    subdirs = [folder[0] for folder in os.walk(directory)]

    for subdir in subdirs:
        files = next(os.walk(subdir))[2]
        for _file in files:
            if extension is not None:
                if _file.endswith(extension):
                    tree.append(os.path.join(subdir, _file))
            else:
                tree.append(os.path.join(subdir, _file))
    return tree


def get_wav_duration(filename: str) -> int:
    """Get the time duration of a wav file"""
    with wave.open(filename, 'rb') as f:
        return f.getnframes() // f.getframerate()


def energy_in_db(signal: np.ndarray) -> float:
    """Return the energy of the input signal in dB.
    
    Args:
        signal (np.ndarray): The input signal.

    Returns:
        float: The energy in dB.
    """
    return 20 * np.log10(np.sum(signal**2))


def time_offset_modulation(signal: np.ndarray, time_index: int, sr: int = 8000, max_offset: float = 0.25) -> np.ndarray:
    """Given an audio segment of signal returns the signal result with a time offset of +- max_offset ms.
    
    Args:
        signal (np.ndarray): The original signal.
        time_index (int): The starting point (i.e. second) of the audio segment.
        max_offset (float): The maximum offset time difference from the original audio segment.

    Return:
        np.ndarray: The signal corresponding to offset of the original audio segment.
    """

    offset = random.choice([random.uniform(-max_offset, -0.1),
                            random.uniform(0.1, max_offset)]) if time_index else random.uniform(0.1, max_offset)
    offset_samples = int(offset * sr)
    start = time_index * sr + offset_samples

    return signal[start:start + sr]


def extract_mel_spectrogram(signal: np.ndarray,
                            sr: int = 8000,
                            n_fft: int = 1024,
                            hop_length: int = 256,
                            n_mels: int = 256) -> np.ndarray:

    S = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    # convert to dB for log-power mel-spectrograms
    # Scale with ref=np.max, handle for batch dimension
    maxes = np.max(S, axis=(-1, -2), keepdims=True)
    amin = np.full_like(maxes, fill_value=1e-10)
    maxes = np.maximum(maxes, amin)

    S /= maxes

    return librosa.power_to_db(S)


def extract_fbanks(source: torch.Tensor,
                   fbank_mean: float = 15.41663,
                   fbank_std: float = 6.55582,
                   sample_rate: int = 8_000) -> torch.Tensor:

    fbanks = []
    for waveform in source:
        waveform = waveform.unsqueeze(0) * 2**15
        fbank = ta_kaldi.fbank(waveform,
                               num_mel_bins=128,
                               sample_frequency=sample_rate,
                               frame_length=25,
                               frame_shift=10)
        fbanks.append(fbank)
    fbank = torch.stack(fbanks, dim=0)
    fbank = (fbank - fbank_mean) / (2 * fbank_std)
    return fbank


class AudioAugChain():
    """ TODO """

    def __init__(self, noise_path: str, ir_path: str, sr: int = 8000, seed: int = 42, freq_cut_bool=True):

        self.rng = np.random.default_rng(seed=seed)
        self.noise_path = noise_path
        self.ir_path = ir_path
        self.sr = sr
        self.freq_cut_bool = freq_cut_bool

    def _construct_aug_chain(self):
        snr_prob = self.rng.random()

        if snr_prob <= 0.50:
            snr = self.rng.uniform(low=0, high=5)
        elif 0.50 < snr_prob <= 0.80:
            snr = self.rng.uniform(low=5, high=10)
        else:
            snr = self.rng.uniform(low=10, high=15)

        self.augmentation_chain = Compose([
            AddBackgroundNoise(sounds_path=self.noise_path, min_snr_in_db=snr, max_snr_in_db=snr, p=0.8),
            ApplyImpulseResponse(ir_path=self.ir_path, p=1.),
        ])

        cut_freq_prob = self.rng.random()
        if self.freq_cut_bool and cut_freq_prob >= 0.60:
            freq_cuts = Compose([
                LowPassFilter(min_cutoff_freq=2000, max_cutoff_freq=3000, min_rolloff=12, max_rolloff=36, p=1),
                HighPassFilter(max_cutoff_freq=1000, min_cutoff_freq=500, min_rolloff=12, max_rolloff=36, p=1)
            ])

            self.augmentation_chain.transforms.append(freq_cuts)

    def __call__(self, x):
        self._construct_aug_chain()

        return self.augmentation_chain(x, sample_rate=self.sr)


def cutout_spec_augment_mask(rng: Optional[np.random.Generator] = None):

    H, W = 256, 32
    H_max, W_max = H // 2, int(0.9 * W)
    mask = np.ones((1, H, W), dtype=np.float32)

    rng = rng if rng else np.random.default_rng()
    H_start, dH = rng.integers(low=0, high=H_max, size=2)
    W_start = rng.integers(low=0, high=W_max, size=1).item()
    dW = rng.integers(low=0, high=int(0.1 * W), size=1).item()

    mask[:, H_start:H_start + dH, W_start:W_start + dW] = 0

    return mask


def query_sequence_search(D, I):
    compensations = []
    for i, idx in enumerate(I):
        compensations.append([(x - i) for x in idx])
    candidates = np.unique(compensations)
    scores = []
    D_flat = D.flatten()
    I_flat = I.flatten()
    for c in candidates:
        idxs = np.where((c <= I_flat) & (I_flat <= c + len(D)))[0]
        scores.append(np.sum(D_flat[idxs]))
    return candidates[np.argmax(scores)], round(max(scores), 4)


def search_index(idx: int, sorted_arr: np.ndarray):
    candidate_indices = np.where(sorted_arr <= idx)[0]
    return sorted_arr[candidate_indices].max()


def majority_vote_search(d: Dict, I: np.ndarray, sorted_array: np.ndarray):
    preds = []
    I_flat = I.flatten()
    preds = [d[str(search_index(idx, sorted_array))] for idx in I_flat]
    c = Counter(preds)
    return c.most_common()[0][0]


def get_winner(d: Dict, I: np.ndarray, D: np.ndarray, sorted_array: np.ndarray):
    preds = []
    I_flat = I.flatten()
    D_flat_inverse = 1 / D.flatten()
    preds = np.array([d[str(search_index(idx, sorted_array))] for idx in I_flat])
    c = Counter(preds)
    winner = c.most_common()[0][0]
    idxs = np.where(preds == winner)[0]
    # num_matches = c.most_common()[0][1]

    D_shape = D.shape[0] * D.shape[1]

    return winner, (1 / D_shape) * D_flat_inverse[idxs].sum()

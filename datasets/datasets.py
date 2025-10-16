import os
import sys
import pickle
import random

current_file_path = os.path.abspath(__file__)
parent_dir_path = os.path.dirname(os.path.dirname(current_file_path))

sys.path.insert(0, parent_dir_path)
from utils.utils import energy_in_db, AudioAugChain, crawl_directory

import torch
import librosa
from torch.utils.data import Dataset
from numpy.random import default_rng

SEED = 42


class GPUSupportedDynamicAudioDataset(Dataset):
    """Create Dynamic Dataset"""

    def __init__(self, data_paths, noise_path, ir_path, max_offset=0.25, pickle_splits=None, freq_cut_bool=True):
        if isinstance(data_paths, str):
            data_paths = [data_paths]
        if pickle_splits is None:
            pickle_splits = [None] * len(data_paths)
            
        self.data = []
        for data_path, pickle_split in zip(data_paths, pickle_splits):
            if pickle_split:
                with open(pickle_split, "rb") as f:
                    split_wavs = pickle.load(f)
                temp_wavs = [os.path.join(data_path, os.path.join(*f.split(os.sep)[-2:])) for f in split_wavs]
            else:
                temp_wavs = crawl_directory(data_path, extension=".wav")
            self.data.extend(temp_wavs)
        
        self.max_offset = max_offset
        self.AugChain = AudioAugChain(noise_path=noise_path, ir_path=ir_path, freq_cut_bool=freq_cut_bool)

        self.rng = default_rng(SEED)
        self.time_indices_dict = {}
        self.get_energy_index()

    def get_energy_index(self):
        '''
        Keeps only segments where energy > 0.
        Returns a dictionary where the keys are the paths to the audio files 
        and the values are a random time index for each audio file.
        '''
        to_keep = []
        for wav in self.data:
            indices = []

            try:
                signal, sr = librosa.load(wav, sr=8000)
            except Exception as err:
                log_info = f"Error occured on: {os.path.basename(wav)}."
                print(log_info)
                print(f"Exception: {err}")
                print(f'Removed filename: {os.path.basename(wav)}')
            else:
                max_time_index = int(signal.size / sr) - 1
                if max_time_index:
                    for time_index in range(0, max_time_index):
                        energy = energy_in_db(signal[time_index * sr:(time_index + 1) * sr])
                        if energy > 0:
                            indices.append(time_index)
                        else:
                            continue

                    if len(indices) > 0:
                        # keep all the random indices for each song (time_indices_dict)
                        self.time_indices_dict[wav] = indices
                        to_keep.append(wav)
                    else:
                        print(f'File {os.path.basename(wav)} has no segments that have higher energy than zero')
                        print(f'Removed filename: {os.path.basename(wav)}')
                else:
                    print(f'File: {os.path.basename(wav)} has duration less than 1 sec. Skipping...')

        self.data = to_keep

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        song_path = self.data[idx]
        time_index = self.rng.choice(self.time_indices_dict[song_path])

        # Offset
        if self.rng.random() > 0.7:
            offset = random.choice([self.rng.uniform(-self.max_offset, -0.1),
                                    self.rng.uniform(0.1, self.max_offset)]) if time_index else self.rng.uniform(
                                        0.1, self.max_offset)
        else:
            offset = 0.

        # Get signal
        signal, sr = librosa.load(song_path, sr=8000, offset=time_index, duration=1)

        if offset:
            shifted_signal, sr = librosa.load(song_path, sr=8000, offset=time_index + offset, duration=1)
        else:
            shifted_signal = signal

        return {"signal": signal, "shifted_signal": self.AugChain(shifted_signal)}

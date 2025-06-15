import json
import os
import sys
import argparse
import pickle

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_path)

from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import numpy as np
import librosa
from tqdm import tqdm
import torch

from utils.utils import crawl_directory, extract_mel_spectrogram, extract_fbanks
from utils.torch_utils import get_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, help='The configuration json file.')

    return parser.parse_args()


def batch_waveforms_and_extract_spectrograms(batch_waveforms):
    # Batch size (B) x Num Samples
    new_batch = np.vstack(batch_waveforms)

    # Extract mel-spec, B x F x T
    new_batch = extract_mel_spectrogram(new_batch)

    # Convert to tensor, add pseudo-dimension and return, B x 1 x F x T

    return torch.from_numpy(new_batch).unsqueeze(1)


def batch_waveforms_and_extract_fbanks(batch_waveforms):
    # Batch size (B) x Num Samples
    new_batch = np.vstack(batch_waveforms)

    # Extract fbanks, B x T x F
    new_batch = extract_fbanks(torch.from_numpy(new_batch))

    # Convert to tensor, add pseudo-dimension and return, B x 1 x T x F

    return new_batch.unsqueeze(1)


class FileDataset(Dataset):

    def __init__(self, file, sr, hop_size):
        self.y, self.F = librosa.load(file, sr=sr)
        self.H = hop_size
        self.dur = self.y.size // self.F

        # Extract spectrograms
        self._get_audio_fragments()

    def __len__(self):
        return len(self.audio_fragments)

    def __getitem__(self, idx):
        return self.audio_fragments[idx]

    def _get_audio_fragments(self):

        J = int(np.floor((self.y.size - self.F) / self.H)) + 1

        self.audio_fragments = [self.y[j * self.H:j * self.H + self.F] for j in range(J)]


if __name__ == '__main__':

    # parse args
    args = parse_args()
    config_file = args.config
    with open(config_file, "r") as f:
        args = json.load(f)
        print(f'Config:\n{args}\n')

    SR = args["SR"]
    HOP_SIZE = args["HOP SIZE"]
    input_dirs = [os.path.join(project_path, dir) for dir in args["input dirs"]]
    output_dir = os.path.join(project_path, args["output dir"])
    batch_size = args["batch size"]
    pt_file = args["weights"]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_model(model_str=args['model_str'], div_encoder_layer=args.get("div_encoder_layer", True)).to(device)
    model.load_state_dict(torch.load(pt_file, weights_only=True))

    # Specify collate fn
    assert args['feature'] in ["spectrogram", "fbanks"]
    collate_fn = batch_waveforms_and_extract_fbanks if args[
        'feature'] == 'fbanks' else batch_waveforms_and_extract_spectrograms

    print(f'Running on {device}')

    # Check if dir exists
    if not os.path.isdir(output_dir):
        raise FileNotFoundError(f"dir {output_dir} does not exist, please create it and rerun")

    all_songs = []
    for dir in input_dirs:
        all_songs += crawl_directory(dir, extension='wav')
    print(f'All songs: {len(all_songs)}')

    # Filter songs based on given list of songs
    if "filter_pickle" in args.keys():
        with open(args['filter_pickle'], "rb") as f:
            songs_to_keep = pickle.load(f)
        all_songs = [file for file in all_songs if os.path.basename(file) in songs_to_keep]
        print(f"Songs after filter: {len(all_songs)}")

    # Discard already fingerprinted songs
    to_discard = [os.path.basename(song).removesuffix('.npy') + '.wav' for song in crawl_directory(output_dir)]
    all_songs = [song for song in all_songs if os.path.basename(song) not in to_discard]
    print(f'Songs to fingerprint: {len(all_songs)} | Discarded: {len(to_discard)}')

    model.eval()
    fails = 0
    totals = len(all_songs)
    p_bar = tqdm(all_songs, desc='Extracting deep audio fingerprints', total=totals)
    with torch.no_grad():
        for file in p_bar:
            file_dset = FileDataset(file=file, sr=SR, hop_size=HOP_SIZE)
            if file_dset.dur < 1:
                print(f'Song: {os.path.basename(file)} has duration less than 1 sec. Skipping...')
                fails += 1
                continue
            file_dloader = DataLoader(file_dset,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      collate_fn=collate_fn,
                                      drop_last=False,
                                      num_workers=args['num_workers'])
            fingerprints = []

            for X in file_dloader:
                # X is of shape B x 1 x F x T

                X = model(X.to(device))  # Shape: B x D

                fingerprints.append(X.cpu().numpy())
            try:
                fingerprints = np.vstack(fingerprints)  # Shape: Num Fingerprints x D
                np.save(file=os.path.join(output_dir,
                                          os.path.basename(file).removesuffix('.wav') + '.npy'),
                        arr=fingerprints)
            except Exception as e:
                print(f'Failed to save {os.path.basename(file)} | Error: {e}')
                fails += 1
                continue

    print(f'Totals: {totals}\nFails: {fails}')

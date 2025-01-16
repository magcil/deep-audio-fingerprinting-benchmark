import os
import argparse
import json
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import faiss
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import numpy as np

from utils.utils import crawl_directory, extract_mel_spectrogram
from utils.torch_utils import get_model
from generation.generate_fingerprints import FileDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", required=True, help=f"Json config file.")

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    with open(args.config_file, "r") as f:
        config = json.load(f)

    # Get songs to add in wav format
    songs_to_add = crawl_directory(config['data_path'], extension=".wav")

    # Open Json & Faiss Index
    with open(config['json'], 'r') as f:
        json_correspondence = json.load(f)

    index = faiss.read_index(config['index'])
    faiss_indexes = sorted([int(x) for x in json_correspondence.keys()])
    next_index = index.ntotal

    # Initialize model
    model = get_model(model_str=config['architecture'])
    device = config.get("device", "cpu")
    # Load Weights and move to device
    model.load_state_dict(torch.load(config['weights'], weights_only=True))
    model = model.to(device)

    model.eval()
    for file in tqdm(songs_to_add, desc="Inserting songs to Index", total=len(songs_to_add)):
        d_loader = DataLoader(FileDataset(
            file=file,
            sr=config.get("sr", 8_000),
            hop_size=config.get('hop_length', 4000),
        ),
                              batch_size=config.get("batch_size", 32))

        song_fingerprints = []
        with torch.no_grad():
            for batch_audios in d_loader:
                # X shape: B x 32 x 256
                X = extract_mel_spectrogram(batch_audios.numpy())
                X = torch.from_numpy(X).unsqueeze(1).to(device)
                X = model(X).to(device)
                song_fingerprints.append(X.cpu().numpy())

        # Add Fingerprints to Index
        song_fingerprints = np.vstack(song_fingerprints)
        index.add(song_fingerprints)
        # Update Json
        json_correspondence[next_index] = os.path.basename(file[:-4])
        next_index += song_fingerprints.shape[0]

    # Save Index & Json
    faiss.write_index(index, config['index'])
    with open(config['json'], "w") as f:
        json.dump(json_correspondence, f)

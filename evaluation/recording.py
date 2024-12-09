import os
import sys
import argparse
import json
import time
import logging
import datetime
from itertools import product

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_path)

from utils.torch_utils import get_model
from utils.utils import extract_mel_spectrogram, get_winner

import faiss
import torch
import numpy as np
import librosa
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score
from prettytable import PrettyTable
import yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, help='The configuration file of the recording test.')

    return parser.parse_args()


def find_songs_intervals(csv_file):
    d = {}
    start = 0
    for file in csv_file:
        y, sr = librosa.load(file, sr=8000)
        dur = y.size / sr
        d[os.path.basename(file)] = {'start': start, 'end': start + dur}
        start += dur
    return d


if __name__ == '__main__':

    args = parse_args()
    config_file = os.path.join(project_path, args.config)
    with open(config_file, 'r') as f:
        args = yaml.safe_load(f)

    # Initialize logger
    date = datetime.datetime.now().date()
    logging.basicConfig(filename=os.path.join(project_path, 'logs',
                                              args['experiment_name'] + f'_recording_test_{date}.log'),
                        encoding='utf-8',
                        level=logging.INFO,
                        force=True,
                        filemode='w',
                        format='%(asctime)s %(message)s')

    logging.info(f"{5*'*'}Config{5*'*'}:\n")
    for k, v in args.items():
        logging.info(f'{k}: {v}')

    # Set device
    device = args.get("device", "cpu")
    logging.info(f'Running on {device}')

    # Initialize Result Table
    result_table = PrettyTable()
    result_table.field_names = [
        'Architecture', 'Query length [s] / Recording',
        *[os.path.basename(recording) for recording in args['recording_wavs']]
    ]

    # Get Query / SNRs / Models
    query_lengths = args['query_lengths']
    models = args['models']

    # Set sampling rate / hop length
    F, H = args['sr'], args['hop_size']

    # Get recordings
    recordings = [os.path.join(project_path, recording) for recording in args['recording_wavs']]

    # Get true songs to find intervals
    with open(os.path.join(project_path, args['csv_path']), 'r') as f:
        true_songs = [os.path.join(project_path, args['true_songs'], x.rstrip()) for x in f.readlines()]

    songs_intervals = find_songs_intervals(true_songs)

    for iter, (query_length, recording_path) in enumerate(product(query_lengths, recordings), 1):

        # Load recording
        recording, sr = librosa.load(recording_path, sr=F)

        # Model rows
        if (iter % len(recordings)) == 1:
            model_rows = {model_num: [query_length] for model_num in models.keys()}

        # Iterate over all models
        for model_num, model_info in models.items():

            # Initialize model
            model = get_model(model_str=model_info['architecture']).to(device)
            # Load weights
            model.load_state_dict(
                torch.load(os.path.join(project_path, model_info['weights']), map_location=device, weights_only=True))

            # Get Faiss and Json
            index = faiss.read_index(os.path.join(project_path, model_info['index']))
            with open(os.path.join(project_path, model_info['json'])) as f:
                json_correspondence = json.load(f)
                sorted_arr = np.sort(np.array(list(map(int, json_correspondence.keys()))))

            index.nprobes = model_info['nprobes']
            neighbors = model_info['neighbors']

            # Set model name
            model_name = model_info['architecture'] + " (filters)" if model_info[
                'filters'] else model_info['architecture'] + " (no filters)"

            # Log config
            logging.info(
                f"Offline test of model {model_name} | Query: {query_length} | Recording: {recording_path} | Vectors: {index.ntotal}"
            )

            # To calculate accuracy
            y_true, y_pred = [], []
            inference_time, query_time = [], []

            model.eval()
            with torch.no_grad():
                for song in tqdm(true_songs, desc='Processing songs'):

                    label = os.path.basename(song)
                    start, end = songs_intervals[label]['start'], songs_intervals[label]['end']
                    iters = int((end - start) / query_length)
                    q, r = divmod(start * sr, sr)
                    start = int(q * sr) + int(r)

                    for seg in range(iters):

                        # Slice recording
                        rec_slice = recording[start + seg * query_length * F:start + (seg + 1) * query_length * F]

                        # Inference
                        tic = time.perf_counter()
                        J = int(np.floor((rec_slice.size - F) / H)) + 1
                        xq = [
                            np.expand_dims(extract_mel_spectrogram(rec_slice[j * H:j * H + F]), axis=0)
                            for j in range(J)
                        ]
                        xq = np.stack(xq)
                        out = model(torch.from_numpy(xq).to(device))
                        inference_time.append(1000 * (time.perf_counter() - tic))

                        # Retrieval
                        tic = time.perf_counter()
                        D, I = index.search(out.cpu().numpy(), neighbors)
                        pred, score = get_winner(json_correspondence, I, D, sorted_arr)
                        query_time.append(1000 * (time.perf_counter() - tic))

                        y_true.append(label.removesuffix('.wav'))
                        y_pred.append(pred)

            # Log model individual result
            logging.info(f"\n\n{7*'*'}Results{7*'*'}\n\n")

            acc = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='macro', labels=list(set(y_true)), zero_division=0)
            recall = recall_score(y_true, y_pred, average='macro', labels=list(set(y_true)), zero_division=0)
            logging.info(f"\nAccuracy score: {acc*100:.2f}%\nPrecision score: {precision*100:.2f}%" +
                         f"\nRecall score: {recall*100:.2f}%\n")
            total_time = [x + y for x, y in zip(inference_time, query_time)]
            logging.info(
                f'Inference Time: {np.mean(inference_time)}\nQuery Time: {np.mean(query_time)}\nTotal: {np.mean(total_time)}'
            )

            # Write row
            model_rows[model_num].append(acc)

        if (iter % len(recordings)) == 0:
            for j, (k, v) in enumerate(model_rows.items(), 1):
                model_name = models[k]['architecture'] + " (filters)" if models[k][
                    'filters'] else models[k]['architecture'] + " (no filters)"

                if j == len(model_rows):
                    result_table.add_row([model_name, *v], divider=True)
                else:
                    result_table.add_row([model_name, *v])

    logging.info(f"Aggregated Results\n\n{result_table}")

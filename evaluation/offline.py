import argparse
import os
import sys
import json
import time
import logging
from itertools import product
import datetime

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_path)

import torch
import faiss
from audiomentations import AddBackgroundNoise
from tqdm import tqdm
import librosa
import numpy as np
from prettytable import PrettyTable
import yaml

from utils.torch_utils import get_model
from utils.utils import crawl_directory, extract_mel_spectrogram, query_sequence_search, search_index
from utils.metrics import summary_metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluation of a DNN on a list of songs for given Query lengths and SNRs.')
    parser.add_argument('-c', '--config', required=True, help='The evaluation configuration file.')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    config_file = os.path.join(project_path, args.config)
    with open(config_file, 'r') as f:
        args = yaml.safe_load(f)

    # Initialize logger
    date = datetime.datetime.now().date()
    logging.basicConfig(filename=os.path.join(project_path, 'logs',
                                              args['experiment_name'] + f'_offline_test_{date}.log'),
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
    result_table.field_names = ['Architecture', 'Query length [s] / SNR [dB]', *args['snrs']]

    # Get Query / SNRs / Models
    query_lengths = args['query_lengths']
    snrs = args['snrs']
    models = args['models']

    # Get songs for test / background noises
    test_songs = crawl_directory(os.path.join(project_path, args['test_songs']), extension='wav')
    background_noises = os.path.join(project_path, args['background_noises'])

    # Set sampling rate / hop length
    F, H = args['sr'], args['hop_size']

    # Iterate over all combinations (Query, SNR)
    for iter, (query_length, snr) in enumerate(product(query_lengths, snrs), start=1):
        # Set Noise augmentation
        b_noise = AddBackgroundNoise(sounds_path=background_noises, min_snr_in_db=snr, max_snr_in_db=snr, p=1.)

        # Model rows
        if (iter % len(snrs)) == 1:
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
                f"Offline test of model {model_name} | Query: {query_length} | SNR: {snr} | Vectors: {index.ntotal}")

            # Hold Statistics
            y_true, y_pred = [], []
            offset_true, offset_pred = [], []
            query_times, inference_times = [], []

            for audio_file in tqdm(test_songs):

                # Read Audio
                y, sr = librosa.load(audio_file, sr=F)

                # Calculate num segs based on query length
                segs = (y.size // F) // query_length

                model.eval()

                with torch.no_grad():
                    for seg in range(segs):

                        # Inference
                        tic = time.perf_counter()
                        y_slice = b_noise(y[seg * query_length * F:(seg + 1) * query_length * F], sample_rate=F)
                        J = int(np.floor((y_slice.size - F) / H)) + 1
                        xq = [
                            np.expand_dims(extract_mel_spectrogram(signal=y_slice[j * H:j * H + F]), axis=0)
                            for j in range(J)
                        ]
                        xq = np.stack(xq)
                        out = model(torch.from_numpy(xq).to(device))
                        inference_times.append(1000 * (time.perf_counter() - tic))

                        # Retrieval
                        tic = time.perf_counter()
                        D, I = index.search(out.cpu().numpy(), neighbors)
                        idx, d = query_sequence_search(D, I)
                        start_idx = search_index(idx, sorted_arr)
                        query_times.append(1000 * (time.perf_counter() - tic))

                        y_true.append(os.path.basename(audio_file).removesuffix('.wav'))
                        y_pred.append(json_correspondence[str(start_idx)])
                        offset_true.append(seg * query_length)
                        offset_pred.append((idx - start_idx) * H / F)

            top_1, res = summary_metrics(np.array(y_true), np.array(y_pred), np.array(offset_true),
                                         np.array(offset_pred))
            total_times = list(map(lambda x: x[0] + x[1], zip(inference_times, query_times)))

            # Log results
            logging.info(f"\n\n{7*'*'}Results{7*'*'}\n\n")
            logging.info(res)
            logging.info(f"Inference Time: {np.mean(inference_times)}")
            logging.info(f"Query Time: {np.mean(query_times)} [s]")
            logging.info(f"Total: {np.mean(total_times)}\n\n")

            # Write row
            model_rows[model_num].append(top_1)

        if (iter % len(snrs)) == 0:
            for j, (k, v) in enumerate(model_rows.items(), 1):
                model_name = models[k]['architecture'] + " (filters)" if models[k][
                    'filters'] else models[k]['architecture'] + " (no filters)"

                if j == len(model_rows):
                    result_table.add_row([model_name, *v], divider=True)
                else:
                    result_table.add_row([model_name, *v])

    logging.info(f"Aggregated Results\n\n{result_table}")

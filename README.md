# deep_audio_fingerprinting_benchmark
Repository for research in deep audio fingerprinting for recognizing songs through the microphone. GitHub repository of the paper titled: 

Robustness of Deep Learning - Based Systems for Song Identification through a Real-World Evaluation Protocol.

## Env Setup

Create a python environment with python version `3.9.20` (any python `3.9.xx` environment should work w/o problems). If you are using conda use

```bash
conda create -n deep_audio_fingerprinting python==3.9.20
```

Activate the environment and install the requirements: `conda activate deep_audio_fingerprinting && pip install -r requirements.txt`.

**Note**: To install PyAudio you'll need to have installed the portaudio library: <a href="https://people.csail.mit.edu/hubert/pyaudio/#downloads">portaudio installation for PyAudio</a>.

## Train a DNN with Contrastive Loss

Two CNN-based architectures can be trained with contrastive loss: The Neural Fingerprinter as presented in [1], [2], and ResNet50 with attention mechanism as presented in [3]. To start the training you'll need to

1. Download a collection of songs in `.wav` format. In the paper we use the full split (`fma_full.zip`) of the <a href="https://github.com/mdeff/fma">FMA Dataset</a>.
2. Create two pickle files (`train.pkl`, `val.pkl`) each containing a python list with the wav files of the training/validation splits of the collection of songs.
3. Download a set of background noises splitted into three sets (`background_train`, `background_val`, `background_test`).
4. Download a set of impulse responses splitted into two sets (`impulse_train`, `impulse_val`).

Then, create a `.json` training configuration file of the following form:

```python
{
    "epochs": 120,
    "patience": 30,
    "batch_size": 256,
    "model_name": "Name of your .pt file",
    "optimizer": "Lamb",
    "output_path": "pretrained_models/ (path to store the .pt file)",
    "data_path": "Abs Path of songs in .wav format",
    "lr":1e-3,
    "background_noise_train": "Abs Path of background_train",
    "background_noise_val": "Abs Path of background_val",
    "impulse_responses_train": "Abs Path of impulse_train",
    "impulse_responses_val": "Abs Path of impulse_val",
    "train_pickle": "Abs Path to train.pkl",
    "val_pickle": "Abs Path to val.pkl",
    "model_str": "fingerprinter (or audsearch)",
    "freq_cut_bool": True
}
```

The key "model_str" can have two options: either "fingerprinter" or "audsearch" to specify the architecture of the CNN. In our paper, we find that fingerprinter performs slightly better. The key "freq_cut_bool" chooses between using our proposed cut-off frequency augmentation (set to `true`) or not (set to `false`). In our paper, we achieve the best performance in our proposed evaluation protocol by using cut-off frequency augmentation.

Then hit the command:

```bash
python training/trainer.py --config <abs_path_to_json_config.json>
```

## Fingerprint Extraction

Once the model is trained you need to extract the fingerprints of each song. In our case, we employ an 128-dimensional embedding vector for each 1 sec audio fragment. For the fingerpint extraction you'll need to create a `.json` of the following form:


```python
{
    "SR": 8000,
    "HOP SIZE": 4000,
    "input dirs": ["Abs path to folder of songs_1", "Abs path to folder of songs_2"],
    "batch size": 256,
    "weights": "Abs Path to .pt file",
    "output dir": "Abs Path to store the fingerprints",
    "num_workers": 8,
    "model_str": "fingerprinter (or audsearch)"
}
```

The `SR` key corresponds to the chosen sampling rate while "HOP SIZE" controls the overlap ratio in the fingerprint extraction. The key "input dirs" expects a sequence of Abs paths containing the directories of the songs to be stored in the database. Once you have the `.json` configuration run

```bash
python generation/generate_fingerprints.py --config <abs_path_to_json_fingerprint_extraction.json>
```

## Database Indexing

We use the <a href="https://github.com/facebookresearch/faiss">Faiss</a> library for the database indexing. A faiss index is created with the specified format, and a json structure keeps track of the song order in the database. To create the faiss index and the json correspondence run the command

```bash
python generation/generate_index.py --config <abs_path_to_json_faiss_generation.json>
```

where `faiss_generation.json` has the following format

```python
{
    "input_dir": "Abs path to fingerprints",
    "output_dir": "data/",
    "name": "IVF200PQ32_model_name",
    "index": "IVF200,PQ32",
    "d": 128
}
```

The key "name" prefixes the index/json files that will be created upon successful execution. The key "index" describes the format of the index as presented in faiss (see: <a href="https://github.com/facebookresearch/faiss/wiki/The-index-factory">https://github.com/facebookresearch/faiss/wiki/The-index-factory</a>)

## Adding Songs to Index

To add new songs to an already existing index create `.json` format of the following form:

```python
{
    "data_path": "Abs Path to directory containing the songs",
    "json": "Abs Path to faiss json",
    "index": "Abs Path to faiss index",
    "architecture": "fingerprinter (or audsearch)",
    "device": "cuda",
    "sr": 8000,
    "hop_length": 4000,
    "weights": "Abs Path to .pt file",
    "batch_size": 32
}
```

Then, run the command

```bash
python utils.wavs_to_faiss_index.py -c <abs_path_to_json_config.json>
```

This action will add the songs contained in `data_path` to the faiss index by updating the corresponding json.


## Real-World Protocol

In our proposed protocol we use the following 15 songs:

1. <a href="https://www.youtube.com/watch?v=y4zdDXPYo0I">ColdPlay - Viva La Vida</a>
2. <a href="https://www.youtube.com/watch?v=ayVLYgjd9Rk">Creedence Clearwater Revival - Looking Out My Back Door</a>
3. <a href="https://www.youtube.com/watch?v=ogLs8LvLKbA"> Brad Paisley - We Danced</a>
4. <a href="https://www.youtube.com/watch?v=Jhpi-idv2F0"> Sam Smith - How do you Sleep</a>
5. <a href="https://www.youtube.com/watch?v=8qzeP-FBJHI"> Vivaldi - The Four Seasons:Alegro</a>
6. <a href="https://www.youtube.com/watch?v=k2C5TjS2sh4"> Roxette - It Must Have Been Love </a>
7. <a href="https://www.youtube.com/watch?v=5ETENrv8cnU"> FireHouse - Love of a Lifetime </a>
8. <a href="https://www.youtube.com/watch?v=C1AHec7sfZ8"> Frank Sinatra - I've got you under my skin</a>
9. <a href="https://www.youtube.com/watch?v=YUtHjOvPKT0"> Pink - You and Your and Hand</a>
10. <a href="https://www.youtube.com/watch?v=ILVPOIFsv04"> Rihanna - Rude Boy </a>
11. <a href="https://www.youtube.com/watch?v=TmNmJegX_FM"> Brenda Lee - Sweet's Nothing </a>
12. <a href="https://www.youtube.com/watch?v=76SDedMYoOU"> Chicago - I Don't Wanna Live Without Your Love</a>
13. <a href="https://www.youtube.com/watch?v=QQU8lazOsKc"> Ice Cube - A Bird in the Hand</a>
14. <a href="https://www.youtube.com/watch?v=W2z2reKU8yo"> Alexia - Summer is Crazy </a>
15. <a href="https://www.youtube.com/watch?v=4-TbQnONe_w"> Billie Eilish - Bad Guy </a>

In `recordings` folder you can find three recordings generated for the purposes of the evaluation: `low_t.wav`, `mid_t.wav`, and `high_t.wav`. Each of these recordings have been generated by sequentially playing these 15 songs in exact order as listed above. A background noise audio was simultaneously played with three different distances relative to the recording microphone. `low_t.wav` corresponds to the recording with the highest SNR, and `high_t.wav` to the lowest.

To evaluate the performance of your model you'll need to have the exact `wav` files that are played in these recordings. We have these wav files in `recordings/true_songs/`. Then, you'll need to add these songs to your index (see previous Section). Furthermore, you'll need to pass the csv (`recording_songs.csv`) defining the order of which these songs are played.

**Note:** Results may be a slightly different if you add these songs after the index creation or before. In our case, we include these songs before the indexing creation.

Then, create a `.yaml` of the following format:

```python
# Configuration File for running recording.py

# General Args

experiment_name: "recording_test" # Name of the experiment to create .log file

recording_wavs: ["data/recordings/low_t.wav", "data/recordings/mid_t.wav", "data/recordings/high_t.wav"] # Path to recordings

query_lengths: [1, 2, 3, 4, 5, 10, 15]

true_songs: "data/recordings/true_songs" # Path to test songs
csv_path: "data/recordings/recording_songs.csv"

sr: 8000 # Always 8000 sampling rate
hop_size: 4000 # Hop length - usually 4000
device: "cuda:0" # Cuda device for inference

# Model configs

models:

    model 1:
        architecture: "fingerprinter"
        filters: True
        weights: "pretrained_models/fingerprinter_filters.pt"
        index: "data/IVF200PQ32_fingerprinter.index"
        json: "data/IVF200PQ32_fingerprinter.json"
        nprobes: 5
        neighbors: 4

    # More models can be added in the same format
```

A `.log` folder under your specified name will be created in `logs` directory with the results on the real-world evaluation. In the `.yaml` you can include multiple models following the above format. For example, if you want to include two models you can write:


```python

models:

    model 1:
        architecture: "fingerprinter"
        filters: True
        weights: "pretrained_models/fingerprinter_filters.pt"
        index: "data/IVF200PQ32_fingerprinter.index"
        json: "data/IVF200PQ32_fingerprinter.json"
        nprobes: 5
        neighbors: 4
    
    model 2:
        architecture: "Either fingerprinter or audsearch"
        filters: True (or False) # Whether the model has been trained with cut-off frequency augmentation
        weights: "Abs Path to .pt file"
        index: "Abs Path to Faiss index"
        json: "Abs Path Faiss json"
        nprobes: 5 # Number of probes for searching
        neighbors: 4 # Number of neighbors to retrieve for each query segment
```

## References

[1] <a href="https://arxiv.org/pdf/1711.10958">Gfeller, Beat, et al. "Now playing: Continuous low-power music recognition." arXiv preprint arXiv:1711.10958 (2017).</a>

[2] <a href="https://arxiv.org/pdf/1711.10958">Chang, Sungkyun, et al. "Neural audio fingerprint for high-specific audio retrieval based on contrastive learning." ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2021.</a>

[3] <a href="https://arxiv.org/pdf/2210.08624"> Singh, Anup, Kris Demuynck, and Vipul Arora. "Attention-based audio embeddings for query-by-example." arXiv preprint arXiv:2210.08624 (2022).</a>
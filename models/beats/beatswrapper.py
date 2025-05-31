import os
import sys
import json
from typing import Optional, Dict

import torch
import torch.nn as nn
import librosa
import numpy as np

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from beats_modules.BEATs import BEATsConfig, BEATs

MODEL_CONFIG = {
    'encoder_layers': 12,
    'encoder_embed_dim': 768,
    'encoder_ffn_embed_dim': 3072,
    'encoder_attention_heads': 12,
    'activation_fn': 'gelu',
    'dropout': 0.0,
    'attention_dropout': 0.0,
    'activation_dropout': 0.0,
    'encoder_layerdrop': 0.05,
    'dropout_input': 0.0,
    'layer_norm_first': False,
    'conv_bias': False,
    'conv_pos': 128,
    'conv_pos_groups': 16,
    'relative_position_embedding': True,
    'num_buckets': 320,
    'max_distance': 800,
    'gru_rel_pos': True,
    'deep_norm': True,
    'input_patch_size': 16,
    'layer_wise_gradient_decay_ratio': 0.6,
    'embed_dim': 512,
    'finetuned_model': False,
}


class DivEncLayer(nn.Module):

    def __init__(self, q: int, v: int, unit_dim=[32, 1]):
        super(DivEncLayer, self).__init__()
        self.split_fc_layers = nn.ModuleList()
        self.q = q
        self.unit_dim = unit_dim
        self.v = v
        self._construct_layers()

    def _construct_layers(self):
        for i in range(self.q):
            seq = nn.Sequential()
            seq.append(nn.Linear(self.v, self.unit_dim[0]))
            seq.append(nn.ELU())
            seq.append(nn.LayerNorm(self.unit_dim[0]))
            seq.append(nn.Linear(self.unit_dim[0], self.unit_dim[1]))
            self.split_fc_layers.append(seq)

    def _split_encoding(self, x_slices):
        out = []
        for i in range(self.q):
            out.append(self.split_fc_layers[i](x_slices[:, i, :]))
        return torch.concat(out, dim=1)

    def forward(self, x):
        # x: BxD, D=1024
        x = torch.reshape(x, (x.shape[0], self.q, -1))
        return self._split_encoding(x)


class BEATsWrapper(nn.Module):

    # Initialize BEATs Model
    def __init__(self, backbone_config: Dict = MODEL_CONFIG, div_encoder_layer: bool = True):

        super(BEATsWrapper, self).__init__()

        # Initialize BEATs Encoder
        cfg = BEATsConfig(cfg=MODEL_CONFIG)
        self.encoder = BEATs(cfg=cfg, preprocess_flag=False)
        self.div_encoder_layer = div_encoder_layer

        if div_encoder_layer:
            self.projection_head = DivEncLayer(q=128, v=int(768 / 128))

    def load_pretrained_encoder(self, weights: str):

        self.encoder.load_state_dict(torch.load(weights, weights_only=True))

    def freeze_encoder_weights(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor):
        # x: B x T x F
        x = self.encoder(x)
        # x: B x N x 768
        x = x.mean(1)
        # x: B x 768
        if self.div_encoder_layer:
            return self.projection_head(x)
        else:
            return x

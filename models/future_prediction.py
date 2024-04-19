
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import logging
from Informer2020.models.model import Informer

import hydra
from hydra.types import TargetConf
from omegaconf import DictConfig, OmegaConf
from common.cluster import KmeansAssigner
from random import choice

import time

input_size = 3000
class Identity(nn.Module):
    """Wrapper around the Identity fn to drop target_shape etc."""
    def __init__(self, in_features):
        super().__init__()
        self.in_features = in_features

    def forward(self, feats, target_shape=None):
        del target_shape  # not needed here
        return feats, feats, {}, {}

    @property
    def output_dim(self):
        return self.in_features


class MLP(nn.Module):
    def __init__(self, in_features, num_layers=2):
        super().__init__()
        self.in_features = in_features
        layers = [[nn.Linear(in_features, in_features),
                   nn.ReLU(inplace=True)] for _ in range(num_layers)]
        # Flatten, remove the last ReLU, and create a sequential
        self.model = nn.Sequential(
            *([item for sublist in layers for item in sublist][:-1]))

    def forward(self, feats, target_shape=None):
        del target_shape
        return feats, self.model(feats), {}, {}

    @property
    def output_dim(self):
        return self.in_features

class AVTh(nn.Module):
    """AVT head architecture."""
    def __init__(
            self,
            in_features: int,
            output_len: int = -1,
            output_len_eval: int = -1,  # Same as output_len, used during eval
            avg_last_n: int = -1,
            inter_dim: int = 768,
            future_pred_loss: TargetConf = None,
            return_past_too: bool = False,
            drop_last_n: int = 0,
            quantize_before_rollout: bool = False,
            # This is only relevant when in_features=1 and input is
            # clustered, or if on the fly cluster assgn is requested
            assign_to_centroids: str = None,
            num_cluster_centers: int = 50000,
            freeze_encoder_decoder: bool = False,

            informer_c_out: int = 768,
            informer_seq_len: int = 96,
            informer_label_len: int = 48,
            informer_pred_len: int =1, 
            informer_factor: int = 5,
            informer_d_model: int = 512, 
            informer_n_heads: int = 8, 
            informer_e_layers: int = 2, # informer_e_layers,
            informer_d_layers: int = 1, 
            informer_d_ff: int = 1028,
            informer_dropout: float = 0.05, 
            informer_attn: str = 'full',#'prob',
            informer_embed: str = 'timeF',
            informer_freq: str = 'h',
            informer_activation: str='gelu',
            informer_output_attention: bool = False,
            informer_distil: bool = False,#True,
            informer_mix: bool = True,

            # new args by Maxence Boels
            informer: TargetConf = None,
            decoder: TargetConf = None,
            r2a2_model: TargetConf = None,
            r2a2_main_head: TargetConf = None,
            r2a2_heatmap_head: TargetConf = None,
            **kwargs,
            ):
        super().__init__()
        key_feature_length = 64
        self.assign_to_centroids = assign_to_centroids
        if self.assign_to_centroids:
            # Since we will be assign the features
            assert in_features != 1
            self.assigner = KmeansAssigner(assign_to_centroids)
            assert self.assigner.num_clusters == num_cluster_centers
        else:
            self.future_pred_loss = None
        self.return_past_too = return_past_too
        self.drop_last_n = drop_last_n

        self.quantize_before_rollout = quantize_before_rollout
        self.orders = [1,1,1,2,3,4,5,6]

        #-----------------select params method-----------------#
        # keep r2d2 as default even for skit reproduction
        # self.forward_method = "r2d2" # options: skit_x_ant, skit_v_ant, r2d2_x_ant, r2d2_v_ant
        self.fusion_method = "add_separate" # options: add, add_forget, add_sub, add_sub_mul, add_sub_mul_div, add_cat, add_sub_mul_cat, add_sub_mul_2
        self.pooling_method = "key_recorder" # options: clean_pooling, key_recorder
        self.future_model = "mlp" # options: decoder, mlp
        #-----------------shared-----------------#
        self.projection = nn.Sequential(
            nn.Linear(512,64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64,1),
        )

        self.future_query = nn.Embedding(1, 512)

        self.present_classifier = nn.Linear(512,7)
        self.present_preanticip_classifier = nn.Linear(512,7)

        #-----------------r2d2-----------------#
        self.r2a2_model = hydra.utils.instantiate(r2a2_model, _recursive_=False)
        
    def forward(self, obs_video, train_mode=True):
        return self.r2a2_model(obs_video, train_mode=train_mode)


    @property
    def output_dim(self):
        if self.in_features == 1:
            return self.inter_dim  # since it will return encoded features
        # else, it will decode it back to the original feat dimension
        return self.in_features

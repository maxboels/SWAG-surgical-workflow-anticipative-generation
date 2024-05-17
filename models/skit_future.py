
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import logging
from Informer2020.models.model import Informer

import hydra
from hydra.types import TargetConf
from omegaconf import DictConfig, OmegaConf

import time
        
class KeyRecorder(nn.Module):
    def __init__(self,
                dim: int = 512, 
                reduc_dim: int = 64, 
                sampling_rate: int = 10, 
                local_size: int = 20,
                return_max_now_reduc: bool = False,
                relu_norm: bool = True,
                **kwargs) -> None:
        super().__init__()
        self.return_max_now_reduc = return_max_now_reduc

        if relu_norm:
            self.linear_reduction = nn.Sequential(
                nn.Linear(dim, reduc_dim),
                nn.ReLU(),
                nn.LayerNorm(reduc_dim) # replace LayerNorm with Sonething for the Temporality
            )
        else:
            self.linear_reduction = nn.Sequential(
                nn.Linear(dim, reduc_dim),
                nn.Sigmoid()
            )
        
        self._linear_expand = nn.Sequential(
            nn.Linear(reduc_dim, dim), 
            nn.ReLU(),
            nn.LayerNorm(dim),
        )
        self._sampling_rate = sampling_rate
        self._local_size = local_size
        self.dropout = nn.Dropout(0.1)
        
    def compare_past_present(self, past_pooled_reduc: torch.Tensor, past_present: torch.Tensor) -> torch.Tensor:
        present = past_present[:, -self._local_size:, :]
        global_max = 0 + present
        for i in range(present.size(1)):
            past_pooled_reduc = torch.max(past_pooled_reduc, present[:,i,:])
            global_max[:,i,:] = past_pooled_reduc
        return global_max, past_pooled_reduc.unsqueeze(1)
        
    def forward(self, obs_frames, future=None) -> torch.Tensor:

        if future is not None:
            obs_frames = torch.cat([obs_frames, future], dim=1)

        compressed_feats = self.linear_reduction(obs_frames)
        past_pooled_reduc = torch.max(compressed_feats[:, :-self._local_size :self._sampling_rate,:], dim=1)[0]
        global_reduc_max, now_reduc_max = self.compare_past_present(past_pooled_reduc, compressed_feats)

        global_max = self._linear_expand(global_reduc_max)
        # global_max = self.dropout(global_max)
        
        # comment out option
        now_reduc_max = compressed_feats[:, -1:, :] # no pooling, just the last frame
        
        if self.return_max_now_reduc:
            return global_max, now_reduc_max
        
        return global_max, None

class FusionHead(nn.Module):
    def __init__(
        self, 
        dim: int = 512, 
        **kwargs
    ):
        super().__init__()
        self.use_dropout = False
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fc_layer = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.LayerNorm(dim)
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, key_feats, feats):
        feats = self.norm1(key_feats + feats)
        if self.use_dropout:
            feats = self.norm2(self.dropout(self.fc_layer(feats)) + feats)
        else:
            feats = self.norm2(self.fc_layer(feats) + feats)
        return feats

class SKITFuture(nn.Module):
    """ Single Pass Decoder for future prediction.
    
    Instead of using an auto-regressive deocder, we use a single pass decoder for future prediction.

    """
    def __init__(
        self,
        input_dim: int = 768,
        present_length: int = 20,
        past_sampling_rate: int = 10,
        d_model: int = 512,
        dec_dim: int = 512,
        num_curr_classes: int = 7,
        num_future_classes: int = 8,
        max_anticip_time: int = 18,
        anticip_time: int = 60,
        pooling_dim: int = 64,
        decoder: TargetConf = None,
        fusion_head: TargetConf = None,
        informer: TargetConf = None,
        **kwargs
    ):
        super().__init__()
        # self.decoder_type = decoder_type
        self.input_dim = input_dim
        self.d_model = d_model
        self.present_length = present_length
        self.past_sampling_rate = past_sampling_rate
        self.max_anticip_time = max_anticip_time        # is in minutes
        self.anticip_time = anticip_time                # is in seconds
        self.num_ant_queries = int(self.max_anticip_time * 60 / self.anticip_time)
        # other params
        self.relu_norm = True
        self.frame_level = True

        self.proj_layer = nn.Linear(d_model, dec_dim)
        self.informer = hydra.utils.instantiate(informer, _recursive_=False)
        self.fusion_head1 = hydra.utils.instantiate(fusion_head, _recursive_=False)
        self.tokens_pooler = KeyRecorder(dim=d_model, reduc_dim=pooling_dim, sampling_rate=10, local_size=20,
                                             relu_norm=self.relu_norm)
        self.frame_decoder = hydra.utils.instantiate(decoder, _recursive_=False)

        self.curr_frames_classifier = nn.Linear(d_model, num_curr_classes)
        self.future_action_classifier = nn.Linear(dec_dim, num_future_classes) # optional +1 for the EOS (end of sequence token)
        self.input_queries = nn.Parameter(torch.randn(1, self.num_ant_queries, d_model))

        self.mean_inference_time = 0.0

    def encoder(self, obs_video):
        """Follow the skit implementation."""
        B, _, _ = obs_video.shape
        past = obs_video[:, :-self.present_length]
        present = obs_video[:, -self.present_length:]
        past = past.contiguous().view(-1, self.present_length, self.input_dim)
        past = self.informer(past, past)
        past = past.contiguous().view(B, -1, self.d_model)
        present = self.informer(present, present)
        past_present = torch.cat((past, present), dim=1)
        return past_present
    
    def forward(self, obs_video, train_mode=True):

        print(f"[SKIT-F] obs_video: {obs_video.shape}")
        outputs = {}
        enc_out = self.encoder(obs_video)  # sliding window encoder
        print(f"[SKIT-F] enc_out: {enc_out.shape}")

        enc_out_pooled, now_reduc_max = self.tokens_pooler(enc_out)
        print(f"[SKIT-F] enc_out_pooled: {enc_out_pooled.shape}")

        enc_out_local = enc_out[:, -self.present_length:]
        print(f"[SKIT-F] enc_out_local: {enc_out_local.shape}")
        
        enc_out = self.fusion_head1(enc_out_pooled, enc_out_local)
        print(f"[SKIT-F] enc_out_fused: {enc_out.shape}")

        curr_frames_cls = self.curr_frames_classifier(enc_out)
        print(f"[SKIT-F] curr_frames: {curr_frames_cls.shape}")
        outputs["curr_frames"] = curr_frames_cls


        # start time tracking if inference mode
        iter_times = []
        if not train_mode:
            start_time = time.time()

        input_queries = self.input_queries.expand(enc_out.shape[0], -1, -1)
        print(f"[SKIT-F] input_queries: {input_queries.shape}")
        print(f"[SKIT-F] enc_out: {enc_out.shape}")
        next_action = self.frame_decoder(input_queries, enc_out)
        print(f"[SKIT-F] decoder out: {next_action.shape}")
        next_frames_cls = self.future_action_classifier(next_action)
        outputs["future_frames"] = next_frames_cls
        print(f"[SKIT-F] next_frames: {next_frames_cls.shape}")

        # end time tracking if inference mode
        if not train_mode:
            iter_time = time.time() - start_time
            # repeat iter time 18 times in list
            iter_times = [iter_time] * self.num_ant_queries
            iter_times.append(iter_time)
            outputs["iter_times"] = iter_times
            
        return outputs
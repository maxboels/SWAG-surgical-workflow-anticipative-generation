# Copyright (c) Maxence Boels

import torch
import torch.nn as nn
import math

import hydra
from hydra.types import TargetConf
from omegaconf import OmegaConf

import torch.nn as nn
import torch.nn.functional as F

from typing import List


class KeyRecorder(nn.Module):
    def __init__(self,
                dim: int = 512, 
                reduc_dim: int = 64, 
                sampling_rate: int = 10, 
                local_size: int = 20,
                pooling_method: str = 'max_abs',
                return_max_now_reduc: bool = False,
                reduction: str = "linear_relu",
                **kwargs) -> None:
        super().__init__()
        self.return_max_now_reduc = return_max_now_reduc
        self.pooling_method = pooling_method


        if reduction=="linear_relu":
            self.linear_reduction = nn.Sequential(
                nn.Linear(dim, reduc_dim),
                nn.ReLU(),
                nn.LayerNorm(reduc_dim) # replace LayerNorm with Sonething for the Temporality
            )
        elif reduction=="linear_sigmoid":
            self.linear_reduction = nn.Sequential(
                nn.Linear(512, reduc_dim),
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
    
    def max_frequency(self, x, norm_values=True, norm_dim=2):
        """
        Focuses on the relative presence of the values in the sequence more than the absolute values.
        Best to normalize over the dimensionality of the sequence to avoid biasing the results towards
        longer sequences and compare values across the dimensionality of the sequence.

        x: (batch_size, time, dim)
        norm_values: normalize the values across the dim dimension
        norm_dim: dimension to normalize the values across
        """
        x = F.normalize(x, p=2, dim=norm_dim)
        x = torch.cumsum(x, dim=1)
        x = torch.max(x, dim=1)[0]
        if norm_values:
            x = F.normalize(x, p=2, dim=1)

        return x
    
    def past_pooling(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: (batch_size, time, dim)
        """
        past = obs[:, :-self._local_size :self._sampling_rate,:]
        if self.pooling_method == 'max_abs':
            return torch.max(past, dim=1)[0]
        elif self.pooling_method == 'max_cumsum':
            return self.max_frequency(past)
        else:
            raise NotImplementedError(f"Pooling method {self.pooling_method} not implemented")
    
    def compare_past_present(self, past_pooled_reduc: torch.Tensor, past_present: torch.Tensor) -> torch.Tensor:
        """
        Return the max between local and global max, and the max of frame at time t.

        Args:
            past_pooled_reduc: (batch_size, dim)
            past_present: (batch_size, time, dim)

        Returns:
            global_max: (batch_size, time, dim)
            now_max: (batch_size, 1, dim)
        """
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
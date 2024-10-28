# Copyright (c) Facebook, Inc. and its affiliates.
#exp70
"""
Modular implementation of the basic train ops
"""
import os
from typing import Dict, Union, Tuple
import torch
import torch.nn as nn
import hydra
from hydra.types import TargetConf

# from common import utils
from R2A2.common.utils import seq_accuracy, seq_accuracy_nans

from datasets.base_video_dataset import FUTURE_PREFIX
from models.base_model import PAST_LOGITS_PREFIX
from loss_fn.multidim_xentropy import MultiDimCrossEntropy, MultiDimNN, PositionalCrossEntropyLoss
from loss_fn.distance_loss import DistanceLoss
import numpy as np

from R2A2.eval.eval_metrics import accuracy_n_pred
from R2A2.eval.plot_segments.plot_values import store_append_h5, store_training_videos_max
from R2A2.train.losses.ce_mse_consistency import CEConsistencyMSE
from R2A2.train.losses.remaining_time_loss import RemainingTimeLoss, ClassicRemainingTimeLoss, InMAEZoneSensitiveLoss

class NoLossAccuracy(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        return {}, {}


class BasicLossAccuracy(nn.Module):
    def __init__(self, dataset, device,
                base_rtd_loss="mse", # mse, mae, smooth_l1
                rem_time_loss_fn="horizon", # horizon, full_horizon, in_mae_zone_sensitive
                weight_type='exponential',
                gamma=0.5,
                mean_normalize_weights=False,
                time_horizon=5,
                loss_w_curr=0.5, loss_w_next=0.5, loss_w_feats=0.0, loss_w_remaining_time=1.0,
                regression_loss_scale=0.01):  # Add this parameter
        super().__init__()
        self.device = device
        self.loss_w_curr = loss_w_curr
        self.loss_w_next = loss_w_next
        self.loss_w_remaining_time = loss_w_remaining_time
        self.loss_w_feats = loss_w_feats
        self.regression_loss_scale = regression_loss_scale  # Initialize it
        #-----------------select params parameters for loss and accuracy-----------------
        self.model = "skit_v_ant" # options: "skit-x-ant", "skit-v-ant", "r2d2-x-ant", "r2d2-v-ant
        self.past_sampling_rate = 1
        self.target_type = "next_target" # options: "future_classes", "next_target", "future_feats"
        #------------------------------------------------------------------

        self.rem_time_loss_fn = rem_time_loss_fn
        self.time_horizon = time_horizon
        # Class weights
        if hasattr(dataset, "curr_class_weights"):
            curr_class_weights = dataset.curr_class_weights.to(device).float()
        else:
            curr_class_weights = torch.ones(7).to(device).float()
        print(f"[LOSS] class weights: {curr_class_weights}")

        if hasattr(dataset, "next_class_weights"):
            next_class_weights = dataset.next_class_weights.to(device).float()
        else:
            next_class_weights = curr_class_weights
        print(f"[LOSS] next class weights: {next_class_weights}")
        
        # Loss functions
        self.ce_loss_fn_curr    = nn.CrossEntropyLoss(weight=curr_class_weights, reduction='none', ignore_index=-1)

        if self.rem_time_loss_fn=="horizon":
            self.rtd_loss_fn = RemainingTimeLoss(h=time_horizon, 
                                                base_rtd_loss=base_rtd_loss, 
                                                weight_type=weight_type,
                                                gamma=gamma)
        elif self.rem_time_loss_fn=="classic":
            self.rtd_loss_fn = ClassicRemainingTimeLoss()

        elif self.rem_time_loss_fn=="in_mae_zone_sensitive":
            self.rtd_loss_fn = InMAEZoneSensitiveLoss(h=time_horizon, 
                                                 high_weight=2.8, 
                                                 low_weight=1.0, 
                                                 gamma=1.0, 
                                                 use_exponential=True)

        else:
            raise ValueError(f"Unknown time_regression: {self.rem_time_loss_fn}")
        
        # loss_fn = RemainingTimeLoss(h=5, weight_type='logarithmic', log_scale=10)


        if hasattr(dataset, "sampler_with_position"):
            sampler_with_position = dataset.sampler_with_position
            self.ce_loss_fn_next    = PositionalCrossEntropyLoss(weights_sampler=sampler_with_position,
                                                                class_weight=dataset.class_weight,
                                                                reduction='none', ignore_index=-1)
            self.ce_loss_fn_future  = PositionalCrossEntropyLoss(weights_sampler=sampler_with_position,
                                                                class_weight=dataset.class_weight,
                                                                reduction='none', ignore_index=-1)
        else:
            self.ce_loss_fn_next    = nn.CrossEntropyLoss(weight=next_class_weights, reduction='none', ignore_index=-1)
            self.ce_loss_fn_future  = nn.CrossEntropyLoss(weight=next_class_weights, reduction='none', ignore_index=-1)
            


    def forward(self, outputs, targets):
        losses = {}
        accuracies = {}

        for key in outputs:
            print(f"[LOSS] {key} output: {outputs[key].shape}") # (B, N, C)
            if key in targets:
                print(f"[LOSS] {key} target: {targets[key].shape}") # (B, N)

            if key == 'curr_frames':
                losses[key + '_loss'] = self.ce_loss_fn_curr(outputs[key].permute(0, 2, 1), targets[key]).mean() * self.loss_w_curr
                print(f"[LOSS] {key}_loss: {losses[key + '_loss']}")

                accuracies[key + '_acc'] = seq_accuracy_nans(outputs[key], targets[key])
                print(f"[LOSS] {key}_acc: {accuracies[key + '_acc']}")
            
            elif key == 'next_frames': # for the auto-regressive approach
                losses[key + '_loss'] = self.ce_loss_fn_next(outputs[key].permute(0, 2, 1), targets[key]).mean() * self.loss_w_next
                print(f"[LOSS] {key}_loss: {losses[key + '_loss']}")

                accuracies[key + '_acc'] = seq_accuracy_nans(outputs[key], targets[key])
                print(f"[LOSS] {key}_acc: {accuracies[key + '_acc']}")
            
            elif key == "feature_loss":
                losses[key + '_loss'] = outputs[key].mean() * self.loss_w_feats
                print(f"[LOSS] {key}_loss: {losses[key + '_loss']}")
            
            elif key == "future_frames": # for the single-pass approach
                losses[key + '_loss'] = self.ce_loss_fn_future(outputs[key].permute(0, 2, 1), targets[key]).mean() * self.loss_w_next
                print(f"[LOSS] {key}_loss: {losses[key + '_loss']}")

                accuracies[key + '_acc'] = seq_accuracy_nans(outputs[key], targets[key])
                print(f"[LOSS] {key}_acc: {accuracies[key + '_acc']}")
            
            elif key == "remaining_time":
                losses[key + '_loss'] = self.rtd_loss_fn(outputs[key], targets[key]).mean() * self.loss_w_remaining_time * self.regression_loss_scale
                print(f"[LOSS] {key}_loss: {losses[key + '_loss']}")
            else:
                raise ValueError(f"Unknown key: {key}")
        
        # total loss
        losses['total_loss'] = torch.sum(torch.stack([losses[key + '_loss'] for key in outputs.keys()]))

        print(f"[LOSS] total_loss: {losses['total_loss']}")

        return losses, accuracies



class Basic:
    def __init__(self,
        model,
        horizon,
        device,
        dataset,
        cls_loss_acc_fn: TargetConf,
    ):
        super().__init__()
        
        self.model = model
        self.h = horizon
        self.device = device
        self.cls_loss_acc_fn = hydra.utils.instantiate(cls_loss_acc_fn,
                                                        dataset, device,
                                                        time_horizon=horizon)

                                                       
    def __call__(
            self,
            data: dict,
            train_mode: bool = True):
        targets = {}

        if train_mode:
            self.model.train()
        else:
            self.model.eval()

        # Forward pass (training only)
        outputs = self.model(data['video'], data['curr_frames_tgt'], train_mode)
        
        if train_mode:
            for key in outputs.keys():
                if key == "feature_loss":
                    continue
                elif key == "remaining_time":
                    targets[key] = data[key+f"_{self.h}_tgt"]
                else:
                    targets[key] = data[key+"_tgt"]
                
                print(f"[TRAIN] {key} output: {outputs[key].shape}")
                
            # Compute loss and accuracy
            losses, accs = self.cls_loss_acc_fn(outputs, targets)
        else:
            losses = {}
            accs = {}

        return data, outputs, losses, accs

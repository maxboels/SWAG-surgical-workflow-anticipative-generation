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
from loss_fn.multidim_xentropy import MultiDimCrossEntropy, MultiDimNN
from loss_fn.distance_loss import DistanceLoss
import numpy as np

from R2A2.eval.eval_metrics import accuracy_n_pred
from R2A2.eval.plot_segments.plot_values import store_append_h5, store_training_videos_max
from R2A2.train.losses.ce_mse_consistency import CEConsistencyMSE


class NoLossAccuracy(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        return {}, {}


class BasicLossAccuracy(nn.Module):
    def __init__(self, dataset, device, balance_classes=False):
        super().__init__()
        self.device = device

        #-----------------select params parameters for loss and accuracy-----------------
        self.model = "skit_v_ant" # options: "skit-x-ant", "skit-v-ant", "r2d2-x-ant", "r2d2-v-ant
        self.past_sampling_rate = 1
        self.target_type = "next_target" # options: "future_classes", "next_target", "future_feats"
        #------------------------------------------------------------------
        
        if hasattr(dataset, "class_weights"):
            weights_train = dataset.class_weights.to(device).float()
        else:
            weights_train = torch.ones(7).to(device).float()
        print(f"[LOSS] class weights: {weights_train}")
        # cholec_weights_train = torch.from_numpy(np.asarray([
        #     1.6411019141231247,
        #     0.19090963801041133,
        #     1.0,
        #     0.2502662616859295,
        #     1.9176363911137977,
        #     0.9840248158200853,
        #     2.174635818337618,
        #     ])).to(device).float() # removed last weight
        
        if hasattr(dataset, "next_classes_weights"):
            next_classes_weights = dataset.next_classes_weights.to(device).float()
        else:
            next_classes_weights = torch.ones(7).to(device).float()
        print(f"[LOSS] next_classes_weights: {next_classes_weights}")
        
        self.ce_loss_fn = nn.CrossEntropyLoss(weight=weights_train, reduction='none', ignore_index=-1)
        self.cls_criterion = MultiDimCrossEntropy(weight=weights_train, reduction = 'none', ignore_index= -1)
        self.smoothl1loss = nn.MSELoss(reduction = 'none')
        self.DistanceLoss = DistanceLoss()
        self.duration_criterion = nn.MSELoss(reduction = 'none')
        self.future_ce_criterion = MultiDimCrossEntropy(weight=next_classes_weights, reduction = 'none', ignore_index= -1)
        self.future_mse_criterion = nn.MSELoss(reduction = 'none')
        self.ce_mse_consistency_ms = CEConsistencyMSE(ignore_idx=-1, 
                                                      ce_weight=weights_train,
                                                       mse_fraction=0.10, 
                                                       mse_clip_val=2.0, 
                                                       num_classes=7)

    def forward(self, outputs, targets):
        losses = {}
        accuracies = {}

        for key in outputs:
            print(f"[LOSS] output ({key}): {outputs[key].shape}")
            print(f"[LOSS] target ({key}): {targets[key].shape}")

            losses[key + '_loss'] = self.ce_loss_fn(outputs[key].permute(0, 2, 1), targets[key]).mean()
            print(f"[LOSS] {key}_loss: {losses[key + '_loss']}")

            accuracies[key + '_acc'] = seq_accuracy_nans(outputs[key], targets[key]) # mean included
            print(f"[LOSS] {key}_acc: {accuracies[key + '_acc']}")

        # total loss
        losses['total_loss'] = losses['curr_frames_loss'] * 0.5 + losses['next_frames_loss'] * 0.5
        print(f"[LOSS] total_loss: {losses['total_loss']}")

        return losses, accuracies



class Basic:
    def __init__(self,
        model,
        device,
        dataset,
        cls_loss_acc_fn: TargetConf,
        reg_criterion: TargetConf = None):
        super().__init__()
        
        self.model = model
        self.device = device
        self.cls_loss_acc_fn = hydra.utils.instantiate(cls_loss_acc_fn,
                                                       dataset, device)
        del reg_criterion  # not used here

    def _basic_preproc(self, data, train_mode):
        self.train_mode = train_mode
        if not isinstance(data, dict):
            video, target, heatmap_target = data
            # Make a dict so that later code can use it
            data = {}
            data['video'] = video
            data['curr_frames_tgt'] = target
            data['idx'] = -torch.ones_like(target)
            data['heatmap'] = heatmap_target
        if train_mode:
            self.model.train()
        else:
            self.model.eval()
        return data

    def __call__(
            self,
            data: Union[Dict[str, torch.Tensor],  # If dict
                        Tuple[torch.Tensor, torch.Tensor]],  # vid, target
            train_mode: bool = True):
        """
        Args:
            data (dict): Dictionary of all the data from the data loader
        """
        targets = {}
        data = self._basic_preproc(data, train_mode)

        # Forward pass
        outputs = self.model(data['video'], train_mode=train_mode)
        
        if train_mode:
            for key in outputs.keys():
                targets[key] = data[key+'_tgt']
            losses, accs = self.cls_loss_acc_fn(outputs, targets)
        else:
            losses = {}
            accs = {}

        return data, outputs, losses, accs

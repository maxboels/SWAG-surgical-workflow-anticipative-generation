# Yang Liu online


# Copyright (c) Facebook, Inc. and its affiliates.
#exp70
"""
Modular implementation of the basic train ops
"""
from typing import Dict, Union, Tuple
import torch
import torch.nn as nn
import hydra
from hydra.types import TargetConf

from common import utils

from datasets.base_video_dataset import FUTURE_PREFIX
from models.base_model import PAST_LOGITS_PREFIX
from loss_fn.multidim_xentropy import MultiDimCrossEntropy, MultiDimNN
from loss_fn.temporal_xentropy import TemporalCrossEntropy
from loss_fn.distance_loss import DistanceLoss
from loss_fn.ce_mse_consistency import CEConsistencyMSE
import numpy as np

from plot_segments.plot_values import plot_tensors

class TrainLossAcc(nn.Module):
    def __init__(self,
                device,
                ce_mse_past: TargetConf,
                ce_mse_present: TargetConf,
                # default args
                balance_classes=False,
                target_pos_weights=False,
                delay=0,
                anticip_time=0,
                rec_targets= "local",
                enc_target_idx=20,
                num_present_targets=20,
                num_future_targets=20,
                future_target=0,
                past_sampling_rate=10,
                present_length=20,
                ):
        super().__init__()
        self.device = device
        self.balance_classes = balance_classes
        self.target_pos_weights = target_pos_weights
        self.anticip_time = anticip_time
        self.rec_targets = rec_targets
        self.enc_target_idx = enc_target_idx
        self.num_present_targets = num_present_targets
        self.num_future_targets = num_future_targets

        self.past_sampling_rate = past_sampling_rate
        self.present_length = present_length

        weights_train = np.asarray(
            [1.6411019141231247,
            0.19090963801041133,
            1.0,
            0.2502662616859295,
            1.9176363911137977,
            0.9840248158200853,
            2.174635818337618
            ])
        weights_train = torch.from_numpy(weights_train).float().to(device)
        if self.target_pos_weights:
            x = np.linspace(0, self.num_future_targets-1, self.num_future_targets)
            factor = 0.5
            sigma = num_future_targets / (num_future_targets * (1-factor))
            mean = future_target # set the mean to the target frame index
            target_pos_weights = np.exp(-(x-mean)**2/(2*sigma**2))
            target_pos_weights /= np.max(target_pos_weights) # normalize to max value of 1
            target_pos_weights = torch.from_numpy(target_pos_weights).float().to(device)
            print(f"target_weights: {target_pos_weights}")
        else:
            target_pos_weights = None
        
        # frame-level losses with consistency
        self.ce_mse_past = hydra.utils.instantiate(ce_mse_past, ce_weight=weights_train, _recursive_=False,)
        self.ce_mse_present = hydra.utils.instantiate(ce_mse_present, ce_weight=weights_train, _recursive_=False,)
        # segment-level losses (observed and future segments)
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, outputs, train_targets):
        losses_dict = {"total_loss": 0.0}
        accuracies_dict = {}
        losses_weights_dict = {}

        # PAST RECOGNITION -> high interval sampling and  of 10 frames, so low consistency loss weight.
        if outputs["past_recognition"] is not None:
            pred = outputs['past_recognition'] # pred: (4, 64, 7, 320)
            targets = train_targets["recognition"][:, :-self.present_length:self.past_sampling_rate] # (B, T)
            loss = self.ce_mse_past(pred, targets) * 0.15 # large number of frames (320). So we can use a lower weight.
            losses_dict[f'ce_mse_loss_past'] = loss

        # PRESENT RECOGNITION WITH PAST CONTEXT
        if outputs["recognition_dec"] is not None:
            if "observed_multi_class" in train_targets.keys():
                pred = outputs['recognition_dec'].squeeze(-1)
                targets = train_targets["observed_multi_class"][:, -1, :]
                loss = self.bce(pred, targets) * 0.25
                losses_dict[f'observed_multi_class'] = loss
            else:
                pred = outputs['recognition_dec'].transpose(1, 2)
                targets = train_targets["recognition"][:, -self.num_present_targets:]
                loss = self.ce_mse_present(pred, targets) * 0.25
                losses_dict[f'present_decoder_loss'] = loss
                acc = utils.accuracy(pred, targets, ignore_index=-1)
                accuracies_dict[f'recognition_present_deocoder'] = acc

        # PRESENT ANTICIPATIVE RECOGNITION
        if outputs['recognition'] is not None:
            pred = outputs['recognition'].transpose(1, 2)
            targets = train_targets["recognition"][:, -self.num_present_targets:]
            loss = self.ce_mse_present(pred, targets) * 0.50
            losses_dict[f'present_loss'] = loss
            acc = utils.accuracy(pred, targets, ignore_index=-1)
            accuracies_dict[f'recognition'] = acc

        # ANTICIPATION
        if outputs['anticipation'] is not None:
            if "future_multi_class" in train_targets.keys():
                pred = outputs["anticipation"].squeeze(-1)
                targets = train_targets["future_multi_class"][:, 0, :]
                loss = self.bce(pred, targets) * 0.5
                losses_dict[f'future_multi_class'] = loss
            else:
                pred = outputs['anticipation'].transpose(1, 2)
                targets = train_targets["anticipation"][:, :self.num_future_targets]
                loss = torch.mean(self.anticip_criterion(pred, targets)) * 0.5
                losses_dict[f'anticipation_loss'] = loss
                acc = utils.accuracy(pred, targets, ignore_index=-1)
                accuracies_dict[f'anticipation'] = acc
                
        for key in losses_dict.keys():
            losses_dict[f'total_loss'] += losses_dict[key]
            print(f"loss ({key}): {losses_dict[key]}")
        print(f"total_loss: {losses_dict[f'total_loss']}")

        return losses_dict, accuracies_dict

class Evaluation(nn.Module):
    """
    Return the predictions and targets for evaluation
    """

    def __init__(self,
                eval_idx=-1,
                num_eval_vids=40,
                ):
        super().__init__()
        self.eval_idx = eval_idx # -1
        self.num_eval_vids = num_eval_vids

        # recognition
        self.acc_vid = {str(i): [] for i in range(0, num_eval_vids)}
        self.label_phase_vid = {str(i): [] for i in range(0, num_eval_vids)}
        self.pred_phase_vid = {str(i): [] for i in range(0, num_eval_vids)}
        self.label_phase = []
        self.pred_phase = []

        # anticipation
        self.acc_vid_a = {str(i): [] for i in range(0, num_eval_vids)}
        self.label_phase_vid_a = {str(i): [] for i in range(0, num_eval_vids)}
        self.pred_phase_vid_a = {str(i): [] for i in range(0, num_eval_vids)}
        self.label_phase_a = []
        self.pred_phase_a = []
    
    def reset_metrics(self):
        """
        Reset the evaluation metrics after each epoch
        """
        # recognition
        self.acc_vid = {str(i): [] for i in range(0, self.num_eval_vids)}
        self.label_phase_vid = {str(i): [] for i in range(0, self.num_eval_vids)}
        self.pred_phase_vid = {str(i): [] for i in range(0, self.num_eval_vids)}                
        self.label_phase = []
        self.pred_phase = []

        # anticipation
        self.acc_vid_a = {str(i): [] for i in range(0, self.num_eval_vids)}
        self.label_phase_vid_a = {str(i): [] for i in range(0, self.num_eval_vids)}
        self.pred_phase_vid_a = {str(i): [] for i in range(0, self.num_eval_vids)}
        self.label_phase_a = []
        self.pred_phase_a = []
    
    def get_targets(self, data, eval_target, eval_idx):
        """
        One evaluation at a time: either recognition or anticipation.
        """
        targets = data[eval_target].detach().cpu().numpy()
        print(f"[Evaluation] {eval_target} tgt shape: {targets.shape}")
        if eval_target=='recognition':
            targets = targets[:, -1]
        elif eval_target=='anticipation':
            targets = targets[:, 0] # NOTE: index = 0 since we already shifted the future video by anticip_time.
        elif eval_target=='future_seg_cls':
            targets = targets[:, 0]
        elif eval_target=='recognition_asr':
            targets = targets[:, -1]
        else:
            raise ValueError("Unknown eval_typ for get_targets")

        return targets
    
    def get_predictions(self, outputs, eval_pred, eval_idx=-1):
        """
        One evaluation at a time: either recognition or anticipation.
        outputs: dict of lists with length = ccit
        """
        pred = outputs[eval_pred].detach().cpu().numpy()
        print(f"[Evaluation] {eval_pred} pred shape: {pred.shape}") # 64, 10, 7
        if eval_pred=='recognition':
            pred = np.argmax(pred[:, -1, :], axis=-1)
        elif eval_pred=='anticipation':
            pred = np.argmax(pred[:, 0, :], axis=-1) # NOTE: index = 0 since we already shifted the future video by anticip_time.
        elif eval_pred=='future_seg_cls':
            pred = np.argmax(pred[:, 0, :], axis=-1)
        elif eval_pred=='recognition_asr':
            # NOTE: all predictions are for the last frame. So we can sum over the frames.
            pred = np.argmax(np.sum(pred, axis=1), axis=-1)  # sum over the classes
        else:
            raise ValueError("Unknown eval_typ for get_predictions")
        
        return pred

    
    def metrics_logger(self, outputs, data, eval_type="recognition"):
        predict_batch = self.get_predictions(outputs, eval_type, self.eval_idx)
        targets_batch = self.get_targets(data, eval_type, self.eval_idx)

        with torch.no_grad():
            for i, (uid, target) in enumerate(zip(data['uid'], targets_batch)):
                vid = str(uid.detach().cpu().numpy())
                if target < 0:
                    continue
                predict = predict_batch[i]
                val = int(predict == target) # 1 if correct, 0 if incorrect
                if eval_type == "recognition":
                    self.acc_vid[vid].append(val)
                    self.label_phase_vid[vid].append(target)
                    self.pred_phase_vid[vid].append(predict)
                    self.label_phase.append(target)
                    self.pred_phase.append(predict)
                elif eval_type == "anticipation":
                    self.acc_vid_a[vid].append(val)
                    self.label_phase_vid_a[vid].append(target)
                    self.pred_phase_vid_a[vid].append(predict)
                    self.label_phase_a.append(target)
                    self.pred_phase_a.append(predict)
        

    def get_metrics(self, eval_type="recognition"):
        """
        Return the metrics
        """
        if eval_type=="recognition":
            return self.acc_vid, self.label_phase_vid, self.pred_phase_vid, self.label_phase, self.pred_phase
        elif eval_type=="anticipation":
            return self.acc_vid_a, self.label_phase_vid_a, self.pred_phase_vid_a, self.label_phase_a, self.pred_phase_a


class TrainEvalOps():
    def __init__(self,
                 model,
                 device,
                 train_targets,
                 eval_types,
                 train_loss_acc: TargetConf,
                 eval_pred_target: TargetConf
                 ):
        super().__init__()

        self.model = model
        self.device = device
        self.train_targets = train_targets
        self.eval_types = eval_types
        self.plot_every = 1500

        self.train_loss_acc = hydra.utils.instantiate(train_loss_acc,
                                                        _recursive_=False,
        )
        self.eval_pred_target = hydra.utils.instantiate(eval_pred_target,
                                                        _recursive_=False,
        )
    
    def reset_eval_metrics(self):
        self.eval_pred_target.reset_metrics()
    
    def metrics_logger(self, outputs, data, eval_type="recognition"):
        self.eval_pred_target.metrics_logger(outputs, data, eval_type)

    def get_eval_metrics(self, eval_type="recognition"):
        return self.eval_pred_target.get_metrics(eval_type)

    def __call__(self, data: Dict[str, torch.Tensor], train_mode: bool = True):
        """
        Args:
            data (dict): Dictionary of all the data from the data loader
        """
        video = data['video'].to(self.device, non_blocking=True)
        if "pad_mask_rec" in data.keys():
            pad_mask = data['pad_mask_rec'].to(self.device, non_blocking=True)
        else:
            pad_mask = None

        if "future_video" in data.keys():
            future_video = data['future_video'].to(self.device, non_blocking=True)
        else:
            future_video = None

        train_targets = {}
        # filter the targets in data based on the train_targets list
        for key in self.train_targets:
            train_targets[key] = data[key].to(self.device, non_blocking=True)
        print(f"train_targets: {train_targets.keys()}")

        past_classes, present_dec_classes, present_classes, future_classes, loss_features = self.model(
            video,
            future_video=future_video,
            pad_mask=pad_mask,
        )


        # get() values

        outputs = {
            "past_recognition": past_classes,
            "recognition_dec": present_dec_classes,
            'recognition': present_classes,
            'anticipation': future_classes,
            'loss_features': loss_features,
        }

        if train_mode:
            losses, accuracies = self.train_loss_acc(outputs, train_targets)
            return data, outputs, losses, accuracies
        else:
            for eval_type in self.eval_types:
                self.metrics_logger(outputs, data, eval_type)
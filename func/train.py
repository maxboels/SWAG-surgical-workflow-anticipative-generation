# Copyright (c) Facebook, Inc. and its affiliates.

"""Training code."""
import pickle
from PIL import Image
import json
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from typing import Union, Sequence
from random import shuffle
import datetime
import os
import time
import sys
import logging
import itertools
import operator
import psutil
import h5py
import subprocess
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import cv2
# Need to import this here, as with pytorch 1.7.1 (or some other CLIP dep)
# it's giving a segmentation fault
# https://github.com/pytorch/pytorch/issues/30651
# Needs to imported before torchvision it seems
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.utils.data
from torch.utils.data.dataloader import default_collate
from torch import nn
import torchvision
import torchvision.datasets.video_utils
from torchvision.datasets.samplers import (DistributedSampler,
                                           UniformClipSampler,
                                           RandomClipSampler)
import torch.distributed as dist
import hydra
from omegaconf import OmegaConf

from models import base_model
from common import scheduler, utils, transforms as T
from common.log import MetricLogger, setup_tbx, get_default_loggers
# from datasets.data import get_dataset
# from notebooks import utils as nb_utils
from datasets.base_video_dataset import Medical_Dataset, SelectDataset
# from models.ResNet import resnet_lstm
from sklearn import metrics
__all__ = ['main', 'evaluate', 'train_one_epoch', 'initial_setup']
RESULTS_SAVE_DIR = 'results'  # Don't put a "/" at the end, will add later
CKPT_FNAME = 'checkpoint.pth'
DATASET_TRAIN_CFG_KEY = 'dataset_train'
DATASET_EVAL_CFG_KEY = 'dataset_eval'
STR_UID_MAXLEN = 64  # Max length of the string UID stored in H5PY


#--maxence boels impots
# from R2A2.eval.plot_video_preds import get_plotter
from R2A2.eval.plot_segments.plot_values import plot_maxpooled_videos
from R2A2.eval.seg_eval import MetricsSegments
from R2A2.eval.plot_segments.plot_video import plot_video_segments
from R2A2.eval.plot_metrics import plot_figure, plot_box_plot_figure, plot_cumulative_time
from R2A2.eval.plot_video_3D import plot_video_contour_3D, plot_video_scatter_3D
from  R2A2.eval.plot_2d_video_classification import plot_classification_video

from R2A2.eval.plot_remaining_time import plot_remaining_time_video
from R2A2.eval.plot_video_combined import plot_video_combined

from R2A2.eval.convert_tasks import regression2classification, classification2regression
from R2A2.eval.classprobs2reg_v3 import find_time_to_next_occurrence

from loss_fn.mae import anticipation_mae

from R2A2.eval.evaluation_metrics import compute_f1_score, compute_rmse_transition_times, \
                                        compute_transition_times, compute_accuracy, \
                                        calculate_metrics, plot_performance_over_time, \
                                        segment_continuity_score, temporal_consistency_score, \
                                        class_distribution_divergence, aggregate_metrics


#--maxence boels- debugging cuda error
import os
os.environ['TORCH_USE_CUDA_DSA'] = '1'
# import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
#--maxence boels-

def check_numpy_to_list(dictionay):
    for key in dictionay.keys():
        if isinstance(dictionay[key], np.ndarray):
            dictionay[key] = dictionay[key].tolist()
            print(f"converted {key} np.ndarray to list")
    return dictionay

def store_checkpoint(fpaths: Union[str, Sequence[str]], model, optimizer,
                     lr_scheduler, epoch):
    """
    Args:
        fpaths: List of paths or a single path, where to store.
        model: the model to be stored
        optimizer, lr_scheduler
        epoch: How many epochs have elapsed when this model is being stored.
    """
    model_without_ddp = model
    if isinstance(model, nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    checkpoint = {
        "model": model_without_ddp.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "epoch": epoch,
    }
    if not isinstance(fpaths, list):
        fpaths = [fpaths]
    for fpath in fpaths:
        logging.info('Storing ckpt at epoch %f to %s', epoch, fpath)
        utils.save_on_master(checkpoint, fpath)

def _get_memory_usage_gb():
    mem = psutil.virtual_memory()
    return mem.used / (1024**3)

def train_one_epoch(
        model,
        step_now,
        device,
        train_eval_op,
        optimizer,
        lr_scheduler,
        dataloader_train,
        epoch: int,
        partial_epoch: float,
        tb_writer,
        logger,
        last_saved_time,
        # kwargs:
        print_freq,
        print_large_freq,
        grad_clip_params,
        loss_wts,  # All the loss wts go here
        save_freq: float,  # num epochs to save at. Could be fractional.
        save_freq_min: float,  # Save a checkpoint every this many minutes
        save_intermediates: bool,
):
    """
    Args:
        epoch (int) defines how many full epochs have finished
        partial_epoch (float): Defines the ratio of the last epoch that was
            finished before the current model was written out
    """
    header = 'Epoch: [{}]'.format(epoch)
    batches_per_epoch = len(dataloader_train)
    # Run the data loader for the partial epochs
    partial_iters = int(batches_per_epoch * partial_epoch)
    if partial_iters > 0:
        # TODO: Figure a better way to do this ... too slow
        for i, _ in tqdm(enumerate(dataloader_train),
                         desc=(f'Loading and throwing data for '
                               f'{partial_epoch:0.8f} epochs or '
                               f'{partial_iters} iters'),
                         total=partial_iters):
            if i >= partial_iters:
                break
    if save_freq:
        save_freq_steps = int(save_freq * batches_per_epoch)
        logger.info('Storing checkpoints every %0.8f epochs, or '
                    '%d steps', save_freq, save_freq_steps)
    if save_freq_min:
        logger.info('Storing checkpoints every %0.2f mins', save_freq_min)
    
    start_time = time.time()
    for i, data in enumerate(
            dataloader_train
            # metric_logger.log_every(dataloader_train, print_freq, header),
            #partial_iters
            ):
        step_id = epoch * batches_per_epoch + i
        cur_epoch = step_id / batches_per_epoch
        time_now = datetime.datetime.now()
        mins_since_last_saved = (time_now -
                                 last_saved_time).total_seconds() / 60.0
        if (save_freq and step_id % save_freq_steps == 0) or (
                save_freq_min and (mins_since_last_saved >= save_freq_min)):
            # Not storing in the main checkpoint, keeping that only for the
            # models at full epoch boundaries. So set save_intermediates true
            # to save models at these points
            ckpt_names = []
            if save_intermediates:
                ckpt_names.append(f'checkpoint_ep{cur_epoch:.8f}.pth')
            store_checkpoint(ckpt_names, train_eval_op.model, optimizer,
                             lr_scheduler, cur_epoch)
            last_saved_time = time_now

        data, _, losses, accuracies = train_eval_op(data,
                                                    train_mode=True)

        remaining_time = (time.time() - start_time) / (i + 1) * (len(dataloader_train) - i - 1) / 60.0

        if i % 100 == 0:
            logger.info(f"[TRAINING] Step {i}/{len(dataloader_train)} | "
                f"Loss: {losses['total_loss'].item():.2f} | "
                f"Acc (curr_frames): {accuracies['curr_frames_acc']:.2f} | "
                f"time-left: {remaining_time:.2f} min")

        loss = losses["total_loss"]

        optimizer.zero_grad()
        loss.backward()

        if not isinstance(loss, float):
            loss = loss.item()

        optimizer.step()

        batch_size = dataloader_train.batch_size
        # metric_logger.update(loss=loss,
        #                      lr=optimizer.param_groups[0]['lr'])

        # TENSORBOARD LOGGING
        for key, val in losses.items():
            tb_writer.add_scalar(f'train_per_iter/loss/{key}', val.item(), step_id)
        for key, val in accuracies.items():
            tb_writer.add_scalar(f'train_per_iter/acc/{key}', val, step_id)
        tb_writer.add_scalar('train_per_iter/lr', optimizer.param_groups[0]['lr'], step_id)


        if not isinstance(lr_scheduler.base_scheduler,
                        scheduler.ReduceLROnPlateau):
        # If it is, then that is handled in the main training loop,
        # since it uses the validation accuracy to step down
            lr_scheduler.step()
    return last_saved_time, step_now

def is_better(condition, score, best_score):
    if condition == 'lower':
        return score < best_score
    else:
        return score > best_score


def evaluate(cfg, model, train_eval_op, device, step_now, dataloaders: list, tb_writer, logger, epoch: float,
    eval_horizons: int = [18],
    anticip_time: int = 60,
    max_anticip_time: int = 18,
    store=False, 
    store_endpoint='logits', 
    only_run_featext=False,
    best_score=np.inf,
    main_metric='inMAE',
    horizon=90,
    probs_to_regression_method: str = 'first_occurrence',
    confidence_threshold: float = 0.5,
    do_classification: bool = True,
    do_regression: bool = True,
    ):

    # create results folder if not exists
    if not os.path.exists(f'./results'):
        os.makedirs(f'./results')
    
    step_size = int(anticip_time/60)
    max_num_steps = int((max_anticip_time * 60) / anticip_time)
    start = int(anticip_time / 60)
    logger.info(f"[EVAL] Epoch: {epoch} | "
                f"Anticipation time (s): {anticip_time} | "
                f"Max anticipation time (m): {max_anticip_time} | "
                f"Step size (m): {step_size} | "
                f"Max auto-reg. steps: {max_num_steps}")

    model.eval()
    # -----------------select params----------------- #
    save_best_video_preds = True
    save_all_metrics = False

    num_classes = 7
    plot_video_freq = 5
    # -----------------select params----------------- #

    # init video metrics
    all_videos_results = OrderedDict()
    all_videos_mean_acc_curr    = []
    all_videos_mean_acc_future  = []
    all_videos_acc_future       = []
    all_videos_cum_acc_future   = []
    all_videos_rmse_future      = []
    all_videos_mean_f1_curr     = []
    all_videos_mean_f1_future   = []
    all_videos_mean_cum_iter_time = []
    eval_start_time = time.time()

    # best_score = 0.2
    best_epoch = 0

    # init video gt and preds
    video_ids = []
    all_video_frame_preds = {}
    all_video_frame_rec = {}
    all_video_tgts_preds = {}
    all_video_tgts_rec = {}
    all_video_mean_curr_acc = {}
    all_video_mean_cum_acc_future = {}
    # regression remaining time
    
    all_video_remaining_time_preds = {}
    all_video_remaining_time_tgts = {}
    for h in eval_horizons:
        all_video_remaining_time_tgts[f'{h}'] = {}

    all_vids_acc = []
    all_vids_prec = []
    all_vids_recall = []
    all_vids_f1 = []

    all_metrics = []
    all_frame_metrics = []

    # ignore the EOS class for the metrics (comparing with previous works)
    ignore_classes = [-1, 7]

    # init metrics fn
    max_horizon = max(eval_horizons)
    for h in eval_horizons:
        locals()[f"mae_metric_{h}"] = anticipation_mae(h=h, ignore_index=ignore_classes)

    # init metrics    
    MAEs = ["wMAE_class", "inMAE_class", "outMAE_class", "expMAE_class", 
            "wMAE", "inMAE", "outMAE", "expMAE"]

    all_videos_metrics = {}
    for h in eval_horizons:
        for metric in MAEs:
            all_videos_metrics[f'{metric}_{h}'] = []
    
    curr_frames = False
    future_frames = False
    remaining_time = False

    store = True


    # FOR EACH VIDEO LOADER
    for video_idx, data_loader in enumerate(dataloaders):
        dataset = data_loader.dataset.dataset_name
        vid_start_time = time.time()
        video_results = OrderedDict()
        video_length = len(data_loader.dataset)
        video_id = data_loader.dataset.video_indices[0] # only one per video loader in test set
        video_ids.append(video_id)
        batch_size = data_loader.batch_size

        # eval_horizons = data_loader.dataset.eval_horizons

        logger.info(f"[EVAL] video (i={video_idx}) id={video_id} | video_length={video_length}")

        num_ant_classes = num_classes + 1  # Add end-of-surgery class

        iters_times = []

        # recognition
        video_frame_rec     = np.full((video_length, 1), -1) # classes are intgers
        video_tgts_rec      = np.full((video_length, 1), -1) # classes are intgers
        # anticipations
        video_frame_preds   = np.full((video_length, max_num_steps), -1) # classes are intgers
        video_tgts_preds    = np.full((video_length, max_num_steps), -1) # classes are intgers

        # recognition and antcipations probs
        # WARNING: make sure to init with float32 to avoid issues with interpolation towards zero.
        # ERROR: np.full((video_length, max_num_steps+1, num_ant_classes), 0) -> int64
        video_frame_probs_preds = np.full((video_length, max_num_steps+1, num_ant_classes), 0.0, dtype=np.float32)
        # video_frame_probs_preds = np.zeros((video_length, max_num_steps+1, num_ant_classes), dtype=np.float32)
        
        # save remaining time gt and predictions
        # NOTE: make sure the targets are floating point values and not integers
        # as they represent the remaining time in minutes (continuous values)
        video_remaining_time_tgts = {}
        video_remaining_time_preds_h = {}
        for h in eval_horizons:
            video_remaining_time_tgts[f'{h}'] = torch.full((video_length, 1, num_ant_classes), -1, dtype=torch.float32)
        video_remaining_time_preds = torch.full((video_length, 1, num_ant_classes), -1, dtype=torch.float32)

        # # create for full remaining time targets
        # video_remaining_time_full_tgts = torch.full((video_length, 1, num_ant_classes), -1, dtype=torch.float32)
        # video_remaining_time_full_preds = torch.full((video_length, 1, num_ant_classes), -1, dtype=torch.float32)

        video_mean_cum_iter_time = []

        start_idx = 0
        end_idx = start_idx + batch_size

        regression_loss = nn.SmoothL1Loss(reduction='none')
        reg_losses = []
        
        # eval loop
        for b, data in enumerate(data_loader):

            batch_size = data['video'].shape[0]
            first_frame_idx = data['frame_idx'][0].detach().cpu().numpy()
            curr_frame = data['frame_idx'][-1].detach().cpu().numpy()

            # phases_rem_time = data[f'gt_remaining_time_{anticip_time}'].detach().cpu().numpy()

            with torch.no_grad():
                        
                outputs = model(data['video'], data['curr_frames_tgt'], train_mode=False)

                # RECOGNITION
                if "curr_frames" in outputs.keys():
                    probs = torch.softmax(outputs['curr_frames'][:, -1: , :], dim=2).detach().cpu().numpy() # shape (N, 1, C)
                    # print(f"curr_frames (probs): {probs}")
                    # logger.info(f"[EVAL] curr_frames (probs): {probs.shape}")
                    video_frame_rec[start_idx:end_idx] = np.argmax(probs, axis=2)
                    video_frame_probs_preds[start_idx:end_idx, :1, :-1] = probs # shape (N, 1, C)
                    curr_frames = True
                                    
                video_tgts_rec[start_idx:end_idx] = data['curr_frames_tgt'].detach().cpu().numpy()[:,-1:]
                
                # FUTURE FRAMES CLASSIFICATION
                if "future_frames" in outputs.keys():
                    probs = torch.softmax(outputs['future_frames'], dim=2).detach().cpu().numpy() # shape (N, T, C+1)
                    # print(f"future_frames (probs): {probs}")
                    # logger.info(f"[EVAL] future_frames (logits): {outputs['future_frames'].shape}")
                    # logger.info(f"[EVAL] future_frames (probs): {probs.shape}")
                    video_frame_preds[start_idx:end_idx] = np.argmax(probs, axis=2)
                    video_frame_probs_preds[start_idx:end_idx, 1:, :] = probs # shape (N, T, C+1)
                    future_frames = True

                video_tgts_preds[start_idx:end_idx] = data["future_frames_tgt"].detach().cpu().numpy()
                    
                # REMAINING TIME REGRESSION
                if "remaining_time" in outputs.keys():
                    video_remaining_time_preds[start_idx:end_idx] = outputs[f'remaining_time'].detach().cpu()
                    remaining_time = True
                
                # if "remaining_time_full_tgt" in data.keys():
                #     video_remaining_time_full_preds[start_idx:end_idx] = outputs[f'remaining_time_full'].detach().cpu()
                #     remaining_time = True


                # save the remaining time targets for regression and classification
                # if eval_horizons list is not empty
                # if len(eval_horizons) > 0:
                for h in eval_horizons:
                    max_horizon = max(eval_horizons)
                    video_remaining_time_tgts[f'{h}'][start_idx:end_idx] = data[f'remaining_time_{max_horizon}_tgt'].detach().cpu()

                if "iters_time" in outputs.keys():
                    iters_time = outputs["iters_time"]  # list with n AR steps
                    iters_times.append(iters_time)      # list of lists
                if "iter_times" in outputs.keys():
                    iter_times = outputs["iter_times"]
                    video_mean_cum_iter_time.append(iter_times)
                
                logger.info(f"[TESTING] video: {video_id} | frame: {curr_frame} / {video_length}")
                
                start_idx += batch_size
                end_idx += batch_size
                logger.info(f"[TESTING] video: {video_id} | start_idx: {start_idx} | end_idx: {end_idx}")
        
        # END OF CURRENT VIDEO LOOP
        vid_test_time = time.time() - vid_start_time
        test_time = time.time() - eval_start_time
        logger.info(f"[TESTING] video: {video_id} | "
                    f"video test time: {vid_test_time/60:.2f} min | "
                    f"total test time: {test_time/60:.2f} min")
        
        # save video probs to numpy instead of keeping in memory
        if store and cfg.test_only:
            if not os.path.exists(f'./probs'):
                os.makedirs(f'./probs')
            np.save(f'./probs/video_probs_ep{epoch}_vid{video_id}.npy', video_frame_probs_preds)
            logger.info(f"[TESTING] Saved video probs to: video_probs_ep{epoch}_vid{video_id}.npy")
            
            # raise error if only zero values in probs
            if np.all(video_frame_probs_preds == 0):
                raise ValueError(f"video_frame_probs_preds are all zeros")

        # STORE VIDEO RESULTS
        video_results["video_id"] = video_id        

        # ---------------------------------------------------------------------
        # EVEN IF THE MODEL OUTPUTS ONE TASK, WE CAN EVALUATE BOTH TASKS
        # BY CONVERTING THE PROBABILITIES TO REGRESSION VALUES AND VICE VERSA
        # ---------------------------------------------------------------------
        # THE OBJECTIVE IS TO EVALUATE THE MODEL PERFORMANCE ON BOTH TASKS
        # SEPARATELY AND TOGETHER
        # ---------------------------------------------------------------------

        # convert regression to classification task
        if not future_frames and remaining_time:
            video_frame_preds = regression2classification(video_remaining_time_preds, horizon_minutes=max_num_steps)
            logger.info(f"[TESTING] video_frame_preds (regression2classification): {video_frame_preds.shape}")

        if cfg.test_only:
            # save the classification predictions to numpy
            if not os.path.exists(f'./class_preds'):
                os.makedirs(f'./class_preds')
            np.save(f'./class_preds/video_frame_preds_ep{epoch}_vid{video_id}.npy', video_frame_preds)

        # convert the class probabilities to class remaining time regression values
        if future_frames and not remaining_time:      
            max_horizon = max(eval_horizons)      
            for h in eval_horizons:
                video_remaining_time_preds = find_time_to_next_occurrence(video_frame_probs_preds, horizon_minutes=max_horizon)
                logger.info(f"[TESTING] video_remaining_time_preds (classification2regression) h={h}: {video_remaining_time_preds.shape}")
                video_remaining_time_preds_h[h] = video_remaining_time_preds

                if cfg.test_only:
                    # save the remaining time predictions to numpy
                    if not os.path.exists(f'./rtd_preds'):
                        os.makedirs(f'./rtd_preds')
                    np.save(f'./rtd_preds/video_remaining_time_preds_ep{epoch}_vid{video_id}_h{h}.npy', video_remaining_time_preds)

                    # save the ground truth remaining time to numpy if not already saved
                    if not os.path.exists(f'./rtd_tgts'):
                        os.makedirs(f'./rtd_tgts')
                    np.save(f'./rtd_tgts/video_remaining_time_tgts_vid{video_id}_h{h}.npy', video_remaining_time_tgts[f'{h}'])

        else:
            # same preds for all horizons
            for h in eval_horizons:
                video_remaining_time_preds_h[h] = video_remaining_time_preds

                if cfg.test_only:
                    # save the remaining time predictions to numpy
                    if not os.path.exists(f'./rtd_preds'):
                        os.makedirs(f'./rtd_preds')
                    np.save(f'./rtd_preds/video_remaining_time_preds_ep{epoch}_vid{video_id}_h{h}.npy', video_remaining_time_preds)
        
                    # save the ground truth remaining time to numpy if not already saved
                    if not os.path.exists(f'./rtd_tgts'):
                        os.makedirs(f'./rtd_tgts')
                    np.save(f'./rtd_tgts/video_remaining_time_tgts_vid{video_id}_h{h}.npy', video_remaining_time_tgts[f'{h}'])

        # store the remaining time predictions dict
        all_video_remaining_time_preds[video_id] = video_remaining_time_preds_h

        # compute the metrics for the remaining time regression
        for h in eval_horizons:
            video_remaining_time_tgts_h = video_remaining_time_tgts[f'{h}']
            all_video_remaining_time_tgts[f'{h}'][video_id] = video_remaining_time_tgts_h

            # compute the metrics for the remaining time regression
            mae_results = locals()[f'mae_metric_{h}'](video_remaining_time_preds_h[h], video_remaining_time_tgts_h)

            # store the metrics for the current video
            for metric, value in mae_results.items():
                if isinstance(value, torch.Tensor):
                    value = value.cpu().numpy()
                all_videos_metrics[f'{metric}_{h}'].append(value)

            
            # overall metrics (without the excluded class: EOS)
            wMAE    = mae_results['wMAE']
            inMAE   = mae_results['inMAE']
            outMAE  = mae_results['outMAE']
            expMAE  = mae_results['expMAE'] # within last 10% before event occurrence
                        
            logger.info(f"[TESTING] video: {video_id} | "
                        f"horizon: {h} | "
                        f"wMAE_{h}: {wMAE:.4f} | "
                        f"inMAE_{h}: {inMAE:.4f} | "
                        f"outMAE_{h}: {outMAE:.4f} | "
                        f"expMAE_{h}: {expMAE:.4f}")

        if cfg.test_only and store:
            if not os.path.exists(f'./class_preds'):
                os.makedirs(f'./class_preds')
            if not os.path.exists(f'./class_tgts'):
                os.makedirs(f'./class_tgts')
            np.save(f"./class_preds/video_frame_rec_{video_id}_ep{epoch}.npy", video_frame_rec)
            np.save(f"./class_preds/video_frame_preds_{video_id}_ep{epoch}.npy", video_frame_preds)
            np.save(f"./class_tgts/video_tgts_rec_{video_id}_ep{epoch}.npy", video_tgts_rec)
            np.save(f"./class_tgts/video_tgts_preds_{video_id}_ep{epoch}.npy", video_tgts_preds)
            logger.info(f"[TESTING] video: {video_id} saved numpy arrays")
            
        all_video_frame_rec[video_id]   = video_frame_rec
        all_video_frame_preds[video_id] = video_frame_preds

        # targets for recognition and anticipation
        all_video_tgts_rec[video_id]    = video_tgts_rec
        all_video_tgts_preds[video_id] = video_tgts_preds
        
        # concatenate the recognition and predictions y_true and y_pred
        y_true = np.concatenate((video_tgts_rec, video_tgts_preds), axis=1)
        y_pred = np.concatenate((video_frame_rec, video_frame_preds), axis=1)

        # all metrics for predictions
        metrics, frame_metrics = calculate_metrics(y_true=y_true, y_pred=y_pred)
        all_metrics.append(metrics)
        all_frame_metrics.append(frame_metrics)
        # store the metrics for the video
        video_accuracy = []
        video_prec = []
        video_recall = []
        video_f1 = []

        # compute the accuracy for the video (frame-level flattened)
        video_accuracy = accuracy_score(y_true.flatten(), y_pred.flatten())
        all_vids_acc.append(video_accuracy)

        # compute the weighted precision, recall, f1 per anticipated segment
        for i in range(video_frame_preds.shape[0]):
            y_true_i = y_true[i]
            y_pred_i = y_pred[i]
            precision, recall, f1, _ = precision_recall_fscore_support(y_true_i, y_pred_i, average='weighted', zero_division=0)
            video_prec.append(precision)
            video_recall.append(recall)
            video_f1.append(f1)
        all_vids_prec.append(np.mean(video_prec))
        all_vids_recall.append(np.mean(video_recall))
        all_vids_f1.append(np.mean(video_f1))
    
        # Accuracy for recognition and predictions (Keep Time Dimension)
        acc_curr_frames         = compute_accuracy(video_frame_rec, video_tgts_rec, return_mean=False)
        acc_future_frames       = compute_accuracy(video_frame_preds, video_tgts_preds, return_mean=False)

        # Compute Root Mean Squared Error
        # Concatenate recognition and predictions
        # tgts = np.concatenate((video_tgts_rec, video_tgts_preds), axis=1)
        # preds = np.concatenate((video_frame_rec, video_frame_preds), axis=1)
        # rmse_future_transitions = compute_rmse_transition_times(tgts, preds, max_duration=18)
        
        # global video mean accuracy
        video_mean_curr_acc         = np.round(np.nanmean(acc_curr_frames), decimals=4).tolist()
        mean_acc_future_frames      = np.round(np.nanmean(acc_future_frames), decimals=4).tolist()
        cum_acc_future_frames       = np.round(np.nancumsum(acc_future_frames) / np.arange(1,len(acc_future_frames)+1), decimals=4).tolist()
        video_mean_cum_acc_future   = np.round(np.nanmean(cum_acc_future_frames), decimals=4).tolist()

        # store the mean accuracy for the video
        all_video_mean_curr_acc[video_id] = video_mean_curr_acc
        all_video_mean_cum_acc_future[video_id] = video_mean_cum_acc_future

        # iter times per index in list of lists
        mean_iter_times = np.round(np.mean(video_mean_cum_iter_time, axis=0), decimals=4).tolist()
        all_videos_mean_cum_iter_time.append(mean_iter_times)

        # Video-level results
        video_results['video_mean_curr_acc']       = video_mean_curr_acc
        video_results['mean_acc_future_frames']     = mean_acc_future_frames
        video_results['video_mean_cum_acc_future'] = video_mean_cum_acc_future

        all_videos_mean_acc_curr.append(video_mean_curr_acc)
        # all_videos_rmse_future.append(rmse_future_transitions)

        # keep temporal dimension
        all_videos_acc_future.append(acc_future_frames)
        all_videos_cum_acc_future.append(cum_acc_future_frames)

        with open(f'./results/per_video_ep{epoch}.json', 'a+') as f:
            video_results = check_numpy_to_list(video_results)
            json.dump([video_results], f)
            f.write(',\n')

        logger.info(f"[TESTING] video: {video_id} | "
                    f"video_mean_curr_acc: {video_mean_curr_acc} | "
                    f"mean_acc_future_frames: {mean_acc_future_frames} | "
                    f"video_mean_cum_acc_future: {video_mean_cum_acc_future}")

    
    # END OF ALL VIDEOS LOOP
    
    # Aggregate metrics across all videos
    agg_metrics, agg_frame_metrics = aggregate_metrics(all_metrics, all_frame_metrics)

    # keep time dimension over all videos
    all_videos_mean_acc_future_t       = np.round(np.nanmean(all_videos_acc_future, axis=0), decimals=4).tolist()
    all_videos_mean_cum_acc_future_t   = np.round(np.nanmean(all_videos_cum_acc_future, axis=0), decimals=4).tolist()

    # concat the current accuracy and future accuracies
    acc_curr                = np.round(np.nanmean(all_videos_mean_acc_curr), decimals=4).tolist()
    acc_future              = np.round(np.nanmean(all_videos_mean_acc_future_t), decimals=4).tolist()
    acc_curr_and_future_t   = np.round(np.nanmean([acc_curr] + all_videos_mean_acc_future_t), decimals=4).tolist()
    
    # Epoch level results
    all_videos_results["epoch"]             = epoch
    all_videos_results["acc_curr"]          = acc_curr
    all_videos_results["acc_future"]        = acc_future
    all_videos_results["acc_curr_future"]   = acc_curr_and_future_t

    # Regression remaining time
    for h in eval_horizons:
        for metric in mae_results.keys():
            # take the mean over all videos and keep the class dimension
            all_videos_results[f"{metric}_{h}"] = np.round(np.nanmean(all_videos_metrics[f'{metric}_{h}'], axis=0), decimals=4).tolist()
            logger.info(f"[TESTING] {metric}_{h}: {all_videos_results[f'{metric}_{h}']}")

            # get standard deviation over all videos
            std_metric = np.round(np.nanstd(all_videos_metrics[f'{metric}_{h}'], axis=0), decimals=4).tolist()
            all_videos_results[f"{metric}_{h}_std"] = std_metric
            logger.info(f"[TESTING] {metric}_{h}_std: {std_metric}")
    
    # all_videos_results["rmse_future"]       = np.round(np.nanmean(all_videos_rmse_future), decimals=4).tolist()

    # Classification (keep temporal dimension)
    all_videos_results["acc_future_t"]      = all_videos_mean_acc_future_t
    
    # Precision, Recall, F1
    all_videos_results.update(agg_metrics)
    all_videos_results.update(agg_frame_metrics) # keep the temporal dimension

    # frame-level metrics
    all_videos_results["accuracy"]              = np.round(np.nanmean(all_vids_acc), decimals=4).tolist()
    all_videos_results["precision_weighted"]    = np.round(np.nanmean(all_vids_prec), decimals=4).tolist()
    all_videos_results["recall_weighted"]       = np.round(np.nanmean(all_vids_recall), decimals=4).tolist()
    all_videos_results["f1_weighted"]           = np.round(np.nanmean(all_vids_f1), decimals=4).tolist()
    
    # set metric to optimize
    h = eval_horizons[-1]
    if main_metric in [f'wMAE_{h}', f'inMAE_{h}', f'outMAE_{h}', f'expMAE_{h}']:
        condition = 'lower'
    elif main_metric in ['acc_curr', 'acc_future', 'acc_curr_future']:
        condition = 'higher'
    else:
        raise ValueError(f"main_metric: {main_metric} not recognized")
    
    main_score_per_horizon = []
    mean_main_metric = "inMAE"
    for h in eval_horizons:
        main_score_per_horizon.append(all_videos_results[f'{mean_main_metric}_{h}'])
    score = np.mean(main_score_per_horizon)
    logger.info(f"[TESTING] Epoch: {epoch} |" 
                f"mean_main_metric: {mean_main_metric} |"
                f"mean score: {score}")

    # check if best epoch
    # score = all_videos_results[main_metric]
    is_best_epoch = is_better(condition, score, best_score)
    logger.info(f"[TESTING] Epoch: {epoch} | is best epoch: {is_best_epoch} with {mean_main_metric}: {score}")

    if epoch % plot_video_freq == 0 or is_best_epoch:
        # PLOT AND SAVE VIDEO DATA if best
        for video_id in video_ids:

            # plot the classification task
            video_frame_preds           = all_video_frame_preds[video_id]
            video_frame_rec             = all_video_frame_rec[video_id]
            video_tgts_preds            = all_video_tgts_preds[video_id]
            video_tgts_rec              = all_video_tgts_rec[video_id]
            video_mean_curr_acc         = all_video_mean_curr_acc[video_id]
            video_mean_cum_acc_future   = all_video_mean_cum_acc_future[video_id]
            
            # plot_video_scatter_3D(video_frame_preds, video_frame_rec, video_tgts_preds, video_tgts_rec, 
            #                     anticip_time, 
            #                     horizon=cfg.eval_horizons[-1],
            #                     video_idx=video_id, 
            #                     dataset=dataset,
            #                     epoch=epoch,
            #                     sampling_rate=60, # current frames axis (seconds to minutes)
            #                     padding_class=-1, # padding class
            #                     eos_class=7,
            #                     num_classes=7,
            #                     video_mean_curr_acc=video_mean_curr_acc,
            #                     video_mean_cum_acc_future=video_mean_cum_acc_future,
            #                     )
            
            
            # merge recognition and anticipation tasks
            gt_classification = np.concatenate((video_tgts_rec, video_tgts_preds), axis=1)
            pred_classification = np.concatenate((video_frame_rec, video_frame_preds), axis=1)
            # print(f"[TESTING] gt_classification: {gt_classification.shape}")
            # print(f"[TESTING] pred_classification: {pred_classification.shape}")

            for h in eval_horizons:

                # Ensure output directory exists
                if not os.path.exists(f"./plots/{dataset}/{h}"):
                    os.makedirs(f"./plots/{dataset}/{h}")
                
                gt_remaining_time = all_video_remaining_time_tgts[f'{h}'][video_id]
                pred_remaining_time = all_video_remaining_time_preds[video_id][h]

                # plot classification task
                plot_classification_video(gt_classification, pred_classification,
                        h=h,
                        num_obs_classes=num_classes, 
                        video_idx=video_id, 
                        epoch=epoch,
                        dataset=dataset, 
                        save_video=False,
                        x_sampling_rate=5, 
                        gif_fps=40,
                        use_scatter=True)
                
                # plot regression remaining time
                plot_remaining_time_video(gt_remaining_time, pred_remaining_time,
                                            h=h,
                                            num_obs_classes=num_classes, 
                                            video_idx=video_id, 
                                            epoch=epoch, 
                                            dataset=dataset,
                                            save_video=False)

                # plot both classification and regression tasks
                plot_video_combined(gt_remaining_time, pred_remaining_time, gt_classification, pred_classification,
                        h=h,
                        num_obs_classes=num_classes,
                        video_idx=video_id, 
                        epoch=epoch,
                        dataset=dataset, 
                        save_video=False,
                        x_sampling_rate=1,
                        y_sampling_rate=1,
                        gif_fps=40,
                        use_scatter=True)
    
    # if wMAE then we should update if lower not higher like for accuracy
    if is_best_epoch:
        best_epoch = epoch
        logger.info(f"[TESTING] Best epoch: {best_epoch} | "
                    f"New Best score: {score}")
        if save_all_metrics:
            np.save(f"all_videos_mean_acc_future_t_ep{epoch}.npy", all_videos_acc_future)
            np.save(f"all_videos_mean_cum_acc_future_t_ep{epoch}.npy", all_videos_cum_acc_future)
        
            if save_best_video_preds:
                np.save(f"video_frame_rec_{video_id}_ep{epoch}.npy", video_frame_rec)
                np.save(f"video_tgts_rec_{video_id}_ep{epoch}.npy", video_tgts_rec)
                np.save(f"video_frame_preds_{video_id}_ep{epoch}.npy", video_frame_preds)
                np.save(f"video_tgts_preds_{video_id}_ep{epoch}.npy", video_tgts_preds)
                logger.info(f"[TESTING] video: {video_id} saved numpy arrays")


    tb_writer.add_scalar(f'test/acc_curr', all_videos_results["acc_curr"], step_now)
    tb_writer.add_scalar(f'test/acc_curr_future', all_videos_results["acc_curr_future"], step_now)

    # compute the mean accuracy through all the videos and keep the time dimension
    all_videos_results["cum_acc_future_t"]      = all_videos_mean_cum_acc_future_t
    all_videos_results["mean_cum_iter_time"]    = np.round(np.mean(all_videos_mean_cum_iter_time, axis=0), decimals=4).tolist()

    with open(f'all_videos_results.json', 'a+') as f:
        all_videos_results = check_numpy_to_list(all_videos_results)
        json.dump([all_videos_results], f)
        f.write(',\n')
                
    # main overall metrics
    logger.info(f"[TESTING] Epoch: {epoch} | "
                f"acc_curr: {all_videos_results['acc_curr']} | "
                f"acc_future: {all_videos_results['acc_future']} | "
                f"acc_curr_future: {all_videos_results['acc_curr_future']}"
                )

    for h in eval_horizons:    
        logger.info(f"[TESTING] "
                    f"wMAE_{h}: {all_videos_results[f'wMAE_{h}']} | "
                    f"inMAE_{h}: {all_videos_results[f'inMAE_{h}']} | "
                    f"outMAE_{h}: {all_videos_results[f'outMAE_{h}']} | "
        )

    return all_videos_results, step_now+1, is_best_epoch


def initial_setup(cfg, logger):
    # torchvision.set_video_backend(cfg.pytorch.video_backend)

    if cfg.data_parallel:
        dist_info = {}
        dist_info['distributed'] = False
        dist_info['world_size'] = torch.cuda.device_count()
        # In DDP we set these params for a single process
        cfg.train.batch_size *= dist_info['world_size']
        cfg.eval.batch_size *= dist_info['world_size']
    else:
        dist_info = utils.init_distributed_mode(logger,
                                                dist_backend=cfg.dist_backend)
    logger.info("Dist info:", dist_info)
    logger.info("torch version: %s", torch.__version__)
    logger.info("torchvision version: %s", torchvision.__version__)
    logger.info("hydra version: %s", hydra.__version__)

    device = torch.device('cuda')

    torch.backends.cudnn.benchmark = True
    tb_writer = setup_tbx('logs/', SummaryWriter)
    return dist_info, device, tb_writer


def init_model(model, ckpt_path, modules_to_keep, logger):
    """Initialize model with weights from ckpt_path.
    Args:
        ckpt_path (str): A string with path to file
        modules_to_keep (str): A comma sep string with the module name prefix
            that should be loaded from the checkpoint
    """
    logger.debug('Initing %s with ckpt path: %s, using modules in it %s',
                 model, ckpt_path, modules_to_keep)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if 'model' in checkpoint.keys():
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint.keys():
        state_dict = checkpoint['state_dict']
    elif 'classy_state_dict' in checkpoint.keys():
        state_dict = checkpoint['classy_state_dict']
        # This is likely coming from a VISSL codebase, so the actual trunk
        # params will be as follows. Ideally support this more generally TODO
        state_dict = state_dict['base_model']['model']['trunk']
    else:
        state_dict = checkpoint
    if modules_to_keep:
        # Keep only the elements of state_dict that match modules to keep.
        # Also, remove that prefix from the names
        filtered_state_dict = {}
        for key, val in state_dict.items():
            for mod_name in modules_to_keep.split(','):
                if key.startswith(mod_name):
                    filtered_state_dict[key[len(mod_name):]] = val
                    continue
        state_dict = filtered_state_dict
    # Ignore any parameters/buffers (bn mean/var) where shape does not match
    for name, param in itertools.chain(model.named_parameters(),
                                       model.named_buffers()):
        if name in state_dict and state_dict[name].shape != param.shape:
            logger.warning('Ckpt shape mismatch for %s (%s vs %s). Ignoring.',
                           name, state_dict[name].shape, param.shape)
            del state_dict[name]
    missing_keys, unexp_keys = model.load_state_dict(state_dict, strict=False)
    logger.warning('Could not init from %s: %s', ckpt_path, missing_keys)
    logger.warning('Unused keys in %s: %s', ckpt_path, unexp_keys)


def collate_fn_remove_audio(batch):
    """Remove audio from the batch.
    Also remove any None(s) -- those were data points I wasn't able to read.
    Not needed, and it doesn't batch properly since it is different length.
    """
    batch = list(filter(lambda x: x is not None, batch))
    if isinstance(batch[0], tuple):
        batch = [(d[0], d[2]) for d in batch]
    return default_collate(batch)


def _get_resize_shape(data_cfg):
    scale_h = data_cfg.scale_h
    scale_w = data_cfg.scale_w
    if isinstance(scale_w, int) and scale_w == -1:
        resize_shape = scale_h
    else:
        assert (not isinstance(scale_h, int) or scale_h != -1), (
            'If using -1, must be used for scale_w. The smaller side will be '
            'scaled by that size.')
        resize_shape = (scale_h, scale_w)
    return resize_shape


def _get_pixel_mean_std(data_cfg):
    return {'mean': tuple(data_cfg.mean), 'std': tuple(data_cfg.std)}


def _set_all_bn_to_not_track_running_mean(model):
    """
    Set all batch norm layers to not use running mean.
    """
    for module in model.modules():
        # This should be able to capture any BatchNorm1d, 2d, 3d etc.
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.track_running_stats = False
    return model


def main(cfg):
    logger = logging.getLogger(__name__)
    dist_info, device, tb_writer = initial_setup(cfg, logger)

    logger.info(f'[MAIN] eval horizons for regression: {cfg.eval_horizons}')

    # Data loading code
    logger.info("Loading data")

    logger.info("Loading datasets")
    st = time.time()

    # separate these into get transforms
    # TODO: This is gotten too complex: clean up, make interface better
    transform_train =[
        T.ToTensorVideo(),
        T.Resize(_get_resize_shape(cfg.data_train)),
        T.RandomHorizontalFlipVideo(cfg.data_train.flip_p),
        T.ColorJitterVideo(brightness=cfg.data_train.color_jitter_brightness,
                           contrast=cfg.data_train.color_jitter_contrast,
                           saturation=cfg.data_train.color_jitter_saturation,
                           hue=cfg.data_train.color_jitter_hue),
        torchvision.transforms.Lambda(
            lambda x: x * cfg.data_train.scale_pix_val),
        torchvision.transforms.Lambda(lambda x: x[[2, 1, 0], ...])
        if cfg.data_train.reverse_channels else torchvision.transforms.Compose(
            []),
        T.NormalizeVideo(**_get_pixel_mean_std(cfg.data_train)),
    ]
    if cfg.data_train.crop_size is not None:
        transform_train.append(
            T.RandomCropVideo(
                (cfg.data_train.crop_size, cfg.data_train.crop_size)), )
    transform_train = torchvision.transforms.Compose(transform_train)
    #transform_eval_pre = []
    transform_eval = [
        T.ToTensorVideo(),
        T.Resize(_get_resize_shape(cfg.data_train)),
        torchvision.transforms.Lambda(
            lambda x: x * cfg.data_train.scale_pix_val),
        torchvision.transforms.Lambda(lambda x: x[[2, 1, 0], ...]) if
        cfg.data_eval.reverse_channels else torchvision.transforms.Compose([]),
        T.NormalizeVideo(**_get_pixel_mean_std(cfg.data_train)),
    ]
    if cfg.data_eval.crop_size is not None:
        transform_eval.append(
            T.CenterCropVideo(
                (cfg.data_train.crop_size, cfg.data_train.crop_size)), )
    transform_eval = torchvision.transforms.Compose(transform_eval)


    # if len(datasets_train) > 1:
    #     dataset_train = torch.utils.data.ConcatDataset(datasets_train)
    
    # ------------- SELECT VIDEOS MANNUALLY -------------
    dataset_name = cfg.dataset_name # 'cholec80'
    # train_videos_ids = np.arange(cfg.train_start, cfg.train_end + 1).tolist() # 1, 40+1).tolist() # 1-40 ( last is not included)
    # test_videos_ids  = np.arange(cfg.test_start, cfg.test_end + 1).tolist() # 40, 80+1).tolist() # 41-80 ( last is not included)
    # --------------------------------------------------

    # Define the splits with 60-20 for train-test
    if dataset_name == 'cholec80':
        splits = [
            ([],[]),
            (np.arange(1, 61).tolist(), np.arange(61, 81).tolist()),
            (np.arange(21, 81).tolist(), np.arange(1, 21).tolist()),
            (np.concatenate((np.arange(1, 21), np.arange(41, 81))).tolist(), np.arange(21, 41).tolist()),
            (np.concatenate((np.arange(1, 41), np.arange(61, 81))).tolist(), np.arange(41, 61).tolist())
        ]
    elif dataset_name == 'autolaparo21':
        # 14-7 split
        splits = [
            (np.arange(1, 15).tolist(), np.arange(15, 22).tolist()),
            (np.arange(1, 15).tolist(), np.arange(15, 22).tolist()),
        ]
    else:
        raise ValueError(f"Dataset: {dataset_name} not recognized")

    # if cfg.split_idx == 0 then use the first split
    if cfg.split_idx == 0:
        train_videos_ids, test_videos_ids = splits[1]

    train_videos_ids, test_videos_ids = splits[cfg.split_idx]
    logger.info(f"Running split {cfg.split_idx}")
    logger.info(f"Train videos: {train_videos_ids}")
    logger.info(f"Test videos: {test_videos_ids}")

    # read dataframe once
    dataset = SelectDataset(dataset_name, logger)
    dataframe = dataset.get_dataframe()
    
    # process dataframe
    dataset_train = Medical_Dataset(cfg,
        dataframe=dataframe,
        train_mode='train',
        dataset_name=dataset_name,
        video_indices=train_videos_ids,
        transform=transform_eval, 
        device=device, 
        logger=logger)

    # list comprehension to load test datasets
    datasets_test = [
        Medical_Dataset(cfg,
            dataframe=dataframe,
            train_mode='test',
            dataset_name=dataset_name,
            video_indices=[video_idx], # needs to be a list
            transform=transform_eval, 
            device=device, 
            logger=logger)
        for video_idx in test_videos_ids
    ]

    logger.info(f"Time to load datasets: {(time.time() - st) / 60:.2f} min")

    logger.info("Creating data loaders")
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=cfg.train.batch_size,
        # sampler=train_sampler,
        num_workers=0,#cfg.data_train.workers,
        pin_memory=False,  # usually hurts..
        shuffle=True,
        collate_fn=collate_fn_remove_audio,
    )

    dataloaders_test = [
        torch.utils.data.DataLoader(
            dataset_test,
            batch_size=cfg.eval.batch_size or cfg.train.batch_size * 4,
            # sampler=test_samplers[key],
            num_workers=0,              #cfg.data_eval.workers,
            pin_memory=False,           # Usually hurts..
            shuffle=False,
            collate_fn=collate_fn_remove_audio)
        for dataset_test in datasets_test
    ]

    num_classes = {'one': 7 } # todo: remove in base model and class_mappings
    logger.info('Creating model with %s classes', num_classes)
    model = base_model.BaseModel(cfg.model,
                                 num_classes=num_classes,
                                 class_mappings=dataset_train.class_mappings)
    logger.debug('Model: %s', model)
    if dist_info['distributed'] and cfg.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)


    model.to(device)
    num_tokens = len(dataset_train)  # Assuming each sample is one token, adjust if necessary
    num_params = sum(p.numel() for p in model.parameters())
    ratio = num_tokens / num_params
    logger.info('[MAIN] Number of tokens: %d', num_tokens)
    logger.info('[MAIN] Number of parameters: %d', num_params)
    logger.info('[MAIN] Token-to-Parameter Ratio is ideally between [1:1 to 10:1]: %f', ratio)

    if ratio < 1 or ratio > 10:
        logger.warning('[MAIN] Data-to-Parameter-Ratio is outside the interval [1:1 to 10:1]')
    
    if cfg.opt.classifier_only:
        assert len(cfg.opt.lr_wd) == 1
        assert cfg.opt.lr_wd[0][0] == 'classifier'
        model = _set_all_bn_to_not_track_running_mean(model)
    params = []
    for this_module_names, this_lr, this_wd in cfg.opt.lr_wd:
        if OmegaConf.get_type(this_module_names) != list:
            this_module_names = [this_module_names]
        this_modules = [
            operator.attrgetter(el)(model) if el != '__all__' else model
            for el in this_module_names
        ]
        this_params_bias_bn = {}
        this_params_rest = {}
        for this_module_name, this_module in zip(this_module_names,
                                                 this_modules):
            for name, param in this_module.named_parameters():
                # ignore the param without grads
                if not param.requires_grad:
                    continue
                # May not always have a ".bias" if it's the last element, and no
                # module name
                if name.endswith('bias') or ('.bn' in name):
                    this_params_bias_bn[this_module_name + '.' + name] = param
                else:
                    this_params_rest[this_module_name + '.' + name] = param
        this_scaled_lr = this_lr * dist_info['world_size']
        if cfg.opt.scale_lr_by_bs:
            this_scaled_lr *= cfg.train.batch_size
        params.append({
            'params': this_params_rest.values(),
            'lr': this_scaled_lr,
            'weight_decay': this_wd,
        })
        logger.info('Using LR %f WD %f for parameters', params[-1]['lr'], params[-1]['weight_decay'])
        params.append({
            'params': this_params_bias_bn.values(),
            'lr': this_scaled_lr,
            'weight_decay': this_wd * cfg.opt.bias_bn_wd_scale,
        })
        logger.info('Using LR %f WD %f for parameters', params[-1]['lr'], params[-1]['weight_decay'])
    # Remove any parameters for which LR is 0; will save GPU usage
    params_final = []
    for param_lr in params:
        #print(param_lr)
        if param_lr['lr'] != 0.0:
            params_final.append(param_lr)
        else:
            for param in param_lr['params']:
                param.requires_grad = False

    optimizer = hydra.utils.instantiate(cfg.opt.optimizer, params_final)

    # convert scheduler to be per iteration,
    # not per epoch, for warmup that lasts
    # between different epochs
    print(dist_info['world_size'])
    print(len(dataloader_train))
    main_scheduler = hydra.utils.instantiate(
        cfg.opt.scheduler,
        optimizer,
        iters_per_epoch=len(dataloader_train),
        world_size=dist_info['world_size'])
    lr_scheduler = hydra.utils.instantiate(cfg.opt.warmup,
                                           optimizer,
                                           main_scheduler,
                                           iters_per_epoch=len(dataloader_train),
                                           world_size=dist_info['world_size'])

    if cfg.finetune_ckpt == "last":
        saved_ckpt = 'checkpoint.pth'
    elif cfg.finetune_ckpt == "best":
        saved_ckpt = 'checkpoint_best.pth'
    else:
        saved_ckpt = f'checkpoint_{cfg.finetune_ckpt}.pth'
    start_epoch = 0
    if os.path.isfile(saved_ckpt):
        checkpoint = torch.load(saved_ckpt, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict = False)
        #optimizer.load_state_dict(checkpoint['optimizer'])
        #lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        #start_epoch = checkpoint['epoch']
        logger.warning('Loaded model from %s (ep %f)', saved_ckpt, start_epoch)
    else:
        logger.warning('No checkpoint found at %s', saved_ckpt)

    if dist_info['distributed'] and not cfg.eval.eval_fn.only_run_featext:
        # If only feat ext, then each gpu is going to test separately anyway,
        # no need for communication between the models
        logger.info('Wrapping model into DDP')
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist_info['gpu']],
            output_device=dist_info['gpu'])

    elif cfg.data_parallel:
        logger.info('Wrapping model into DP')
        device_ids = range(dist_info['world_size'])

    # TODO add an option here to support val mode training
    # Passing in the training dataset_train, since that will be used for computing
    # weights for classes etc.
    train_eval_op = hydra.utils.instantiate(cfg.train_eval_op,
                                            model,
                                            cfg.eval_horizons,
                                            device,
                                            dataset_train,
                                            _recursive_=False)

    # define the score to optimize
    main_metric = cfg.main_metric
    
    if main_metric in ['wMAE', 'inMAE', 'outMAE', 'expMAE']:
        # minimization
        best_score = np.inf
        main_metric = f"{cfg.main_metric}_{cfg.eval_horizons[-1]}"
    elif main_metric in ['acc_curr', 'acc_future', 'acc_curr_future']:
        # maximization
        best_score = 0.0
    else:
        raise ValueError(f"main_metric: {main_metric} not recognized")
    
    # Testing only
    if cfg.test_only:
        logger.info("Starting test_only")
        all_videos_results, step_val_now, is_best_epoch = hydra.utils.call(
            cfg.eval.eval_fn,
            cfg,
            model,
            train_eval_op, 
            device, 
            1,
            dataloaders_test, 
            tb_writer, 
            logger, 
            1,
            best_score=best_score,
            main_metric=main_metric,
            horizon=cfg.eval_horizons[:],
            probs_to_regression_method=cfg.probs_to_regression_method,
            confidence_threshold= 0.5
        )
        print("\n\nTest successful\n\n")
        return

    logger.info("Start training")
    start_time = time.time()

    # Get training metric logger
    stat_loggers = get_default_loggers(tb_writer, start_epoch, logger)
    partial_epoch = start_epoch - int(start_epoch)
    start_epoch = int(start_epoch)
    last_saved_time = datetime.datetime(1, 1, 1, 0, 0)
    epoch = 1  # Since using this var to write the checkpoint output, so init to sth
    step_now = 3000*start_epoch
    step_val_now =0
    acc_vs_params = {}
    for epoch in range(start_epoch, cfg.train.num_epochs):
        if dist_info['distributed'] and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        if epoch>=0:
            # Train the model
            last_saved_time, step_now = hydra.utils.call(
                cfg.train.train_one_epoch_fn,
                model,
                step_now,
                device,
                train_eval_op, 
                optimizer,
                lr_scheduler, 
                dataloader_train, 
                epoch,
                partial_epoch,
                tb_writer, 
                logger,
                last_saved_time,)
            
            partial_epoch = 0  # Reset, for future epochs
            store_checkpoint([CKPT_FNAME], model, optimizer, lr_scheduler,
                            epoch + 1)

        # Evaluate the model     
        all_videos_results, step_val_now, is_best_epoch = hydra.utils.call(
            cfg.eval.eval_fn,
            cfg,
            model,
            train_eval_op, 
            device, 
            step_val_now,
            dataloaders_test, 
            tb_writer, 
            logger,
            epoch + 1,
            best_score=best_score,
            main_metric=main_metric,
            horizon=cfg.eval_horizons[:], # previously [0]
            probs_to_regression_method=cfg.probs_to_regression_method,
            confidence_threshold= 0.5
        )

        # Store the all_videos_results per number of parameters and tokens
        with open('./results/acc_vs_params.json', 'a+') as f:
            acc_vs_params['epoch'] = epoch
            acc_vs_params['num_params'] = num_params
            acc_vs_params['num_tokens'] = num_tokens
            acc_vs_params.update(all_videos_results)          # accuaries is a dict
            json.dump([acc_vs_params], f)
            f.write(',\n')
        
        # Store the best model MINIMIZING the main_metric
        if is_best_epoch:
            store_checkpoint(f'checkpoint_best.pth', model, optimizer, lr_scheduler, epoch + 1)
            best_score = all_videos_results[main_metric]
        if isinstance(lr_scheduler.base_scheduler, scheduler.ReduceLROnPlateau):
            lr_scheduler.step(all_videos_results[main_metric])

        # reset all meters in the metric logger
        for log in stat_loggers:
            stat_loggers[log].reset_meters()
    
    # Store the final model to checkpoint
    store_checkpoint([CKPT_FNAME], model, optimizer, lr_scheduler, epoch + 1)

    total_time = (time.time() - start_time) / 60
    logger.info(f'Total Training & Evaluation time: {total_time:.2f} min')

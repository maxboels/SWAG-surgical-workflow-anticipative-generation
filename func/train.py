# Copyright (c) Facebook, Inc. and its affiliates.

"""Training code."""
import pickle
from PIL import Image
import json
from matplotlib import pyplot as plt
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
from R2A2.eval.plot_video_3D import plot_video_contour_3D


#--maxence boels- debugging cuda error
import os
os.environ['TORCH_USE_CUDA_DSA'] = '1'
# import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
#--maxence boels-

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

        data, _, losses, accuracies = train_eval_op(data, train_mode=True)

        remaining_time = (time.time() - start_time) / (i + 1) * (len(dataloader_train) - i - 1) / 60.0

        if i % 100 == 0:
            logger.info(f"[TRAINING] Step {i}/{len(dataloader_train)} | "
                f"Loss: {losses['total_loss'].item():.2f} | "
                f"Acc (curr_frames): {accuracies['curr_frames_acc']:.2f} | "
                f"Acc (next_frames): {accuracies['next_frames_acc']:.2f} | "
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

import numpy as np
from sklearn.metrics import f1_score

def compute_accuracy(inputs, targets, return_list=True, ignore_index=-1, return_mean=True):
    # Ensure inputs and targets are numpy arrays
    inputs = np.array(inputs)
    targets = np.array(targets)

    # Check if inputs and targets are empty
    if inputs.size == 0 or targets.size == 0:
        raise ValueError("Inputs and targets cannot be empty")

    # Check if inputs and targets have the same length
    assert inputs.shape[0] == targets.shape[0], "Inputs and targets must have the same length"

    # Reshape inputs and targets if necessary
    if len(inputs.shape) == 1:
        inputs = inputs.reshape(-1, 1)
    if len(targets.shape) == 1:
        targets = targets.reshape(-1, 1)

    # Check if reshaped inputs and targets have the same shape
    assert inputs.shape == targets.shape, "Reshaped inputs and targets must have the same shape"

    # Convert inputs and targets to integer type
    inputs = inputs.astype(int)
    targets = targets.astype(int)

    # Create a mask to ignore padded values
    mask = targets != ignore_index

    # Compute the accuracy for each frame, excluding ignored indices
    accuracy = np.zeros_like(targets, dtype=float)
    valid_frames = mask.sum(axis=0)
    accuracy[mask] = (inputs[mask] == targets[mask]).astype(float)

    # Compute the mean accuracy along the last dimension, excluding frames with all padded targets
    accuracy = np.divide(accuracy.sum(axis=0), valid_frames, out=np.zeros_like(valid_frames, dtype=float), where=valid_frames!=0)

    # Set accuracy to NaN for frames with all padded targets
    accuracy[valid_frames == 0] = np.nan

    # Compute mean over frames indices but ignore NaNs
    if return_mean:
        accuracy = np.nanmean(accuracy)

    # Round the accuracy to 4 decimal places
    accuracy = np.round(accuracy, decimals=4)

    if return_list:
        return accuracy.tolist()
    return accuracy

def compute_f1_score(inputs, targets, return_list=True, ignore_index=-1, return_mean=True):
    # Ensure inputs and targets are numpy arrays
    inputs = np.array(inputs)
    targets = np.array(targets)

    # Check if inputs and targets are empty
    if inputs.size == 0 or targets.size == 0:
        raise ValueError("Inputs and targets cannot be empty")

    # Check if inputs and targets have the same length
    assert inputs.shape[0] == targets.shape[0], "Inputs and targets must have the same length"

    # Reshape inputs and targets if necessary
    if len(inputs.shape) == 1:
        inputs = inputs.reshape(-1, 1)
    if len(targets.shape) == 1:
        targets = targets.reshape(-1, 1)

    # Check if reshaped inputs and targets have the same shape
    assert inputs.shape == targets.shape, "Reshaped inputs and targets must have the same shape"

    # Convert inputs and targets to integer type
    inputs = inputs.astype(int)
    targets = targets.astype(int)

    # Create a mask to ignore padded values
    mask = targets != ignore_index

    # Compute true positives, false positives, and false negatives for each frame, excluding ignored indices
    tp = ((inputs == targets) & mask).astype(float)
    fp = ((inputs != targets) & mask).astype(float)
    fn = ((inputs != targets) & mask).astype(float)

    # Compute precision, recall, and F1 score for each frame
    precision = np.zeros(targets.shape[1], dtype=float)
    recall = np.zeros(targets.shape[1], dtype=float)
    f1_score = np.zeros(targets.shape[1], dtype=float)

    for i in range(targets.shape[1]):
        frame_mask = mask[:, i]
        frame_inputs = inputs[:, i]
        frame_targets = targets[:, i]

        tp = ((frame_inputs == frame_targets) & frame_mask).astype(float)
        fp = ((frame_inputs != frame_targets) & frame_mask).astype(float)
        fn = ((frame_inputs != frame_targets) & frame_mask).astype(float)

        valid_frames = frame_mask.sum()

        precision[i] = np.divide(tp.sum(), tp.sum() + fp.sum(), out=np.zeros_like(tp.sum()), where=(tp.sum() + fp.sum()) != 0)
        recall[i] = np.divide(tp.sum(), tp.sum() + fn.sum(), out=np.zeros_like(tp.sum()), where=(tp.sum() + fn.sum()) != 0)
        f1_score[i] = np.divide(2 * precision[i] * recall[i], precision[i] + recall[i], out=np.zeros_like(precision[i]), where=(precision[i] + recall[i]) != 0)

        if valid_frames == 0:
            f1_score[i] = np.nan
    
    # Compute mean over frames indices but ignore NaNs
    if return_mean:
        f1_score = np.nanmean(f1_score)

    # Round the F1 score to 4 decimal places
    f1_score = np.round(f1_score, decimals=4)

    if return_list:
        return f1_score.tolist()
    return f1_score

def check_numpy_to_list(dictionay):
    for key in dictionay.keys():
        if isinstance(dictionay[key], np.ndarray):
            dictionay[key] = dictionay[key].tolist()
            print(f"converted {key} np.ndarray to list")
    return dictionay

def evaluate(model, train_eval_op, device, step_now, dataloaders: list, tb_writer, logger, epoch: float,
            anticip_time: int,
            max_anticip_time: int, 
             store=False, store_endpoint='logits', only_run_featext=False,
             best_acc1=0.0,):
    
    step_size = max_anticip_time / anticip_time
    max_num_preds = int(max_anticip_time / step_size)
    x_values = np.arange(1, max_anticip_time+1, step_size).tolist()

    model.eval()
    # -----------------select params----------------- #
    num_classes = 7
    # -----------------select params----------------- #

    all_videos_results = OrderedDict()
    all_videos_mean_acc_curr    = []
    all_videos_mean_acc_future  = []
    all_videos_acc_future       = []
    all_videos_cum_acc_future   = []
    all_videos_mean_f1_curr     = []
    all_videos_mean_f1_future   = []

    all_videos_mean_cum_iter_time = []

    eval_start_time = time.time()

    best_video_idx = 60                     # NOTE: cherry-picked video index 

    # FOR EACH VIDEO LOADER
    for data_loader in dataloaders:
        vid_start_time = time.time()
        video_results = OrderedDict()
        video_length = len(data_loader.dataset)
        video_id = data_loader.dataset.video_indices[0] # int type

        iters_times = []

        # init new video buffers
        video_frame_rec     = np.full((video_length, 1), -1)
        video_tgts_rec      = np.full((video_length, 1), -1)
        video_frame_preds   = np.full((video_length, max_num_preds), -1)
        video_tgts_preds    = np.full((video_length, max_num_preds), -1)
        video_seg_preds     = np.full((video_length, 1), -1)
        video_tgts_preds_seg = np.full((video_length, 1), -1)

        start_idx = 0
        end_idx = 0
        
        # eval loop
        for b, data in enumerate(data_loader):

            batch_size = data['video'].shape[0]
            first_frame_idx = data['frame_idx'][0].detach().cpu().numpy()
            curr_frame = data['frame_idx'][-1].detach().cpu().numpy()
            end_idx = start_idx + batch_size

            with torch.no_grad():
                        
                outputs = model(data['video'], train_mode=False)

                # FRAME LEVEL STATE RECOGNITION
                # (there is no autoregressive prediction here)
                preds = np.argmax(outputs['curr_frames'].detach().cpu().numpy()[:,-1:,:], axis=-1)
                targets = data['curr_frames_tgt'].detach().cpu().numpy()[:,-1:]
                video_frame_rec[start_idx:end_idx] = preds
                video_tgts_rec[start_idx:end_idx] = targets
                
                # AR FRAME LEVEL ACTION PREDICTION
                # (possible auto-regressive predictions)
                if "future_frames" in outputs.keys():
                    preds = np.argmax(outputs['future_frames'].detach().cpu().numpy(), axis=2)
                    targets = data["future_frames_tgt"].detach().cpu().numpy()
                    video_frame_preds[start_idx:end_idx] = preds
                    video_tgts_preds[start_idx:end_idx] = targets # NOTE: those targets are for inference only
                    
                # SEGMENT LEVEL PREDICTION
                # (possible auto-regressive predictions)
                if "future_segmts_cls" in outputs.keys():
                    preds = np.argmax(outputs['future_segmts_cls'].detach().cpu().numpy(), axis=2)
                    targets = data["future_segmts_tgt"].detach().cpu().numpy()
                    video_seg_preds[start_idx:end_idx] = preds
                    video_tgts_preds_seg[start_idx:end_idx] = targets
                
                if "iters_time" in outputs.keys():
                    iters_time = outputs["iters_time"]  # list with n AR steps
                    iters_times.append(iters_time)      # list of lists
                
                logger.info(f"[TESTING] video: {video_id} | "
                            f"frame: {curr_frame} / {video_length}")

            # update start index
            start_idx += batch_size
        
        # Video-level results

        # If performance increases with none fixed context length, then tracking the time per iteration makes sense
        iters_times = np.mean(iters_times, axis=0)
        cum_iters_times = np.round(np.cumsum(iters_times, axis=0), decimals=4).tolist()
        all_videos_mean_cum_iter_time.append(cum_iters_times)

        vid_test_time = time.time() - vid_start_time
        test_time = time.time() - eval_start_time
        logger.info(f"[TESTING] video: {video_id} | "
                    f"video test time: {vid_test_time/60:.2f} min | "
                    f"total test time: {test_time/60:.2f} min")

        # Store the results
        video_results["video_id"] = video_id
            
        # compute eval metrics for each video
        # NOTE: make sure to ignore the -1 class for both padding the last class and video length

        # Keep Time Dimension
        acc_curr_frames         = compute_accuracy(video_frame_rec, video_tgts_rec, return_mean=False)      # potential nans
        acc_future_frames       = compute_accuracy(video_frame_preds, video_tgts_preds, return_mean=False)  # potential nans

        # global mean accuracy
        mean_acc_curr_frames        = np.round(np.nanmean(acc_curr_frames), decimals=4).tolist()
        mean_acc_future_frames      = np.round(np.nanmean(acc_future_frames), decimals=4).tolist()
        cum_acc_future_frames       = np.round(np.nancumsum(acc_future_frames) / np.arange(1,len(acc_future_frames)+1), decimals=4).tolist()
        mean_cum_acc_future_frames  = np.round(np.nanmean(cum_acc_future_frames), decimals=4).tolist()

        # Video-level results
        video_results['mean_acc_curr_frames']       = mean_acc_curr_frames
        video_results['mean_acc_future_frames']     = mean_acc_future_frames
        video_results['mean_cum_acc_future_frames'] = mean_cum_acc_future_frames

        for key in video_results.keys():
            print(f"{key} is data type: {type(video_results[key])}")
            print(f"{key}: {video_results[key]}")
        
        # video-level metrics
        all_videos_mean_acc_curr.append(mean_acc_curr_frames)
        # all_videos_mean_acc_future.append(mean_acc_future_frames)
        # all_videos_mean_cum_acc_future.append(mean_cum_acc_future_frames)

        # keep temporal dimension
        all_videos_acc_future.append(acc_future_frames)
        all_videos_cum_acc_future.append(cum_acc_future_frames)

        with open(f'per_video_ep{epoch}.json', 'a+') as f:
            video_results = check_numpy_to_list(video_results)
            json.dump([video_results], f)
            f.write(',\n')

        logger.info(f"[TESTING] video: {video_id} | "
                    f"mean_acc_curr_frames: {mean_acc_curr_frames}, "
                    f"mean_acc_future_frames: {mean_acc_future_frames}")
        print(f"mean_acc_curr_frames: {mean_acc_curr_frames}, mean_acc_future_frames: {mean_acc_future_frames}")

        
        # PLOTTING VIDEO RESULTS (3D (static or animated)
        plot_vid_ids = [41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 60, 70, 80]
        if video_id in plot_vid_ids and epoch == 1: # starts at 0
            # 2. Qualitative: Visualize targets and predictions for each video
            plot_video_contour_3D(video_frame_preds, video_frame_rec, video_tgts_preds, video_tgts_rec, video_id)

            save_numpy_arrays = False
            if save_numpy_arrays:
                np.save(f"video_frame_rec_{video_id}_ep{epoch}.npy", video_frame_rec)
                np.save(f"video_tgts_rec_{video_id}_ep{epoch}.npy", video_tgts_rec)
                np.save(f"video_frame_preds_{video_id}_ep{epoch}.npy", video_frame_preds)
                np.save(f"video_tgts_preds_{video_id}_ep{epoch}.npy", video_tgts_preds)
                logger.info(f"[TESTING] video: {video_id} saved numpy arrays")
            
    # keep time dimension over all videos
    all_videos_mean_acc_future_t       = np.round(np.nanmean(all_videos_acc_future, axis=0), decimals=4).tolist()
    all_videos_mean_cum_acc_future_t   = np.round(np.nanmean(all_videos_cum_acc_future, axis=0), decimals=4).tolist()
    
    # Epoch level results
    all_videos_results["epoch"]                         = epoch
    all_videos_results["all_videos_mean_acc_curr"]      = np.round(np.nanmean(all_videos_mean_acc_curr), decimals=4).tolist()
    all_videos_results["all_videos_mean_acc_future"]    = np.round(np.nanmean(all_videos_mean_acc_future_t), decimals=4).tolist()
    all_videos_results["all_videos_mean_cum_acc_future"]= np.round(np.nanmean(all_videos_mean_cum_acc_future_t), decimals=4).tolist()

    tb_writer.add_scalar(f'test/all_videos_mean_acc_curr', all_videos_results["all_videos_mean_acc_curr"], step_now)
    tb_writer.add_scalar(f'test/all_videos_mean_acc_future', all_videos_results["all_videos_mean_acc_future"], step_now)

    # compute the mean accuracy through all the videos and keep the time dimension
    all_videos_results["acc_future_t"]          = all_videos_mean_acc_future_t
    all_videos_results["cum_acc_future_t"]      = all_videos_mean_cum_acc_future_t
    all_videos_results["mean_cum_iter_time"]    = np.round(np.mean(all_videos_mean_cum_iter_time, axis=0), decimals=2).tolist()

    with open(f'all_videos_results.json', 'a+') as f:
        all_videos_results = check_numpy_to_list(all_videos_results)
        json.dump([all_videos_results], f)
        f.write(',\n')
    

    # PLOTTING
    if all_videos_results["all_videos_mean_acc_future"] > best_acc1:
        # plot the mean accuracy over the videos
        y_values = {"Cholec80": all_videos_mean_acc_future_t}
        plot_figure(x_values, y_values,
                    title=f'Planning Evaluation (mean. acc. {np.nanmean(all_videos_mean_acc_future_t):.4f})',
                    x_axis_title='Predicted Sequence Length (in minutes)',
                    y_axis_title='Mean Accuracy', file_name='planning_evaluation_mean_acc.png')
        
        # plot the mean accuracy over the videos
        y_values = {"Cholec80": all_videos_mean_cum_acc_future_t}
        plot_figure(x_values, y_values,
                    title=f'Planning Evaluation (cumm. acc. {np.nanmean(all_videos_mean_cum_acc_future_t):.4f})',
                    x_axis_title='Predicted Sequence Length (in minutes)',
                    y_axis_title='Mean Cummulative Accuracy', file_name='planning_evaluation_mean_cumm_acc.png')
        
        # plot box plots (keeping the video-level information)
        y_values = {"Cholec80": all_videos_acc_future}
        plot_box_plot_figure(x_values, y_values,
                            file_name='plot_acc_pred_box.png', 
                            title='Furure Phase Planning Accuracies for all videos (N=50)', 
                            x_axis_title='Future Predictions (in minutes)', 
                            y_axis_title='Mean Accuracy')
        
        # Inference time for deployment at 1fps
        plot_cumulative_time(all_videos_results["mean_cum_iter_time"])

    accuracies = {
        "acc_cur": all_videos_results["all_videos_mean_acc_curr"],
        "acc_fut": all_videos_results["all_videos_mean_acc_future"]
    }

    return accuracies, step_now+1


def evaluate_and_save_metrics(
        eval_type,
        label_phase_vid,
        pred_phase_vid,
        accuracy_vid,
        edit_score_vid,
        f_score_vid,
        count,
        label_phase,
        pred_phase,
        acc_vid,
        epoch,
    ):

    # TODO: check if same as above
    # flatten the sequences into list of lists
    if isinstance(pred_phase_vid, dict):
        pred_phase_vid = [pred_phase_vid[k] for k in pred_phase_vid]
        ground_truth = [label_phase_vid[k] for k in label_phase_vid]

    # video level: MetricsSegments (MB)
    metrics_seg = MetricsSegments()
    acc, edit, f1s = metrics_seg.get_metrics(pred_phase_vid, ground_truth)
    print(f"Video-level Acc: {acc}")
    print(f"Video-level Edit: {edit}")
    print(f"Video-level F1s: {f1s}")
    for overlap, f1 in zip([0.1, 0.25, 0.5], f1s):
        print(f"F1 @ {overlap}: {f1}")

    # video level: MetricsSegments (MB)
    test_video_results = {}
    print(f"Number of test videos: {count}")
    test_video_results['epoch'] = int(epoch)
    # test_video_results['video_acc'] = float("{:.2f}".format(acc*100))
    test_video_results['accuracy'] = float("{:.2f}".format(accuracy_vid/count*100))
    test_video_results['edit_score'] = float("{:.2f}".format(edit_score_vid/count)) # already in percentage
    test_video_results['f_score'] = [float("{:.2f}".format(f/count*100)) for f in f_score_vid]
    # test_video_results['recall_phase'] = float("{:.2f}".format(metrics.recall_score(label_phase, pred_phase, average='macro')*100))
    # test_video_results['precision_phase'] = float("{:.2f}".format(metrics.precision_score(label_phase, pred_phase, average='macro')*100))
    # test_video_results['jaccard_phase'] = float("{:.2f}".format(metrics.jaccard_score(label_phase, pred_phase, average='macro')*100))

    # log video level accuracy
    cum_acc = 0.0
    count = 0
    for key, value in acc_vid.items():
        # MB: skip the training videos
        if len(value)<1:
            continue
        # log the accuracy for each video
        test_video_results[key] = float("{:.2f}".format(np.mean(value)*100))
        count += 1
        cum_acc += test_video_results[key]
    test_video_results['accuracy_mean'] = float("{:.2f}".format(cum_acc/count))
    print(f"Number of test videos: {count}")
    print(f"test_video_results: {test_video_results}")

    # create json file if not exist and save dict to json and start a new line / row and comma
    with open(f'r2a2_eval_metrics_{eval_type}.json', 'a+') as f:
        json.dump([test_video_results], f)
        f.write(',\n')
    
    return test_video_results

def plot_and_evaluate_phase(eval_type, label_phase_vid, pred_phase_vid, num_classes=7):
    """
    Args:
        eval_type (str): 'curr_state_rec' or 'prediction'
        label_phase_vid (dict): {vid_id: [phase1, phase2, ...]}
        pred_phase_vid (dict): {vid_id: [phase1, phase2, ...]}
    """
    edit_score_vid = 0.0
    precision_vid = 0.0
    recall_vid = 0.0
    f_score_vid = [0.0, 0.0, 0.0]
    accuracy_vid = 0.0
    count = 0

    for vid_id, value in label_phase_vid.items():
        print(f"[Video {vid_id}]: ")
        if len(value)<1:
            continue

        pred = np.array(pred_phase_vid[vid_id]).tolist()
        gt = np.array(label_phase_vid[vid_id]).tolist()
        save_path = os.getcwd()+'/pred/'+eval_type+'/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if not os.path.exists(save_path+f"{eval_type}_{vid_id}.json"):
            e = 0
            video_preds = {str(e): pred}
            json.dump(video_preds, open(save_path+f"{eval_type}_{vid_id}.json", 'w')) 
        else:
            video_preds = json.load(open(save_path+f"{eval_type}_{vid_id}.json", 'r'))
            # jump gt key and take the last epoch and add 1
            e = int(list(video_preds.keys())[-2]) + 1
            video_preds[str(e)] = pred

        # update gt key
        if 'gt' in video_preds:
            del video_preds['gt']
        video_preds['gt'] = gt

        # write new json file with appended predictions
        json.dump(video_preds, open(save_path+f"{eval_type}_{vid_id}.json", 'w'))
        # read new json file
        input_dict = json.load(open(save_path+f"{eval_type}_{vid_id}.json", 'r'))
        colors = plt.get_cmap('tab20')(list(range(num_classes))) # also: 'tab20', 'tab20b', 'tab20c'

        plot_video_segments(input_dict, colors, vid_id, save_path, eval_type)

        # MB: metrics
        metrics_seg = MetricsSegments()

        # f1
        for i, overlap in enumerate([0.1, 0.25, 0.5]):
            tp, fp, fn = metrics_seg.f_score(pred_phase_vid[vid_id], label_phase_vid[vid_id], overlap=overlap)
            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            f_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
            print(f"F1 @ {overlap}: {f_score:.3f}")
            f_score_vid[i] += f_score
        # edit
        edit_score = metrics_seg.edit_score(pred_phase_vid[vid_id], label_phase_vid[vid_id])
        print(f"Edit score: {edit_score:.3f}")
        edit_score_vid += edit_score
        # accuracy
        accuracy = metrics_seg.accuracy(pred_phase_vid[vid_id], label_phase_vid[vid_id])
        print(f"Accuracy: {accuracy:.3f}")
        accuracy_vid += accuracy
        count += 1

    return accuracy_vid, edit_score_vid, f_score_vid, count

def plot_recognition_prediction(rec_out, rec_target, pred_out, pred_target, vid, save_path):
    """
    Args:
        rec_out (list)
    """
    rec_out = np.array(rec_out)
    rec_target = np.array(rec_target)
    pred_out = np.array(pred_out)
    pred_target = np.array(pred_target)
    print(f"rec_out: {rec_out.shape}, rec_target: {rec_target.shape}, pred_out: {pred_out.shape}, pred_target: {pred_target.shape}")
    video_length = len(rec_out)
    num_pred = pred_out.shape[0] # fixed error
    height = 200
    # add a white border between segments (rows)
    white_border = - np.ones((video_length, 1))
    dilation_factor = int(height / (2 + num_pred))
    print(f"Video: {vid}, video_length: {video_length}, num_pred: {num_pred}")
    # add the curr_state_rec and prediction to a numpy array
    video_classes = - np.ones((video_length, 4 + 2 * num_pred))
    video_classes[:, 0] = rec_out
    video_classes[:, 1] = white_border[:, 0]
    video_classes[:, 2] = rec_target
    video_classes[:, 3] = white_border[:, 0]
    for i in range(num_pred):
        video_classes[:, 4+2*i] = pred_target[:, i]
        video_classes[:, 5+2*i] = white_border[:, 0]
    # dilate the video_classes array to make it visually appealing
    video_classes = np.repeat(video_classes, dilation_factor, axis=1)
    # plot the curr_state_rec and prediction
    plt.figure(figsize=(20, 10), dpi=300)
    plt.imshow(video_classes.T, cmap="tab20")
    plt.yticks([])
    plt.savefig(save_path+str(vid)+'_rec_pred.jpg', bbox_inches='tight')
    plt.close()


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
    train_videos_ids = np.arange(cfg.train_start, cfg.train_end).tolist() # 1, 41).tolist() # 1-40 ( last is not included)
    test_videos_ids  = np.arange(cfg.test_start, cfg.test_end).tolist() # 41, 81).tolist() # 41-80 ( last is not included)
    # --------------------------------------------------

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

    last_saved_ckpt = CKPT_FNAME
    start_epoch = 0
    if os.path.isfile(last_saved_ckpt):
        checkpoint = torch.load(last_saved_ckpt, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict = False)
        #optimizer.load_state_dict(checkpoint['optimizer'])
        #lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        #start_epoch = checkpoint['epoch']
        logger.warning('Loaded model from %s (ep %f)', last_saved_ckpt,
                       start_epoch)

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
                                            device,
                                            dataset_train,
                                            _recursive_=False)

    if cfg.test_only:
        logger.info("Starting test_only")
        hydra.utils.call(
            cfg.eval.eval_fn, 
            model,
            train_eval_op, device, 1,
            dataloaders_test, 
            tb_writer, 
            logger, 
            1)
        return

    logger.info("Start training")
    start_time = time.time()

    # Get training metric logger
    stat_loggers = get_default_loggers(tb_writer, start_epoch, logger)
    best_acc1 = 0.2
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
                      
        if cfg.train.eval_freq and epoch % cfg.train.eval_freq == 0:
            accuracies, step_val_now = hydra.utils.call(
                cfg.eval.eval_fn, 
                model,
                train_eval_op, 
                device, 
                step_val_now,
                dataloaders_test, 
                tb_writer, 
                logger,
                epoch + 1,
                best_acc1=best_acc1)

            # Store the accuracies per number of parameters and tokens
            with open('acc_vs_params.json', 'a+') as f:
                acc_vs_params['epoch'] = epoch
                acc_vs_params['num_params'] = num_params
                acc_vs_params['num_tokens'] = num_tokens
                acc_vs_params.update(accuracies)          # accuaries is a dict
                json.dump([acc_vs_params], f)
                f.write(',\n')

        else:
            accuracies["acc_fut"] = 0
        
        # Store the best model
        if accuracies["acc_fut"] >= best_acc1:
            store_checkpoint('checkpoint_best.pth', model, optimizer,
                             lr_scheduler, epoch + 1)
            best_acc1 = accuracies["acc_fut"]
        if isinstance(lr_scheduler.base_scheduler,
                      scheduler.ReduceLROnPlateau):
            lr_scheduler.step(accuracies["acc_fut"])

        # reset all meters in the metric logger
        for log in stat_loggers:
            stat_loggers[log].reset_meters()
    # Store the final model to checkpoint
    store_checkpoint([CKPT_FNAME], model, optimizer, lr_scheduler, epoch + 1)

    total_time = (time.time() - start_time) / 60
    logger.info(f'Total Training & Evaluation time: {total_time:.2f} min')

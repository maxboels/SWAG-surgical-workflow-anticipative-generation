# Copyright (c) Facebook, Inc. and its affiliates.

"""Training code."""
import pickle
from PIL import Image
import json
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

# added or modified by max ################################## below this line
from models import base_model

################################################# above this line


from common import scheduler, utils, transforms as T
from common.log import MetricLogger, setup_tbx, get_default_loggers
from datasets.data import get_dataset
from notebooks import utils as nb_utils
from datasets.base_video_dataset import Medical_Dataset
from models.ResNet import resnet_lstm
from sklearn import metrics

from eval.seg_eval import MetricsSegments
# maxboels imports
from R2A2.eval.plot_segments.plot_video import plot_video_segments
import matplotlib.pyplot as plt
from R2A2.eval.anticipate_store_retrieve import AnticipateStoreRetrieve

__all__ = ['main', 'evaluate', 'train_one_epoch', 'initial_setup']
RESULTS_SAVE_DIR = 'results'  # Don't put a "/" at the end, will add later
CKPT_FNAME = 'checkpoint.pth'
DATASET_TRAIN_CFG_KEY = 'dataset_train'
DATASET_EVAL_CFG_KEY = 'dataset_eval'
STR_UID_MAXLEN = 64  # Max length of the string UID stored in H5PY


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
        logging.info('Storing ckpt at epoch %s to %s', epoch, fpath)
        utils.save_on_master(checkpoint, fpath)


def _store_video_logs(data, key, step_id, print_large_freq, metric_logger):
    """
    Args:
        data[key] -> video (B, #clips, 3, T, H, W)
    """
    if metric_logger.writer is None:
        return
    if step_id % print_large_freq != 0:
        return
    if key not in data:
        return
    video = data[key]
    if video.ndim != 6:
        return
    ## Store the videos
    # Swap dims to get N*#clips,T,C,H,W format used by tensorboard
    video = torch.flatten(video, 0, 1)
    vid_log = torch.transpose(video, 1, 2)
    vid_log = vid_log - vid_log.min()
    vid_log = vid_log / vid_log.max()
    kwargs = {}
    if 'video_info' in data:
        # Can't specify different frame rate for videos, so use the min
        kwargs['fps'] = max(
            data['video_info']['video_fps'].min().cpu().numpy().tolist(), 4)
    metric_logger.writer.add_video(key, vid_log, step_id, **kwargs)


def _store_scalar_logs(name, val, step_id, print_freq, metric_logger):
    if metric_logger.writer is None:
        return
    if step_id % print_freq != 0:
        return
    metric_logger.writer.add_scalar(name, val, step_id)



def train_one_epoch(
        model,
        step_now,
        device,
        train_eval_op, # see train_eval_op.py and its classes
        optimizer,
        lr_scheduler,
        data_loader,
        epoch: int,
        partial_epoch: float,
        metric_logger,
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
    Training loop for one epoch


    Args:
        epoch (int) defines how many full epochs have finished
        partial_epoch (float): Defines the ratio of the last epoch that was
            finished before the current model was written out
    """
    header = 'Epoch: [{}]'.format(epoch)
    batches_per_epoch = len(data_loader)
    # Run the data loader for the partial epochs
    partial_iters = int(batches_per_epoch * partial_epoch)
    if partial_iters > 0:
        # TODO: Figure a better way to do this ... too slow
        for i, _ in tqdm(enumerate(data_loader),
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
    
    epoch_start_time = time.time()
    last_data_loader_time = epoch_start_time
    
    # DataLoader
    for i, data in enumerate(data_loader):
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
        
        iter_start = time.time()
        dataloader_time = iter_start - last_data_loader_time

        data, _, losses, accuracies = train_eval_op(data, train_mode=True)

        # time
        forward_time = time.time() - iter_start

        loss = losses['total_loss']

        b = 0.0001
        loss = (torch.sum(loss)-b).abs() + b
        print(f"loss: {loss}")

        if "recognition" in accuracies:
            print(f"accuracies (recognition): {accuracies['recognition']}")
        if "anticipation" in accuracies:
            print(f"accuracies (anticipation): {accuracies['anticipation']}")
        
        if torch.isnan(loss):
            raise ValueError('The loss is NaN!')

        # ZERO GRADIENTS
        optimizer.zero_grad()
        # BACKWARD PASS
        loss.backward()

        if i % print_freq == 0:
            logger.info(f'Time for forward and loss: {forward_time:.2f}s, data loader: {dataloader_time:.2f}s, at iter {i}/{batches_per_epoch}')


        # Clip the gradients if asked for
        if grad_clip_params['max_norm'] is not None:
            params_being_optimized = []
            for param_group in optimizer.param_groups:
                params_being_optimized += param_group['params']
            assert len(params_being_optimized) > 0, 'Shouldnt be training else'
            torch.nn.utils.clip_grad_norm_(params_being_optimized, **grad_clip_params)
            print(f"CLIIPED GRADIENTS: default is false")

        optimizer.step()

        batch_size = data_loader.batch_size
        
        # Store logs
        for loss_key, loss_val in losses.items():
            _store_scalar_logs(f'train_per_iter/loss/{loss_key}', loss_val, step_id, print_freq, metric_logger)
        for acc_key, acc_val in accuracies.items():
            _store_scalar_logs(f'train_per_iter/acc/{acc_key}', acc_val, step_id, print_freq, metric_logger)
        _store_scalar_logs('train_per_iter/lr', optimizer.param_groups[0]['lr'], step_id, print_freq, metric_logger)
        # Store the videos
        [_store_video_logs(data, key, step_id, print_large_freq, metric_logger) for key in data if key.endswith('video')]
        if not isinstance(lr_scheduler.base_scheduler,
                        scheduler.ReduceLROnPlateau):
            lr_scheduler.step()
        
        last_data_loader_time = time.time()


    
    # Log the time taken for the epoch in minutes
    time_taken = f"{(time.time() - epoch_start_time) / 60.0:.1f}"
    logger.info(f'Training Time (epoch {epoch}): {time_taken} mins')
    
    return last_saved_time, step_now

def evaluate(
        eval_types,
        batch_size,
        device,
        step_now,
        data_loader,
        tb_writer,
        logger,
        epoch: float,  # Can be a partial epoch
        train_eval_op: object,
        ):
    
    """
    Run evaluation loop and then return the main metric for the scheduler

    Args:
        data_loader: A dict from key (name) to a data loader. Allows to
            multiple dataloaders for testing on.
        only_run_featext (bool): Set this to true and it will return after the
            features are extracted and won't compute final numbers etc. So
            it will never try to sync processes etc, which leads to crashes.
    """
    eval_metrics = dict()
    for eval_type in eval_types:
        eval_metrics[eval_type] = 0.0

    test_video_results = OrderedDict()
    all_metric_loggers = {}
    final_accuracies = {}
    log_print_freq = 20
    step_key = 0
    mark_idx = 0
    prev_vid = "0"
    data_key = '41_80'

    logger.info('Running evaluation for {0}{1}'.format(
        DATASET_EVAL_CFG_KEY, data_key))
    header = f'[{data_key}] Test:'
    metric_logger = MetricLogger(delimiter='  ',
                                    writer=tb_writer,
                                    stat_set='val' + data_key,
                                    logger=logger)
    all_metric_loggers[data_key] = metric_logger
    this_save_dir = RESULTS_SAVE_DIR + data_key + '/'
    
    # reset the eval metrics for both recognition and anticipation at the beginning of each epoch
    train_eval_op.reset_eval_metrics()

    # eval loop
    for data in metric_logger.log_every(data_loader, log_print_freq, header):
        with torch.no_grad():
            # logging in object instance
            train_eval_op(data, train_mode=False)
    
    # get the eval metrics
    for eval_type in eval_metrics.keys():
        # get the eval metrics
        acc_vid, label_phase_vid, pred_phase_vid, label_phase, pred_phase = train_eval_op.get_eval_metrics(
            eval_type=eval_type)
        # print the eval metrics in the log
        accuracy_vid, edit_score_vid, f_score_vid, count = plot_and_evaluate_phase(
            eval_type, label_phase_vid, pred_phase_vid)
        # save metrics
        test_video_results = evaluate_and_save_metrics(
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
        )
        
        eval_metrics[eval_type] = test_video_results['accuracy'] # float

    # scheduler accuracy metric
    main_metric = eval_metrics["recognition"]
    print(f"main_metric: {main_metric} for scheduler")
    
    return main_metric, step_now+1

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
    test_video_results['video_acc'] = float("{:.2f}".format(acc*100))
    test_video_results['accuracy'] = float("{:.2f}".format(accuracy_vid/count*100))
    test_video_results['edit_score'] = float("{:.2f}".format(edit_score_vid/count)) # already in percentage
    test_video_results['f_score'] = [float("{:.2f}".format(f/count*100)) for f in f_score_vid]
    test_video_results['recall_phase'] = float("{:.2f}".format(metrics.recall_score(label_phase, pred_phase, average='macro')*100))
    test_video_results['precision_phase'] = float("{:.2f}".format(metrics.precision_score(label_phase, pred_phase, average='macro')*100))
    test_video_results['jaccard_phase'] = float("{:.2f}".format(metrics.jaccard_score(label_phase, pred_phase, average='macro')*100))

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
    with open(f'eval_metrics_{eval_type}.json', 'a+') as f:
        json.dump([test_video_results], f)
        f.write(',\n')
    
    return test_video_results

def plot_and_evaluate_phase(eval_type, label_phase_vid, pred_phase_vid):
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
        colors = plt.get_cmap('tab20')(list(range(7))) # also: 'tab20', 'tab20b', 'tab20c'

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

def initial_setup(cfg, logger):
    # torchvision.set_video_backend(cfg.pytorch.video_backend)

    if cfg.data_parallel:
        dist_info = {}
        dist_info['distributed'] = False
        dist_info['world_size'] = torch.cuda.device_count()
        # In DDP we set these params for a single process
        cfg.train.batch_size *= dist_info['world_size']
        cfg.evaluate.batch_size *= dist_info['world_size']
    else:
        dist_info = utils.init_distributed_mode(logger,
                                                dist_backend=cfg.dist_backend)
    logger.info("Dist info:", dist_info)
    logger.info("torch version: %s", torch.__version__)
    logger.info("torchvision version: %s", torchvision.__version__)
    logger.info("hydra version: %s", hydra.__version__)

    device = torch.device('cuda')

    torch.backends.cudnn.benchmark = True
    writer = setup_tbx('logs/', SummaryWriter)
    return dist_info, device, writer


def main(cfg):

    print("Python Version:", sys.version)
    print("PyTorch Version:", torch.__version__)
    logger = logging.getLogger(__name__)
    dist_info, device, writer = initial_setup(cfg, logger)
    logger.info("Loading datframe")
    start = time.time()
    datasets_train = Medical_Dataset(cfg, 'train', 'DATA/cholec80/', transform = None, device = device)
    logger.info("Init Training data: %.2f (s)", time.time() - start)
    start = time.time()
    datasets_test = Medical_Dataset(cfg, 'test', 'DATA/cholec80/', transform = None, device = device)
    logger.info("Init Testing data: %.2f (s)", time.time() - start)
    start = time.time()
    # train data loader
    data_loader_train = torch.utils.data.DataLoader(
        datasets_train,
        batch_size=cfg.train.batch_size,
        sampler=None,
        num_workers=0,
        pin_memory=False,  # usually hurts..
        shuffle=True
    )
    # test data loader
    data_loader_test = torch.utils.data.DataLoader(
            datasets_test,
            batch_size=cfg.evaluate.batch_size,
            sampler=None,
            num_workers=0,#cfg.data_eval.workers,
            pin_memory=False,  # Usually hurts..
            shuffle=False,
        )
    logger.info("Data loaders (train and test): %.2f (s)", time.time() - start)
    num_classes = {'one':7}
    logger.info('Creating model with %s classes', num_classes)

    model = hydra.utils.instantiate(cfg.mat, 
                                    _recursive_=False)
    
    logger.debug('Model: %s', model)

    ############################################################################# above this line

    model.to(device)
    
    params = []
    # [[__all__, 0.1, 0.0001]] # 0.0003, 1.0e-05]]
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
        this_scaled_lr = this_lr * dist_info['world_size'] # world_size = 1
        if cfg.opt.scale_lr_by_bs:
            this_scaled_lr *= cfg.train.batch_size
        params.append({
            'params': this_params_rest.values(),
            'lr': this_scaled_lr,
            'weight_decay': this_wd,
        })
        logger.info('Using LR %f WD %f for parameters ... MB removed this_params_bias_bn.keys()', params[-1]['lr'],
                    params[-1]['weight_decay']) # MB removed this_params_rest.keys()
        params.append({
            'params': this_params_bias_bn.values(),
            'lr': this_scaled_lr,
            'weight_decay': this_wd * cfg.opt.bias_bn_wd_scale,
        })
        logger.info('Using LR %f WD %f for parameters ... MB removed this_params_bias_bn.keys()', params[-1]['lr'],
                    params[-1]['weight_decay']) # MB removed this_params_bias_bn.keys()
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
    
    main_scheduler = hydra.utils.instantiate(cfg.opt.scheduler,
                                            optimizer,
                                            iters_per_epoch=len(data_loader_train),
                                            world_size=dist_info['world_size']
                                            )
    lr_scheduler = hydra.utils.instantiate(cfg.opt.warmup,
                                           optimizer,
                                           main_scheduler,
                                           iters_per_epoch=len(data_loader_train),
                                           world_size=dist_info['world_size']
                                           )

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

    logger.info('Wrapping model into DP')

    # func.train_eval_ops.TrainEvalOps
    train_eval_op = hydra.utils.instantiate(cfg.train_eval_op,
                                            model,
                                            device,
                                            _recursive_=False) # false
    # TEST EVALUATION ONLY
    if cfg.test_only:
        logger.info("Starting test_only")
        hydra.utils.call(cfg.evaluate,
                        eval_types=cfg.evaluate.eval_types,
                        device=device,
                        step_now=1,
                        data_loader=data_loader_test,
                        tb_writer=writer,
                        logger=logger,
                        epoch=1,
                        train_eval_op=train_eval_op)
        return

    logger.info("Start training")
    start_time = time.time()

    # Get training metric logger
    stat_loggers = get_default_loggers(writer, start_epoch, logger)
    best_acc1 = 0.8
    partial_epoch = start_epoch - int(start_epoch)
    start_epoch = int(start_epoch)
    last_saved_time = datetime.datetime(1, 1, 1, 0, 0)
    epoch = 1  # Since using this var to write the checkpoint output, so init to sth
    step_now = 3000*start_epoch
    step_val_now =0
    for epoch in range(start_epoch, cfg.train.num_epochs):
        
        # TRAINING
        if epoch>=0:
            last_saved_time, step_now = hydra.utils.call(cfg.train.train_one_epoch_fn,
                    model,
                    step_now,
                    device,
                    train_eval_op,
                    optimizer,
                    lr_scheduler, 
                    data_loader_train, 
                    epoch,
                    partial_epoch,
                    stat_loggers["train"], 
                    logger,
                    last_saved_time)
            
            partial_epoch = 0  # Reset, for future epochs
            store_checkpoint([CKPT_FNAME], model, optimizer, lr_scheduler,
                            epoch + 1)
        
        # EVALUATION
        if cfg.train.eval_freq and epoch % cfg.train.eval_freq == 0:
            # NOTE: MUST map the arguments to the correct names
            acc1, step_val_now = hydra.utils.call(cfg.evaluate,
                    eval_types=cfg.evaluate.eval_types,
                    device=device, 
                    step_now=step_val_now,
                    data_loader=data_loader_test, 
                    tb_writer=writer, 
                    logger=logger, 
                    epoch=epoch + 1,
                    train_eval_op=train_eval_op)
        else:
            acc1 = 0
        if acc1 >= best_acc1:
            store_checkpoint('checkpoint_best.pth', model, optimizer,
                             lr_scheduler, epoch + 1)
            best_acc1 = acc1
        if isinstance(lr_scheduler.base_scheduler,
                      scheduler.ReduceLROnPlateau):
            lr_scheduler.step(acc1)

        # reset all meters in the metric logger
        for log in stat_loggers:
            stat_loggers[log].reset_meters()
    # Store the final model to checkpoint
    store_checkpoint([CKPT_FNAME], model, optimizer, lr_scheduler, epoch + 1)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time %s', total_time_str)

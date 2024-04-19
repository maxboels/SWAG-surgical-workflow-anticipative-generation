# Copyright (c) Facebook, Inc. and its affiliates.

from __future__ import print_function
from typing import List, Dict

import errno
import os
from pathlib import Path
import logging
import submitit
import cv2

import torch
import torch.distributed as dist

def seq_accuracy_nans(preds, targets, padding_value=-1, mean=True):
    """
    Calculate the accuracy per element in the sequence.
    
    Args:
        preds (torch.Tensor): Predicted values with shape (Batch, Seq_length, Classes).
        targets (torch.Tensor): Target values with shape (Batch, Seq_length).
        padding_value (int, optional): Value used for padding. Elements with this value are ignored. Defaults to 0.
    
    Returns:
        torch.Tensor: Accuracy per element in the sequence with shape (Seq_length,).
    """
    # Check if the input shapes are correct
    if preds.dim() != 3:
        raise ValueError(f"Expected preds to have 3 dimensions, but got {preds.dim()}")
    if targets.dim() != 2:
        raise ValueError(f"Expected targets to have 2 dimensions, but got {targets.dim()}")
    if preds.shape[:2] != targets.shape:
        raise ValueError(f"Mismatch between preds shape ({preds.shape[:2]}) and targets shape ({targets.shape})")
    
    # Get the predicted classes
    pred_classes = torch.argmax(preds, dim=-1)
    
    # Create a mask to ignore padding elements
    mask = (targets != padding_value)
    
    # Calculate the accuracy per element
    correct = (pred_classes == targets) * mask
    total = mask.sum(dim=0)
    accuracy = correct.sum(dim=0) / total
    
    # Handle division by zero for sequences with all padding elements
    accuracy[total == 0] = torch.nan

    if not mean:
        return accuracy.float()
    return torch.nanmean(accuracy).item()
    
def seq_accuracy(preds, targets, padding_value=-1, mean=True):
    """
    Calculate the accuracy per element in the sequence.

    Args:
        preds (torch.Tensor): Predicted values with shape (Batch, Seq_length, Classes).
        targets (torch.Tensor): Target values with shape (Batch, Seq_length).
        padding_value (int, optional): Value used for padding. Elements with this value are ignored. Defaults to 0.

    Returns:
        torch.Tensor: Accuracy per element in the sequence with shape (Seq_length,).
    """
    # Check if the input shapes are correct
    if preds.dim() != 3:
        raise ValueError(f"Expected preds to have 3 dimensions, but got {preds.dim()}")
    if targets.dim() != 2:
        raise ValueError(f"Expected targets to have 2 dimensions, but got {targets.dim()}")
    if preds.shape[:2] != targets.shape:
        raise ValueError(f"Mismatch between preds shape ({preds.shape[:2]}) and targets shape ({targets.shape})")

    # Get the predicted classes
    pred_classes = torch.argmax(preds, dim=-1)

    # Create a mask to ignore padding elements
    mask = (targets != padding_value)

    # Calculate the accuracy per element
    correct = (pred_classes == targets) * mask
    total = mask.sum(dim=0)
    accuracy = correct.sum(dim=0) / total

    # Handle division by zero for sequences with all padding elements
    accuracy[total == 0] = torch.nan

    if not mean:
        return accuracy
    return accuracy.mean()

def accuracy(inputs, targets):
    """ Computes the accuracy for frames with valid labels (i.e., target != -1).
    Input:
        inputs: (B, C, T)
        targets: (B, T)
    Output:
        accuracy: Scalar value representing the mean accuracy
    """
    # Create a mask to identify valid target frames
    mask = (targets != -1)

    # Apply the mask to flatten the inputs and targets tensors
    inputs_flat = inputs[mask]
    targets_flat = targets[mask]

    # Compute the predicted class for each valid frame
    predicted_flat = torch.argmax(inputs_flat, dim=1)

    # Calculate the number of correct predictions
    correct = (predicted_flat == targets_flat).sum().item()

    # Calculate the accuracy
    total = targets_flat.size(0)
    accuracy = correct / total

    return accuracy

def new_accuracy(output, target, ignore_index=None):
    """
    Computes the accuracy for frames with valid labels (i.e., target != ignore_index).

    Input:
        output: (B, T, C)
        target: (B, T)
        ignore_index: Index to ignore in the target (default: None)

    Output:
        accuracy: Scalar value representing the mean accuracy
    """
    if ignore_index is not None:
        valid_mask = target != ignore_index
        if not valid_mask.any():
            return torch.tensor(0.0, device=output.device)
        output = output[valid_mask]
        target = target[valid_mask]

    pred = output.argmax(dim=-1)  # (B, T)
    accuracy = (pred == target).float().mean()

    return accuracy.item()

def old_accuracy(output, target, ignore_index=None):
    """Computes the accuracy for frames with valid labels (i.e. target >= 0).
    Input:
        output: (B, C, T)
        target: (B, T)
    Output:
        accuracy: (1)
    """
    if torch.all(target < 0):
        return torch.zeros(output.shape[0], device=output.device)
    # get the predicted classes for those frames
    pred = output.argmax(dim=1) # (B, T)
    # print("accuracy pred.shape", pred.shape)
    # get frames where target is valid
    target_mask = target != ignore_index
    if ignore_index is not None:
        # set ignored frames to -1, so they are not counted in the accuracy if predicted correctly
        # but they will be counted as errors if predicted incorrectly
        pred[target == ignore_index] = -1
    # get the accuracy for the valid frames
    accuracy = (pred[target_mask] == target[target_mask]).float()
    # print("accuracy accuracy.shape", accuracy.shape)

    # return the mean accuracy over all frames and batches
    return accuracy.mean().unsqueeze(0)


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master, logger):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
    if not is_master:
        # Don't print anything except FATAL
        logger.setLevel(logging.ERROR)
        logging.basicConfig(level=logging.ERROR)
    else:
        logger.setLevel(logging.INFO)
        logging.basicConfig(level=logging.INFO)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(logger, dist_backend='nccl'):
    dist_info = dict(
        distributed=False,
        rank=0,
        world_size=1,
        gpu=0,
        dist_backend=dist_backend,
        dist_url=get_init_file(None).as_uri(),
    )
    # If launched using submitit, get the job_env and set using those
    try:
        job_env = submitit.JobEnvironment()
    except RuntimeError:
        job_env = None
    if job_env is not None:
        dist_info['rank'] = job_env.global_rank
        dist_info['world_size'] = job_env.num_tasks
        dist_info['gpu'] = job_env.local_rank
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist_info['rank'] = int(os.environ["RANK"])
        dist_info['world_size'] = int(os.environ['WORLD_SIZE'])
        dist_info['gpu'] = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        dist_info['rank'] = int(os.environ['SLURM_PROCID'])
        dist_info['gpu'] = dist_info['rank'] % torch.cuda.device_count()
    elif 'rank' in dist_info:
        pass
    else:
        print('Not using distributed mode')
        dist_info['distributed'] = False
        return dist_info

    dist_info['distributed'] = True

    torch.cuda.set_device(dist_info['gpu'])
    dist_info['dist_backend'] = dist_backend
    print('| distributed init (rank {}): {}'.format(dist_info['rank'],
                                                    dist_info['dist_url']),
          flush=True)
    torch.distributed.init_process_group(backend=dist_info['dist_backend'],
                                         init_method=dist_info['dist_url'],
                                         world_size=dist_info['world_size'],
                                         rank=dist_info['rank'])
    setup_for_distributed(dist_info['rank'] == 0, logger)
    return dist_info


def get_shared_folder(name) -> Path:
    # Since using hydra, which figures the out folder
    return Path('./').absolute()


def get_init_file(name):
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder(name)), exist_ok=True)
    init_file = get_shared_folder(name) / 'sync_file_init'
    return init_file


def gather_tensors_from_all(tensor: torch.Tensor) -> List[torch.Tensor]:
    """
    Wrapper over torch.distributed.all_gather for performing
    'gather' of 'tensor' over all processes in both distributed /
    non-distributed scenarios.
    """
    if tensor.ndim == 0:
        # 0 dim tensors cannot be gathered. so unsqueeze
        tensor = tensor.unsqueeze(0)

    if is_dist_avail_and_initialized():
        gathered_tensors = [
            torch.zeros_like(tensor)
            for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(gathered_tensors, tensor)
    else:
        gathered_tensors = [tensor]

    return gathered_tensors


def gather_from_all(tensor: torch.Tensor) -> torch.Tensor:
    gathered_tensors = gather_tensors_from_all(tensor)
    gathered_tensor = torch.cat(gathered_tensors, 0)
    return gathered_tensor


def get_video_info(video_path: Path, props: List[str]) -> Dict[str, float]:
    """
    Given the video, return the properties asked for
    """
    output = {}
    cam = cv2.VideoCapture(str(video_path))
    output['frames'] = cam.get(cv2.CAP_PROP_FRAME_COUNT)
    if 'fps' in props:
        output['fps'] = cam.get(cv2.CAP_PROP_FPS)
    if 'len' in props:
        fps = cam.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            output['len'] = 0
        else:
            output['len'] = (out['frames'] / fps)
    
    cam.release()
    return output

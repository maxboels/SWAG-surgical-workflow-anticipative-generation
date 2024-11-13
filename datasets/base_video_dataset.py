# cut data augmentation muti-video
import math
import pickle
import os
import numpy as np
import cv2
from torch.utils.data import Dataset
import pathlib
import random
from collections import OrderedDict
import operator
from typing import Tuple, Union, Sequence, Dict
import torch
import torch.nn as nn
import torchvision
from PIL import Image
import json
import time

import h5py
import glob

#-- mboels
from R2A2.dataset.next_class_and_duration_v2 import get_next_classes_and_durations
# temp
from matplotlib import pyplot as plt

#num_dataset = len(dict_data) #print(dict_data.keys())
abs_path = pathlib.Path(__file__).parent.parent.absolute()
FUTURE_PREFIX = 'future'
phase_segments = 16
fps = 1

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return np.array(img.convert('RGB'))

def plot_remaining_time(phase_labels, gt_remaining_time, h, num_obs_classes, video_idx=0, dataset="Cholec80"):
    fig, axs = plt.subplots(num_obs_classes + 1, 1, figsize=(15, 3*(num_obs_classes + 1)), sharex=True)
    time_steps = np.arange(len(phase_labels))
    colors = plt.cm.tab10(np.linspace(0, 1, num_obs_classes + 1))

    y_min, y_max = -0.1, h + 0.5
    for i in range(num_obs_classes + 1):
        remaining_time = gt_remaining_time[i].numpy()
        axs[i].plot(time_steps, remaining_time, label=f'Class {i if i < num_obs_classes else "EOS"}', linewidth=2.5, color=colors[i])

        regression_mask = (remaining_time > 0) & (remaining_time < h)
        axs[i].fill_between(time_steps, y_min, y_max, where=regression_mask, color='lightgray', alpha=0.3)

        active_mask = remaining_time == 0
        axs[i].fill_between(time_steps, y_min, y_max, where=active_mask, color=colors[i], alpha=0.3)

        axs[i].set_ylim(y_min, y_max)
        axs[i].set_ylabel("Ant. Time (m)")
        axs[i].legend(loc='upper right')
        axs[i].grid(True, linestyle='--', alpha=0.7)

        regression_ranges = np.where(np.diff(np.concatenate(([False], regression_mask, [False]))))[0].reshape(-1, 2)
        active_ranges = np.where(np.diff(np.concatenate(([False], active_mask, [False]))))[0].reshape(-1, 2)

        for start, end in regression_ranges:
            axs[i].text((start + end) / 2, y_max, f'<{h} min.', ha='center', va='bottom', fontsize=8, color='gray')

        for start, end in active_ranges:
            axs[i].text((start + end) / 2, y_max, 'Active', ha='center', va='bottom', fontsize=8, color=colors[i])

    axs[-1].set_xlabel("Video Time Steps (seconds)")
    plt.tight_layout()
    plt.savefig(f"./plots/{dataset}/rtd/remaining_time_gt_{video_idx}.png")
    plt.show()
    plt.close()

def gt_remaining_time_full(phase_labels, num_classes=7):
    seq_len = phase_labels.shape[0]
    remaining_time = torch.full((seq_len, num_classes + 1), float('inf'), device=phase_labels.device, dtype=torch.float32)
    
    for phase in range(num_classes):
        phase_indices = torch.where(phase_labels == phase)[0]
        if len(phase_indices) == 0:
            continue
        
        # Find contiguous segments
        segments = []
        segment_start = phase_indices[0]
        for i in range(1, len(phase_indices)):
            if phase_indices[i] != phase_indices[i-1] + 1:
                segments.append((segment_start, phase_indices[i-1]))
                segment_start = phase_indices[i]
        segments.append((segment_start, phase_indices[-1]))
        
        for segment_start, segment_end in segments:
            for i in range(segment_start):
                remaining_time[i, phase] = min(remaining_time[i, phase], (segment_start - i) / 60.0)
            remaining_time[segment_start:segment_end+1, phase] = 0.0
    
    # Handle EOS class
    for i in range(seq_len):
        remaining_time[i, -1] = (seq_len - i) / 60.0
    
    return remaining_time

def gt_remaining_time_capped(phase_labels, h=5, num_classes=7):
    seq_len = phase_labels.shape[0]
    remaining_time = torch.full((seq_len, num_classes + 1), h, device=phase_labels.device, dtype=torch.float32)
    print(f"seq_len: {seq_len}, num_classes: {num_classes}, h: {h}")
    
    for phase in range(num_classes):
        phase_indices = torch.where(phase_labels == phase)[0]
        if len(phase_indices) == 0:
            continue
        
        # Find contiguous segments
        segments = []
        segment_start = phase_indices[0]
        for i in range(1, len(phase_indices)):
            if phase_indices[i] != phase_indices[i-1] + 1: # not contiguous
                segments.append((segment_start, phase_indices[i-1]))
                segment_start = phase_indices[i]
        segments.append((segment_start, phase_indices[-1]))
        print(f"phase: {phase}, segments: {segments}")
        
        # Handle segments
        for segment_start, segment_end in segments:
            pre_segment_start = max(0, segment_start - int(h*60))
            
            # Remaining time before the segment starts
            for i in range(pre_segment_start, segment_start):
                # Avoid overwriting if a smaller value is already present
                remaining_time[i, phase] = min(remaining_time[i, phase], (segment_start - i) / 60.0)
            remaining_time[segment_start:segment_end+1, phase] = 0.0

    # Handle EOS class
    eos_start = max(0, seq_len - int(h*60))
    print(f"eos_start: {eos_start} = {seq_len} - {h*60}")
    for i in range(eos_start, seq_len):
        remaining_time[i, -1] = (seq_len - i) / 60.0
    
    return remaining_time

def video_end_regression_values(video_length, eos_length=20*60):
    """ Here, 60 frames is 1/30 of the eos length. And 3 minutes is 1/10 of the eos length.
    """
    values = []
    start_before_eos = video_length - eos_length  # Point at which to start decrementing values

    for i in range(video_length):
        if i >= start_before_eos:
            # Calculate how many seconds into the eos_length we are
            seconds_into_eos = i - start_before_eos + 1
            # Normalize the remaining time in the context of eos_length
            rem_time_norm = 1 - (seconds_into_eos / eos_length)
        else:
            # Before the eos_length, the value is kept at 1
            rem_time_norm = 1.0
        values.append(rem_time_norm)

    return values




class GaussianMixtureSamplerWithPosition:
    def __init__(self, class_freq_positions, lookahead=18):
        self.class_freq_positions = class_freq_positions
        self.lookahead = lookahead
        self.probabilities = self._compute_probabilities()
    
    def _compute_probabilities(self):
        probabilities = {}
        for cls, freq_list in self.class_freq_positions.items():
            probabilities[int(cls)] = []
            for freq_dict in freq_list:
                total_count = sum(freq_dict.values())
                probabilities[int(cls)].append({int(k): v / total_count for k, v in freq_dict.items()})
        return probabilities
    
    def return_class_weights(self, current_class, position):
        if current_class not in self.probabilities:
            raise ValueError(f"Class {current_class} not found in class frequencies.")
        
        return self.probabilities[current_class][position]
    
    def sample(self, current_class, num_samples=18):
        if current_class not in self.probabilities:
            raise ValueError(f"Class {current_class} not found in class frequencies.")
        
        samples = []
        for j in range(num_samples):
            possible_values = list(self.probabilities[current_class][j].keys())
            probabilities = list(self.probabilities[current_class][j].values())
            samples.append(np.random.choice(possible_values, p=probabilities))
        
        return samples


class SelectDataset():
    def __init__(self, dataset_name="cholec80", logger=None):
        self.dataset_name = dataset_name
        self.logger = logger

        if self.dataset_name == "autolaparo21":
            self.frames_fps = 1
            self.dataset_path = f"{abs_path}/R2A2/dataset/{self.dataset_name}/"
            self.data_root = '/nfs/home/mboels/datasets/autolaparo21/'
            self.path_and_label_dir = os.path.join(self.data_root, 'dataframes', f'autolaparo_df_255px_{self.frames_fps}fps.pkl')
        elif self.dataset_name == "cholec80":
            self.frames_fps = 25
            self.dataset_path = f"{abs_path}/R2A2/dataset/{self.dataset_name}/"
            self.data_root = '/nfs/home/yangliu/dataset/VandArecognition/cholec80/'
            self.path_and_label_dir = os.path.join(self.data_root, 'dataframes', f'CholecFrames_{self.frames_fps}fps_250px_df.pkl')
        elif self.dataset_name == "cholect50":
            self.frames_fps = 25
            self.dataset_path = f"{abs_path}/R2A2/dataset/{self.dataset_name}/"
            self.data_root = '/nfs/home/yangliu/dataset/VandArecognition/cholect50/'
            self.path_and_label_dir = os.path.join(self.data_root, 'dataframes', f'CholecFrames_{self.frames_fps}fps_250px_df.pkl')
        else:
            raise ValueError("dataset_name not found")

        start_time = time.time()
        with open(self.path_and_label_dir, 'rb') as fo:
            self.df = pickle.load(fo)
        time_elapsed = time.time() - start_time
        self.logger.info(f"[SELECT DATASET] ALL loaded dataframe in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")
        self.logger.info(f"[SELECT DATASET] ALL loaded dataframe: {self.df.shape}")

        # replace the labels to start from 0 to num_classes-1 (currently 1 to num_classes)
        if self.df['class'].min() == 1:
            self.logger.info(f"[SELECT DATASET] labels are starting from 1")
            self.df['class'] = self.df['class'] - 1
            # overwrite the dataframe with the new labels
            with open(self.path_and_label_dir, 'wb') as f:
                pickle.dump(self.df, f)
                self.logger.info(f"[SELECT DATASET] saved dataframe with updated labels: {self.df.shape}")
        else:
            self.logger.info(f"[SELECT DATASET] labels are already starting from 0")
        #-----------------mb added-----------------
        self.logger.info(f"[SELECT DATASET] ALL unique classes: {np.unique(self.df['class'].tolist())}")
        self.logger.info(f"[SELECT DATASET] aLL dataframe (head): {self.df.head()}")
        self.logger.info(f"[SELECT DATASET] ALL dataframe (tail): {self.df.tail()}")
    
    def get_dataframe(self):
        return self.df


# Helper functions
def calculate_total_padded_values(video_duration_seconds, fps, future_sampling_rate, future_samples):
    video_frames = video_duration_seconds * fps
    future_clip_count = video_frames // future_sampling_rate
    padded_values_per_clip = future_samples - 1
    total_padded_values = future_clip_count * padded_values_per_clip
    return total_padded_values

def compute_num_eos_values(anticip_time=60, full_anticip_time_minutes=18, auto_regressive=True, num_videos=40):
    """Calculate the number of eos values needed for anticipation time for the last samples

    Args:
        anticip_time (int): Anticipation time in seconds
        full_anticip_time_minutes (int): Full anticipation time in minutes
        num_videos (int): Number of videos
        num_targets (int, optional): Number of targets per token. Defaults to 18.
    """
    if auto_regressive:
        total_eos_tokens = anticip_time * num_videos
    else:
        total_eos_tokens = full_anticip_time_minutes * 60 * num_videos
    return total_eos_tokens

class Medical_Dataset(Dataset):
        def __init__(self, cfg, dataframe, train_mode='train', dataset_name="cholec80", video_indices=None,
                    transform=None, device=None, logger=None, **kwargs):
            self.df = dataframe
            self.label_type = ['one']
            self.train_mode = train_mode
            self.video_indices = video_indices
            self.transform = transform
            self.device = device
            self.logger = logger
            # self.project_path = "/nfs/home/mboels/projects/SKiT_video_augmentation"
            self.dataset_local_path = f'/nfs/home/mboels/projects/SuPRA/datasets/{dataset_name}/'
            self.save_video_labels_to_npy = cfg.save_video_labels_to_npy

            # TASKS
            self.do_regression      = cfg.do_regression
            self.do_classification  = cfg.do_classification


            self.model_name = cfg.model_name
            self.debug = False
            self.dataset_name = dataset_name                                        # autolaparo21, cholec80, cholect50
            self.num_classes = 7                                                    # AutoLapro: 7, Cholec80: 7
            self.attn_window_size = 20
            self.ctx_length = cfg.ctx_length                                        # 3000 or 1500 frames
            
            self.max_ctx_tokens = int(cfg.ctx_length/cfg.anticip_time)              # Compression Tokens: 100 or 50 tokens for curr and next frames
            
            self.ctx_pooling = cfg.ctx_pooling
            self.num_ctx_tokens = cfg.num_ctx_tokens
            
            self.anticip_time = cfg.anticip_time
            self.max_anticip_sec = int(cfg.max_anticip_time * 60)  # 50 or 25 frames per token
            self.num_future_tokens = int(self.max_anticip_sec / cfg.anticip_time)   # 60 or 30 tokens (num auto-regressive steps)
            self.eos_class = cfg.eos_class
            self.eos_classification = cfg.eos_classification
            # new approach
            self.eos_regression = cfg.eos_regression
            self.eos_reg_length = int(self.num_future_tokens/4) * cfg.anticip_time + 1 # 30m * 180s = 5400 frames
            self.plot_eos_times = False     # TODO: Use Gaussian
            #----------------- end select arguments-----------------
            self.reduc_feats_folder = "train_targets_x_mp/"
            self.num_next_labels = 6
            self.last_seg_class = 7
            self.predict_next_phase = True
            self.predict_next_segment = False
            self.get_target_feats = False
            self.replace_current_with_next_class = False
            self.future_segmts_tgt = False

            self.eos_weight = cfg.eos_weight # "count" or "mean"
            self.class_weight = cfg.class_weight # "positional" or "frequency", positional_inverse_frequency
            
            # observed targets
            if self.model_name in ["supra", "lstm"]:
                # self.curr_frames_tgt = "global" # "global" or "local"
                self.next_frames_tgt = "next"
                self.auto_regressive = True
            else:
                # self.curr_frames_tgt = "local"
                self.next_frames_tgt = "future"
                self.auto_regressive = False                
            
             # "global" or "local"
            params = {
                "debug": self.debug,
                "dataset_name": self.dataset_name,
                "reduc_feats_folder": self.reduc_feats_folder,
                "num_classes": self.num_classes,
                "num_next_labels": self.num_next_labels,
                "last_seg_class": self.last_seg_class,
                "predict_next_phase": self.predict_next_phase,
                "get_target_feats": self.get_target_feats,
                "replace_current_with_next_class": self.replace_current_with_next_class
            }
            if self.train_mode == "train":
                self.logger.info(f"[DATASET] params: {params}")


            self.eval_horizons = cfg.eval_horizons
            self.num_obs_classes = self.num_classes
            
            #-----------------end select arguments-----------------
            if self.dataset_name == "autolaparo21":
                self.frames_fps = 1
                self.dataset_path = f"{abs_path}/R2A2/dataset/{self.dataset_name}/"
                self.data_root = '/nfs/home/yangliu/dataset/AutoLaparo_Task1/'
                # self.path_and_label_dir = os.path.join(self.data_root, 'dataframes', f'autolaparo_df_255px_{self.frames_fps}fps.pkl')
            elif self.dataset_name == "cholec80":
                self.frames_fps = 25
                self.dataset_path = f"{abs_path}/R2A2/dataset/{self.dataset_name}/"
                self.data_root = '/nfs/home/yangliu/dataset/VandArecognition/cholec80/'
                # self.path_and_label_dir = os.path.join(self.data_root, 'dataframes', f'CholecFrames_{self.frames_fps}fps_250px_df.pkl')
            elif self.dataset_name == "cholect50":
                self.frames_fps = 25
                self.dataset_path = f"{abs_path}/R2A2/dataset/{self.dataset_name}/"
                self.data_root = '/nfs/home/yangliu/dataset/VandArecognition/cholect50/'
                # self.path_and_label_dir = os.path.join(self.data_root, 'dataframes', f'CholecFrames_{self.frames_fps}fps_250px_df.pkl')
            else:
                raise ValueError("dataset_name not found")
            self.extracted_feats_path = self.dataset_path + f"vit_extracted_feats/"
            self.key_feats_path = self.dataset_path + f"key_feats/" + self.reduc_feats_folder

            # TODO: REMOVE
            self.df.loc[:, 'frame'] = (self.df.image_path.str[-10:-4]).astype(int)

            # SPLITS
            self.df_split = self.get_data_split(
                dataset_name=dataset_name, 
                split=train_mode, 
                video_indices=video_indices
            )
            
            if self.debug:
                self.df_split = self.df_split[:1000]
            
            print(f"[DATASET] unique classes: {np.unique(self.df_split['class'].tolist())}")
            print(f"[DATASET] {train_mode} video_indices: {video_indices}")
            self.logger.info(f"[DATASET] {train_mode} size: {len(self.df_split)}")
            self.logger.info(f"[DATASET] {train_mode} video_indices: {video_indices}")
                

            self.frame_indices = []
            self.video_idx = []
            self.video_lengths = []
            self.video_pool = {}
            self.list_pre = {}
            self.branch_dict = {}

            # MAIN CLASS METHOD
            self.init_video_data(video_indices=video_indices, mode=train_mode)
            self.data_order = list(range(len(self.frame_indices)))

            if self.train_mode == 'train':
                random.shuffle(self.data_order)

        def save_video_durations(self, video_indices, output_file):
            video_durations = self.get_video_durations(video_indices)
            with open(output_file, 'w') as f:
                json.dump(video_durations, f)
            self.logger.info(f"Saved video durations to {output_file}")

        def get_video_durations(self, video_indices):
            video_durations = {}
            for video_indx in video_indices:
                df_video = self.df_split[self.df_split.video_idx == video_indx]
                video_length = len(df_video)
                video_durations[video_indx] = video_length
            return video_durations

        def get_data_split(self, dataset_name="cholec80", split="train", video_indices=None):
            """
                autolapro21: 0-14, 14-21
                cholec80: 1-41, 41-81
            """
            self.logger.info(f"[DATASET] df unique videos: {self.df.video_idx.unique()}")
            self.logger.info(f"[DATASET] split: {split} video_indices: {video_indices}")
            df = self.df[self.df.video_idx.isin(video_indices)]
            assert len(df.video_idx.unique()) == len(video_indices), "video indices not found"
            df = df.reset_index() if not split == "train" else df
            return df

        def init_video_data(self, video_indices=None, mode='train'):
            self.videos = []
            self.labels = []
            self.gt_remaining_time = []
            self.gt_remaining_time_full = []
            self.next_segments_class = []
            self.heatmap = []
            self.eos_video_targets = []
            # mb added
            self.target_feats = []
            new_id = 0  
            video_durations = {}

            # initialize class counts for 1fps
            train_class_count_1fps = {}
            
            for video_indx in video_indices:
                self.logger.info(f"[DATASET] video_idx: {video_indx}")

                # VIDEO DATAFRAME
                df_video = self.df_split[self.df_split.video_idx==video_indx]

                # SAMPLE VIDEO DATAFRAME AT 1FPS (25fps to 1fps)
                df_video = df_video[df_video.frame % self.frames_fps==0].reset_index(drop = True)
                self.logger.info(f"[DATASET] Sampled video_idx: {video_indx} at {self.frames_fps} fps to {len(df_video)} frames")
                video_length = len(df_video)
                video_durations[video_indx] = video_length
                if video_length==0:
                    print(f'[SKIP] video_idx:{video_indx} video_length:{video_length}') 
                    continue
                path_list = df_video.image_path.tolist()
                phase_labels = df_video['class'].tolist()

                if self.save_video_labels_to_npy:
                    np.save(self.dataset_local_path + 'labels' + '/' + f"video_{video_indx}_labels.npy", phase_labels)
                    self.logger.info(f"[DATASET] Saved labels for video {video_indx}")


                #-----------------mb added-----------------
                if self.eos_regression:
                    eos_regression_list = video_end_regression_values(video_length, eos_length=self.eos_reg_length)
                    eos_regression_tensor = torch.Tensor(eos_regression_list).to(self.device)
                    self.eos_video_targets.append(eos_regression_tensor)
                    if video_indx == video_indices[-1] and self.plot_eos_times:
                        plt.plot(eos_regression_list)
                        plt.xlabel('Frame number')
                        plt.ylabel('Remaining time')
                        plt.title(f'Remaining time regression values for video {video_indx}')
                        plt.savefig(f"remaining_time_video_{video_indx}.png")
                        plt.close()

                # to tensors
                phase_labels = torch.Tensor(phase_labels).long()
                self.logger.info(f"[DATASET] video {video_indx} phase_labels: {phase_labels.size()}")
                
                new_id += 1
                num=0
                # TODO: challenge this loop
                for i in range(0, video_length):
                    # maping between frames and video ids
                    self.frame_indices.append(i) # [0, 1, 2, 3, 0, 1, 0, 1, 2, ...]
                    self.video_idx.append(len(self.videos))
                    self.video_lengths.append(len(df_video))

                with torch.no_grad():
                    self.labels.append(phase_labels)
                    with open(self.extracted_feats_path + f"video_{video_indx}.pkl", 'rb') as f:
                        video = pickle.load(f)
                    video = torch.from_numpy(video).float().to(self.device)
                    self.videos.append(video)
                
                # regression labels (remaining time until occurrence of each phases)
                gt_remaining_time = {}
                # for h in self.eval_horizons[-1]:
                max_horizon = self.eval_horizons[-1]
                if max_horizon == 1000:
                    # remaining time duration for each phase (full duration values)
                    gt_remaining_time = gt_remaining_time_full(phase_labels, num_classes=self.num_obs_classes)
                    self.gt_remaining_time.append(gt_remaining_time)
                    self.logger.info(f"[DATASET] gt_remaining_time h={max_horizon}: {gt_remaining_time.size()}")
                else:
                    gt_remaining_time[max_horizon] = gt_remaining_time_capped(phase_labels, h=max_horizon, num_classes=self.num_obs_classes)
                    self.logger.info(f"[DATASET] gt_remaining_time h={max_horizon}: {gt_remaining_time[max_horizon].size()}")

                self.gt_remaining_time.append(gt_remaining_time)
                self.logger.info(f"\n")
            
                # save video gt remaining time to json
                gt_remaining_time = {k: v.cpu().numpy().tolist() for k, v in gt_remaining_time.items()}
                with open(f"{self.dataset_local_path}gt_remaining_time_{video_indx}.json", 'w') as f:
                    json.dump(gt_remaining_time, f)
                    self.logger.info(f"[DATASET] Saved gt_remaining_time for video {video_indx}")    

            # save video durations to json
            video_durations_file = f"{self.dataset_local_path}video_durations.json"
            if os.path.exists(video_durations_file):
                with open(video_durations_file, 'r') as f:
                    existing_video_durations = json.load(f)
            else:
                existing_video_durations = {}
            existing_video_durations.update(video_durations)
            with open(video_durations_file, 'w') as f:
                json.dump(existing_video_durations, f)
                self.logger.info(f"[DATASET] Saved video durations")

            # End of video loop
            self.avg_video_length = np.mean(list(video_durations.values())) / 60
            self.logger.info(f"[DATASET] number of {self.train_mode} videos: {len(self.videos)}")
            self.logger.info(f"[DATASET] avg video length: {self.avg_video_length:.2f} minutes")

            # PADDING VALUES based on video dims
            self.zeros = torch.zeros([video.size(0), self.ctx_length, video.size(2), video.size(3)]).float().to(self.device)
            self.ones = torch.ones([self.ctx_length,]).to(self.device).long()

            if self.eos_class== -1:
                self.eos = - torch.ones([self.ctx_length,]).to(self.device).long()
            elif self.eos_class== 7:
                self.eos = torch.full((self.max_anticip_sec,), self.eos_class).long().to(self.device)
            else:
                raise ValueError("eos_padding not found")

            # Class weights after sampling at 1fps
            if self.train_mode == 'train':
                # concatenate all labels in training dataset
                classes = torch.cat(self.labels, 0).cpu().numpy()
                self.logger.info(f"[DATASET] num classes at 1fps in {self.train_mode}: {classes.shape}")

                # CURR AND NEXT CLASS WEIGHTS
                classes_counts_dict = dict(zip(*np.unique(classes, return_counts=True)))
                self.logger.info(f"[DATASET] classes_counts_dict: {classes_counts_dict}")
                self.curr_class_weights = self._compute_class_weights(classes_counts_dict)
                self.logger.info(f"[DATASET] curr_class_weights: {self.curr_class_weights}")

                if self.class_weight in ["positional", "positional_inverse_frequency"]:
                    # Class weights for an observed class and different future predicted index positions
                    path_class_freq = f"/nfs/home/mboels/projects/SuPRA/datasets/{self.dataset_name}/naive2_{self.dataset_name}_class_freq_positions.json"
                    # load from json file
                    with open(path_class_freq, 'r') as f:
                        class_freq_positions = json.load(f)
                    class_freq_positions = {int(k): [{int(inner_k): inner_v for inner_k, inner_v in freq_dict.items()} for freq_dict in v] for k, v in class_freq_positions.items()}
                    self.logger.info(f"[DATASET] class_freq_positions: {class_freq_positions}")
                    # convert the probabilities to class weights
                    self.sampler_with_position = GaussianMixtureSamplerWithPosition(class_freq_positions, lookahead=18)
                    
                    # self.logger.info(f"[DATASET] next_class_weights: {self.next_class_weights}")
                else:
                    if self.eos_class == 7 and self.eos_classification:
                        if self.eos_weight=="count":
                            self.logger.info(f"[DATASET] eos_weight: {self.eos_weight}")
                            num_eos_vals = compute_num_eos_values(
                                anticip_time=self.anticip_time,
                                full_anticip_time_minutes=18,
                                auto_regressive=self.auto_regressive,
                                num_videos=len(video_indices)
                            )
                            self.logger.info(f"[DATASET] num_eos_vals: {num_eos_vals}")
                        elif self.eos_weight=="mean":
                            num_eos_vals = np.mean(list(classes_counts_dict.values()))
                            self.logger.info(f"[DATASET] num_eos_vals: {num_eos_vals}")
                        else:
                            raise ValueError("eos_weight not found")
                        
                        classes_counts_dict[self.num_classes] = int(num_eos_vals)
                        self.logger.info(f"[DATASET] classes_counts_dict: {classes_counts_dict}")
                        self.next_class_weights = self._compute_class_weights(classes_counts_dict)
                        self.logger.info(f"[DATASET] next_class_weights: {self.next_class_weights}")
            
        def _compute_class_weights(self, class_counts, normalize=False):
            total_samples = sum(class_counts.values())
            num_classes = len(class_counts)
            class_weights = torch.zeros(num_classes)
            for cls_id, count in class_counts.items():
                class_weights[cls_id] = total_samples / (num_classes * count)
            if normalize:
                # Optional normalization (if you want the weights to sum to 1)
                class_weights = class_weights / class_weights.sum()
            return class_weights
        
        def get_next_label(self, phase_labels, num_next_labels, end_class=7):
            """Next labels for each frame in the video
            Returns: a tensor of size (video_length, num_next_labels)
            """
            next_labels_tensor = - torch.ones((len(phase_labels), num_next_labels)).long()
            def find_next_classes(index, labels, num_next_class=1, end_class_id=7):
                next_classes = []
                current_class = labels[index]
                for i in range(index + 1, len(labels)):
                    if current_class != labels[i]:
                        next_classes.append(labels[i])
                        current_class = labels[i]
                    if len(next_classes) == num_next_class:
                        break
                if len(next_classes) < num_next_class:
                    next_classes += [end_class_id] * (num_next_class - len(next_classes))
                next_classes = torch.Tensor(next_classes).long()
                return next_classes
            for i in range(len(phase_labels) - 1):
                next_labels_tensor[i] = find_next_classes(i, phase_labels, num_next_labels, end_class)
            next_labels_tensor[-1] = torch.Tensor([end_class] * num_next_labels).long()
            return next_labels_tensor

        def get_short_video(self, video, start_idx=0):
            end_idx = video.size(1)
            num_take = 100
            num = math.ceil((end_idx-1-start_idx)/(num_take-1.0))
            num = max(1,num)
            start = max((end_idx-1)%num, (end_idx - 1) - (num_take-1)*num)
            video_short = video[:,start::num,:]
            rests = num_take - video_short.size(1)
            if rests>0:
                video_short = torch.cat((self.zeros[:,:rests,:,:], video_short),1)
            return video_short

  
        def class_mappings(self) -> Dict[Tuple[str, str], torch.FloatTensor]:
            return {}
        
        def get_video_label(self, video_idx, start_idx, end_idx):
            for i in range(start_idx, end_idx):
                # maping between frames and video ids
                self.frame_indices.append(i) # [0, 1, 2, 3, 0, 1, 0, 1, 2, ...]
                self.video_idx.append(video_idx)
                self.video_lengths.append(end_idx)
        
        def get_video_data(self, video_idx, frame_idx):
            video               = self.videos[video_idx]
            video_targets       = self.labels[video_idx]
            gt_remaining_time   = self.gt_remaining_time[video_idx]
            # gt_remaining_time_full = self.gt_remaining_time_full[video_idx]
            if self.eos_regression:
                eos_values = self.eos_video_targets[video_idx]
                print(f"[DATASET] eos_values: {eos_values.shape}")
            # video_next_seg_targets = self.next_segments_class[video_idx]
            print(f"[DATASET] video_idx: {video_idx} (t={frame_idx}) original: {video.size()}")
            print(f"[DATASET] video targets: {video_targets.size()}")
            # print(f"[DATASET] full video_next_seg_targets: {video_next_seg_targets.size()}")
            
            # for h in self.eval_horizons:
            print(f"[DATASET] gt_remaining_time {self.eval_horizons[-1]}: {gt_remaining_time[self.eval_horizons[-1]].size()}")
            
            # # print(f"[DATASET] gt_remaining_time_full: {gt_remaining_time_full.size()}")
            # # save gt remaining time full save to json
            # # gt_remaining_time_full = gt_remaining_time_full.cpu().numpy()
            # # save to json
            # with open(f"{self.dataset_local_path}gt_remaining_time_full_{video_idx}.json", 'w') as f:
            #     json.dump(gt_remaining_time_full.tolist(), f)


            # VIDEO FEED
            # NOTE: index 0 is included in the frame id so it is not necessary to add 1 !!!
            starts = max(0, frame_idx - self.ctx_length) # context length
            video_now = video[:, starts: frame_idx, :, :]
            missing = self.ctx_length - video_now.size(1)
            if missing > 0:
                video_now = torch.cat((self.zeros[:, :missing, :, :], video_now), 1)
            print(f"[DATASET] video (t={frame_idx}) with ctx_length: {video_now.size()}")

            # ----------------- DATA DICTIONARY -----------------
            data_now = OrderedDict()
            data_now['video_idx'] = video_idx
            data_now['frame_idx'] = frame_idx
            data_now['video_lengths'] = self.video_lengths[video_idx]
            data_now['video'] = video_now.to(self.device)

            # ----------------- TRAINING LABELS -----------------
            
            # CURRENT FRAMES TARGETS
            if self.ctx_pooling == "local":
                self.num_ctx_tokens = self.max_ctx_tokens # keep all context tokens for training
                curr_frame_spacing = self.anticip_time
                ends = frame_idx + 1
                remainder = frame_idx  % self.anticip_time
                # the target is the last frame in the attn window
                starts = max(remainder, frame_idx - self.ctx_length + self.anticip_time)
            elif self.ctx_pooling == "global":
                curr_frame_spacing = 1
                ends = frame_idx + 1
                starts = max(0, frame_idx - self.num_ctx_tokens + 1)
                # self.num_ctx_tokens = self.attn_window_size
            else:
                raise ValueError("curr_frames_tgt not found")
            
            # CURRENT FRAMES TARGETS
            print(f"[DATASET] curr_frames_tgt starts: {starts} ends: {ends}")
            curr_frames_tgt = video_targets[starts: ends: curr_frame_spacing].to(self.device)
            print(f"[DATASET] curr_frames_tgt: {curr_frames_tgt.size()}")
            print(f"[DATASET] curr_frames_tgt: {torch.arange(starts, ends, curr_frame_spacing)}")
            print(f"[DATASET] num_ctx_tokens: {self.num_ctx_tokens}")
            diff = self.num_ctx_tokens - curr_frames_tgt.size(0)
            if diff > 0:
                print(f"[DATASET] missing: {diff}")
                curr_frames_tgt = torch.cat(( -self.ones[:diff].long(), curr_frames_tgt), 0)
            elif diff < 0:
                print(f"[DATASET] excess: {diff}")
                curr_frames_tgt = curr_frames_tgt[-self.num_ctx_tokens:]
            data_now['curr_frames_tgt'] = curr_frames_tgt.to(self.device).long()
            print(f"[DATASET] curr_frames_tgt (new): {curr_frames_tgt.size()}")

            # REGRESSION REMAINING TIME TARGETS
            if self.do_regression or self.do_classification:
                # for h in self.eval_horizons:
                h = self.eval_horizons[-1]
                remaining_time = gt_remaining_time[h][frame_idx, :].unsqueeze(0)
                data_now[f'remaining_time_{h}_tgt'] = remaining_time.to(self.device).float()
                print(f"[DATASET] remaining_time_{h}_tgt: {remaining_time.size()}")
                print(f"[DATASET] remaining_time_{h}_tgt: {remaining_time}")
                
                # # FULL REMAINING TIME TARGETS
                # data_now['remaining_time_full_tgt'] = gt_remaining_time_full[frame_idx, :].unsqueeze(0).to(self.device).float()
                # print(f"[DATASET] remaining_time_full_tgt: {gt_remaining_time_full[frame_idx, :].unsqueeze(0).size()}")

            
            if self.eos_regression:
                curr_eos_values = eos_values[starts: ends: self.anticip_time]
                if missing > 0:
                    curr_eos_values = torch.cat((self.ones[:missing].float(), curr_eos_values), 0)
                data_now['curr_time2eos_tgt'] = curr_eos_values.to(self.device).float()
                print(f"[DATASET] curr_time2eos_tgt: {curr_eos_values.size()}")

            assert curr_frames_tgt.size(0) == self.num_ctx_tokens, f"num_ctx_tokens size not equal to curr_frames_tgt: {curr_frames_tgt.size(0)}"

            if self.train_mode == 'train':

                # NEXT FRAMES TARGETS (SHIFTED BY 2x Window size or 2x Frames per tokens)
                if self.next_frames_tgt=="next":
                    excess_right = max(0, (frame_idx + self.anticip_time + 1) - video_targets.size(0))
                    ends = min(video_targets.size(0), frame_idx + self.anticip_time + 1)
                    if self.ctx_pooling=="local":
                        start_offset = frame_idx  % self.anticip_time
                        starts = max(start_offset, frame_idx - self.ctx_length + (2*self.anticip_time))
                        print(f"[DATASET] next_frames_tgt starts: {starts}, ends: {ends}")
                    elif self.ctx_pooling=="global":
                        starts = frame_idx + self.anticip_time
                        print(f"[DATASET] next_frames_tgt starts: {starts}, ends: {ends}")
                    else:
                        raise ValueError("next_frames_tgt not found")
                    
                    next_frames_tgt = video_targets[starts: ends: self.anticip_time].to(self.device)
                    print(f"[DATASET] next_frames_tgt: {next_frames_tgt.size()}")
                    if starts < ends:
                        print(f"[DATASET] next_frames_tgt: {torch.arange(starts, ends, self.anticip_time)}")

                    # We keep all the tokens for the training with supra
                    # if self.ctx_pooling=="local":
                    #     next_frames_tgt = next_frames_tgt[-self.num_ctx_tokens:]
                    #     print(f"[DATASET] next_frames_tgt (trimmed to num_ctx_tokens): {next_frames_tgt.size()}")

                    if excess_right > 0:
                        # only add 1 eos token since we use anticip window size
                        next_frames_tgt = torch.cat((next_frames_tgt, self.eos[:1].long()), 0)
                        print(f"[DATASET] EOS padding for next_frames_tgt: {excess_right}")
                    missing = self.num_ctx_tokens - next_frames_tgt.size(0)
                    if missing > 0:
                        next_frames_tgt = torch.cat(( -self.ones[:missing].long(), next_frames_tgt), 0)
                    data_now['next_frames_tgt'] = next_frames_tgt.to(self.device).long()
                    print(f"[DATASET] next_frames_tgt: {next_frames_tgt.size()}")

                    assert next_frames_tgt.size(0) == self.num_ctx_tokens, f"next_frames_tgt size not equal to num_ctx_tokens: {next_frames_tgt.size(0)}"

                elif self.next_frames_tgt=="future":
                    # FUTURE FRAMES TARGETS (SHIFTED BY WINDOW SIZE FROM CURRENT FRAME)
                    starts = frame_idx + self.anticip_time
                    ends = min(video_targets.size(0), starts + self.max_anticip_sec)
                    print(f"[DATASET] future_frames_tgt starts: {starts} ends: {ends}, anticip_time: {self.anticip_time}")
                    future_frames_tgt = video_targets[starts: ends: self.anticip_time].to(self.device)
                    if starts < ends:
                        print(f"[DATASET] future_frames_tgt: {torch.arange(starts, ends, self.anticip_time)}")
                    missing = self.num_future_tokens - future_frames_tgt.size(0)
                    if missing > 0:
                        future_frames_tgt = torch.cat((future_frames_tgt, self.eos[:missing].long()), 0)
                        print(f"[DATASET] EOS padding for future_frames_tgt: {missing}")
                    data_now['future_frames_tgt'] = future_frames_tgt.to(self.device).long()
                    print(f"[DATASET] future_frames_tgt: {future_frames_tgt.size()}")

                    assert future_frames_tgt.size(0) == self.num_future_tokens, f"future_frames_tgt size not equal to num_future_tokens: {future_frames_tgt.size(0)}"
                
                else:
                    raise ValueError("next_frames_tgt not found")
            
            # ----------------- INFERENCE LABELS -----------------
            else:

                # FUTURE FRAMES TARGETS (SHIFTED BY WINDOW SIZE FROM CURRENT FRAME)
                starts = frame_idx + self.anticip_time
                ends = min(video_targets.size(0), starts + self.max_anticip_sec)
                print(f"[DATASET] future_frames_tgt starts: {starts} ends: {ends}, anticip_time: {self.anticip_time}")
                future_frames_tgt = video_targets[starts: ends: self.anticip_time].to(self.device)
                if starts < ends:
                    print(f"[DATASET] future_frames_tgt: {torch.arange(starts, ends, self.anticip_time)}")
                missing = self.num_future_tokens - future_frames_tgt.size(0)
                if missing > 0:
                    future_frames_tgt = torch.cat((future_frames_tgt, self.eos[:missing].long()), 0)
                    print(f"[DATASET] EOS padding for future_frames_tgt: {missing}")
                data_now['future_frames_tgt'] = future_frames_tgt.to(self.device).long()
                print(f"[DATASET] future_frames_tgt: {future_frames_tgt.size()}")

                assert future_frames_tgt.size(0) == self.num_future_tokens, f"future_frames_tgt size not equal to num_future_tokens: {future_frames_tgt.size(0)}"
            
            print(f"\n")

            return data_now
        
        def __getitem__(self, idx):
            idx = self.data_order[idx] # shuffled or not unique frame indices
            data_dict = {}

            video_idx = self.video_idx[idx]
            frame_idx = self.frame_indices[idx]
            # get video idx
            data_dict['uid'] = video_idx
            data_dict['video_idx'] = video_idx

            data_dict['frame_idx'] = frame_idx
            data_dict['video_lengths'] = self.video_lengths[idx]

            data_now = self.get_video_data(video_idx, frame_idx)

            data_dict.update(data_now)

            return data_dict
        
        def __len__(self):
            return (int)(len(self.data_order))

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


def compute_num_eos_values(anticip_time, num_videos):
    """Calculate the number of eos values needed for anticipation time for the last samples
    """
    return anticip_time/2 * (anticip_time + 1) * num_videos


class SelectDataset():
    def __init__(self, dataset_name="cholec80", logger=None):
        self.dataset_name = dataset_name
        self.logger = logger

        if self.dataset_name == "autolaparo21":
            self.frames_fps = 1
            self.dataset_path = f"{abs_path}/R2A2/dataset/{self.dataset_name}/"
            self.data_root = '/nfs/home/yangliu/dataset/AutoLaparo_Task1/'
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
        self.logger.info(f"[DATASET] ALL loaded dataframe in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")
        self.logger.info(f"[DATASET] ALL loaded dataframe: {self.df.shape}")

        # replace the labels to start from 0 to num_classes-1 (currently 1 to num_classes)
        if self.df['class'].min() == 1:
            self.df['class'] = self.df['class'] - 1
            # overwrite the dataframe with the new labels
            with open(self.path_and_label_dir, 'wb') as f:
                pickle.dump(self.df, f)
                print(f"[DATASET] saved dataframe with updated labels: {self.df.shape}")
        else:
            print(f"[DATASET] labels are already starting from 0")
        #-----------------mb added-----------------
        print(f"[DATASET] ALL unique classes: {np.unique(self.df['class'].tolist())}")
        print(f"[DATASET] aLL dataframe (head): {self.df.head()}")
        print(f"[DATASET] ALL dataframe (tail): {self.df.tail()}")
    
    def get_dataframe(self):
        return self.df

def calculate_total_padded_values(video_duration_seconds, fps, future_sampling_rate, future_samples):
    video_frames = video_duration_seconds * fps
    future_clip_count = video_frames // future_sampling_rate
    padded_values_per_clip = future_samples - 1
    total_padded_values = future_clip_count * padded_values_per_clip
    return total_padded_values


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
            #-----------------select arguments-----------------
            self.debug = False
            self.dataset_name = dataset_name                                        # autolaparo21, cholec80, cholect50
            self.num_classes = 7                                                    # AutoLapro: 7, Cholec80: 7
            self.attn_window_size = 20
            self.ctx_length = cfg.ctx_length                                        # 3000 or 1500 frames
            self.num_ctx_tokens = int(cfg.ctx_length/cfg.anticip_time)                 # Compression Tokens: 100 or 50 tokens for curr and next frames
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
            self.curr_frames_tgt = "global" # "global" or "local"
            self.next_frames_tgt = "global" # "global" or "local"
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
            #-----------------end select arguments-----------------
            if self.dataset_name == "autolaparo21":
                self.frames_fps = 1
                self.dataset_path = f"{abs_path}/R2A2/dataset/{self.dataset_name}/"
                self.data_root = '/nfs/home/yangliu/dataset/AutoLaparo_Task1/'
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
            self.init_video_data(video_indices=video_indices)
            self.data_order = list(range(len(self.frame_indices)))

            if self.train_mode == 'train':
                random.shuffle(self.data_order)
                classes = self.df_split['class'].tolist()

                # CUUR AND NEXT CLASS WEIGHTS
                classes_counts_dict = dict(zip(*np.unique(classes, return_counts=True)))
                self.curr_class_weights = self._compute_class_weights(classes_counts_dict)
                self.logger.info(f"[DATASET] curr_class_weights: {self.curr_class_weights}")

                if self.eos_class == 7 and self.eos_classification:
                    num_eos_vals = compute_num_eos_values(self.anticip_time, len(video_indices))
                    classes_counts_dict[self.num_classes] = int(num_eos_vals)
                    self.next_class_weights = self._compute_class_weights(classes_counts_dict)
                    self.logger.info(f"[DATASET] next_class_weights: {self.next_class_weights}")

        def get_data_split(self, dataset_name="cholec80", split="train", video_indices=None):
            """
                autolapro21: 0-14, 14-21
                cholec80: 0-40, 40-80
            """
            df = self.df[self.df.video_idx.isin(video_indices)]
            assert len(df.video_idx.unique()) == len(video_indices), "video indices not found"
            df = df.reset_index() if not split == "train" else df
            return df

        def init_video_data(self, video_indices=None):
            self.videos = []
            self.labels = []
            self.next_segments_class = []
            self.heatmap = []
            self.eos_video_targets = []
            # mb added
            self.target_feats = []
            new_id = 0  
            video_stats = {
                "video_lengths": [],
            }
            
            for video_indx in video_indices:
                df_video = self.df_split[self.df_split.video_idx==video_indx]
                df_video = df_video[df_video.frame%self.frames_fps==0].reset_index(drop = True)
                self.logger.info(f"[DATASET] Sampling video_idx: {video_indx} at {self.frames_fps} fps to {len(df_video)} frames")
                video_length = len(df_video)
                video_stats['video_lengths'].append(video_length)
                if video_length==0:
                    print(f'[SKIP] video_idx:{video_indx} video_length:{video_length}') 
                    continue
                path_list = df_video.image_path.tolist()
                label_list = df_video['class'].tolist()

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
                label_list = torch.Tensor(label_list).long()
                self.logger.info(f"[DATASET] video {video_indx} label_list: {label_list.size()}")
                
                new_id += 1
                num=0
                # TODO: challenge this loop
                for i in range(0, video_length):
                    # maping between frames and video ids
                    self.frame_indices.append(i) # [0, 1, 2, 3, 0, 1, 0, 1, 2, ...]
                    self.video_idx.append(len(self.videos))
                    self.video_lengths.append(len(df_video))

                with torch.no_grad():
                    self.labels.append(label_list)
                    with open(self.extracted_feats_path + f"video_{video_indx}.pkl", 'rb') as f:
                        video = pickle.load(f)
                    video = torch.from_numpy(video).float().to(self.device)
                    self.videos.append(video)

            print(f"[DATASET] label_list sample: {self.labels[-1]}")
            self.logger.info(f"[DATASET] number of {self.train_mode} videos: {len(self.videos)}")
            self.logger.info(f"[DATASET] avg video length: {np.mean(video_stats['video_lengths']) / 60:.2f} minutes")

            # PADDING VALUES based on video dims
            self.zeros = torch.zeros([video.size(0), self.ctx_length, video.size(2), video.size(3)]).float().to(self.device)
            self.ones = torch.ones([self.ctx_length,]).to(self.device).long()

            if self.eos_class== -1:
                self.eos = - torch.ones([self.ctx_length,]).to(self.device).long()
            elif self.eos_class== 7:
                self.eos = torch.full((self.max_anticip_sec,), self.eos_class).long().to(self.device)
            else:
                raise ValueError("eos_padding not found")
            
        def _compute_class_weights(self, class_counts):
            total_samples = sum(class_counts.values())
            num_classes = len(class_counts)
            class_weights = torch.zeros(num_classes)
            for cls_id, count in class_counts.items():
                class_weights[cls_id] = total_samples / (num_classes * count)
            return class_weights

        
        def get_next_label(self, label_list, num_next_labels, end_class=7):
            """Next labels for each frame in the video
            Returns: a tensor of size (video_length, num_next_labels)
            """
            next_labels_tensor = - torch.ones((len(label_list), num_next_labels)).long()
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
            for i in range(len(label_list) - 1):
                next_labels_tensor[i] = find_next_classes(i, label_list, num_next_labels, end_class)
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
            video = self.videos[video_idx]
            video_targets = self.labels[video_idx]
            if self.eos_regression:
                eos_values = self.eos_video_targets[video_idx]
                print(f"[DATASET] eos_values: {eos_values.shape}")
            # video_next_seg_targets = self.next_segments_class[video_idx]
            print(f"[DATASET] video_idx: {video_idx} (t={frame_idx}) with shape: {video.size()}")
            print(f"[DATASET] video targets: {video_targets.size()}")
            # print(f"[DATASET] full video_next_seg_targets: {video_next_seg_targets.size()}")

            # VIDEO FEED
            # NOTE: index 0 is included in the frame id so it is not necessary to add 1 !!!
            starts = max(0, frame_idx - self.ctx_length) # context length
            video_now = video[:, starts: frame_idx, :, :]
            missing = self.ctx_length - video_now.size(1)
            if missing > 0:
                video_now = torch.cat((self.zeros[:, :missing, :, :], video_now), 1)
            print(f"[DATASET] video (t={frame_idx}) : {video_now.size()}")

            # ----------------- DATA DICTIONARY -----------------
            data_now = OrderedDict()
            data_now['video_idx'] = video_idx
            data_now['frame_idx'] = frame_idx
            data_now['video_lengths'] = self.video_lengths[video_idx]
            data_now['video'] = video_now.to(self.device)

            # ----------------- TRAINING LABELS -----------------
            
            # CURRENT FRAMES TARGETS
            if self.curr_frames_tgt == "global":
                ends = frame_idx + 1
                remainder = frame_idx  % self.anticip_time
                # the target is the last frame in the attn window
                starts = max(remainder, frame_idx - self.ctx_length + self.anticip_time)
            elif self.curr_frames_tgt == "local":
                # self.anticip_time = 1
                self.num_ctx_tokens = self.attn_window_size
                ends = frame_idx +1
                starts = max(0, frame_idx - self.attn_window_size)
            else:
                raise ValueError("curr_frames_tgt not found")
            print(f"[DATASET] curr_frames_tgt starts: {starts} ends: {ends}")
            curr_frames_tgt = video_targets[starts: ends: self.anticip_time].to(self.device)
            print(f"[DATASET] curr_frames_tgt: {curr_frames_tgt.size()}")
            print(f"[DATASET] curr_frames_tgt: {torch.arange(starts, ends, self.anticip_time)}")
            missing = self.num_ctx_tokens - curr_frames_tgt.size(0)
            print(f"[DATASET] missing: {missing}")
            if missing > 0:
                curr_frames_tgt = torch.cat(( -self.ones[:missing].long(), curr_frames_tgt), 0)
            data_now['curr_frames_tgt'] = curr_frames_tgt.to(self.device).long()
            print(f"[DATASET] curr_frames_tgt: {curr_frames_tgt.size()}")

            if self.eos_regression:
                curr_eos_values = eos_values[starts: ends: self.anticip_time]
                if missing > 0:
                    curr_eos_values = torch.cat((self.ones[:missing].float(), curr_eos_values), 0)
                data_now['curr_time2eos_tgt'] = curr_eos_values.to(self.device).float()
                print(f"[DATASET] curr_time2eos_tgt: {curr_eos_values.size()}")

            assert curr_frames_tgt.size(0) == self.num_ctx_tokens, f"curr_frames_tgt size not equal to num_ctx_tokens: {curr_frames_tgt.size(0)}"
            
            if self.train_mode == 'train':

                # NEXT FRAMES TARGETS (SHIFTED BY 2x Window size or 2x Frames per tokens)
                if self.next_frames_tgt=="global":
                    excess_right = max(0, (frame_idx + self.anticip_time + 1) - video_targets.size(0))
                    ends = min(video_targets.size(0), frame_idx + self.anticip_time + 1)
                    start_offset = frame_idx  % self.anticip_time
                    starts = max(start_offset, frame_idx - self.ctx_length + (2*self.anticip_time))
                elif self.next_frames_tgt=="local":
                    starts = max(0, frame_idx - self.attn_window_size)
                    ends = frame_idx
                    # self.anticip_time = 1
                    self.num_ctx_tokens = self.attn_window_size
                else:
                    raise ValueError("next_frames_tgt not found")
                print(f"[DATASET] next_frames_tgt starts: {starts}, ends: {ends}")
                next_frames_tgt = video_targets[starts: ends: self.anticip_time].to(self.device)
                print(f"[DATASET] next_frames_tgt: {next_frames_tgt.size()}")
                print(f"[DATASET] next_frames_tgt: {torch.arange(starts, ends, self.anticip_time)}")
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

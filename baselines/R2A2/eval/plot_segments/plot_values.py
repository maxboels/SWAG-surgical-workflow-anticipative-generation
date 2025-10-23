import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import os
import numpy as np
import h5py




def plot_maxpooled_videos(save_path, aspect='auto'):
    """Plot maxpooled videos from h5 files.
    """
    video_files = [f for f in os.listdir(save_path) if f.endswith('.h5')]
    print(f"Plotting maxpooled videos from {len(video_files)} files")
    for file_name in video_files:
        print(f"Plotting maxpooled video {file_name}")
        video_idx = file_name.split('.h5')[0]
        video_path = os.path.join(save_path, file_name)
        with h5py.File(video_path, 'r') as fin:
            data = fin['data'][:]
        # plot the maxpooled video
        plt.figure(figsize=(25,10))
        plt.imshow(data.T, aspect=aspect)
        plt.colorbar(fraction=0.046, pad=0.012)
        plt.tight_layout()
        plt.savefig(f"{save_path}/{video_idx}.png", bbox_inches='tight')
        # plt.show()


def get_smooth_tensor(numpy_tensor, numpy_target, smoothing="median"):
    """Get smooth tensor for each dimension of the outputs tensor. Slice if different targets in the batch.
    Args:
        numpy_tensor (np.ndarray): outputs tensor, shape (B, D)
        numpy_target (np.ndarray): targets tensor, shape (B, 1)
    Returns:
        numpy_tensor (np.ndarray): median tensor for each dimension of the outputs tensor, shape (B, D)
    """
    for target in np.unique(numpy_target):
        mask = (numpy_target == target).squeeze()
        print(f"mask shape: {mask.shape}")
        if smoothing == "median":
            smooth_tensor = np.median(numpy_tensor[mask], axis=0)
        elif smoothing == "mean":
            smooth_tensor = np.mean(numpy_tensor[mask], axis=0)
        
        # repeat the smooth tensor for each frame in the batch and replace the original tensor
        numpy_tensor[mask] = np.repeat(smooth_tensor[np.newaxis, :], repeats=mask.sum(), axis=0)
    return numpy_tensor


def store_append_h5(outputs, target, heatmap, save_path, video_indices, video_lengths, smooth_values=False):
    """Append outputs and its heatmap to video files.
    We save each frame and its heatmap as a new line in the file.

    Args:
        outputs (torch.Tensor): shape (B, T, D)
        heatmap (torch.Tensor): shape (B, T)
        save_path (str): path to directory where files will be saved
        video_indices (torch.Tensor): tensor of video indices in the batch, shape (B, T)
    """
    # assert outputs.shape[1] == heatmap.shape[1], "outputs and heatmap must have the same number of frames"
    scaling_factor = 5
    for video_idx in torch.unique(video_indices):
        output_fpath = os.path.join(save_path, f'video_{video_idx.item()}.h5')
        video_mask = (video_indices == video_idx)
        numpy_tensor = outputs[video_mask, -1, :].detach().cpu().numpy()
        numpy_target = target[video_mask, -1:].detach().cpu().numpy()
        numpy_heatmap = heatmap[video_mask, -1:].detach().cpu().numpy() * scaling_factor
        numpy_heatmap = np.clip(numpy_heatmap, a_min=0, a_max=scaling_factor)

        if smooth_values:
            numpy_tensor = get_smooth_tensor(numpy_tensor, numpy_target, smoothing="median")

        # concatenate outputs and heatmap on the dim dimension
        numpy_tensor = np.concatenate((numpy_tensor, numpy_heatmap), axis=1)
        print(f"Storing outputs of shape {numpy_tensor.shape} to {output_fpath}")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(output_fpath):
            with h5py.File(output_fpath, 'w') as fout:
                fout.create_dataset('data', data=numpy_tensor, chunks=True, maxshape=(None, numpy_tensor.shape[1]))
        # if file length is less than the video length, append the tensor to the file
        else:
            with h5py.File(output_fpath, 'r') as fin:
                file_length = fin['data'].shape[0]
            if file_length < video_lengths[video_mask][0].item():
                with h5py.File(output_fpath, 'a') as fout:
                    if fout['data'].chunks is None:
                        raise TypeError("Only chunked datasets can be resized")
                    fout['data'].resize((fout['data'].shape[0] + numpy_tensor.shape[0]), axis=0)
                    fout['data'][-numpy_tensor.shape[0]:] = numpy_tensor
            else:
                # clean and overwrite the file
                with h5py.File(output_fpath, 'w') as fout:
                    fout.create_dataset('data', data=numpy_tensor, chunks=True, maxshape=(None, numpy_tensor.shape[1]))


def store_training_videos_max(outputs, heatmap, save_path, video_indices, frame_indices, video_lengths):
    """Store outputs and its heatmap to video files.
    We save each frame and its heatmap as a new line in the file.
    However, this function needs to either sort the appended frames based on frame_indices in each video file
    or the frames need to be stored at a fixed pre-allocted position.

    Args:
        outputs (torch.Tensor): shape (B, 1, D)
        heatmap (torch.Tensor): shape (B, T)
    """
    dim = outputs.shape[-1]
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print(f"unique video indices: {torch.unique(video_indices)}")
    for video_idx in torch.unique(video_indices):
        output_fpath = os.path.join(save_path, f'video_{video_idx.item()}.h5')
        print(f"output_fpath: {output_fpath}")
        # select the video length of the current video
        mask = (video_indices == video_idx)
        print(f"mask: {mask}")
        video_length = video_lengths[mask][0].item()
        print(f"video_length: {video_length}")
        # open file if it exists
        if os.path.exists(output_fpath):
            numpy_video = h5py.File(output_fpath, 'r')['data'][:]
        else:
            numpy_video = np.zeros((video_length, dim))
        video_mask = (video_indices == video_idx)
        print(f"video_mask shape: {video_mask.shape}")
        video_frames = frame_indices[video_mask]
        print(f"video_frames: {video_frames}")
        frames = outputs[video_mask, -1, :].detach().cpu().numpy()
        for batch_idx, frame_pos in enumerate(video_frames):
            numpy_video[frame_pos] = frames[batch_idx]
        with h5py.File(output_fpath, 'w') as fout:
            fout.create_dataset('data', data=numpy_video)


def plot_tensors(outputs, data, batch_idx, sampling_rate=10, aspect='auto'):
    """
    Plotting the global tensor and the maxpooled tensor
    (outputs, data)
    """
    # set font size for the entire plot
    plt.rcParams.update({'font.size': 14})
    
    save_path = os.getcwd()+'/plot_tensors/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # if more than one element in batch_idx, make a for loop, add a dim in case the tensor has 0-dim
    # reshape batch_idx into a 1D tensor
    batch_idx = batch_idx.view(-1)
    print(f"Plotting tensors for batch_idx {batch_idx}")
    for idx in batch_idx:

        # Unpacking the data
        video_idx = data['video_idx'][idx].item()
        frame_idx = data['frame_idx'][idx].item()

        print(f"Plotting tensors for video {video_idx} and frame {frame_idx}")

        # Unpacking the data
        enc_out_reduc_past = outputs['enc_out_reduc_past'][idx].detach().cpu().numpy()
        enc_out_reduc_futr = outputs['enc_out_reduc_futr'][idx].detach().cpu().numpy()
        dec_out_reduc_futr = outputs['dec_out_reduc_futr'][idx].detach().cpu().numpy()
        
        # set figure size
        plt.figure(figsize=(25,10))

        # Encoded past
        ax0 = plt.subplot(2,2,(1,2))
        im0 = ax0.imshow(enc_out_reduc_past[::sampling_rate, :].T, aspect=aspect)
        ax0.set_title(f"Past Frames Encoded")

        # # Decoded future
        ax1 = plt.subplot(2,2,3)
        im1 = ax1.imshow(dec_out_reduc_futr.T, aspect=aspect)
        ax1.set_title(f"Future Frames Decoded (Pred)")

        # # Encoded future
        ax2 = plt.subplot(2,2,4)
        im2 = ax2.imshow(enc_out_reduc_futr.T, aspect=aspect)
        ax2.set_title(f"Future Frames Encoded (GT)")

        # Add colorbar to the right of the last plot
        plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.012)

        plt.tight_layout()
        
        plt.savefig(f"{save_path}/video_{video_idx}_frame_{frame_idx}.png", bbox_inches='tight')
        # plt.show()



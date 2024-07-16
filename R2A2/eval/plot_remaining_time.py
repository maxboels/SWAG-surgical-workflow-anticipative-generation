import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation, writers
from matplotlib.patheffects import withStroke
import os
import matplotlib.colors as mcolors


def plot_remaining_time_video(gt_remaining_time, pred_remaining_time, h, num_obs_classes, video_idx=0, epoch=0, dataset="cholec80", save_video=False):
    

    color_scheme = 'plasma'
    shift_colors = False

    fig, axs = plt.subplots(num_obs_classes + 1, 1, figsize=(12, 1.5*(num_obs_classes + 1)), sharex=True)
    plt.subplots_adjust(bottom=0.1, hspace=0.05)
    time_steps = np.arange(gt_remaining_time.shape[0])
    # colors = plt.cm.tab10(np.linspace(0, 1, num_obs_classes + 1))

    # Create color scheme with increased contrast
    if color_scheme == 'plasma':
        cmap = get_color_scheme(color_scheme, num_obs_classes, brightness_factor=1.0, shift_colors=shift_colors)
        colors = cmap(np.linspace(0, 1, num_obs_classes + 1))
    else:
        colors = plt.cm.tab10(np.linspace(0, 1, num_obs_classes + 1))
        cmap = mcolors.ListedColormap(colors)

    # create rtd folder if doesnt exist
    if not os.path.exists(f"./plots/{dataset}/rtd/"):
        os.makedirs(f"./plots/{dataset}/rtd/")

    # if tensor then convert to numpy
    if torch.is_tensor(gt_remaining_time):
        gt_remaining_time = gt_remaining_time.cpu().numpy()
    if torch.is_tensor(pred_remaining_time):
        pred_remaining_time = pred_remaining_time.cpu().numpy()
    # if has 3 dimensions then remove the second one
    if len(gt_remaining_time.shape) == 3:
        gt_remaining_time = gt_remaining_time[:, 0, :]
    if len(pred_remaining_time.shape) == 3:
        pred_remaining_time = pred_remaining_time[:, 0, :]

    y_min, y_max = -0.1, h + 0.5
    gt_lines = []
    pred_lines = []
    vlines = []
    for i in range(num_obs_classes + 1):
        gt_time = gt_remaining_time[:, i]
        pred_time = pred_remaining_time[:, i]
        
        gt_line, = axs[i].plot(time_steps, gt_time, label=f'GT Class {i if i < num_obs_classes else "EOS"}', linewidth=2.5, color=colors[i])
        pred_line, = axs[i].plot(time_steps, pred_time if not save_video else np.full_like(pred_time, np.nan), label=f'Pred Class {i if i < num_obs_classes else "EOS"}', linewidth=2.5, color=colors[i], linestyle='--', alpha=0.7)
        gt_lines.append(gt_line)
        pred_lines.append(pred_line)

        regression_mask = (gt_time > 0) & (gt_time < h)
        axs[i].fill_between(time_steps, y_min, y_max, where=regression_mask, color='lightgray', alpha=0.3)

        active_mask = gt_time == 0
        axs[i].fill_between(time_steps, y_min, y_max, where=active_mask, color=colors[i], alpha=0.3)

        axs[i].set_ylim(y_min, y_max)
        axs[i].set_ylabel("Ant. Time (m)")
        axs[i].legend(loc='upper right')
        axs[i].grid(True, linestyle='--', alpha=0.7)

        vline = axs[i].axvline(x=0, color='r', linestyle='--')
        vlines.append(vline)

        regression_ranges = np.where(np.diff(np.concatenate(([False], regression_mask, [False]))))[0].reshape(-1, 2)
        active_ranges = np.where(np.diff(np.concatenate(([False], active_mask, [False]))))[0].reshape(-1, 2)

        for start, end in regression_ranges:
            axs[i].text((start + end) / 2, y_max, f'<{h} min.', ha='center', va='bottom', fontsize=8, color='gray',
                        path_effects=[withStroke(linewidth=2, foreground='white')])

        for start, end in active_ranges:
            axs[i].text((start + end) / 2, y_max, 'Active', ha='center', va='bottom', fontsize=8, color=colors[i],
                        path_effects=[withStroke(linewidth=2, foreground='white')])

    axs[-1].set_xlabel("Video Time Steps (seconds)")
    
    if not save_video:
        plt.tight_layout()
        output_file = f"./plots/{dataset}/rtd/video{video_idx}_ep{epoch}_phase_rtd.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        print(f"Static figure saved to {output_file}")
    else:
        text_annotations = [ax.text(0.02, 0.95, '', transform=ax.transAxes, verticalalignment='top',
                                    path_effects=[withStroke(linewidth=3, foreground='white')]) for ax in axs]

        def animate(t):
            for vline in vlines:
                vline.set_xdata([t, t])
            
            for i, (gt_line, pred_line, text_ann) in enumerate(zip(gt_lines, pred_lines, text_annotations)):
                gt_y = gt_line.get_ydata()[t]
                pred_y = pred_remaining_time[t, i]
                text_ann.set_text(f'GT: {gt_y:.2f}, Pred: {pred_y:.2f}')
                
                pred_data = pred_line.get_ydata()
                pred_data[:t+1] = pred_remaining_time[:t+1, i]
                pred_data[t+1:] = np.nan
                pred_line.set_ydata(pred_data)
            
            if t % 10 == 0:
                print(f"Video {video_idx} - Rendering frame {t}/{len(time_steps)}")
            
            return vlines + pred_lines + text_annotations

        print("Starting animation rendering...")
        ani = FuncAnimation(fig, animate, frames=np.arange(len(time_steps)), interval=50, blit=True)

        print("Saving animation to video file...")
        Writer = writers['ffmpeg']
        writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)
        output_file = f"./plots/{dataset}/rtd/video{video_idx}_ep{epoch}_phase_rtd.mp4"
        ani.save(output_file, writer=writer)

        plt.close()
        print(f"Video saved to {output_file}")


def get_color_scheme(scheme_name, num_classes, brightness_factor=1.0, shift_colors=True):
    def create_colormap(colors, name='custom_cmap'):
        return mcolors.LinearSegmentedColormap.from_list(name, colors)

    def adjust_brightness(color, factor):
        rgb = np.array(mcolors.to_rgb(color))
        hsv = mcolors.rgb_to_hsv(rgb)
        hsv[2] = np.clip(hsv[2] * factor, 0, 1)
        rgb = mcolors.hsv_to_rgb(hsv)
        return np.clip(rgb, 0, 1)  # Ensure RGB values are within 0-1 range

    schemes = {
        'plasma': plt.cm.plasma,
        'viridis': plt.cm.viridis,
        'inferno': plt.cm.inferno,
        'magma': plt.cm.magma,
        'cividis': plt.cm.cividis,
        'cool': plt.cm.cool,
        'warm': plt.cm.autumn,
        'rainbow': plt.cm.rainbow,
        'terrain': plt.cm.terrain,
        'ocean': plt.cm.ocean,
        'spectral': plt.cm.Spectral,
        'pastel': create_colormap(['#FFB3BA', '#BAFFC9', '#BAE1FF', '#FFFFBA', '#FFDFBA']),
        'neon': create_colormap(['#FF00FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF0000']),
        'earth': create_colormap(['#8B4513', '#228B22', '#4682B4', '#D2691E', '#556B2F']),
        'jewel': create_colormap(['#50C878', '#4B0082', '#E0115F', '#00FFFF', '#FFD700']),
        'retro': create_colormap(['#FF6B6B', '#4ECDC4', '#45B7D1', '#FDCB6E', '#6C5CE7'])
    }

    if scheme_name not in schemes:
        raise ValueError(f"Unknown color scheme: {scheme_name}")

    cmap = schemes[scheme_name]
    if shift_colors:
        color_positions = np.linspace(0, 1, num_classes + 1)  # +1 to include EOS class
    else:
        color_positions = np.linspace(0, 1, num_classes)
    colors = cmap(color_positions)

    # Adjust brightness of all colors
    adjusted_colors = [adjust_brightness(color, brightness_factor) for color in colors]

    # Shift colors and assign the first color to EOS class
    if shift_colors:
        eos_color = adjusted_colors[0]
        adjusted_colors = adjusted_colors[1:] + [eos_color]
    else:
        # Add greengray for EOS class
        greengray = [0.65, 0.70, 0.70]
        eos_color = adjust_brightness(greengray, brightness_factor)
        adjusted_colors.append(np.append(eos_color, 1))  # Add alpha channel


    return mcolors.ListedColormap(adjusted_colors)

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from matplotlib.patheffects import withStroke
import seaborn as sns

def load_data(video_idx, dataset, folder):
    preds_file_name = f"video_frame_preds_{video_idx}_ep1.npy"
    recs_file_name = f"video_frame_rec_{video_idx}_ep1.npy"
    tgt_preds_file_name = f"video_tgts_preds_{video_idx}_ep1.npy"
    tgt_recs_file_name = f"video_tgts_rec_{video_idx}_ep1.npy"

    # Method 1: SKiT - Single Pass
    preds_baseline = np.load(os.path.join(folder, skit_exp_name, preds_file_name))
    recs_baseline = np.load(os.path.join(folder, skit_exp_name, recs_file_name))
    sp = np.concatenate([recs_baseline, preds_baseline], axis=1)

    # Method 2: SuPRA - Auto-Regressive
    preds2 = np.load(os.path.join(folder, supra_exp_name, preds_file_name))
    recs = np.load(os.path.join(folder, supra_exp_name, recs_file_name))
    ar = np.concatenate([recs, preds2], axis=1)

    # Ground Truth
    tgt_preds = np.load(os.path.join(folder, supra_exp_name, tgt_preds_file_name))
    tgt_recs = np.load(os.path.join(folder, supra_exp_name, tgt_recs_file_name))
    gt = np.concatenate([tgt_recs, tgt_preds], axis=1)

    return sp, ar, gt

def ground_truth_remaining_time(phase_labels, h=5, num_classes=7):
    seq_len = phase_labels.shape[0]
    remaining_time = torch.full((seq_len, num_classes + 1), h, device=phase_labels.device, dtype=torch.float32)
    
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
            pre_segment_start = max(0, segment_start - int(h*60))
            
            for i in range(pre_segment_start, segment_start):
                remaining_time[i, phase] = min(remaining_time[i, phase], (segment_start - i) / 60.0)
            remaining_time[segment_start:segment_end+1, phase] = 0.0
    
    # Handle EOS class
    eos_start = seq_len - int(h*60)
    for i in range(eos_start, seq_len):
        remaining_time[i, -1] = (seq_len - i) / 60.0
    
    return remaining_time

def generate_simulated_predictions(gt_remaining_time, noise_level=0.1):
    """
    Generate simulated predictions based on ground truth with some added noise.
    Ensure no NaN values are produced.
    """
    predictions = gt_remaining_time.clone()
    noise = torch.randn_like(predictions) * noise_level * gt_remaining_time.max()
    predictions += noise
    predictions.clamp_(min=0, max=gt_remaining_time.max())
    return predictions


def print_mean_remaining_times(mean_remaining_times):
    num_classes = mean_remaining_times.shape[0]
    print("Mean Remaining Times:")
    print("---------------------")
    for observed_class in range(num_classes):
        print(f"When observing class {observed_class}:")
        for other_class in range(num_classes):
            if other_class != observed_class:
                print(f"  Mean remaining time for class {other_class}: {mean_remaining_times[observed_class, other_class]:.2f}")
        print()


# Main execution
if __name__ == "__main__":
    # Parameters
    dataset = "autolaparo21" # cholec80, autolaprao21
    dataset_short = "c80"
    model_name = "sup"
    num_obs_classes = 7
    h = 18  # Horizon in minutes
    sampling_rate = 5 # from 1fps to 0.2fps
    epoch = 0
    video_last = 18 # 41 or 22
    save_video = False

    # Experiment names
    supra_exp_name = "sup_best_al21_ct144_at60_ls442.txt/local/"
    skit_exp_name = "skit_best_al21_ct144_at60_ls442_eosw1.txt/local/"

    # exp_name = f"{model_name}_best_{dataset_short}_ct144_at60_ls442_eosw.txt/local/"
    folder = f"./supra/best_models/{dataset}/"

    all_gt_remaining_times = []

    for video_idx in range(17, video_last):  # Process videos 1 to 41
        # tgt_preds_file_name = f"video_tgts_rec_{video_idx}_ep1.npy"
        data_path=f"datasets/{dataset}/labels/video_{video_idx}_labels.npy"
        phase_labels = np.load(data_path)

        phase_labels = torch.tensor(phase_labels)
        phase_labels = phase_labels[::sampling_rate]

        gt_remaining_time = ground_truth_remaining_time(phase_labels, h, num_obs_classes)
        all_gt_remaining_times.append(gt_remaining_time.numpy())

        # Generate simulated predictions (you may want to replace this with actual predictions)
        pred_remaining_time = generate_simulated_predictions(gt_remaining_time)

        # Plot remaining time for each video (option to create video animation)
        if True:
            plot_remaining_time_video(gt_remaining_time.numpy(), pred_remaining_time.numpy(), 
                h=h, num_obs_classes=num_obs_classes, video_idx=video_idx, epoch=epoch, dataset=dataset, save_video=save_video)

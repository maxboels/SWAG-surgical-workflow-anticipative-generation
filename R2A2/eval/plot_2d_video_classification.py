import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from matplotlib.patheffects import withStroke
import matplotlib.colors as mcolors
import os

def plot_classification_video(gt_classification, pred_classification,
                        h, num_obs_classes, video_idx, epoch, dataset, save_video=True,
                        x_sampling_rate=5, gif_fps=40, use_scatter=True):
    
    # Configuration
    n_yticks = 3
    # scale plot
    scatter_size = 180 * (18 / h)**2
    color_scheme = 'plasma'
    shift_colors = False

    # Classification task
    gt_classification = gt_classification[::x_sampling_rate, :h]
    pred_classification = pred_classification[::x_sampling_rate, :h]

    # Set figure size and layout
    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 0.05], hspace=0.3)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    cbar_ax = fig.add_subplot(gs[2])

    # Create color scheme with increased contrast
    if color_scheme == 'plasma':
        cmap = get_color_scheme(color_scheme, num_obs_classes, brightness_factor=1.0, shift_colors=shift_colors)
    else:
        colors = plt.cm.tab10(np.linspace(0, 1, num_obs_classes + 1))
        cmap = mcolors.ListedColormap(colors)

    # Ensure output directory exists
    if not os.path.exists(f"./plots/{dataset}/combined/"):
        os.makedirs(f"./plots/{dataset}/combined/")

    y_min, y_max = -0.5, h + 0.5
    
    # Classification task plots
    def plot_classification_data(data, ax):
        if use_scatter:
            x_values = np.repeat(np.arange(data.shape[0]), data.shape[1])
            y_values = np.tile(np.arange(data.shape[1]), data.shape[0]) * h / data.shape[1]
            scatter = ax.scatter(x_values, y_values, c=data.flatten(), cmap=cmap, marker='o', s=scatter_size, vmin=0, vmax=num_obs_classes)
            return scatter
        else:
            im = ax.imshow(data.T, aspect='auto', cmap=cmap, interpolation='nearest', vmin=0, vmax=num_obs_classes, 
                        extent=[0, data.shape[0], 0, h], origin='lower')
            return im

    scatter_pred_classification = plot_classification_data(pred_classification, ax1)
    scatter_gt_classification = plot_classification_data(gt_classification, ax2)

    for ax in [ax1, ax2]:
        ax.set_ylim(0, h)
        ax.set_yticks(np.linspace(0, h, n_yticks))
        ax.set_yticklabels([f'{y:.0f}' for y in np.linspace(0, h, n_yticks)])
        ax.set_ylabel("Future (m)")
        ax.grid(True, linestyle='--', alpha=0.7)

    ax1.set_title('Single-Pass Decoding')
    ax2.set_title('Ground Truth Classes')
    ax2.set_xlabel("Video Time Steps (seconds)")
    
    # Add colorbar with adjusted size
    cbar = plt.colorbar(scatter_gt_classification, cax=cbar_ax, orientation='horizontal', aspect=25)
    cbar.set_label('Classes')

    plt.tight_layout()
    
    if not save_video:
        plt.tight_layout()
        output_file = f"./plots/{dataset}/combined/video{video_idx}_ep{epoch}_h{h}_sr{x_sampling_rate}_combined.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        print(f"Static figure saved to {output_file}")
    else:
        def animate(t):
            if use_scatter:
                x_values = np.repeat(np.arange(t+1), pred_classification.shape[1])
                y_values = np.tile(np.arange(pred_classification.shape[1]), t+1) * h / pred_classification.shape[1]

                scatter_pred_classification.set_offsets(np.column_stack((x_values, y_values)))
                scatter_gt_classification.set_offsets(np.column_stack((x_values, y_values)))

                scatter_pred_classification.set_array(pred_classification[:t+1].flatten())
                scatter_gt_classification.set_array(gt_classification[:t+1].flatten())
            else:
                scatter_pred_classification.set_extent([0, t+1, 0, h])
                scatter_gt_classification.set_extent([0, t+1, 0, h])
                
                scatter_pred_classification.set_data(pred_classification[:t+1].T)
                scatter_gt_classification.set_data(gt_classification[:t+1].T)
            
            if t % 10 == 0:
                print(f"Video {video_idx} - Rendering frame {t}/{len(time_steps)}")
            
            return [scatter_pred_classification, scatter_gt_classification]

        print("Starting animation rendering...")
        ani = FuncAnimation(fig, animate, frames=np.arange(len(time_steps)), interval=50, blit=True)

        print("Saving animation to video file...")
        Writer = writers['ffmpeg']
        writer = Writer(fps=gif_fps, metadata=dict(artist='Me'), bitrate=1800)
        output_file = f"./plots/{dataset}/combined/video{video_idx}_ep{epoch}_h{h}_sr{x_sampling_rate}_combined.mp4"
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


if __name__ == "__main__":
    # Parameters
    dataset = "autolaparo21"
    dataset_short = "al21"
    model_name = "sup"
    num_obs_classes = 7
    h = 5  # Horizon in minutes
    sampling_rate = 5
    epoch = 0
    video_idx = 21
    save_video = False
    gif_fps = 40
    use_scatter = True

    folder = f"./supra/best_models/{dataset}/"

    # Experiment names
    supra_exp_name = "sup_best_al21_ct144_at60_ls442.txt/local/"
    skit_exp_name = "skit_best_al21_ct144_at60_ls442_eosw1.txt/local/"

    # Load classification data
    sp_data, ar_data, gt_data = load_data(video_idx, dataset, folder)

    pred_classification = sp_data
    gt_classification = gt_data

    print(f"GT Classification: {gt_classification.shape}")
    print(f"Pred Classification: {pred_classification.shape}")

    # Plot combined video
    plot_classification_video(gt_classification, pred_classification,
                        h=h, num_obs_classes=num_obs_classes, video_idx=video_idx, epoch=epoch,
                        dataset=dataset, save_video=save_video,
                        x_sampling_rate=sampling_rate, gif_fps=gif_fps,
                        use_scatter=use_scatter)
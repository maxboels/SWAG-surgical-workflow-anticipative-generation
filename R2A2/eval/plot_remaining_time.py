import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from matplotlib.patheffects import withStroke

from loss_fn.mae import anticipation_mae

def plot_remaining_time_video(gt_remaining_time, pred_remaining_time, h, num_obs_classes, video_idx=0, dataset="Cholec80", save_video=True):
    fig, axs = plt.subplots(num_obs_classes + 1, 1, figsize=(12, 1.5*(num_obs_classes + 1)), sharex=True)
    plt.subplots_adjust(bottom=0.1, hspace=0.05)
    time_steps = np.arange(gt_remaining_time.shape[0])
    colors = plt.cm.tab10(np.linspace(0, 1, num_obs_classes + 1))

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
        output_file = f"./plots/{dataset}/rtd/remaining_time_gt_pred_static_{video_idx}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        print(f"Static figure saved to {output_file}")
    else:
        text_annotations = [ax.text(0.02, 0.95, '', transform=ax.transAxes, verticalalignment='top',
                                    path_effects=[withStroke(linewidth=3, foreground='white')]) for ax in axs]

        def animate(t):
            for vline in vlines:
                vline.set_xdata([t, t])  # Set as a list to avoid deprecation warning
            
            for i, (gt_line, pred_line, text_ann) in enumerate(zip(gt_lines, pred_lines, text_annotations)):
                gt_y = gt_line.get_ydata()[t]
                pred_y = pred_remaining_time[t, i]
                text_ann.set_text(f'GT: {gt_y:.2f}, Pred: {pred_y:.2f}')
                
                # Update prediction line to show only up to current time step
                pred_data = pred_line.get_ydata()
                pred_data[:t+1] = pred_remaining_time[:t+1, i]
                pred_data[t+1:] = np.nan
                pred_line.set_ydata(pred_data)
            
            if t % 10 == 0:  # Print progress every 10 frames
                print(f"Rendering frame {t}/{len(time_steps)}")
            
            return vlines + pred_lines + text_annotations

        print("Starting animation rendering...")
        ani = FuncAnimation(fig, animate, frames=np.arange(len(time_steps)), interval=50, blit=True)

        # Save the animation as a video file
        print("Saving animation to video file...")
        Writer = writers['ffmpeg']
        writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)
        output_file = f"./plots/{dataset}/rtd/remaining_time_gt_pred_video_{video_idx}.mp4"
        ani.save(output_file, writer=writer)

        plt.close()
        print(f"Video saved to {output_file}")


def ground_truth_remaining_time(phase_labels, h=5, num_classes=7):
    seq_len = phase_labels.shape[0]
    remaining_time = torch.full((seq_len, num_classes + 1), h, device=phase_labels.device, dtype=torch.float32)
    
    for phase in range(num_classes):
        phase_indices = torch.where(phase_labels == phase)[0]
        if len(phase_indices) == 0:
            continue
        phase_start = phase_indices[0]
        phase_end = phase_indices[-1]

        pre_phase_start = max(0, phase_start - int(h*60))

        for i in range(pre_phase_start, phase_start):
            remaining_time[i, phase] = (phase_start - i) / 60.0
        remaining_time[phase_start:phase_end+1, phase] = 0.0
    
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


# Main execution
if __name__ == "__main__":
    # Parameters
    dataset = "Cholec80"
    dataset_short = "c80"
    model_name = "sup"
    num_obs_classes = 7
    h = 3
    sampling_rate = 5 # from 1fps to 0.2fps
    save_video = False

    mae_metric_18 = anticipation_mae(h=18*60)  # 18 minutes in seconds
    mae_metric_5 = anticipation_mae(h=5*60)   # 5 minutes in seconds
    mae_metric_3 = anticipation_mae(h=3*60)   # 3 minutes in seconds
    mae_metric_2 = anticipation_mae(h=2*60)   # 2 minutes in seconds

    all_videos_wMAE_18 = []
    all_videos_inMAE_18 = []
    all_videos_pMAE_18 = []
    all_videos_eMAE_18 = []

    all_videos_wMAE_5 = []
    all_videos_inMAE_5 = []
    all_videos_pMAE_5 = []
    all_videos_eMAE_5 = []

    all_videos_wMAE_3 = []
    all_videos_inMAE_3 = []
    all_videos_pMAE_3 = []
    all_videos_eMAE_3 = []

    all_videos_wMAE_2 = []
    all_videos_inMAE_2 = []
    all_videos_pMAE_2 = []
    all_videos_eMAE_2 = []



    exp_name = f"{model_name}_best_{dataset_short}_ct144_at60_ls442_eosw.txt/local/"
    folder = f"./supra/best_models/{dataset}/"

    for video_idx in np.arange(41, 42):
        tgt_preds_file_name = f"video_tgts_rec_{video_idx}_ep1.npy"
        phase_labels = np.load(folder + exp_name + tgt_preds_file_name)

        phase_labels = torch.tensor(phase_labels)
        print("Phase_labels shape:", phase_labels.shape)

        phase_labels = phase_labels[::sampling_rate]
        print("Phase_labels shape after downsampling:", phase_labels.shape)

        gt_remaining_time = ground_truth_remaining_time(phase_labels, h, num_obs_classes)
        print("Ground truth remaining time shape:", gt_remaining_time.shape)

        # Generate simulated predictions
        pred_remaining_time = generate_simulated_predictions(gt_remaining_time)
        print("Predicted remaining time shape:", pred_remaining_time.shape)

        # to gpu if available
        if torch.cuda.is_available():
            gt_remaining_time = gt_remaining_time.to('cuda')
            pred_remaining_time = pred_remaining_time.to('cuda')
        
        print(f"Video {video_idx} - Remaining Time Prediction Evaluation")
        print("-----------------------------------------------------------")
        # Compute metrics
        for horizon in [18, 5, 3, 2]:
            mae_metric = locals()[f'mae_metric_{horizon}']
            wMAE, inMAE, pMAE, eMAE = mae_metric(pred_remaining_time.unsqueeze(1), gt_remaining_time.unsqueeze(1))
            
            # Store metrics
            for metric, value in zip(['wMAE', 'inMAE', 'pMAE', 'eMAE'], [wMAE, inMAE, pMAE, eMAE]):
                locals()[f'all_videos_{metric}_{horizon}'].append(value.item())

        # Print metrics
        for horizon in [18, 5, 3, 2]:
            for metric in ['wMAE', 'inMAE', 'pMAE', 'eMAE']:
                print(f"{metric} ({horizon} min): {np.mean(locals()[f'all_videos_{metric}_{horizon}']):.2f}")

        # to numpy
        gt_remaining_time = gt_remaining_time.cpu().numpy()
        pred_remaining_time = pred_remaining_time.cpu().numpy()

        plot_remaining_time_video(gt_remaining_time, pred_remaining_time, h, num_obs_classes, video_idx, dataset, 
            save_video=save_video)


                        
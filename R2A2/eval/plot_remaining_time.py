import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation, writers
from matplotlib.patheffects import withStroke
import os

def plot_remaining_time_video(gt_remaining_time, pred_remaining_time, h, num_obs_classes, video_idx=0, epoch=0, dataset="cholec80", save_video=False):
    
    fig, axs = plt.subplots(num_obs_classes + 1, 1, figsize=(12, 1.5*(num_obs_classes + 1)), sharex=True)
    plt.subplots_adjust(bottom=0.1, hspace=0.05)
    time_steps = np.arange(gt_remaining_time.shape[0])
    colors = plt.cm.tab10(np.linspace(0, 1, num_obs_classes + 1))

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

    y_min, y_max = -0.5, h + 0.5
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
        # plt.show()
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

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def aggregate_video_data(gt_remaining_times, h, num_obs_classes):
    num_classes = num_obs_classes + 1  # Include EOS class
    total_occurrences = np.zeros((num_classes, num_classes))
    total_minutes = np.zeros((num_classes, num_classes))

    for gt_remaining_time in gt_remaining_times:
        for t in range(gt_remaining_time.shape[0]):
            observed_class = np.argmin(gt_remaining_time[t, :])
            
            for target_class in range(num_classes):
                total_occurrences[observed_class, target_class] += 1
                total_minutes[observed_class, target_class] += gt_remaining_time[t, target_class]

    # Compute mean remaining times
    mean_remaining_times = total_minutes / (total_occurrences + 1e-10)  # Add small value to avoid division by zero

    # Convert to probabilities of class occurance (P(T_i <= h | X_{1:t}))
    conditional_probs = 1 - np.clip(mean_remaining_times / h, 0, 1)

    return conditional_probs

def plot_conditional_probabilities_heatmap(conditional_probs, h, dataset):
    num_classes = conditional_probs.shape[0]
    
    plt.figure(figsize=(12, 10))
    
    cmap = 'YlOrRd'
    
    sns.heatmap(conditional_probs, 
                annot=True, 
                fmt=".2f", 
                cmap=cmap, 
                vmin=0, 
                vmax=1, 
                cbar_kws={'label': f'P(Target class occurs within {h} min | Observed class)'})
    
    plt.title(f"Conditional Class Probabilities - {dataset} Training Set")
    plt.xlabel("Target Class")
    plt.ylabel("Observed Class")
    
    plt.xticks(np.arange(num_classes) + 0.5, range(num_classes))
    plt.yticks(np.arange(num_classes) + 0.5, range(num_classes))
    
    plt.tight_layout()
    
    output_file = f"./plots/{dataset}/rtd/conditional_probabilities_heatmap.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Conditional probabilities heatmap saved to {output_file}")
    
    plt.show()
# --------------------

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from matplotlib.patheffects import withStroke
import seaborn as sns

from loss_fn.mae import anticipation_mae

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from matplotlib.patheffects import withStroke

def plot_class_probabilities_video(gt_remaining_time, pred_remaining_time, h, num_obs_classes, video_idx=0, dataset="cholec80", save_video=True):
    fig, axs = plt.subplots(num_obs_classes + 1, 1, figsize=(12, 1.5*(num_obs_classes + 1)), sharex=True)
    plt.subplots_adjust(bottom=0.1, hspace=0.05)
    time_steps = np.arange(gt_remaining_time.shape[0])
    colors = plt.cm.tab10(np.linspace(0, 1, num_obs_classes + 1))

    def remaining_time_to_probability(remaining_time):
        prob = 1 - (remaining_time / h)
        prob = np.clip(prob, 0, 1)
        return prob

    gt_prob = remaining_time_to_probability(gt_remaining_time)
    pred_prob = remaining_time_to_probability(pred_remaining_time)

    y_min, y_max = 0, 1
    gt_lines = []
    pred_lines = []
    vlines = []
    for i in range(num_obs_classes + 1):
        gt_line, = axs[i].plot(time_steps, gt_prob[:, i], label=f'GT Class {i if i < num_obs_classes else "EOS"}', linewidth=2.5, color=colors[i])
        pred_line, = axs[i].plot(time_steps, pred_prob[:, i] if not save_video else np.full_like(pred_prob[:, i], np.nan), label=f'Pred Class {i if i < num_obs_classes else "EOS"}', linewidth=2.5, color=colors[i], linestyle='--', alpha=0.7)
        gt_lines.append(gt_line)
        pred_lines.append(pred_line)

        # Add background formatting
        regression_mask = (gt_remaining_time[:, i] > 0) & (gt_remaining_time[:, i] < h)
        axs[i].fill_between(time_steps, y_min, y_max, where=regression_mask, color='lightgray', alpha=0.3)

        active_mask = gt_remaining_time[:, i] == 0
        axs[i].fill_between(time_steps, y_min, y_max, where=active_mask, color=colors[i], alpha=0.3)

        axs[i].set_ylim(y_min, y_max)
        axs[i].set_ylabel(r'$P(T_i \leq %d \mid X_{1:t})$' % h, fontsize=10)
        axs[i].legend(loc='upper right')
        axs[i].grid(True, linestyle='--', alpha=0.7)

        vline = axs[i].axvline(x=0, color='r', linestyle='--')
        vlines.append(vline)

        # Add text annotations for regression and active ranges
        regression_ranges = np.where(np.diff(np.concatenate(([False], regression_mask, [False]))))[0].reshape(-1, 2)
        active_ranges = np.where(np.diff(np.concatenate(([False], active_mask, [False]))))[0].reshape(-1, 2)

        for start, end in regression_ranges:
            axs[i].text((start + end) / 2, y_max, f'<{h} min.', ha='center', va='bottom', fontsize=8, color='gray',
                        path_effects=[withStroke(linewidth=2, foreground='white')])

        for start, end in active_ranges:
            axs[i].text((start + end) / 2, y_max, 'Observed', ha='center', va='bottom', fontsize=8, color=colors[i],
                        path_effects=[withStroke(linewidth=2, foreground='white')])

    axs[-1].set_xlabel("Video Time Steps (seconds)")
    
    # Center the text annotation explaining the probability
    fig.text(0.5, 0.99, f"Probability of observing class i within the next {h} minutes,\n"
                        f"given the observed sequence of frames up to time t", 
             va='top', ha='center', fontsize=10, bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    if not save_video:
        plt.tight_layout()
        output_file = f"./plots/{dataset}/rtd/class_probabilities_gt_pred_static_{video_idx}.png"
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
                gt_y = gt_prob[t, i]
                pred_y = pred_prob[t, i]
                text_ann.set_text(f'GT: {gt_y:.2f}, Pred: {pred_y:.2f}')
                
                pred_data = pred_line.get_ydata()
                pred_data[:t+1] = pred_prob[:t+1, i]
                pred_data[t+1:] = np.nan
                pred_line.set_ydata(pred_data)
            
            if t % 10 == 0:
                print(f"Rendering frame {t}/{len(time_steps)}")
            
            return vlines + pred_lines + text_annotations

        print("Starting animation rendering...")
        ani = FuncAnimation(fig, animate, frames=np.arange(len(time_steps)), interval=50, blit=True)

        print("Saving animation to video file...")
        Writer = writers['ffmpeg']
        writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)
        output_file = f"./plots/{dataset}/rtd/class_probabilities_gt_pred_video_{video_idx}.mp4"
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

def calculate_mean_remaining_times(gt_remaining_time):
    num_classes = gt_remaining_time.shape[1] - 1  # Excluding EOS class
    mean_remaining_times = np.zeros((num_classes, num_classes))
    counts = np.zeros((num_classes, num_classes))

    for t in range(gt_remaining_time.shape[0]):
        observed_class = np.argmin(gt_remaining_time[t, :num_classes])
        
        for other_class in range(num_classes):
            if other_class != observed_class:
                mean_remaining_times[observed_class, other_class] += gt_remaining_time[t, other_class]
                counts[observed_class, other_class] += 1

    # Avoid division by zero
    counts = np.maximum(counts, 1)
    mean_remaining_times /= counts

    return mean_remaining_times

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

def plot_mean_remaining_times_heatmap(mean_remaining_times, video_idx, dataset):
    num_classes = mean_remaining_times.shape[0]
    
    # Create a mask to hide the diagonal (which is always 0)
    mask = np.eye(num_classes, dtype=bool)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(mean_remaining_times, annot=True, fmt=".2f", cmap="YlOrRd", mask=mask)
    
    plt.title(f"Mean Remaining Times Heatmap - {dataset} Video {video_idx}")
    plt.xlabel("Target Class")
    plt.ylabel("Observed Class")
    
    # Set tick labels
    plt.xticks(np.arange(num_classes) + 0.5, range(num_classes))
    plt.yticks(np.arange(num_classes) + 0.5, range(num_classes))
    
    plt.tight_layout()
    
    # Save the plot
    output_file = f"./plots/{dataset}/rtd/mean_remaining_times_heatmap_{video_idx}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to {output_file}")
    
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_mean_probabilities_heatmap(mean_probabilities, h, video_idx, dataset):
    num_classes = mean_probabilities.shape[0]
    
    # Create a mask to hide the diagonal
    mask = np.eye(num_classes, dtype=bool)
    
    plt.figure(figsize=(12, 10))
    
    # Use a custom colormap for better contrast
    cmap = plt.cm.get_cmap('RdYlBu_r')
    
    # Adjust the color scale to improve contrast
    vmin = max(mean_probabilities.min(), 0.5)  # Set minimum to 0.5 or the actual minimum if higher
    vmax = 1.0
    
    sns.heatmap(mean_probabilities, 
                annot=True, 
                fmt=".2f", 
                cmap=cmap, 
                mask=mask, 
                vmin=vmin, 
                vmax=vmax, 
                cbar_kws={'label': f'Probability of observing within {h} minutes'})
    
    plt.title(f"Mean Probabilities Heatmap - {dataset} Video {video_idx}")
    plt.xlabel("Target Class")
    plt.ylabel("Observed Class")
    
    # Set tick labels
    plt.xticks(np.arange(num_classes) + 0.5, range(num_classes))
    plt.yticks(np.arange(num_classes) + 0.5, range(num_classes))
    
    plt.tight_layout()
    
    # Save the plot
    output_file = f"./plots/{dataset}/rtd/mean_probabilities_heatmap_{video_idx}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Probability heatmap with improved contrast saved to {output_file}")
    
    plt.show()

def calculate_mean_probabilities(gt_remaining_time, h):
    num_classes = gt_remaining_time.shape[1] - 1  # Excluding EOS class
    mean_probabilities = np.zeros((num_classes, num_classes))
    counts = np.zeros((num_classes, num_classes))

    for t in range(gt_remaining_time.shape[0]):
        observed_class = np.argmin(gt_remaining_time[t, :num_classes])
        
        for other_class in range(num_classes):
            if other_class != observed_class:
                probability = 1 - (gt_remaining_time[t, other_class] / h)
                probability = np.clip(probability, 0, 1)
                mean_probabilities[observed_class, other_class] += probability
                counts[observed_class, other_class] += 1

    # Avoid division by zero
    counts = np.maximum(counts, 1)
    mean_probabilities /= counts

    return mean_probabilities

def print_mean_probabilities(mean_probabilities):
    num_classes = mean_probabilities.shape[0]
    print("Mean Probabilities:")
    print("-------------------")
    for observed_class in range(num_classes):
        print(f"When observing class {observed_class}:")
        for other_class in range(num_classes):
            if other_class != observed_class:
                print(f"  Mean probability of observing class {other_class}: {mean_probabilities[observed_class, other_class]:.2f}")
        print()


# Main execution
if __name__ == "__main__":
    # Parameters
    dataset = "autolaparo21" # cholec80, autolaprao21
    dataset_short = "c80"
    model_name = "sup"
    num_obs_classes = 7
    h = 5  # Horizon in minutes
    sampling_rate = 5 # from 1fps to 0.2fps
    epoch = 0
    video_last = 15 # 41 or 22
    save_video = True

    # exp_name = f"{model_name}_best_{dataset_short}_ct144_at60_ls442_eosw.txt/local/"
    folder = f"./supra/best_models/{dataset}/"

    all_gt_remaining_times = []

    horizons = [18, 5, 3, 2]
    metrics = ['wMAE', 'inMAE', 'pMAE', 'eMAE']

    for horizon in horizons:
        locals()[f'mae_metric_{horizon}'] = anticipation_mae(h=horizon*60)  # Convert horizon to seconds

    for horizon in horizons:
        for metric in metrics:
            locals()[f'all_videos_{metric}_{horizon}'] = []

    for video_idx in range(1, video_last):  # Process videos 1 to 41
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


    # Aggregate data and create conditional probabilities heatmap
    conditional_probs = aggregate_video_data(all_gt_remaining_times, h, num_obs_classes)

    # save file
    np.save(f"./plots/{dataset}/rtd/rem_time_{h}_cond_probs_train.npy", conditional_probs)

    plot_conditional_probabilities_heatmap(conditional_probs, h, dataset)

    print("Analysis complete. Check the 'plots' directory for all generated visualizations.")
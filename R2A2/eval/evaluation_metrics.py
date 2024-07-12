import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from scipy.stats import entropy
import matplotlib.pyplot as plt

import numpy as np
from sklearn.metrics import f1_score

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from scipy.stats import entropy
import matplotlib.pyplot as plt

import json
import os
import numpy as np


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

def compute_transition_times(sequence):
    transition_times = {}
    prev_class = sequence[0] # current frame recognition class
    for i in range(1, len(sequence)):
        if sequence[i] != prev_class:
            transition_times[sequence[i]] = i
            prev_class = sequence[i]
    return transition_times

def compute_rmse_transition_times(targets, predictions, max_duration=18, min_duration=0):
    total_mse = 0
    count = 0

    for i in range(len(targets)):
        target_seq = targets[i]
        pred_seq = predictions[i]
        # print(f"target_seq: {target_seq}")
        # print(f"pred_seq:   {pred_seq}")

        
        target_transitions = compute_transition_times(target_seq)
        pred_transitions = compute_transition_times(pred_seq)
        # print(f"")
        # print

        # Filter transitions within max_duration
        target_transitions = {c: t for c, t in target_transitions.items() if t <= max_duration}
        pred_transitions = {c: t for c, t in pred_transitions.items() if t <= max_duration}
        
        all_classes = set(target_transitions.keys()).union(pred_transitions.keys())
        
        for c in all_classes:
            target_time = target_transitions.get(c, max_duration)
            pred_time = pred_transitions.get(c, max_duration)

            if pred_time != max_duration and target_time != max_duration:
                error = (target_time - pred_time) ** 2
            else:
                error = min((target_time - min_duration) ** 2, (max_duration - target_time) ** 2)

            # error = (target_time - pred_time) ** 2
            total_mse += error
            count += 1
        # print(f"error: {np.sqrt(error)} minutes")
        
        # break

    rmse = (total_mse / count) ** 0.5 if count != 0 else 0
    return rmse

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

def calculate_metrics(y_true, y_pred):
    """
    Calculate various metrics for the video classification task using both
    flattened and frame-by-frame approaches.
    
    Args:
    y_true (np.array): Ground truth labels, shape (video_length, 18)
    y_pred (np.array): Predicted labels, shape (video_length, 18)
    
    Returns:
    dict: A dictionary containing calculated metrics and frame-by-frame metrics
    """
    metrics = {}
    
    # Flattened metrics
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    metrics['flat_accuracy'] = accuracy_score(y_true_flat, y_pred_flat)
    metrics['flat_macro_f1'] = f1_score(y_true_flat, y_pred_flat, average='macro', zero_division=0)
    metrics['flat_weighted_f1'] = f1_score(y_true_flat, y_pred_flat, average='weighted', zero_division=0)
    metrics['flat_macro_precision'] = precision_score(y_true_flat, y_pred_flat, average='macro', zero_division=0)
    metrics['flat_weighted_precision'] = precision_score(y_true_flat, y_pred_flat, average='weighted', zero_division=0)
    metrics['flat_macro_recall'] = recall_score(y_true_flat, y_pred_flat, average='macro', zero_division=0)
    metrics['flat_weighted_recall'] = recall_score(y_true_flat, y_pred_flat, average='weighted', zero_division=0)
    
    # Frame-by-frame metrics
    frame_metrics = {
        'accuracy': [],
        'macro_f1': [],
        'weighted_f1': [],
        'macro_precision': [],
        'weighted_precision': [],
        'macro_recall': [],
        'weighted_recall': []
    }
    
    for i in range(y_true.shape[1]):  # Iterate over each frame in the anticipated window
        y_true_frame = y_true[:, i]
        y_pred_frame = y_pred[:, i]
        
        frame_metrics['accuracy'].append(accuracy_score(y_true_frame, y_pred_frame))
        frame_metrics['macro_f1'].append(f1_score(y_true_frame, y_pred_frame, average='macro', zero_division=0))
        frame_metrics['weighted_f1'].append(f1_score(y_true_frame, y_pred_frame, average='weighted', zero_division=0))
        frame_metrics['macro_precision'].append(precision_score(y_true_frame, y_pred_frame, average='macro', zero_division=0))
        frame_metrics['weighted_precision'].append(precision_score(y_true_frame, y_pred_frame, average='weighted', zero_division=0))
        frame_metrics['macro_recall'].append(recall_score(y_true_frame, y_pred_frame, average='macro', zero_division=0))
        frame_metrics['weighted_recall'].append(recall_score(y_true_frame, y_pred_frame, average='weighted', zero_division=0))
    
    # Average the frame-by-frame metrics
    for metric, values in frame_metrics.items():
        metrics[f'frame_{metric}_avg'] = np.mean(values)
        metrics[f'frame_{metric}_std'] = np.std(values)
    
    # Other metrics (unchanged)
    metrics['confusion_matrix'] = confusion_matrix(y_true_flat, y_pred_flat)
    metrics['segment_continuity'] = segment_continuity_score(y_true, y_pred)
    metrics['temporal_consistency'] = temporal_consistency_score(y_pred)
    metrics['class_distribution_divergence'] = class_distribution_divergence(y_true, y_pred)
    
    return metrics, frame_metrics


import seaborn as sns

def plot_performance_over_time(frame_metrics):
    """
    Plot the performance of various metrics over the anticipated time window.
    
    Args:
    frame_metrics (dict): Dictionary containing frame-by-frame metrics
    """
    time_steps = range(1, 19)  # 18 minutes
    
    plt.figure(figsize=(15, 10))
    
    # Use a color palette for other metrics
    colors = sns.color_palette("husl", len(frame_metrics) - 1)
    color_iter = iter(colors)
    
    # Plot accuracy separately with a distinct style
    plt.plot(time_steps, frame_metrics['accuracy'], label='accuracy', 
             linewidth=3, color='black', linestyle='--', marker='o')
    
    # Plot other metrics
    for metric, values in frame_metrics.items():
        if metric != 'accuracy':
            plt.plot(time_steps, values, label=metric, color=next(color_iter))
    
    plt.xlabel('Minutes into the future')
    plt.ylabel('Metric Value')
    plt.title('Performance Metrics over Anticipated Time Window')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)  # Assuming all metrics are between 0 and 1
    
    # Add horizontal lines for better readability
    plt.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    plt.axhline(y=0.25, color='gray', linestyle=':', alpha=0.5)
    plt.axhline(y=0.75, color='gray', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

def segment_continuity_score(y_true, y_pred):
    """
    Calculate a score based on the continuity of predicted segments.
    
    This function compares the lengths of continuous segments in the
    ground truth and predictions, rewarding longer matching segments.
    """
    score = 0
    for t in range(y_true.shape[0]):
        true_segment = y_true[t]
        pred_segment = y_pred[t]
        
        # Find the longest matching subsequence
        longest_match = 0
        current_match = 0
        for i in range(18):
            if true_segment[i] == pred_segment[i]:
                current_match += 1
                longest_match = max(longest_match, current_match)
            else:
                current_match = 0
        
        # Score is the square of the longest match, normalized
        score += (longest_match ** 2) / 324  # 324 is 18^2
    
    return score / y_true.shape[0]

def temporal_consistency_score(y_pred):
    """
    Calculate a score based on the temporal consistency of predictions.
    
    This function rewards predictions that are consistent over time,
    penalizing frequent changes in predicted classes.
    """
    changes = np.sum(np.diff(y_pred, axis=0) != 0)
    max_possible_changes = y_pred.shape[0] * y_pred.shape[1]
    return 1 - (changes / max_possible_changes)

def class_distribution_divergence(y_true, y_pred):
    """
    Calculate the divergence between true and predicted class distributions.
    
    This function uses KL divergence to measure how different the
    predicted class distribution is from the true distribution.
    """
    true_dist = np.bincount(y_true.flatten(), minlength=9) / len(y_true.flatten())
    pred_dist = np.bincount(y_pred.flatten(), minlength=9) / len(y_pred.flatten())
    
    # Add small epsilon to avoid division by zero in KL divergence
    epsilon = 1e-10
    true_dist += epsilon
    pred_dist += epsilon
    
    return entropy(true_dist, pred_dist)


def calculate_metrics(y_true, y_pred):
    """
    Calculate various metrics for the video classification task using both
    flattened and frame-by-frame approaches.
    
    Args:
    y_true (np.array): Ground truth labels, shape (video_length, 18)
    y_pred (np.array): Predicted labels, shape (video_length, 18)
    
    Returns:
    dict: A dictionary containing calculated metrics and frame-by-frame metrics
    """
    metrics = {}
    
    # Flattened metrics
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    metrics['flat_accuracy']            = np_round(accuracy_score(y_true_flat, y_pred_flat))
    metrics['flat_macro_f1']            = np_round(f1_score(y_true_flat, y_pred_flat, average='macro', zero_division=0))
    metrics['flat_weighted_f1']         = np_round(f1_score(y_true_flat, y_pred_flat, average='weighted', zero_division=0))
    metrics['flat_macro_precision']     = np_round(precision_score(y_true_flat, y_pred_flat, average='macro', zero_division=0))
    metrics['flat_weighted_precision']  = np_round(precision_score(y_true_flat, y_pred_flat, average='weighted', zero_division=0))
    metrics['flat_macro_recall']        = np_round(recall_score(y_true_flat, y_pred_flat, average='macro', zero_division=0))
    metrics['flat_weighted_recall']     = np_round(recall_score(y_true_flat, y_pred_flat, average='weighted', zero_division=0))
    
    # Frame-by-frame metrics
    frame_metrics = {
        'accuracy': [],
        'macro_f1': [],
        'weighted_f1': [],
        'macro_precision': [],
        'weighted_precision': [],
        'macro_recall': [],
        'weighted_recall': []
    }
    
    for i in range(y_true.shape[1]):  # Iterate over each anticipation time index
        y_true_frame = y_true[:, i]
        y_pred_frame = y_pred[:, i]
        
        frame_metrics['accuracy'].append(np_round(accuracy_score(y_true_frame, y_pred_frame)))
        frame_metrics['macro_f1'].append(np_round(f1_score(y_true_frame, y_pred_frame, average='macro', zero_division=0)))
        frame_metrics['weighted_f1'].append(np_round(f1_score(y_true_frame, y_pred_frame, average='weighted', zero_division=0)))
        frame_metrics['macro_precision'].append(np_round(precision_score(y_true_frame, y_pred_frame, average='macro', zero_division=0)))
        frame_metrics['weighted_precision'].append(np_round(precision_score(y_true_frame, y_pred_frame, average='weighted', zero_division=0)))
        frame_metrics['macro_recall'].append(np_round(recall_score(y_true_frame, y_pred_frame, average='macro', zero_division=0)))
        frame_metrics['weighted_recall'].append(np_round(recall_score(y_true_frame, y_pred_frame, average='weighted', zero_division=0)))
    
    # Average the 
    for metric, values in frame_metrics.items():
        metrics[f'{metric}_avg'] = np_round(np.mean(values))
        metrics[f'{metric}_std'] = np_round(np.std(values))
    
    # Other metrics (unchanged)
    metrics['confusion_matrix']                 = confusion_matrix(y_true_flat, y_pred_flat)
    metrics['segment_continuity']               = np_round(segment_continuity_score(y_true, y_pred))
    metrics['temporal_consistency']             = np_round(temporal_consistency_score(y_pred))
    metrics['class_distribution_divergence']    = np_round(class_distribution_divergence(y_true, y_pred))
    
    return metrics, frame_metrics


import seaborn as sns

def plot_performance_over_time(frame_metrics):
    """
    Plot the performance of various metrics over the anticipated time window.
    
    Args:
    frame_metrics (dict): Dictionary containing frame-by-frame metrics
    """
    length = len(frame_metrics['accuracy'])
    time_steps = range(0, length)
    
    plt.figure(figsize=(15, 10))
    
    # Use a color palette for other metrics
    colors = sns.color_palette("husl", len(frame_metrics) - 1)
    color_iter = iter(colors)

    min_y = 1
    max_y = 0
    y_space = 0.05
    
    # Plot accuracy separately with a distinct style
    plt.plot(time_steps, frame_metrics['accuracy'], label='accuracy', 
             linewidth=3, color='black', linestyle='--', marker='o')
    min_y = min(min_y, min(frame_metrics['accuracy']))
    max_y = max(max_y, max(frame_metrics['accuracy']))
    
    # Plot other metrics
    for metric, values in frame_metrics.items():
        if metric != 'accuracy':
            plt.plot(time_steps, values, label=metric, color=next(color_iter))
            min_y = min(min_y, min(values)) - y_space
            max_y = max(max_y, max(values)) + y_space
    
    plt.xlabel('Minutes into the future')
    plt.ylabel('Metric Value')
    plt.title('Performance Metrics over Anticipated Time Window')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.ylim(min_y, max_y)  # Assuming all metrics are between 0 and 1
    
    # Add horizontal lines for better readability
    plt.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    plt.axhline(y=0.25, color='gray', linestyle=':', alpha=0.5)
    plt.axhline(y=0.75, color='gray', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    # plt.show()
    plt.savefig('performance_over_time.png')

def segment_continuity_score(y_true, y_pred):
    """
    Calculate a score based on the continuity of predicted segments.
    
    This function compares the lengths of continuous segments in the
    ground truth and predictions, rewarding longer matching segments.
    """
    score = 0
    for t in range(y_true.shape[0]):
        true_segment = y_true[t]
        pred_segment = y_pred[t]
        
        # Find the longest matching subsequence
        longest_match = 0
        current_match = 0
        for i in range(len(true_segment)):
            if true_segment[i] == pred_segment[i]:
                current_match += 1
                longest_match = max(longest_match, current_match)
            else:
                current_match = 0
        
        # Score is the square of the longest match, normalized
        score += (longest_match ** 2) / 324  # 324 is 18^2
    
    return score / y_true.shape[0]

def temporal_consistency_score(y_pred):
    """
    Calculate a score based on the temporal consistency of predictions.
    
    This function rewards predictions that are consistent over time,
    penalizing frequent changes in predicted classes.
    """
    changes = np.sum(np.diff(y_pred, axis=0) != 0)
    max_possible_changes = y_pred.shape[0] * y_pred.shape[1]
    return 1 - (changes / max_possible_changes)

def class_distribution_divergence(y_true, y_pred):
    """
    Calculate the divergence between true and predicted class distributions.
    
    This function uses KL divergence to measure how different the
    predicted class distribution is from the true distribution.
    """
    true_dist = np.bincount(y_true.flatten(), minlength=9) / len(y_true.flatten())
    pred_dist = np.bincount(y_pred.flatten(), minlength=9) / len(y_pred.flatten())
    
    # Add small epsilon to avoid division by zero in KL divergence
    epsilon = 1e-10
    true_dist += epsilon
    pred_dist += epsilon
    
    return entropy(true_dist, pred_dist)

def aggregate_metrics(all_metrics, all_frame_metrics):
    """
    Aggregate metrics across all videos, handling different-sized confusion matrices.
    
    Args:
    all_metrics (list): List of metric dictionaries for each video
    all_frame_metrics (list): List of frame metric dictionaries for each video
    
    Returns:
    dict, dict: Aggregated metrics and aggregated frame metrics
    """
    agg_metrics = {}
    agg_frame_metrics = {metric: [] for metric in all_frame_metrics[0].keys()}
    
    # Aggregate flattened metrics
    for metric in all_metrics[0].keys():
        if metric != 'confusion_matrix':
            agg_metrics[metric] = np.round(np.mean([m[metric] for m in all_metrics]), 4).tolist()
    
    # Aggregate confusion matrices
    max_classes = max(cm.shape[0] for cm in [m['confusion_matrix'] for m in all_metrics])
    agg_confusion_matrix = np.zeros((max_classes, max_classes), dtype=int)
    
    for m in all_metrics:
        cm = m['confusion_matrix']
        n_classes = cm.shape[0]
        agg_confusion_matrix[:n_classes, :n_classes] += cm
    
    agg_metrics['confusion_matrix'] = agg_confusion_matrix.tolist()
    
    # Aggregate frame metrics
    for metric in agg_frame_metrics.keys():
        # mean over all videos but keep the temporal dimension
        agg_frame_metrics[metric] = np.round(np.mean([fm[metric] for fm in all_frame_metrics], axis=0), 4).tolist()
    
    return agg_metrics, agg_frame_metrics

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


# Example usage
if __name__ == "__main__":
   
    print("\n" + "="*50 + "\n")

    # Example 3: Model predictions
    y_true = "./supra/best_models/Cholec80/skit_c80_base_glob12.txt/local/video_frame_preds_69_ep6.npy"
    y_pred = "./supra/best_models/Cholec80/skit_c80_base_glob12.txt/local/video_tgts_preds_69_ep6.npy"

    y_true = np.load(y_true)
    y_pred = np.load(y_pred)

    metrics, frame_metrics = calculate_metrics(y_true, y_pred)

    print("Flattened Metrics:")
    for metric, value in metrics.items():
        if metric.startswith('flat_'):
            print(f"{metric}: {value:.4f}")
    
    print("\nFrame-by-Frame Metrics (Avg ± Std):")
    for metric, value in metrics.items():
        if metric.startswith('frame_') and metric.endswith('_avg'):
            std_metric = metric.replace('_avg', '_std')
            print(f"{metric}: {value:.4f} ± {metrics[std_metric]:.4f}")
    
    print("\nOther Metrics:")
    for metric, value in metrics.items():
        if not metric.startswith(('flat_', 'frame_')) and metric != 'confusion_matrix':
            print(f"{metric}: {value:.4f}")

    # Print class distribution
    class_counts = np.bincount(y_true.flatten(), minlength=9)
    print("\nClass distribution:")
    for i, count in enumerate(class_counts):
        print(f"Class {i}: {count} ({count/len(y_true.flatten())*100:.2f}%)")
    
    # Plot performance over time
    plot_performance_over_time(frame_metrics)
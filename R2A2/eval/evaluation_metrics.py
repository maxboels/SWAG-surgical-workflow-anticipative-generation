import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from scipy.stats import entropy
import matplotlib.pyplot as plt

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
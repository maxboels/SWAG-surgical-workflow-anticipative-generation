import matplotlib.pyplot as plt
import numpy as np

def plot_box_plot_figure(predicted_segments, data_dict, 
                         file_name='plot_acc_pred_box.png', 
                         title='Prediction Accuracies for all videos (N=40)', 
                         x_axis_title='Future Predictions (in minutes)', 
                         y_axis_title='Mean Accuracy'):
    plt.rcParams.update({'font.size': 18})
    plt.figure(figsize=(12, 8))
    box_width = 0.25

    num_datasets = len(data_dict)
    shift = (box_width + 0.05) * num_datasets / 2
    positions = {dataset_name: [(j - shift + i * (box_width + 0.05)) for j in range(len(predicted_segments))] 
                 for i, dataset_name in enumerate(data_dict)}

    colors = plt.get_cmap('tab10')
    box_patches = []
    for idx, (dataset_name, all_videos_accuracies) in enumerate(data_dict.items()):
        color = colors(idx)
        # Transpose the videos accuracies list to align with predicted segments
        accuracies = list(map(list, zip(*all_videos_accuracies)))
        for pos_idx, acc in enumerate(accuracies):
            # Exclude NaN values and check if list is not empty before plotting
            valid_acc = [a for a in acc if not np.isnan(a)]
            if valid_acc:
                bp = plt.boxplot(valid_acc, positions=[positions[dataset_name][pos_idx]], widths=box_width, patch_artist=True, boxprops=dict(facecolor=color), medianprops=dict(color="black"))
                box_patches.append(bp["boxes"][0])

    tick_positions = [pos for position_list in positions.values() for pos in position_list]
    plt.xticks(tick_positions, [f'{i+1}' if (i+1) % 5 == 0 else '' for i in range(len(predicted_segments))] * num_datasets, rotation=45)

    plt.xlabel(x_axis_title, weight='bold')
    plt.ylabel(y_axis_title, weight='bold')
    plt.title(title, fontdict={'family': 'monospace', 'weight': 'bold', 'size': 20})
    plt.legend(box_patches[:len(data_dict)], data_dict.keys(), loc='upper right')
    plt.grid(color='lightgray', linestyle='--', linewidth=1)
    plt.ylim(0, 1)
    plt.savefig(file_name, dpi=300)
    plt.close()
    # remove all configurations for the next plot
    plt.rcdefaults()


def plot_figure(planning_length, data_dict, 
                title='Planning Evaluation', 
                x_axis_title='Future Predictions (in minutes)',
                y_axis_title='Cummulative Mean Accuracy',
                file_name='planning_evaluation.png'):
    # Increase font size
    plt.rcParams.update({'font.size': 18})

    # Create a new figure with a specific size (in inches)
    plt.figure(figsize=(10, 6))

    # Plot with increased marker size, dashed lines, and hollow markers
    markers = ['^', 'o', 's']  # Define markers for each dataset
    for i, (dataset_name, mean_accuracy) in enumerate(data_dict.items()):
        plt.plot(planning_length, mean_accuracy, marker=markers[i], markersize=14, linestyle='--', fillstyle='none', markeredgewidth=4, label=dataset_name)

    # Legend
    plt.legend()

    # Axes labels
    plt.xlabel(x_axis_title, weight='bold')
    plt.ylabel(y_axis_title, weight='bold')

    # Set title
    plt.title(title, fontdict={'family': 'monospace', 'weight': 'bold', 'size': 20})

    # Set axis limits
    x_extra_space = 0.5
    y_extra_space = 0.025
    plt.xlim(min(planning_length)-x_extra_space, max(planning_length)+x_extra_space)
    all_mean_accuracies = [value for sublist in data_dict.values() for value in sublist]
    plt.ylim(min(all_mean_accuracies)-y_extra_space, max(all_mean_accuracies)+y_extra_space)

    # Grid
    plt.grid(color='lightgray', linestyle=':', linewidth=2.0)

    # Save plot with high resolution
    plt.savefig(file_name, dpi=300)
    
    plt.close()
    # remove all configurations for the next plot
    plt.rcdefaults()

import numpy as np
import matplotlib.pyplot as plt

def plot_cumulative_time(cumulative_times):

    # Plotting
    plt.rcParams.update({'font.size': 18})
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_times, marker='o', markersize=4, linestyle='-', fillstyle='none', markeredgewidth=2, label='Cumulative Time')
    plt.axhline(y=1, color=(1, 0.5, 0.25), linestyle='--')  # Add a red dashed line at y=1
    plt.title('Cumulative Iteration Time (1fps)', fontdict={'family': 'monospace', 'weight': 'bold', 'size': 20})
    plt.xlabel('Iteration Step', weight='bold')
    plt.ylabel('Cumulative Time (seconds)', weight='bold')
    plt.grid(color='lightgray', linestyle=':', linewidth=2.0)
    plt.savefig('plot_cumulative_time.png', dpi=300)
    plt.close()
    # remove all configurations for the next plot
    plt.rcdefaults()


if __name__ == '__main__':
    # Data
    planning_length = np.arange(1, 50+1)
    data_dict = {
        'Cholec80': [0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.44, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.0],
        'AutoLaparo21': [0.9, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
        'CholecT50': [0.98, 0.95, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    }

    # Call the function
    plot_cum_acc_future_frames(planning_length, data_dict)
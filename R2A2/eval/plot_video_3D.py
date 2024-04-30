import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap, Normalize

def plot_video_scatter_3D(preds, recs, tgt_preds, tgt_recs, anticip_time, video_idx=1, sampling_rate=1):
    # Concatenate recordings and predictions
    preds = np.concatenate([recs, preds], axis=1)
    targets = np.concatenate([tgt_recs, tgt_preds], axis=1)

    # Sample data according to the given rate
    preds = preds[::sampling_rate]
    targets = targets[::sampling_rate]

    # Create figure and adjust layout
    fig = plt.figure(figsize=(16, 9))
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.25, top=0.95, wspace=0.1, hspace=0.35)
    fig.suptitle(f'Surgical Phase Rec & Pred (Video {video_idx})', fontdict={'family': 'monospace', 'weight': 'bold', 'size': 20})

    # Define a custom discrete colormap with 8 distinct colors from 'viridis'
    colors = plt.cm.plasma(np.linspace(0, 1, 8)) # viridis, magma, inferno, plasma
    cmap = ListedColormap(colors)

    # Axes for predictions
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.grid(False)
    ax1.set_zticks([])
    # Set the color of the axis lines to light gray
    ax1.xaxis.line.set_color('lightgray')
    ax1.yaxis.line.set_color('lightgray')
    ax1.zaxis.line.set_color('lightgray')

    # Flatten the data for scatter plot, switch x and y
    y_values = np.tile(np.arange(preds.shape[1]), preds.shape[0]) * anticip_time
    x_values = np.repeat(np.arange(preds.shape[0]), preds.shape[1])
    z_values = np.zeros_like(x_values)  # No actual z-values, use zeros

    ax1.scatter(x_values, y_values, z_values, c=preds.flatten(), cmap=cmap, marker='o', s=400)
    ax1.set_xlim(max(x_values), 0)  # Ensure x-axis is inverted
    ax1.set_title("Predictions")
    ax1.set_xlabel("Current Frames (1fps)")
    ax1.set_ylabel("Future Frames (1fps)")
    ax1.set_ylim(0, anticip_time * preds.shape[1])  # Adjust y-axis limits to bring rows closer
    ax1.set_yticks(np.arange(0, anticip_time * preds.shape[1] + 1, anticip_time * 2))  # Adjust y-ticks to match original data scale
    # left side view
    ax1.view_init(elev=23., azim=-60)

    # Axes for targets
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.grid(False)
    ax2.set_zticks([])
    # Set the color of the axis lines to light gray
    ax2.xaxis.line.set_color('lightgray')
    ax2.yaxis.line.set_color('lightgray')
    ax2.zaxis.line.set_color('lightgray')

    ax2.scatter(x_values, y_values, z_values, c=targets.flatten(), cmap=cmap, marker='o', s=400)
    ax2.set_xlim(max(x_values), 0)  # Ensure x-axis is inverted
    ax2.set_title("Targets")
    ax2.set_xlabel("Current Frames (1fps)")
    ax2.set_ylabel("Future Frames (1fps)")
    ax2.set_ylim(0, anticip_time * preds.shape[1])  # Adjust y-axis limits to bring rows closer
    ax2.set_yticks(np.arange(0, anticip_time * preds.shape[1] + 1, anticip_time))  # Adjust y-ticks to match original data scale
    ax2.view_init(elev=23., azim=-60)

    # Color bar setup
    norm = plt.Normalize(vmin=-1, vmax=6)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.25, 0.08, 0.50, 0.02])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', ticks=np.arange(-1, 7))
    cbar.set_label('Classes (phases)')

    plt.savefig(f'video_{video_idx}_scatter3d_preds.png', dpi=300)
    # plt.show()

def plot_video_contour_3D(preds, recs, tgt_preds, tgt_recs, anticip_time, video_idx=1, sampling_rate=1):
    # Concatenate current state with predictions for future states
    preds = np.concatenate([recs, preds], axis=1)
    targets = np.concatenate([tgt_recs, tgt_preds], axis=1)

    # Sampling frames to match the anticipation interval
    preds = preds[::sampling_rate]
    targets = targets[::sampling_rate]

    # Making the data 3D
    preds = np.expand_dims(preds, axis=2)
    targets = np.expand_dims(targets, axis=2)

    # # Create a discrete colormap
    # unique_classes = np.unique(targets)  # Update this if class labels are known
    # cmap = ListedColormap(plt.cm.get_cmap('tab10').colors[:len(unique_classes)])  # Change 'tab10' to any suitable colormap

    fig = plt.figure(figsize=(16, 9))  # Adjust overall figure size
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.25, top=0.95, wspace=0.1, hspace=0.35)  # Adjust plot spacing
    fig.suptitle(f'Surgical Phase Rec & Pred (Video {video_idx})', fontdict={'family': 'monospace', 'weight': 'bold', 'size': 20})

    # Add grid
    # plt.grid(color='lightgray', linestyle=':', linewidth=2.0)
    
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.grid(False)
    ax1.set_zticks([])

    x_values = np.arange(0, preds.shape[1], 1) * anticip_time       # future frames axis
    y_values = np.arange(0, preds.shape[0], 1)                      # current frames axis
    X, Y = np.meshgrid(x_values, y_values)

    pred_plot = ax1.contourf(X, Y, preds[:, :, 0], zdir='z', offset=-10, cmap='magma', alpha=1.0)
    ax1.set(xlim=[0, max(x_values)], ylim=[0, len(y_values)], zlim=[-20, 20])
    ax1.view_init(elev=40., azim=-130)
    ax1.set_title("Predictions")
    ax1.set_xlabel("Future Frames (1fps)")
    ax1.set_ylabel("Current Frames (1fps)")

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.grid(False)
    ax2.set_zticks([])

    target_plot = ax2.contourf(X, Y, targets[:, :, 0], zdir='z', offset=-10, cmap='magma', alpha=0.95)
    ax2.set(xlim=[0, max(x_values)], ylim=[0, len(y_values)], zlim=[-20, 20])
    ax2.view_init(elev=40., azim=-130)
    ax2.set_title("Targets")
    ax2.set_xlabel("Future Frames (1fps)")
    ax2.set_ylabel("Current Frames (1fps)")

    cbar_ax = fig.add_axes([0.25, 0.08, 0.50, 0.02])
    fig.colorbar(target_plot, cax=cbar_ax, orientation='horizontal', label='Classes (phases)')
    plt.savefig(f'video_{video_idx}_preds.png', dpi=300)

if __name__ == '__main__':
    preds = np.load('data/video_frame_preds_60_ep2.npy')
    recs = np.load('data/video_frame_rec_60_ep2.npy')
    tgt_preds = np.load("data/video_tgts_preds_60_ep2.npy")
    tgt_recs = np.load("data/video_tgts_rec_60_ep2.npy")
    
    plot_video_contour_3D(preds, recs, tgt_preds, tgt_recs, video_idx=1)

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap, Normalize

def plot_video_scatter_3D(preds, recs, tgt_preds, tgt_recs, anticip_time,
                          video_idx=1, 
                          epoch=1,
                          sampling_rate=60, # seconds to minutes
                          padding_class=-1,
                          eos_class=7, 
                          num_classes=7,
                          video_mean_curr_acc=0.0,
                          video_mean_cum_acc_future=0.0): # don't include the eos class which is assigned to -1

    suptitle = f"Surgical Phase Rec. & Pred. (Video={video_idx}) \n" \
                f'Current Accuracy: {video_mean_curr_acc:.2f} | ' \
                f"Future Accuracy: {video_mean_cum_acc_future:.2f} \n" \
                f"Anticipative steps: {anticip_time // 60} minutes"
    
    main_fontdict={'family': 'monospace', 'weight': 'bold', 'size': 16}
    axis_fontdict = {'family': 'monospace', 'weight': 'bold', 'size': 12}
    
    xticks_step = 3  # set x ticks as every 3 minutes from 0 to 18
    
    # Concatenate recordings and predictions
    preds = np.concatenate([recs, preds], axis=1)
    targets = np.concatenate([tgt_recs, tgt_preds], axis=1)

    # change all the values equal to 7 into -1
    preds[preds == eos_class] = padding_class
    targets[targets == eos_class] = padding_class

    # Sample data according to the given rate
    preds = preds[::sampling_rate]
    targets = targets[::sampling_rate]

    # Create figure and adjust layout
    fig = plt.figure(figsize=(16, 9))
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.25, top=0.95, wspace=0.1, hspace=0.35)
    fig.suptitle(suptitle, fontdict=main_fontdict)

    # Define a custom discrete colormap with 8 distinct colors from 'viridis'
    colors = plt.cm.plasma(np.linspace(0, 1, num_classes+1)) # viridis, magma, inferno, plasma
    cmap = ListedColormap(colors)

    # Axes for targets
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.grid(False)
    ax2.set_zticks([])
    ax2.xaxis.line.set_color('lightgray')
    ax2.yaxis.line.set_color('lightgray')
    ax2.zaxis.line.set_color('lightgray')

    # Axes for predictions
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.grid(False)
    ax1.set_zticks([])
    ax1.xaxis.line.set_color('lightgray')
    ax1.yaxis.line.set_color('lightgray')
    ax1.zaxis.line.set_color('lightgray')

    # Flatten the data for scatter plot, switch x and y
    y_values = (np.tile(np.arange(preds.shape[1]), preds.shape[0]) * anticip_time) / 60  # Convert to minutes
    x_values = np.repeat(np.arange(preds.shape[0]), preds.shape[1])
    z_values = np.zeros_like(x_values)  # No actual z-values, use zeros

    # Scatter plot targets
    ax2.scatter(x_values, y_values, z_values, c=targets.flatten(), cmap=cmap, marker='o', s=400, 
                vmin=padding_class, vmax=num_classes)
    ax2.set_xlim(0, max(x_values))  # Ensure x-axis is left to right
    # ax2.set_xlim(max(x_values), 0)  # Ensure x-axis is inverted
    ax2.set_title("Targets", fontdict=axis_fontdict)
    ax2.set_xlabel("Observed Frames (1fps)", fontdict=axis_fontdict)
    ax2.set_ylabel("Future Frames (minutes)", fontdict=axis_fontdict)
    ax2.set_ylim(0, anticip_time * preds.shape[1] / 60)  # Adjust y-axis limits to bring rows closer
    # set x ticks as every 3 minutes from 0 to 18
    tick_positions = np.arange(0, preds.shape[0], 1)
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels([f'{i}' if (i) % xticks_step == 0 else '' for i in range(len(tick_positions))], rotation=45)
    ax2.set_yticks(np.arange(0, (anticip_time * preds.shape[1])/60, anticip_time/60))  # Adjust y-ticks to match original data scale
    ax2.view_init(elev=23., azim=-65)

    # Scatter plot predictions
    ax1.scatter(x_values, y_values, z_values, c=preds.flatten(), cmap=cmap, marker='o', s=400, 
                vmin=padding_class, vmax=num_classes)
    ax1.set_xlim(0, max(x_values))  # Ensure x-axis is left to right
    # ax1.set_xlim(max(x_values), 0)  # Ensure x-axis is inverted
    ax1.set_title("Predictions", fontdict=axis_fontdict)
    ax1.set_xlabel("Observed Frames (1fps)", fontdict=axis_fontdict)
    ax1.set_ylabel("Future Frames (minutes)", fontdict=axis_fontdict)
    ax1.set_ylim(0, anticip_time * preds.shape[1] / 60)  # Adjust y-axis limits to bring rows closer
    # set x ticks as every 3 minutes from 0 to 18
    tick_positions = np.arange(0, preds.shape[0], 1)
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels([f'{i}' if (i) % xticks_step == 0 else '' for i in range(len(tick_positions))], rotation=45)
    ax1.set_yticks(np.arange(0, (anticip_time * preds.shape[1])/60, anticip_time/60))  # Adjust y-ticks to match original data scale
    # left side view
    ax1.view_init(elev=23., azim=-65)

    # Set font size for axes
    ax1.tick_params(axis='both', which='major', labelsize=10)
    ax2.tick_params(axis='both', which='major', labelsize=10)

    # Color bar setup
    norm = plt.Normalize(vmin=0, vmax=num_classes)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    # thin color bar
    cbar_ax = fig.add_axes([0.25, 0.08, 0.50, 0.02])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', ticks=np.arange(0, num_classes+1))
    cbar.set_label('Classes (phases)')

    plt.savefig(f'video_{video_idx}_ep{epoch}_scatter3d_preds.png', dpi=300)
    plt.close()
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
    plt.close()

if __name__ == '__main__':
    preds = np.load('data/video_frame_preds_60_ep2.npy')
    recs = np.load('data/video_frame_rec_60_ep2.npy')
    tgt_preds = np.load("data/video_tgts_preds_60_ep2.npy")
    tgt_recs = np.load("data/video_tgts_rec_60_ep2.npy")
    
    plot_video_contour_3D(preds, recs, tgt_preds, tgt_recs, video_idx=1)

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def plot_video_contour_3D(preds, recs, tgt_preds, tgt_recs, video_idx=1, sampling_rate=1):
    # Concatenate current state with predictions for future states
    preds = np.concatenate([recs, preds], axis=1)
    targets = np.concatenate([tgt_recs, tgt_preds], axis=1)

    # Sampling frames to match the anticipation interval
    preds = preds[::(sampling_rate + 1)]
    targets = targets[::(sampling_rate + 1)]

    # Making the data 3D
    preds = np.expand_dims(preds, axis=2)
    targets = np.expand_dims(targets, axis=2)

    # Repeat to enhance visibility on plot
    preds = np.repeat(preds, 10, axis=2)
    targets = np.repeat(targets, 10, axis=2)

    fig = plt.figure(figsize=(16, 9))  # Adjust overall figure size
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.25, top=0.95, wspace=0.1, hspace=0.35)  # Adjust plot spacing
    fig.suptitle(f'Surgical Phase Rec & Pred (Video {video_idx})', fontdict={'family': 'monospace', 'weight': 'bold', 'size': 20})

    # Add grid
    # plt.grid(color='lightgray', linestyle=':', linewidth=2.0)
    
    # Plotting predictions
    ax1 = fig.add_subplot(121, projection='3d')  # First subplot for predictions
    # remove z axis ticks and upper part of the grid to enhance visibility
    ax1.grid(False)
    ax1.set_zticks([])
    # ax1.set_axis_off()
    X, Y = np.meshgrid(np.arange(preds.shape[1]), np.arange(preds.shape[0]))
    pred_plot = ax1.contourf(X, Y, preds[:, :, 0], zdir='z', offset=-10, cmap='magma', alpha=1.0)
    ax1.set(xlim=[0, preds.shape[1]], ylim=[0, preds.shape[0]], zlim=[-20, 20])
    ax1.view_init(elev=40., azim=-130)
    ax1.set_title("Predictions")

    ax2 = fig.add_subplot(122, projection='3d')  # Second subplot for targets
    ax2.grid(False)
    ax2.set_zticks([])
    # ax2.set_axis_off()
    target_plot = ax2.contourf(X, Y, targets[:, :, 0], zdir='z', offset=-10, cmap='magma', alpha=0.95)
    ax2.set(xlim=[0, targets.shape[1]], ylim=[0, targets.shape[0]], zlim=[-20, 20])
    ax2.view_init(elev=40., azim=-130)
    ax2.set_title("Targets")

    # Colorbar with reduced size
    cbar_ax = fig.add_axes([0.25, 0.10, 0.50, 0.02])  # Adjust these values to position and size the colorbar
    fig.colorbar(pred_plot, cax=cbar_ax, orientation='horizontal', label='Classes (phases)')

    plt.savefig(f'video_{video_idx}_rec_pred_3d_volume_corrected.png', dpi=300)
    # plt.show()

if __name__ == '__main__':
    preds = np.load('data/video_frame_preds_60_ep2.npy')
    recs = np.load('data/video_frame_rec_60_ep2.npy')
    tgt_preds = np.load("data/video_tgts_preds_60_ep2.npy")
    tgt_recs = np.load("data/video_tgts_rec_60_ep2.npy")
    
    plot_video_contour_3D(preds, recs, tgt_preds, tgt_recs, video_idx=1)

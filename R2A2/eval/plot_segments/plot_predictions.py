from matplotlib import pyplot as plt
import numpy as np
import torch
import os


def plot_current_and_next_classes(class_list, num_pred):
    video_classes = - torch.ones(video_length, 1 + num_pred)
    # for every transition, get its next class.
    transition_list = []
    prev_class = -1
    for i in range(class_list.shape[0]):
        if prev_class != class_list[i]:
            transition_list.append((i, class_list[i].item()))
            prev_class = class_list[i]
    print(f"transition_list: {transition_list}")
    # store the current class list in the first row of video_classes
    video_classes[:, 0] = class_list[:, 0]
    print(f"video_classes:\n{video_classes}")
    # now, for every frame i, store the next classes in video_classes
    for i in range(video_classes.shape[0]):
        # get the next classes for this frame
        next_classes = []
        for j in range(len(transition_list)):
            if transition_list[j][0] > i:
                next_classes.append(transition_list[j][1])
        # store the next classes in video_classes
        for j in range(len(next_classes)):
            video_classes[i, j+1] = next_classes[j]
            if j+1 == num_pred:
                break
    print(f"video_classes:\n{video_classes}")
    # dilate the video_classes array to make it visually appealing
    # repeat each of the last dimensions 10 times
    video_classes = np.repeat(video_classes, dilation_factor, axis=1)
    # plot image with imshow
    fig, ax = plt.subplots(figsize=(20, 4), dpi=300)
    ax.imshow(video_classes.T, cmap='tab20')
    ax.set_xlabel("Frame", fontsize=14)
    ax.set_ylabel("Class next", fontsize=14)
    ax.set_title("Classes", fontsize=16)
    ax.grid(True, axis='x')
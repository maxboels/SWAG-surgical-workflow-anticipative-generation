import matplotlib.pyplot as plt
import numpy as np
import os



def create_segments(input_list, class_labels=None, color_array=None):
    """
    Compute the segment inrtervals for each class.

    input_list: list of predicted phases or labels
    class_labels: list of class labels from 0 to 6
    color_array: array of colors for plotting

    return:
        segments: list of segments for each class
        c_value: current value i.e. the last value **relative** to the segment start
        c_phase: current phase
    """
    # print('input_list: ', input_list)
    
    # current phase
    c_phase = input_list[0]
    # current value of phase
    c_value = 0
    segment_values=[]
    # for element in input_list:
    for x in input_list:
        # if the input phase is different from the current phase
        if c_phase != x:
            # add the current element index to the list with the corresponding phase label and color
            segment_values.append({'value': c_value, 'label': int(c_phase)})
            # segment_values.append({'value': c_value, 'label':class_labels[c_phase], 'color': color_array[c_phase]}) # before
            # reset the current value
            c_value = 0
            # set the current phase to the breaking phase in list
            c_phase = x
        # else add 1 to the current phase
        c_value+=1
    # add the last element to the list
    segment_values.append({'value': c_value, 'label': int(c_phase)})
    # segment_values.append({'value': c_value, 'label': class_labels[c_phase], 'color': color_array[c_phase]}) # before
    return segment_values


def plot_video_segments(input_dict, colors, vid_id=1000, save_path=None, eval_type="recognition"):
    """
    Plot the video segments.
    
    input_dict: dictionary with label as key and list of predicted phases as value
    colors: dictionary with class labels as key and color as value
    vid_id: video id
    save_path: path to save the plot
    eval_type: recognition or anticipation
    """
    # print('input_dict: ', input_dict)
    segment_dict = {}
    for key, value in input_dict.items():
        if value:
            segment_dict[key] = create_segments(value)

    # set font size for the entire plot
    plt.rcParams.update({'font.size': 14})
    
    fig, ax = plt.subplots(figsize=(20, 4))

    # stack plots vertically
    y_pos = 0.0
    bar_height = 0.8
    text_y_pos = 0.0

    # plot segments with keys as labels
    for key, segment_list in segment_dict.items():
        left_pos = 0
        for idx in range(len(segment_list)):
            segdata = segment_list[idx]
            seglabel = segdata['label']
            segval = segdata['value']
            segcol = colors[segdata['label']]
            ax.barh(y_pos, width=[segval], height=bar_height, align='center', color=segcol, label=seglabel, left=left_pos, linewidth=0.0)
            left_pos += segval
        text_y_pos += y_pos
        y_pos += -1
        
        ax.set_xlim(0, left_pos)
        ax.set_yticks([])
    
    # add full length of the video at the end of the x-axis (but keep the font small)
    ax.set_xticks([0, left_pos])
    # title
    ax.set_title('Video Segmentation: Phase '+eval_type, fontsize=15)
    # add epoch number for each row in the plot
    for i, key in enumerate(segment_dict.keys()):
        ax.text(0.0, -i-0.5, str(key), fontsize=5)
    # remove borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    # subplot config tools: set left margin to 0 and right margin to 1
    plt.subplots_adjust(left=0, right=1, top=0.8, bottom=0.3)

    # ax.grid(True, axis='x')
    # plt.show()
    fig.savefig(save_path+str(vid_id)+'.png', dpi=300, bbox_inches='tight')



def plot_classes_and_durations(classes, durations, num_targets, vid_id=0, path="", relative_to="video_length"):

    # make path if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)

    # set font size for the entire plot
    plt.rcParams.update({'font.size': 14})

    # make the width larget than the height
    fig, axs = plt.subplots(2, 1, figsize=(20, 6))
    labels = ["current class"]
    labels.extend([f"next class {i}" for i in range(1, num_targets)])
    print("classes.shape: ", classes.shape)
    for i in range(classes.shape[1]):
        print(f"i: {i}")
        print(f"labels: {labels}")
        axs[0].plot(classes[:, i], label=labels[i])
    axs[0].set_title("Classes", fontsize=16)
    axs[0].set_xlabel("Frame", fontsize=14)
    axs[0].set_ylabel("Class", fontsize=14)
    axs[0].legend(fontsize=12)

    labels = ["current duration"]
    labels.extend([f"next duration {i}" for i in range(1, num_targets)])
    for i in range(durations.shape[1]):
        axs[1].plot(durations[:, i], label=labels[i])
    axs[1].set_title("Durations relative to "+relative_to, fontsize=16)
    axs[1].set_xlabel("Frame", fontsize=14)
    axs[1].set_ylabel("Duration", fontsize=14)
    axs[1].legend(fontsize=12)

    plt.subplots_adjust(hspace=0.5) # add spacing between subplots

    plt.savefig(f"{path}{vid_id}_class_durations_to_{relative_to}_num_targets{num_targets}.png", dpi=300, bbox_inches='tight')
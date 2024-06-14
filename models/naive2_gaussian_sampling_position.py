import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json



class GaussianMixtureSamplerWithPosition:
    def __init__(self, class_freq_positions, lookahead=18):
        self.class_freq_positions = class_freq_positions
        self.lookahead = lookahead
        self.probabilities = self._compute_probabilities()
    
    def _compute_probabilities(self):
        probabilities = {}
        for cls, freq_list in self.class_freq_positions.items():
            probabilities[int(cls)] = []
            for freq_dict in freq_list:
                total_count = sum(freq_dict.values())
                probabilities[int(cls)].append({int(k): v / total_count for k, v in freq_dict.items()})
        return probabilities
    
    def sample(self, current_class, num_samples=18):
        if current_class not in self.probabilities:
            raise ValueError(f"Class {current_class} not found in class frequencies.")
        
        samples = []
        for j in range(num_samples):
            possible_values = list(self.probabilities[current_class][j].keys())
            probabilities = list(self.probabilities[current_class][j].values())
            samples.append(np.random.choice(possible_values, p=probabilities))
        
        return samples

def get_class_freq_positions(folder_path, video_range=[41,41], eos_class=7, lookahead=18, anticip_time=60):
    """
    Build a naive predictor that samples the next class based on the class frequencies from the training set.
    :param folder_path: Path to the folder containing the data
    :param video_range: List of video indices to consider
    :param eos_class: End of sequence class
    :param lookahead: Number of steps to look ahead in minutes
    :param anticip_time: Anticipation time steps in seconds
    """

    lookahead_1fps = lookahead * 60
    lookahead_tokens = lookahead_1fps // anticip_time

    # Initialize all possible classes
    unique_classes = set()

    # Iterate over the video_range in the training set to find all possible classes
    for video_idx in range(video_range[0], video_range[1]):

        # Load the data
        file_path = f'{folder_path}video_tgts_rec_{video_idx}_ep1.npy'
        data = np.load(file_path)

        # Initialize the class frequencies
        video_unique_classes = np.unique(data)

        # Update the unique classes
        unique_classes.update(video_unique_classes)

    # add the eos class
    unique_classes.add(eos_class)

    # Initialize a dictionary to store frequencies for each class and position
    class_frequencies_with_position = {cls: [defaultdict(int) for _ in range(lookahead_tokens)] for cls in unique_classes}

    # Iterate over the video_range in the training set
    for video_idx in range(video_range[0], video_range[1]):

        # Load the data
        file_path = f'{folder_path}video_tgts_rec_{video_idx}_ep1.npy'
        data = np.load(file_path)

        # pad sequence with eos class at 1fps like the input data
        data = np.vstack((data, np.full((lookahead*60, 1), eos_class)))

        # Initialize the class frequencies
        unique_classes = np.unique(data)

        # Iterate over the data to build the frequencies with position
        padded_sequence_length = data.shape[0]
        for i in range(padded_sequence_length - lookahead_1fps):
            current_class = data[i, 0]
            for j in range(lookahead_tokens):
                next_class = data[i + j*anticip_time, 0]
                class_frequencies_with_position[current_class][j][next_class] += 1

    return class_frequencies_with_position

if __name__ == '__main__':

    video_range = [15, 22] # end is exclusive: use 22 and 81
    dataset = "AutoLaparo21"  # "Cholec80" or "AutoLaparo21"
    folder = f"./supra/best_models/{dataset}/"
    ant_times=[20, 30, 40, 50, 60]
    # subfolder = f"sup_best_al21_ct144_at60_ls442.txt/local/"
    subfolder = f"skit_best_al21_ct144_at60_ls442.txt/local/"

    data_path = f"{folder}{subfolder}"

    class_freq_positions = get_class_freq_positions(data_path, 
                                                    video_range=video_range, 
                                                    eos_class=7,
                                                    lookahead=18,
                                                    anticip_time=60)
    print(f"Naive predictor built for video_range: {video_range}")
    print(f"Anticipation times: {ant_times}")
    print(f"Naive predictor: {class_freq_positions}")

    # save to json, keys to string and values to list
    save_dict = {}
    for k, v in class_freq_positions.items():
        save_dict[str(k)] = [{str(inner_k): float(inner_v) for inner_k, inner_v in freq_dict.items()} for freq_dict in v]

    with open(f"naive2_{dataset}_class_freq_positions.json", "w") as f:
        json.dump(save_dict, f)

    sampler_with_position = GaussianMixtureSamplerWithPosition(class_freq_positions, lookahead=18)

    # Example usage
    predictions_per_class = []
    num_classes = 7
    for current_class in range(num_classes):
        next_classes = sampler_with_position.sample(current_class)
        print(f"Current class: {current_class}")
        print(f"Next classes: {next_classes}")
        predictions_per_class.append(next_classes)
    
    # Plot the predictions
    sns.set()
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for i, next_classes in enumerate(predictions_per_class):
        ax.plot(next_classes, label=f"Class {i}")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Class")
    ax.set_title("Naive Gaussian Mixture Sampling")
    ax.legend()
    plt.show()







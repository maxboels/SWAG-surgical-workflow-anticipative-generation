import torch
import numpy as np
from scipy.interpolate import interp1d

def find_intersections(x, y1, y2, epsilon=1e-8):
    """Find intersections between two curves with improved numerical stability."""
    diff = y1 - y2
    indices = np.where((diff[:-1] * diff[1:]) <= 0)[0]
    
    intersections = []
    for i in indices:
        denominator = diff[i+1] - diff[i]
        if abs(denominator) > epsilon:
            x_intersect = x[i] + (x[i+1] - x[i]) * (0 - diff[i]) / denominator
            intersections.append(x_intersect)
    
    return intersections

def classification2regression(video_anticipation_probs, horizon_minutes=18, prob_threshold=0.05):
    """
    Convert classification probabilities to regression values for a video sequence
    using probability intersections to determine phase transitions.

    Args:
        video_anticipation_probs (torch.Tensor): Tensor of shape (video_length, horizon_minutes+1, num_classes)
        horizon_minutes (int): The time horizon in minutes
        prob_threshold (float): Minimum probability difference to consider a valid transition
    
    Returns:
        torch.Tensor: Tensor of shape (video_length, num_classes) with regression values
    """
    video_length, horizon_steps, num_classes = video_anticipation_probs.shape
    
    output = torch.full((video_length, num_classes), horizon_minutes, dtype=torch.float32)
    time_array = np.linspace(0, horizon_minutes, horizon_steps)
    
    for t in range(video_length):
        probs = video_anticipation_probs[t].numpy()
        current_class = np.argmax(probs[0])
        interp_probs = [interp1d(time_array, probs[:, c], kind='cubic') for c in range(num_classes)]
        
        for c in range(num_classes):
            if c == current_class:
                output[t, c] = 0
                continue
            
            intersections = find_intersections(time_array, interp_probs[current_class](time_array), interp_probs[c](time_array))
            
            if intersections:
                valid_intersections = []
                for intersect in intersections:
                    prob_diff = interp_probs[c](intersect) - interp_probs[current_class](intersect)
                    if prob_diff > prob_threshold:
                        valid_intersections.append((intersect, prob_diff))
                
                if valid_intersections:
                    best_intersect = min(valid_intersections, key=lambda x: x[0])[0]
                    output[t, c] = best_intersect
                else:
                    # If no valid intersection, find point of maximum probability difference
                    prob_diffs = interp_probs[c](time_array) - interp_probs[current_class](time_array)
                    max_diff_index = np.argmax(prob_diffs)
                    if prob_diffs[max_diff_index] > prob_threshold:
                        output[t, c] = time_array[max_diff_index]
    
    return output

# def find_intersections(x, y1, y2, epsilon=1e-8):
#     """Find intersections between two curves with improved numerical stability."""
#     diff = y1 - y2
#     indices = np.where((diff[:-1] * diff[1:]) <= 0)[0]
    
#     intersections = []
#     for i in indices:
#         denominator = diff[i+1] - diff[i]
#         if abs(denominator) > epsilon:
#             x_intersect = x[i] + (x[i+1] - x[i]) * (0 - diff[i]) / denominator
#             intersections.append(x_intersect)
    
#     return intersections

# def find_intersections(x, y1, y2):
#     """Find intersections between two curves."""
#     diff = y1 - y2
#     indices = np.where((diff[:-1] * diff[1:]) <= 0)[0]
    
#     intersections = []
#     for i in indices:
#         x_intersect = x[i] + (x[i+1] - x[i]) * (0 - diff[i]) / (diff[i+1] - diff[i])
#         intersections.append(x_intersect)
    
#     return intersections

# def classification2regression(video_anticipation_probs, horizon_minutes=18):
#     """
#     Convert classification probabilities to regression values for a video sequence
#     using probability intersections to determine phase transitions.

#     Args:
#         video_anticipation_probs (torch.Tensor): Tensor of shape (video_length, horizon_minutes+1, num_classes)
#         horizon_minutes (int): The time horizon in minutes
    
#     Input tensor should have the following dimensions:
#     - video_length: Number of time steps in the video sequence
#     - horizon_minutes+1: Number of time steps in the future to consider for anticipation (including the current time step)
#     - num_classes: Number of anticipation classes
    
#     Returns:
#         torch.Tensor: Tensor of shape (video_length, num_classes) with regression values
#     """
#     video_length, horizon_steps, num_classes = video_anticipation_probs.shape
    
#     # Initialize output tensor
#     output = torch.full((video_length, num_classes), horizon_minutes, dtype=torch.float32)
    
#     # Create a time array
#     time_array = np.linspace(0, horizon_minutes, horizon_steps)
    
#     for t in range(video_length):
#         probs = video_anticipation_probs[t]
        
#         # Find the current most probable class
#         current_class = np.argmax(probs[0])
        
#         # Interpolate probability curves for smoother intersection finding
#         interp_probs = [interp1d(time_array, probs[:, c], kind='cubic') for c in range(num_classes)]
        
#         # Find intersections for each class
#         for c in range(num_classes):
#             if c == current_class:
#                 output[t, c] = 0  # Current class has 0 remaining time
#                 continue
            
#             intersections = find_intersections(time_array, interp_probs[current_class](time_array), interp_probs[c](time_array))
            
#             if intersections:
#                 # Find the first intersection where the class probability becomes higher
#                 for intersect in intersections:
#                     if interp_probs[c](intersect) > interp_probs[current_class](intersect):
#                         output[t, c] = intersect
#                         break
    
#     return output

def regression2classification(regression_values, horizon_minutes=18):
    """Original function (fixed roudning issue)"""

    # If has 3 dimensions and 1 channel, remove the channel dimension
    if len(regression_values.shape) == 3 and regression_values.shape[1] == 1:
        regression_values = regression_values.squeeze(1)

    video_length, num_classes = regression_values.shape
    output = torch.zeros((video_length, horizon_minutes), dtype=torch.long)
    
    for t in range(video_length):
        current_regression = regression_values[t]
        current_classification = torch.zeros(horizon_minutes, dtype=torch.long)
        current_class = torch.argmin(current_regression)
        current_classification[:] = current_class
        
        for c in range(num_classes):
            if c != current_class:
                minute = current_regression[c]
                if minute <= horizon_minutes:
                    minute = round(minute.item())
                    current_classification[minute:] = c
        
        output[t] = current_classification
    
    return output

def regression2classification_error(regression_values, horizon_minutes=18):
    """
    Convert regression values to classification sequence for a video sequence.

    Args:
        regression_values (torch.Tensor): Tensor of shape (video_length, 1, num_classes+1) with regression values
        horizon_minutes (int): The time horizon in minutes

    Returns:
        torch.Tensor: Tensor of shape (video_length, horizon_minutes) with class labels
    """

    # If has 3 dimensions and 1 channel, remove the channel dimension
    if len(regression_values.shape) == 3 and regression_values.shape[1] == 1:
        regression_values = regression_values.squeeze(1)

    video_length, _ = regression_values.shape
    
    # Initialize output tensor for classes (integers)
    output = torch.zeros((video_length, horizon_minutes), dtype=torch.long)
    
    for t in range(video_length):
        # Get the regression values for the current time step
        current_regression = regression_values[t]
        
        # Initialize the classification for this time step
        # with the lowest class for each time step
        current_classification = torch.zeros(horizon_minutes, dtype=torch.long)

        current_classification[:] = current_regression.argmin()
        
        # Select the class with the smallest remaining time
        # and fill the classification sequence from the index of the selected lowest value up to the end
        # then replace the selected value with the maximum value to avoid selecting it again
        while current_regression.min() < horizon_minutes:
            min_index_class = current_regression.argmin()
            min_index_value = int(current_regression[min_index_class])
            current_classification[min_index_value:] = min_index_class
            current_regression[min_index_class] = horizon_minutes

        output[t] = current_classification
    
    return output


# def regression2classification(regression_values, horizon_minutes=18):
#     """
#     Convert regression values to classification sequence for a video sequence.

#     Args:
#         regression_values (torch.Tensor): Tensor of shape (video_length, 1, num_classes+1) with regression values
#         horizon_minutes (int): The time horizon in minutes

#     Returns:
#         torch.Tensor: Tensor of shape (video_length, horizon_minutes) with class labels
#     """

#     # If has 3 dimensions and 1 channel, remove the channel dimension
#     if len(regression_values.shape) == 3 and regression_values.shape[1] == 1:
#         regression_values = regression_values.squeeze(1)

#     video_length, num_classes = regression_values.shape
    
#     # Initialize output tensor for classes (integers)
#     output = torch.zeros((video_length, horizon_minutes), dtype=torch.long)
    
#     for t in range(video_length):
#         # Get the regression values for the current time step
#         current_regression = regression_values[t]
        
#         # Initialize the classification for this time step
#         current_classification = torch.zeros(horizon_minutes, dtype=torch.long)
        
#         # Find the current active class (class with 0 regression value)
#         current_class = torch.argmin(current_regression)
        
#         # Fill in the classification from the bottom up
#         current_classification[:] = current_class
        
#         # Fill in the other classes also from the bottom up
#         for c in range(num_classes):
#             if c != current_class:
#                 minute = current_regression[c] # is float
#                 if minute < horizon_minutes:
#                     # convert the float into nearest integer
#                     minute = int(minute)
#                     current_classification[minute:] = c
        
#         output[t] = current_classification
    
#     return output

def classification_to_remaining_time(class_probs, time_steps, h, num_classes=7, method='first_occurrence', confidence_threshold=0.5):
    """
    Compute the remaining time for each phase based on classification probabilities.
    Args:
    class_probs (torch.Tensor): Classification probabilities, shape (batch_size, num_steps, num_classes)
    time_steps (torch.Tensor): Time steps in minutes, shape (num_steps)
    h (int): Maximum anticipation time in minutes
    num_classes (int): Total number of anticipation classes in the dataset (default: 7)
    method (str): Method to compute remaining time ('first_occurrence', 'mean', 'median')
    confidence_threshold (float): Confidence threshold for the 'first_occurrence' method
    """
    # Excluding the 
    
    batch_size, num_steps, num_ant_classes = class_probs.size()
    remaining_time = torch.full((batch_size, num_classes), h, device=class_probs.device)
    
    for phase in range(num_classes):        
        # TODO: add option with last EOS class regression
        if method == 'class_occurence':
            future_classes = torch.argmax(class_probs, dim=2)
            first_occurrence = torch.argmax(future_classes == phase, dim=1)
            remaining_time[:, phase] = torch.min(time_steps[first_occurrence], torch.tensor(h, device=class_probs.device))

        elif method == 'first_occurrence_and_threshold': # and > confidence_threshold
            first_occurrence = torch.argmax((phase_probs > confidence_threshold).float(), dim=1)
            mask = (first_occurrence == 0) & (phase_probs[:, 0] <= confidence_threshold)
            first_occurrence[mask] = torch.argmax(phase_probs[mask], dim=1)
            remaining_time[:, phase] = torch.min(time_steps[first_occurrence], torch.tensor(h, device=class_probs.device))
        
        elif method == 'mean':
            expected_time = torch.sum(phase_probs * time_steps, dim=1) / torch.sum(phase_probs, dim=1)
            remaining_time[:, phase] = torch.min(expected_time, torch.tensor(h, device=class_probs.device))
        
        elif method == 'median':
            cumulative_probs = torch.cumsum(phase_probs, dim=1)
            median_index = torch.argmin(torch.abs(cumulative_probs - 0.5), dim=1)
            remaining_time[:, phase] = torch.min(time_steps[median_index], torch.tensor(h, device=class_probs.device))
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    return remaining_time

def ground_truth_remaining_time(phase_labels, current_step, h, num_classes=7):
    """
    Compute the remaining time for each phase based on ground truth labels.

    Args:
    phase_labels (torch.Tensor): Ground truth phase labels, shape (batch_size, seq_len)
    h (int): Maximum anticipation time in minutes
    num_classes (int): Total number of anticipation classes in the dataset (default: 7)
    """
    batch_size, seq_len = phase_labels.size()
    remaining_time = torch.full((batch_size, num_classes), h, device=phase_labels.device)
    for b in range(batch_size):
        for phase in range(num_classes):
            # select the future first occurrences of the phase
            future_occurrences = torch.where(phase_labels[b, current_step:] == phase)[0]
            if len(future_occurrences) > 0:
                remaining_time[b, phase] = min(future_occurrences[0].item(), h) # indices are minutes
            else:
                remaining_time[b, phase] = h

    return remaining_time
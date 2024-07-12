import torch

import torch
import numpy as np
from scipy.interpolate import interp1d

def find_intersections(x, y1, y2):
    """Find intersections between two curves."""
    diff = y1 - y2
    indices = np.where((diff[:-1] * diff[1:]) <= 0)[0]
    
    intersections = []
    for i in indices:
        x_intersect = x[i] + (x[i+1] - x[i]) * (0 - diff[i]) / (diff[i+1] - diff[i])
        intersections.append(x_intersect)
    
    return intersections

def classification2regression_intersections(video_anticipation_probs, horizon_minutes=18):
    """
    Convert classification probabilities to regression values for a video sequence
    using probability intersections to determine phase transitions.

    Args:
        video_anticipation_probs (torch.Tensor): Tensor of shape (video_length, horizon_minutes+1, num_classes)
        horizon_minutes (int): The time horizon in minutes
    
    Returns:
        torch.Tensor: Tensor of shape (video_length, num_classes) with regression values
    """
    video_length, horizon_steps, num_classes = video_anticipation_probs.shape
    
    # Initialize output tensor
    output = torch.full((video_length, num_classes), horizon_minutes, dtype=torch.float32)
    
    # Create a time array
    time_array = np.linspace(0, horizon_minutes, horizon_steps)
    
    for t in range(video_length):
        probs = video_anticipation_probs[t].numpy()
        
        # Find the current most probable class
        current_class = np.argmax(probs[0])
        
        # Interpolate probability curves for smoother intersection finding
        interp_probs = [interp1d(time_array, probs[:, c], kind='cubic') for c in range(num_classes)]
        
        # Find intersections for each class
        for c in range(num_classes):
            if c == current_class:
                output[t, c] = 0  # Current class has 0 remaining time
                continue
            
            intersections = find_intersections(time_array, interp_probs[current_class](time_array), interp_probs[c](time_array))
            
            if intersections:
                # Find the first intersection where the class probability becomes higher
                for intersect in intersections:
                    if interp_probs[c](intersect) > interp_probs[current_class](intersect):
                        output[t, c] = intersect
                        break
    
    return output



def regression2classification(regression_values, horizon_minutes=18):
    """
    Convert regression values to classification sequence for a video sequence.

    Args:
        regression_values (torch.Tensor): Tensor of shape (video_length, 1, num_classes) with regression values
        horizon_minutes (int): The time horizon in minutes

    Returns:
        torch.Tensor: Tensor of shape (video_length, horizon_minutes) with class labels
    """

    # If has 3 dimensions and 1 channel, remove the channel dimension
    if len(regression_values.shape) == 3 and regression_values.shape[1] == 1:
        regression_values = regression_values.squeeze(1)

    video_length, num_classes = regression_values.shape
    
    # Initialize output tensor
    output = torch.zeros((video_length, horizon_minutes), dtype=torch.long)
    
    for t in range(video_length):
        # Get the regression values for the current time step
        current_regression = regression_values[t]
        
        # Initialize the classification for this time step
        current_classification = torch.zeros(horizon_minutes, dtype=torch.long)
        
        # Find the current active class (class with 0 regression value)
        current_class = torch.argmin(current_regression)
        
        # Fill in the classification from the bottom up
        current_classification[:] = current_class
        
        # Fill in the other classes also from the bottom up
        for c in range(num_classes):
            if c != current_class:
                minute = current_regression[c] # is float
                if minute < horizon_minutes:
                    # convert the float into nearest integer
                    minute = int(minute)
                    current_classification[minute:] = c
        
        output[t] = current_classification
    
    return output

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
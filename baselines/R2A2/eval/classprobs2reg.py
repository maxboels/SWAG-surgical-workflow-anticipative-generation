import torch
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

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

def visualize_intersections(video_anticipation_probs, regression_values, time_step, horizon_minutes=18):
    """
    Visualize the probability curves, their intersections, and the calculated regression values.

    Args:
        video_anticipation_probs (torch.Tensor): Tensor of shape (video_length, horizon_minutes+1, num_classes)
        regression_values (torch.Tensor): Tensor of shape (video_length, num_classes)
        time_step (int): The time step to visualize
        horizon_minutes (int): The time horizon in minutes
    """
    _, horizon_steps, num_classes = video_anticipation_probs.shape
    time_array = np.linspace(0, horizon_minutes, horizon_steps)
    probs = video_anticipation_probs[time_step].numpy()

    plt.figure(figsize=(12, 6))

    # Plot probability curves
    for c in range(num_classes):
        interp_prob = interp1d(time_array, probs[:, c], kind='cubic')
        plt.plot(time_array, interp_prob(time_array), label=f'Class {c}')

    # Plot regression values
    for c in range(num_classes):
        regression_time = regression_values[time_step, c].item()
        if regression_time < horizon_minutes:
            plt.axvline(x=regression_time, color='k', linestyle='--', alpha=0.5)
            plt.text(regression_time, 0.1, f'Class {c}: {regression_time:.2f}', rotation=90, verticalalignment='bottom')

    plt.xlabel('Time (minutes)')
    plt.ylabel('Probability')
    plt.title(f'Probability Curves and Regression Values at Time Step {time_step}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Test the function and visualize results
if __name__ == "__main__":
    # Generate some example classification probabilities
    video_length = 100
    num_classes = 5
    horizon_minutes = 18
    horizon_steps = horizon_minutes + 1

    video_anticipation_probs = torch.rand(video_length, horizon_steps, num_classes)
    video_anticipation_probs = video_anticipation_probs / video_anticipation_probs.sum(dim=2, keepdim=True)

    # Convert to regression values
    regression_values = classification2regression_intersections(video_anticipation_probs, horizon_minutes)

    # Visualize a specific time step
    time_step_to_visualize = 50  # You can change this to any value between 0 and video_length-1
    visualize_intersections(video_anticipation_probs, regression_values, time_step_to_visualize, horizon_minutes)

    print("Classification probabilities shape:", video_anticipation_probs.shape)
    print("Regression values shape:", regression_values.shape)
    print(f"\nRegression values for time step {time_step_to_visualize}:")
    print(regression_values[time_step_to_visualize])
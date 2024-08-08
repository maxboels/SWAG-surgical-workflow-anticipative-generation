import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def find_time_to_next_occurrence(video_anticipation_probs, horizon_minutes=18):
    """
    Find the time to next occurrence for each class at each time step in the video.

    Args:
        video_anticipation_probs (np.ndarray): Tensor of shape (video_length, horizon_steps, num_classes)
        horizon_minutes (int): The time horizon in minutes
    
    Returns:
        torch.Tensor: Tensor of shape (video_length, num_classes) with time to next occurrence
    """
    video_length, horizon_steps, num_classes = video_anticipation_probs.shape
    
    # Initialize output tensor with horizon_minutes
    output = torch.full((video_length, num_classes), horizon_minutes, dtype=torch.float32)
    
    # Create a time array
    time_array = np.linspace(0, horizon_minutes, horizon_steps)
    
    for t in range(video_length):
        
        probs = video_anticipation_probs[t]

        current_class = np.argmax(probs[0])
        
        # Set time to 0 for the current highest probability class
        output[t, current_class] = 0
        
        # Interpolate probability curves for smoother intersection finding
        interp_probs = [interp1d(time_array, probs[:, c], kind='cubic') for c in range(num_classes)]
        
        # Find the next class occurrence time (remove current class from unpredicted classes)
        unpredicted_classes = set(range(num_classes)) - {current_class}
        current_time = 0
        
        while unpredicted_classes and current_time < horizon_minutes:
            next_intersections = []
            for c in unpredicted_classes:
                intersections = []
                for i in range(1, len(time_array)):
                    if (time_array[i] > current_time and
                        interp_probs[c](time_array[i]) > interp_probs[current_class](time_array[i]) and
                        interp_probs[c](time_array[i-1]) <= interp_probs[current_class](time_array[i-1])):
                        # Linear interpolation to find more precise intersection point
                        x0, x1 = time_array[i-1], time_array[i]
                        y0 = interp_probs[c](x0) - interp_probs[current_class](x0)
                        y1 = interp_probs[c](x1) - interp_probs[current_class](x1)
                        x_intersect = x0 + (x1 - x0) * (-y0 / (y1 - y0))
                        intersections.append(x_intersect)
                if intersections:
                    next_intersections.append((c, min(intersections)))
            
            if not next_intersections:
                break
            
            next_class, next_time = min(next_intersections, key=lambda x: x[1])
            output[t, next_class] = next_time
            unpredicted_classes.remove(next_class)
            current_class = next_class
            current_time = next_time

    # Check if all classes are equal to the max horizon
    if torch.all(output[t] == horizon_minutes):
        raise UserWarning(f"All classes at time step {t} are equal to the maximum horizon.")
    
    return output


def generate_smooth_probs(horizon_steps, num_classes):
    x = np.linspace(0, 1, horizon_steps)
    probs = np.zeros((horizon_steps, num_classes))
    for c in range(num_classes):
        peak = np.random.uniform(0, 1)
        width = np.random.uniform(0.1, 0.3)
        probs[:, c] = np.exp(-((x - peak) / width) ** 2)
    probs /= probs.sum(axis=1, keepdims=True)
    return torch.tensor(probs, dtype=torch.float32)

def visualize_results(video_anticipation_probs, time_to_next, time_step, horizon_minutes):
    _, horizon_steps, num_classes = video_anticipation_probs.shape
    time_array = np.linspace(0, horizon_minutes, horizon_steps)
    probs = video_anticipation_probs[time_step].numpy()

    plt.figure(figsize=(12, 6))

    for c in range(num_classes):
        interp_prob = interp1d(time_array, probs[:, c], kind='cubic')
        plt.plot(time_array, interp_prob(time_array), label=f'Class {c}')
        
        next_occurrence = time_to_next[time_step, c].item()
        plt.axvline(x=next_occurrence, color=f'C{c}', linestyle='--', alpha=0.5)
        plt.text(next_occurrence, 0.1, f'{next_occurrence:.2f}', rotation=90, verticalalignment='bottom')

    plt.xlabel('Time (minutes)')
    plt.ylabel('Probability')
    plt.title(f'Probability Curves and Next Occurrence Times at Time Step {time_step}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    video_length = 100
    num_classes = 5
    horizon_minutes = 18
    horizon_steps = horizon_minutes + 1
    time_step_to_visualize = 0

    video_anticipation_probs = torch.stack([generate_smooth_probs(horizon_steps, num_classes) for _ in range(video_length)])

    time_to_next = find_time_to_next_occurrence(video_anticipation_probs, horizon_minutes)
    
    print("Video anticipation probabilities shape:", video_anticipation_probs.shape)
    print("Time to next occurrence shape:", time_to_next.shape)
    print(f"\nTime to next occurrence for time step {time_step_to_visualize}:")
    print(time_to_next[time_step_to_visualize])

    visualize_results(video_anticipation_probs, time_to_next, time_step_to_visualize, horizon_minutes)
import numpy as np
import torch

class AnticipateStoreRetrieve:
    def __init__(self, num_classes: int, num_frames_anticip: int, anticip_weights: str):
        self.num_classes = num_classes
        self.num_frames_anticip = num_frames_anticip
        self.anticip_weights = anticip_weights
        self.num_frames = 0
        self.predictions = torch.zeros(self.num_frames_anticip, self.num_frames_anticip, self.num_classes)
        # compute the weighted average of the class probabilities
        if self.anticip_weights == "uniform":
            self.weights = torch.ones(self.num_frames_anticip)
        elif self.anticip_weights == "linear":
            self.weights = torch.linspace(0.1, 1, self.num_frames_anticip)
        else:
            raise ValueError(f"Unknown anticipation weights: {self.anticip_weights}")

    def store(self, predictions: np.ndarray):
        """
        Store the predictions at every time step.
        Args:
            predictions: numpy (num_frames_anticpated, num_classes)
        """
        # store the predictions at the current time step until the anticipation window
        self.predictions[self.num_frames] = torch.from_numpy(predictions)
        self.num_frames += 1
    
    def pop(self):
        """
        Remove the oldest predictions from the predictions buffer (after they have been used to make a prediction at time t).
        """
        self.predictions = torch.cat((self.predictions[1:], torch.zeros(1, self.num_frames_anticip, self.num_classes)), dim=0)
        self.num_frames -= 1

    def reset(self):
        """
        Reset the predictions at every new video.
        """
        self.num_frames = 0
        self.predictions = torch.zeros(self.num_frames_anticip, self.num_frames_anticip, self.num_classes)

    def retrieve(self):
        """
        Get the accumulated predictions up to the current time step and remove the oldest predictions from the predictions buffer.

        Returns:
            class_predictions: the class predictions at the current time step with the highest probability through time
        """
        # select the stored anticipations logits for t to t-window (diagnoal of the predictions matrix)
        anticipations = self.predictions.diagonal(offset=0, dim1=0, dim2=1)

        # remove the oldest predictions from the predictions buffer
        self.pop()
        
        # multiply every anticipation with the temporal weights to add more or less confidence to the most recent predictions.
        # and sum the weighted class logits through time to aggregate the predictions
        class_logits = torch.sum(anticipations * self.weights, dim=1) # (C)
        return torch.argmax(class_logits, dim=0).numpy()

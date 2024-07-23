import torch
import torch.nn as nn

class RemainingTimeLoss(nn.Module):
    """
    More sensitive to lower remaining times to encourage the model to predict the remaining time more accurately,
    and deal with the class imbalance problem.
    """
    def __init__(self, beta=1.0, epsilon=1e-6):
        super().__init__()
        self.beta = beta
        self.epsilon = epsilon

    def forward(self, predictions, targets):
        # Calculate base loss (Smooth L1)
        diff = torch.abs(predictions - targets)
        base_loss = torch.where(diff < self.beta, 
                                0.5 * diff ** 2 / self.beta, 
                                diff - 0.5 * self.beta)

        # Calculate weights based on proximity to action
        weights = 1 / (targets + self.epsilon)
        
        # Normalize weights
        weights = weights / weights.sum()

        # Apply weights and calculate mean loss
        weighted_loss = base_loss * weights
        return weighted_loss.mean()
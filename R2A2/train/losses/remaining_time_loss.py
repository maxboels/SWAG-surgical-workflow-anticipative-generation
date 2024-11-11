import torch
import torch.nn as nn


class RemainingTimeLoss(nn.Module):
    def __init__(self, h, beta=1.0, epsilon=1e-6, base_rtd_loss='mse', weight_type='exponential', gamma=0.5, normalize_weights=False):
        super().__init__()
        self.h = h
        self.beta = beta
        self.epsilon = epsilon
        self.weight_type = weight_type
        self.gamma = gamma  # Controls the decay rate of the exponential weighting
        self.normalize_weights = normalize_weights
        self.base_rtd_loss = base_rtd_loss

        self.rsd_weight = 1.5

    def normalize(self, x):
        return x / self.h

    def forward(self, predictions, targets):
        norm_predictions = self.normalize(predictions)
        norm_targets = self.normalize(targets)

        if self.base_rtd_loss == 'smooth_l1':
            diff = torch.abs(norm_predictions - norm_targets)
            base_rtd_loss = torch.where(diff < self.beta, 
                                    0.5 * diff ** 2 / self.beta, 
                                    diff - 0.5 * self.beta)
        elif self.base_rtd_loss == 'mae':
            base_rtd_loss = torch.abs(norm_predictions - norm_targets)
        elif self.base_rtd_loss == 'mse':
            base_rtd_loss = (norm_predictions - norm_targets) ** 2
        else:
            raise ValueError("Invalid base_rtd_loss. Choose 'mse', 'mae', or 'smooth_l1'.")

        if self.weight_type == 'balanced_exponential':
            # New balanced exponential weighting
            weights = torch.exp(-self.gamma * norm_targets)
            # Add a linear component to increase weight for longer times
            weights += (1 - self.gamma) * norm_targets
        elif self.weight_type == 'inverse':
            weights = 1 / (norm_targets + self.epsilon)
        elif self.weight_type == 'exponential':
            weights = torch.exp(-norm_targets)
        elif self.weight_type == 'linear':
            weights = 1 - 0.5 * norm_targets  # Less aggressive linear weighting
        else:
            raise ValueError("Invalid weight_type. Choose 'balanced_exponential', 'inverse', 'exponential', or 'linear'.")

        if self.rsd_weight > 1:
            weights[-1] *= self.rsd_weight # focus a bit more on the end of surgery class

        if self.normalize_weights:
            weights = weights / weights.mean()

        weighted_loss = base_rtd_loss * weights
        return weighted_loss.mean()



import torch
import torch.nn as nn

class InMAEZoneSensitiveLoss(nn.Module):
    def __init__(self, h, high_weight=2.0, low_weight=1.0, gamma=1.0, 
                 use_exponential=False, 
                 normalize_weights=False,
                 rsd_weight=1.5):
        super().__init__()
        self.h = h  # Horizon defining the inMAE zone
        self.high_weight = high_weight  # Weight for errors within the inMAE zone
        self.low_weight = low_weight    # Weight for errors outside the inMAE zone
        self.gamma = gamma              # Controls the decay rate if exponential weighting is used
        self.use_exponential = use_exponential  # Flag to use exponential weighting
        self.normalize_weights = normalize_weights
        self.rsd_weight = rsd_weight

    def forward(self, predictions, targets):
        # Calculate the base loss using MSE
        base_rtd_loss = (predictions - targets) ** 2

        # Define the inMAE zone
        inMAE_zone = (targets >= 0) & (targets < self.h)

        # Initialize weights with low_weight
        weights = torch.full_like(targets, self.low_weight)

        if self.use_exponential:
            # Apply exponential weights within the inMAE zone
            weights[inMAE_zone] = self.high_weight * torch.exp(-self.gamma * (targets[inMAE_zone] / self.h))
        else:
            # Assign high_weight within the inMAE zone
            weights[inMAE_zone] = self.high_weight

        # Multiply the weight for the last class by rsd_weight to increase its importance
        weights[-1] *= self.rsd_weight

        # Normalize weights if required
        if self.normalize_weights:
            weights = weights / weights.mean()

        # Calculate the weighted loss
        weighted_loss = base_rtd_loss * weights
        return weighted_loss.mean()



import torch
import torch.nn as nn

class ExponentialRemainingTimeLoss(nn.Module):
    def __init__(self, h, max_weight=2.0, gamma=5.0, 
                 normalize_weights=False,
                 rsd_weight=1.5):
        super().__init__()
        self.h = h  # Total horizon
        self.max_weight = max_weight  # Maximum weight at targets=0
        self.gamma = gamma            # Controls the rate of weight increase
        self.normalize_weights = normalize_weights
        self.rsd_weight = rsd_weight

    def forward(self, predictions, targets):
        # Calculate the base loss using MSE
        base_rtd_loss = (predictions - targets) ** 2

        # Calculate the weights
        weights = 1 + (self.max_weight - 1) * torch.exp(-self.gamma * targets / self.h)

        # Apply rsd_weight to the last element if necessary
        weights[-1] *= self.rsd_weight

        # Normalize weights if required
        if self.normalize_weights:
            weights = weights / weights.mean()

        # Calculate the weighted loss
        weighted_loss = base_rtd_loss * weights
        return weighted_loss.mean()


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    loss_fn = InMAEZoneSensitiveLoss(h=90.0, max_weight=2.0, gamma=5.0)


    h = 90.0
    targets = torch.linspace(0, h, steps=100)
    gamma = 5.0  # Adjust gamma to control the sharpness of the increase
    max_weight = 2.0

    weights = 1 + (max_weight - 1) * torch.exp(-gamma * targets / h)


    # Balanced Exponential Weights
    gammas = [1.0, 2.0, 3.0, 4.0, 5.0, 10.0]
    plt.figure(figsize=(10, 6))
    for gamma in gammas:
        weights = 1 + (max_weight - 1) * torch.exp(-gamma * targets / h)
        plt.plot(targets.numpy(), weights.numpy(), label=f'gamma={gamma}')
    plt.xlabel('Target Time')
    plt.ylabel('Weight')
    plt.title('Weights vs Target Time for Different gamma Values')
    plt.legend()
    plt.grid(True)
    plt.show()


import torch
import torch.nn as nn

class ClassicRemainingTimeLoss(nn.Module):
    def __init__(self, beta=1.0, epsilon=1e-6, weight_type='balanced_exponential', gamma=0.5, normalize_weights=False):
        super().__init__()
        self.beta = beta
        self.epsilon = epsilon
        self.weight_type = weight_type
        self.gamma = gamma  # Controls the decay rate of the exponential weighting
        self.normalize_weights = normalize_weights

    def forward(self, predictions, targets):
        # Calculate the base loss using MSE
        base_rtd_loss = (predictions - targets) ** 2

        # Apply balanced exponential weighting
        weights = torch.exp(-self.gamma * targets)
        weights += (1 - self.gamma) * targets

        # Normalize weights if required
        if self.normalize_weights:
            weights = weights / weights.mean()

        # Calculate the weighted loss
        weighted_loss = base_rtd_loss * weights
        return weighted_loss.mean()

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
                 rsd_weight=2.0):
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



if __name__ == '__main__':


    # Test the loss function
    import matplotlib.pyplot as plt
    import numpy as np

    h = 5.0
    targets = torch.linspace(0, 20, steps=100)
    loss_fn = InMAEZoneSensitiveLoss(h=h, high_weight=2.8, low_weight=1.0, gamma=1.0, use_exponential=True)

    inMAE_zone = (targets >= 0) & (targets < h)
    weights = torch.full_like(targets, loss_fn.low_weight)
    weights[inMAE_zone] = loss_fn.high_weight * torch.exp(-loss_fn.gamma * (targets[inMAE_zone] / h))

    plt.plot(targets.numpy(), weights.numpy())
    plt.xlabel('Target Time')
    plt.ylabel('Weight')
    plt.title('Weights vs Target Time')
    plt.show()



# class RemainingTimeLoss(nn.Module):
#     """
#     Normalizes inputs and targets to [0, 1] range and applies weighting
#     to be more sensitive to lower remaining times.

#     Inputs:
#     - predictions: remaining_time = (1 - self.sigmoid(next_phase_occurrence)) * self.max_anticip_time
#     - targets: remaining_time [0, self.max_anticip_time]
#     """
#     def __init__(self, h, beta=1.0, epsilon=1e-6, base_rtd_loss='mse', weight_type='exponential', alpha=0.1, normalize_weights=False, log_scale=10):
#         super().__init__()
#         self.h = h
#         self.beta = beta
#         self.epsilon = epsilon
#         self.weight_type = weight_type
#         self.alpha = alpha  # Controls the minimum weight for linear weighting        
#         self.normalize_weights = normalize_weights
#         self.log_scale = log_scale  # Scaling factor for logarithmic weighting

#         self.base_rtd_loss = base_rtd_loss

#     def normalize(self, x):
#         return x / self.h

#     def forward(self, predictions, targets):

#         # Normalize predictions and targets to [0, 1] range instead of [0, h]
#         norm_predictions = self.normalize(predictions)
#         norm_targets = self.normalize(targets)

#         if self.base_rtd_loss == 'smooth_l1':
#             # Calculate base loss (Smooth L1)
#             diff = torch.abs(norm_predictions - norm_targets)
#             base_rtd_loss = torch.where(diff < self.beta, 
#                                     0.5 * diff ** 2 / self.beta, 
#                                     diff - 0.5 * self.beta)
#         elif self.base_rtd_loss == 'mae':
#             # Calculate base loss Mean Absolute Error (MAE) or L1 loss
#             base_rtd_loss = torch.abs(norm_predictions - norm_targets)
#         elif self.base_rtd_loss == 'mse':
#             # Calculate base loss Mean Squared Error (MSE) or L2 loss
#             base_rtd_loss = (norm_predictions - norm_targets) ** 2
#         else:
#             raise ValueError("Invalid base_rtd_loss. Choose 'mse', 'mae', or 'smooth_l1'.")

#         # Calculate weights based on normalized targets
#         if self.weight_type == 'inverse':
#             weights = 1 / (norm_targets + self.epsilon)
#         elif self.weight_type == 'exponential':
#             weights = torch.exp(-norm_targets)
#         elif self.weight_type == 'linear':
#             weights = (1 - ((1 - self.alpha) * norm_targets))
#         elif self.weight_type == 'logarithmic':
#             weights = torch.log(self.log_scale * norm_targets + 1)
#         else:
#             raise ValueError("Invalid weight_type. Choose 'inverse', 'exponential', 'linear', or 'logarithmic'.")

#         # Normalize weights if specified
#         if self.normalize_weights:
#             weights = weights / weights.mean()

#         # Apply weights and calculate mean loss
#         weighted_loss = base_rtd_loss * weights
#         return weighted_loss.mean()

# class RemainingTimeLoss(nn.Module):
#     """
#     More sensitive to lower remaining times to encourage the model to predict the remaining time more accurately,
#     and deal with the class imbalance problem. Normalizes based on h, provides option for Smooth L1 or MSE loss,
#     and includes a scaling factor to adjust loss magnitude.
#     """
#     def __init__(self, h, beta=1.0, epsilon=1e-6, use_smooth_l1=False, scale_factor=1):
#         super().__init__()
#         self.h = h
#         self.beta = beta
#         self.epsilon = epsilon
#         self.use_smooth_l1 = use_smooth_l1
#         self.scale_factor = scale_factor

#     def forward(self, predictions, targets):
#         # Normalize predictions and targets
#         norm_predictions = predictions / self.h
#         norm_targets = targets / self.h

#         if self.use_smooth_l1:
#             # Calculate base loss (Smooth L1)
#             diff = torch.abs(norm_predictions - norm_targets)
#             base_rtd_loss = torch.where(diff < self.beta, 
#                                     0.5 * diff ** 2 / self.beta, 
#                                     diff - 0.5 * self.beta)
#         else:
#             # Calculate base loss (MSE)
#             base_rtd_loss = (norm_predictions - norm_targets) ** 2

#         # Calculate weights based on proximity to action
#         weights = 1 / (norm_targets + self.epsilon)
        
#         # Normalize weights
#         weights = weights / weights.sum()

#         # Apply weights and calculate mean loss
#         weighted_loss = base_rtd_loss * weights

#         # Scale the loss
#         scaled_loss = weighted_loss.mean() * self.scale_factor

#         return scaled_loss

# class RemainingTimeLoss(nn.Module):
#     """
#     More sensitive to lower remaining times to encourage the model to predict the remaining time more accurately,
#     and deal with the class imbalance problem.
#     """
#     def __init__(self, beta=1.0, epsilon=1e-6):
#         super().__init__()
#         self.beta = beta
#         self.epsilon = epsilon

#     def forward(self, predictions, targets):
#         # Calculate base loss (Smooth L1)
#         diff = torch.abs(predictions - targets)
#         base_rtd_loss = torch.where(diff < self.beta, 
#                                 0.5 * diff ** 2 / self.beta, 
#                                 diff - 0.5 * self.beta)

#         # Calculate weights based on proximity to action
#         weights = 1 / (targets + self.epsilon)
        
#         # Normalize weights
#         weights = weights / weights.sum()

#         # Apply weights and calculate mean loss
#         weighted_loss = base_rtd_loss * weights
#         return weighted_loss
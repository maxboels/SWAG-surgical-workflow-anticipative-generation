import torch
import torch.nn as nn

class RemainingTimeLoss(nn.Module):
    """
    Normalizes inputs and targets to [0, 1] range and applies weighting
    to be more sensitive to lower remaining times.
    """
    def __init__(self, h, beta=1.0, epsilon=1e-6, base_rtd_loss='l1', weight_type='exponential', alpha=0.1, normalize_weights=False):
        super().__init__()
        self.h = h
        self.beta = beta
        self.epsilon = epsilon
        # self.use_smooth_l1 = use_smooth_l1
        self.weight_type = weight_type
        self.alpha = alpha  # Controls the minimum weight for linear weighting        
        self.normalize_weights = normalize_weights

        self.base_rtd_loss = base_rtd_loss

    def normalize(self, x):
        return x / self.h

    def forward(self, predictions, targets):
        # Normalize predictions and targets to [0, 1] range
        norm_predictions = self.normalize(predictions)
        norm_targets = self.normalize(targets)

        if self.base_rtd_loss == 'smooth_l1':
            # Calculate base loss (Smooth L1)
            diff = torch.abs(norm_predictions - norm_targets)
            base_rtd_loss = torch.where(diff < self.beta, 
                                    0.5 * diff ** 2 / self.beta, 
                                    diff - 0.5 * self.beta)
        elif self.base_rtd_loss == 'l1':
            # Calculate base loss (L1) or Mean Absolute Error (MAE)
            base_rtd_loss = torch.abs(norm_predictions - norm_targets)
        elif self.base_rtd_loss == 'l2':
            # Calculate base loss (L2) or Mean Squared Error (MSE)
            base_rtd_loss = (norm_predictions - norm_targets) ** 2
        else:
            raise ValueError("Invalid base_rtd_loss. Choose 'l1', 'l2', or 'smooth_l1'.")

        # Calculate weights based on normalized targets
        if self.weight_type == 'inverse':
            weights = 1 / (norm_targets + self.epsilon)
        elif self.weight_type == 'exponential':
            weights = torch.exp(-norm_targets)
        elif self.weight_type == 'linear':
            # scale_factor = 10
            weights = (1 - ((1 - self.alpha) * norm_targets))
        else:
            raise ValueError("Invalid weight_type. Choose 'inverse', 'exponential', or 'linear'.")

        # Normalize weights if specified
        if self.normalize_weights:
            weights = weights / weights.mean()

        # Apply weights and calculate mean loss
        weighted_loss = base_rtd_loss * weights
        return weighted_loss.mean()

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
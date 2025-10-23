import torch
import torch.nn as nn

class anticipation_mae(nn.Module):
    """ Anticipation Mean Absolute Error
    Inputs:
    - output_rsd: (B, 1, C) tensor of predicted remaining time
    - target_rsd: (B, 1, C) tensor of ground truth remaining time
    - h: scalar, maximum time horizon
    - ignore_index: int or list of ints, classes to ignore when computing overall MAE
    """
    def __init__(self, h=None, ignore_index=None):
        super(anticipation_mae, self).__init__()
        self.h = torch.tensor(h).float()
        
        # Ensure ignore_index is a list if not None
        if ignore_index is not None and not isinstance(ignore_index, list):
            self.ignore_index = [ignore_index]
        else:
            self.ignore_index = ignore_index

    def forward(self, output_rsd, target_rsd):
        # Squeeze the second dimension if it's 1
        output_rsd = output_rsd.squeeze(1)
        target_rsd = target_rsd.squeeze(1)

        # Compute pairwise MAE (keeping the same shape)
        mae = torch.abs(output_rsd - target_rsd)  # shape (B, C)

        # Compute masks for different conditions
        mask_out = (target_rsd >= self.h)
        mask_in = (target_rsd < self.h) & (target_rsd > 0)
        mask_exp = (target_rsd < 0.1 * self.h) & (target_rsd > 0)  # expected

        # For per-class MAE calculations
        # Compute sums and counts per class
        in_mae_sum = (mae * mask_in.float()).sum(dim=0)  # shape (C,)
        in_mae_count = mask_in.sum(dim=0)

        out_mae_sum = (mae * mask_out.float()).sum(dim=0) # shape (C,)
        out_mae_count = mask_out.sum(dim=0)

        exp_mae_sum = (mae * mask_exp.float()).sum(dim=0) # shape (C,)
        exp_mae_count = mask_exp.sum(dim=0)

        # Handle zero counts to avoid division by zero
        in_mae_per_class = in_mae_sum / in_mae_count.float()
        in_mae_per_class[in_mae_count == 0] = float('nan')

        out_mae_per_class = out_mae_sum / out_mae_count.float()
        out_mae_per_class[out_mae_count == 0] = float('nan')

        exp_mae_per_class = exp_mae_sum / exp_mae_count.float()
        exp_mae_per_class[exp_mae_count == 0] = float('nan')

        # Compute per-class weighted MAE as the average of in_mae_per_class and out_mae_per_class
        w_mae_per_class = (in_mae_per_class + out_mae_per_class) / 2

        # Exclude ignore_index classes when computing overall MAEs
        if self.ignore_index is not None:
            valid_indices = [i for i in range(mae.size(-1)) if i not in self.ignore_index]
            in_mae_overall = in_mae_per_class[valid_indices].nanmean()
            out_mae_overall = out_mae_per_class[valid_indices].nanmean()
            exp_mae_overall = exp_mae_per_class[valid_indices].nanmean()
        else:
            in_mae_overall = in_mae_per_class.nanmean()
            out_mae_overall = out_mae_per_class.nanmean()
            exp_mae_overall = exp_mae_per_class.nanmean()

        # Compute overall wMAE as the average of in_mae_overall and out_mae_overall
        w_mae_overall = (in_mae_overall + out_mae_overall) / 2

        # Convert to float
        w_mae_overall = w_mae_overall.cpu().item()
        in_mae_overall = in_mae_overall.cpu().item()
        out_mae_overall = out_mae_overall.cpu().item()
        exp_mae_overall = exp_mae_overall.cpu().item()
        w_mae_per_class = w_mae_per_class.cpu().numpy().round(4).tolist()
        in_mae_per_class = in_mae_per_class.cpu().numpy().round(4).tolist()
        out_mae_per_class = out_mae_per_class.cpu().numpy().round(4).tolist()
        exp_mae_per_class = exp_mae_per_class.cpu().numpy().round(4).tolist()

        return {
            'wMAE': w_mae_overall,
            'inMAE': in_mae_overall,
            'outMAE': out_mae_overall,
            'expMAE': exp_mae_overall,
            'wMAE_class': w_mae_per_class,
            'inMAE_class': in_mae_per_class,
            'outMAE_class': out_mae_per_class,
            'expMAE_class': exp_mae_per_class,
        }

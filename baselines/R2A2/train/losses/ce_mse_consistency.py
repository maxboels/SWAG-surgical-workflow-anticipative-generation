from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class CEConsistencyMSE(nn.Module):
    """
    Loss from MS-TCN paper. CrossEntropy + MSE
    https://arxiv.org/abs/1903.01945
    """
    def __init__(self,
                ignore_idx: int,
                ce_weight: torch.Tensor,
                mse_fraction: float,
                mse_clip_val: float,
                num_classes: int):
        super(CEConsistencyMSE, self).__init__()
        self.ce = nn.CrossEntropyLoss(weight=ce_weight, ignore_index=ignore_idx)
        self.mse = nn.MSELoss(reduction='none')
        self.mse_fraction = mse_fraction
        self.mse_clip_val = mse_clip_val
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        :param logits: [n_stages, batch_size, n_classes, seq_len]
        :param targets: [batch_size, seq_len]
        :return:
        """

        loss_dict = {"loss": 0.0, "loss_ce": 0.0, "loss_mse": 0.0}

        if logits.dim() == 4:
            for p in logits:
                # flatten the batch and seq_len dimensions
                loss_dict['loss_ce'] += self.ce(rearrange(p, "b n_classes seq_len -> (b seq_len) n_classes"),
                                                rearrange(targets, "b seq_len -> (b seq_len)"))

                # MSE loss for temporal consistency
                loss_dict['loss_mse'] += torch.mean(torch.clamp(self.mse(F.log_softmax(p[:, :, 1:], dim=1),
                                                                        F.log_softmax(p.detach()[:, :, :-1], dim=1)),
                                                                min=0,
                                                                max=self.mse_clip_val))
        else:
            loss_dict['loss_ce'] = self.ce(rearrange(logits, "b n_classes seq_len -> (b seq_len) n_classes"),
                                           rearrange(targets, "b seq_len -> (b seq_len)"))

            # MSE loss for temporal consistency
            loss_dict['loss_mse'] = torch.mean(torch.clamp(self.mse(F.log_softmax(logits[:, :, 1:], dim=1),
                                                                    F.log_softmax(logits.detach()[:, :, :-1], dim=1)),
                                                            min=0,
                                                            max=self.mse_clip_val))

        
        loss = loss_dict['loss_ce'] + self.mse_fraction * loss_dict['loss_mse']

        return loss




import torch
import torch.nn.functional as F

class TemporalCrossEntropy(torch.nn.Module):
    def __init__(self, weight=None, size_average=None, ignore_index=-1,
                 reduce=None, reduction='none', target_frame_weights=None, temporal_reduction='mean'):
        super(TemporalCrossEntropy, self).__init__()
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.reduce = reduce
        self.reduction = reduction
        self.target_frame_weights = target_frame_weights
        self.temporal_reduction = temporal_reduction

    def forward(self, input, target):
        """
        Computes the forward pass of the TemporalCrossEntropy loss function.

        Args:
            input (torch.Tensor): The input tensor of shape (B, C, T).
            target (torch.Tensor): The target tensor of shape (B, T).

        Returns:
            torch.Tensor: The computed loss tensor.
        """
        assert input.dim() == 3, "Input tensor must have 3 dimensions (B, C, T) but has {}".format(input.dim())
        assert target.dim() == 2, "Target tensor must have 2 dimensions (B, T) but has {}".format(target.dim())
        assert input.size(0) == target.size(0), "Batch size of input and target must match: input: {} != target: {}".format(input.size(0), target.size(0))
        assert input.size(2) == target.size(1), "Temporal dimension of input and target must match: input: {} != target: {}".format(input.size(2), target.size(1))
        if self.weight is not None:
            assert input.size(1) == self.weight.size(0), "Number of classes in input and weight must match: input: {} != weight: {}".format(input.size(1), self.weight.size(0))
        if self.target_frame_weights is not None:
            assert input.size(2) == len(self.target_frame_weights), "Temporal dimension of input and target frame weights must match: input: {} != target frame weights: {}".format(input.size(2), len(self.target_frame_weights))


        # compute loss with weights and target frame weights
        loss = F.cross_entropy(input, target, weight=self.weight, ignore_index=self.ignore_index,
                                 reduction=self.reduction)
        
        # multiply loss by target frame weights
        if self.target_frame_weights is not None:
            loss *= self.target_frame_weights
        # (B, T) -> (0)
        if self.reduction == "mean" and self.temporal_reduction == "mean":
            loss = loss.mean(dim=0)
        # (B, T) -> (B,)
        elif self.reduction == "none" and self.temporal_reduction == "mean":
            loss = loss.mean(dim=1)        
        # (B, T) -> (B, T)
        elif self.reduction == "mean" and self.temporal_reduction == "none":
            loss = loss
        return loss
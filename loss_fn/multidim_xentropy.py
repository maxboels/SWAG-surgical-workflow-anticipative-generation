# Copyright (c) Facebook, Inc. and its affiliates.


"""Cross entropy loss, that works with multi-dim input."""
import torch
import torch.nn as nn
from common.cluster import KmeansAssigner
import torch.nn.functional as F



import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    Cross entropy loss with class weights based on the observed class and the prediction index position.
    """

    def __init__(self, weights_sampler, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights_sampler = weights_sampler

    def forward(self, inp, tgt, *args, **kwargs):
        """
        Select the class weights from a dict for the CE loss based on the observed class.
        Args:
            inp: Predicted logits with shape (B, C, N)
            tgt: Target tensor with shape (B, N)
        """
        batch_size = inp.shape[0]
        num_classes = inp.shape[1]
        num_preds = inp.shape[2]

        losses = []
        
        for batch_idx in range(batch_size):
            for pred_idx in range(num_preds):
                observed_class = tgt[batch_idx, pred_idx].item()
                print(f"Observed class: {observed_class}")
                # if the target is -1, ignore this prediction
                if observed_class == -1:
                    continue
                class_probs = self.weights_sampler.return_class_weights(observed_class, pred_idx)
                print(f"Pred index: {pred_idx}")
                print(f"Class probs: {class_probs}")

                # Initialize class_weights with small values to avoid log(0)
                class_weights = torch.zeros(num_classes) + 0.001

                # Update class_weights based on the sampler
                for key, value in class_probs.items():
                    class_weights[key] = value
                
                # Ensure class_weights are on the same device as input
                class_weights = class_weights.to(inp.device)
                print(f"Class weights: {class_weights}")
                print(f"Class weights shape: {class_weights.shape}")
                print(f"Class weights device: {class_weights.device}")

                # Compute cross-entropy loss for this specific batch and class index
                loss = F.cross_entropy(inp[batch_idx:batch_idx+1, :, pred_idx:pred_idx+1],
                                       tgt[batch_idx:batch_idx+1, pred_idx:pred_idx+1],
                                       weight=class_weights, 
                                       *args, **kwargs)
                losses.append(loss)
        
        # TODO: add the class frequencies to the loss

        return torch.stack(losses).mean()






class MultiDimCrossEntropy(nn.CrossEntropyLoss):
    def forward(self, inp, tgt, *args, **kwargs):
        """
        Args:
            inp: (*, C)
            tgt: (*, )
            Will reshape the flatten initial dimensions and then incur loss
        """
        assert inp.ndim == tgt.ndim + 1
        assert inp.shape[:-1] == tgt.shape
        res = super().forward(inp.reshape(-1, inp.size(-1)), tgt.reshape(
            (-1, )), *args, **kwargs)
        if torch.numel(res) == torch.numel(tgt):
            # Reduction was not done, so reshape back to orig shape
            res = res.reshape(tgt.shape)
        return res

class MultiDimNN(nn.NLLLoss):
    def forward(self, inp, tgt, *args, **kwargs):
        """
        Args:
            inp: (*, C)
            tgt: (*, )
            Will reshape the flatten initial dimensions and then incur loss
        """
        assert inp.ndim == tgt.ndim + 1
        assert inp.shape[:-1] == tgt.shape
        res = super().forward(inp.reshape(-1, inp.size(-1)), tgt.reshape(
            (-1, )), *args, **kwargs)
        if torch.numel(res) == torch.numel(tgt):
            # Reduction was not done, so reshape back to orig shape
            res = res.reshape(tgt.shape)
        return res

class QuantizeAndCrossEntropy(MultiDimCrossEntropy):
    """Given a set of cluster centers, project the features to that before
    incurring the loss."""
    def __init__(self, centroids_fpath, norm=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.assigner = KmeansAssigner(centroids_fpath)
        self.norm = norm

    def forward(self, inp, tgt):
        """
        Args:
            inp: (*, C)
            tgt: (*, C)
            Will reshape the flatten initial dimensions and then incur loss
        """
        # Normalize L2 both target and input, since that's how I'm computing
        # centroids
        if self.norm:
            inp = nn.functional.normalize(inp, dim=-1, p=2)
            tgt = nn.functional.normalize(tgt, dim=-1, p=2)
        # assign the GT and predictions to the centroids
        inp_proj = torch.mm(inp.flatten(0, 1),
                            self.centroids.t()).view(inp.shape[:-1] +
                                                     self.centroids.shape[:1])
        # the weights of project layer are the centroids, so pick from there
        tgt_proj_q = self.assigner(tgt)
        return super().forward(inp_proj, tgt_proj_q)

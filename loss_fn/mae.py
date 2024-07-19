import torch
import torch.nn as nn

import torch
import torch.nn as nn

class anticipation_mae(nn.Module):
    """ Anticipation Mean Absolute Error
    Inputs:
    - output_rsd: (B, T, C) tensor of predicted remaining time
    - target_rsd: (B, T, C) tensor of ground truth remaining time
    """

    def __init__(self, h=18, ignore_index=None):
        super(anticipation_mae, self).__init__()
        self.h = torch.tensor(h).float()
        self.ignore_index = ignore_index

    def forward(self, output_rsd, target_rsd):
        # Squeeze the second dimension if it's 1
        output_rsd = output_rsd.squeeze(1)
        target_rsd = target_rsd.squeeze(1)

        # Handle ignore_index
        if self.ignore_index is not None:
            if self.ignore_index == -1 or self.ignore_index == output_rsd.size(-1) - 1:
                output_rsd = output_rsd[..., :-1]
                target_rsd = target_rsd[..., :-1]
            elif 0 <= self.ignore_index < output_rsd.size(-1):
                output_rsd = torch.cat([output_rsd[..., :self.ignore_index], output_rsd[..., self.ignore_index+1:]], dim=-1)
                target_rsd = torch.cat([target_rsd[..., :self.ignore_index], target_rsd[..., self.ignore_index+1:]], dim=-1)
            else:
                raise ValueError(f"Invalid ignore_index {self.ignore_index}")

        # Clip target values to be within [0, h]
        target_rsd = torch.clamp(target_rsd, 0, self.h)

        # Compute pairwise MAE
        mae = torch.abs(output_rsd - target_rsd)

        # Compute masks for different conditions
        mask_out = (target_rsd == self.h)
        mask_in = (target_rsd < self.h) & (target_rsd > 0)
        mask_exp = (target_rsd < 0.1 * self.h) & (target_rsd > 0)

        # Compute MAEs for different conditions
        in_mae = mae[mask_in].mean() if mask_in.any() else torch.tensor(float('nan'))
        out_mae = mae[mask_out].mean() if mask_out.any() else torch.tensor(float('nan'))
        exp_mae = mae[mask_exp].mean() if mask_exp.any() else torch.tensor(float('nan'))

        # Compute weighted MAE
        w_mae = torch.stack([out_mae, in_mae]).nanmean()

        return w_mae, in_mae, out_mae, exp_mae




class attention_loss(nn.Module):
    def __init__(self, sigma = 0.05):
        super(attention_loss, self).__init__()


        self.sigma = sigma



    def forward(self, a ):

        loss  = self.sigma*torch.sum(a)

        return loss


class framewise_ce(nn.Module):
    def __init__(self, sigma = 0.05):
        super(framewise_ce, self).__init__()


        self.ce = nn.CrossEntropyLoss(size_average=True)



    def forward(self, x, target):

        loss = 0
        for b in range(x.size(0)):

            loss+= self.ce(x[b,:,:],target[b,:])


        return loss


class similarity_loss(nn.Module):
    def __init__(self, sigma = 0.05):
        super(similarity_loss, self).__init__()


        self.similarity_loss = nn.CosineSimilarity(dim=2)



    def forward(self, output,target ):

        loss  = torch.mean(self.similarity_loss(output,target))

        return loss

class class_wise_anticipation_mae(nn.Module):

    def __init__(self, h=7500):
        super(class_wise_anticipation_mae, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.h = torch.tensor(h).float().to(self.device)
        self.lossfunc = anticipation_mae(h)
        self.base = [0]

    def forward(self, output_rsd, target_rsd):

        c = output_rsd.size(-1)
        wMAE_loss_list  = self.base*c
        inMAE_loss_list = self.base*c
        pMAE_loss_list = self.base*c
        eMAE_loss_list = self.base*c

        for i in range(c):

            tmp_output_rsd = output_rsd[:,:,i:i+1]
            tmp_target_rsd = target_rsd[:,:,i:i+1]
            wMAE_loss_list[i] , inMAE_loss_list[i],pMAE_loss_list[i], eMAE_loss_list[i] = self.lossfunc(tmp_output_rsd,tmp_target_rsd)

        output_loss_list = [wMAE_loss_list, inMAE_loss_list,pMAE_loss_list,eMAE_loss_list]
        return output_loss_list

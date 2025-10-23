import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DistributionUncertainty(nn.Module):
    """
    Distribution Uncertainty Module
        Args:
        p   (float): probabilty of foward distribution uncertainty module, p in [0,1].
        dim   (int): dimension of feature map channels
    """

    def __init__(self, p=0.5, eps=1e-6, dim=-1):
        super(DistributionUncertainty, self).__init__()
        self.eps = eps
        self.p = p
        self.factor = 1.0

    def _reparameterize(self, mu, std):
        epsilon = torch.randn_like(std) * self.factor
        return mu + epsilon * std

    def sqrtvar(self, x):
        t = (x.var(dim=0, keepdim=True) + self.eps).sqrt()
        t = t.repeat(x.shape[0], 1)
        return t

    def forward(self, x):
        x = x.permute(0,2,1)
        if (not self.training) or (np.random.random()) > self.p:
            x = x.permute(0,2,1)
            return x

        mean = x.mean(dim=[2], keepdim=False)
        std = (x.var(dim=[2], keepdim=False) + self.eps).sqrt()

        sqrtvar_mu = self.sqrtvar(mean)
        sqrtvar_std = self.sqrtvar(std)

        beta = self._reparameterize(mean, sqrtvar_mu)
        gamma = self._reparameterize(std, sqrtvar_std)

        x = (x - mean.reshape(x.shape[0], x.shape[1], 1)) / std.reshape(x.shape[0], x.shape[1], 1)
        x = x * gamma.reshape(x.shape[0], x.shape[1], 1) + beta.reshape(x.shape[0], x.shape[1], 1)
        x = x.permute(0,2,1)
        return x
class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        #self.dsu1 = DistributionUncertainty(p = 0.5)
        #self.dsu2 = DistributionUncertainty(p = 0.5)
        #self.dsu3 = DistributionUncertainty(p = 0.5)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        
        #if x.size(1) != 1 and x.size(2)!=1:
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])

        #if self.training:
        #    x = self.dsu1(x)
        x = self.norm1(x)
        x = x + self.dropout(self.self_attention(
                x, x, x,
                attn_mask=x_mask
        )[0])
        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))

        return self.norm3(x+y)

class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        y = x
        if self.layers is not None:
            for layer in self.layers:
                x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            y = x
        
        else:
            y = cross
        
        if self.norm is not None:
            y = self.norm(y)

        return y
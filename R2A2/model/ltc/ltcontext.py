import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from . import MultiHeadAttention, AttentionParams, convert_to_patches

import hydra
from hydra.types import TargetConf
from omegaconf import OmegaConf


class BaseAttention(nn.Module):
    """
    Base module for attention.
    """
    def __init__(self,
                 model_dim: int,
                 attn_params: AttentionParams
                 ):
        super().__init__()
        out_proj = nn.Sequential(
            nn.GELU(),
            nn.Conv1d(model_dim // 2, model_dim, kernel_size=1, bias=True)
        )

        self.attn = MultiHeadAttention(model_dim,
                                       attn_params,
                                       out_proj=out_proj,
                                       use_conv1d_proj=True,
                                       dim_key=model_dim // 2,
                                       dim_value=model_dim // 2)


class WindowedAttention(BaseAttention):
    def __init__(self,
                 windowed_attn_w: int,
                 model_dim: int,
                 attn_params: AttentionParams):
        super().__init__(model_dim, attn_params)
        self.windowed_attn_w = windowed_attn_w

    def _reshape(self, x: Tensor, overlapping_patches: bool, masks: Tensor = None):
        """

        :param x:
        :param overlapping_patches:
        :param masks:
        :return:
            patches: a Tensor of shape [batch_size * num_patches, model_dim, patch_size]
            attn_mask: a Tensor of shape [batch_size * num_patches, patch_size]
        """
        patches, masks = convert_to_patches(x, self.windowed_attn_w, masks, overlapping_patches)
        # combine num_windows to the batch dimension
        patches = rearrange(patches, "b d num_patches patch_size -> (b num_patches) d patch_size")
        masks = rearrange(masks,   "b d num_patches patch_size -> (b num_patches) d patch_size")
        return patches.contiguous(), masks.contiguous()

    def _undo_reshape(self, patches: Tensor, batch_size: int, orig_seq_len: int):
        num_patches = patches.shape[0] // batch_size
        x = rearrange(patches,
                      "(b num_patches) d patch_size -> b d (num_patches patch_size)",
                      num_patches=num_patches,
                      b=batch_size)

        x = x[:, :, :orig_seq_len]
        return x.contiguous()

    def _prep_qkv(self, qk: Tensor, v: Tensor, masks: Tensor):
        """
         Prepare the query, key and value for attention
         by applying the transform_func.
        :param qk:
        :param v:
        :param masks:
        :return:
        """
        q, k = qk, qk
        q, _ = self._reshape(q, overlapping_patches=False)
        k, attn_mask = self._reshape(k, overlapping_patches=True, masks=masks)
        if v is not None:
            v, _ = self._reshape(v, overlapping_patches=True)
        else:
            v = k

        return q, k, v, attn_mask

    def forward(self, qk: Tensor, v: Tensor = None, masks: Tensor = None):
        """

        :param qk: Tensor with shape [batch_size, model_dim, seq_len]
        :param v: Tensor with shape [batch_size, model_dim, seq_len]
        :param masks: Tensor with shape [batch_size, 1, seq_len]
        :return:
        """
        batch_size, _, seq_len = qk.shape
        q, k, v, att_mask = self._prep_qkv(qk=qk, v=v, masks=masks)
        windowed_attn = self.attn(query=q, key=k, value=v, att_mask=att_mask)
        out = self._undo_reshape(windowed_attn, batch_size, seq_len)
        if masks is not None:
            out = out * masks
        return out


class LTContextAttention(BaseAttention):
    def __init__(self, long_term_attn_g, model_dim, attn_params):
        super().__init__(model_dim, attn_params)
        self.long_term_attn_g = long_term_attn_g

    def _reshape(self, x: Tensor, masks: Tensor = None):
        """

        :param x:
        :param masks:
        :return:
            lt_patches: a Tensor of shape [batch_size * patch_size, model_dim, num_patches]
            attn_mask: a Tensor of shape [batch_size * patch_size, 1, num_patches]
        """
        patches, masks = convert_to_patches(x, self.long_term_attn_g, masks)

        # transpose and combine patch_size to the batch dimension which means tokens are taken with stride 'patch_size'
        lt_patches = rearrange(patches, "b d num_patches patch_size -> (b patch_size) d num_patches")
        masks = rearrange(masks, "b d num_patches patch_size -> (b patch_size) d num_patches")

        return lt_patches.contiguous(), masks.contiguous()

    def _undo_reshape(self, lt_patches, batch_size, orig_seq_len):
        x = rearrange(lt_patches,
                      "(b patch_size) d num_patches -> b d (num_patches patch_size)",
                      patch_size=self.long_term_attn_g,
                      b=batch_size)
        x = x[:, :, :orig_seq_len]
        return x.contiguous()

    def _prep_qkv(self, qk: Tensor, v: Tensor, masks: Tensor):
        """
         Prepare the query, key and value for attention by applying the transform_func.
        :param qk:
        :param v:
        :param masks:
        :return:
        """
        q, k = qk, qk
        q, _ = self._reshape(q)
        k, attn_mask = self._reshape(k, masks=masks)
        if v is not None:
            v, _ = self._reshape(v)
        else:
            v = k

        return q, k, v, attn_mask

    def forward(self, qk: Tensor, v: Tensor = None, masks: Tensor = None):
        """

        :param qk: Tensor with shape [batch_size, model_dim, seq_len]
        :param v: Tensor with shape [batch_size, model_dim, seq_len]
        :param masks: Tensor with shape [batch_size, 1, seq_len]
        :return:
        """
        batch_size, _, seq_len = qk.shape
        q, k, v, att_mask = self._prep_qkv(qk=qk, v=v, masks=masks)
        lt_attn = self.attn(query=q, key=k, value=v, att_mask=att_mask)
        out = self._undo_reshape(lt_attn, batch_size, seq_len)
        if masks is not None:
            out = out * masks
        return out


class DilatedConv(nn.Module):
    def __init__(self, n_channels: int, dilation: int, kernel_size: int = 3):
        super().__init__()
        self.dilated_conv = nn.Conv1d(n_channels,
                                      n_channels,
                                      kernel_size=kernel_size,
                                      padding=dilation,
                                      dilation=dilation,
                                      )
        self.activation = nn.GELU()

    def forward(self, x, masks):
        """
        x: Tensor with shape [batch_size, n_channels, seq_len]
        masks: Tensor with shape [batch_size, 1, seq_len]
        """
        return self.activation(self.dilated_conv(x)) * masks


class LTCBlock(nn.Module):
    def __init__(self,
                 model_dim: int,
                 dilation: int,
                 windowed_attn_w: int,
                 long_term_attn_g: int,
                 attn_params: AttentionParams,
                 use_instance_norm: bool,
                 dropout_prob: float):
        super(LTCBlock, self).__init__()

        self.dilated_conv = DilatedConv(n_channels=model_dim, dilation=dilation, kernel_size=3)

        if use_instance_norm:
            self.instance_norm = nn.Identity()
        else:
            self.instance_norm = nn.InstanceNorm1d(model_dim)

        self.windowed_attn = WindowedAttention(windowed_attn_w=windowed_attn_w,
                                               model_dim=model_dim,
                                               attn_params=attn_params,
                                               )
        self.ltc_attn = LTContextAttention(long_term_attn_g=long_term_attn_g,
                                           model_dim=model_dim,
                                           attn_params=attn_params,
                                           )

        self.out_linear = nn.Conv1d(model_dim, model_dim, kernel_size=1, bias=True)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, inputs: Tensor, masks: Tensor, prev_stage_feat: Tensor = None):
        """

        :param inputs:
        :param masks:
        :param prev_stage_feat:
        :return:
        """
        # nn.Conv1d
        out = self.dilated_conv(inputs, masks)
        # use previous window as query and current window as key and value
        out = self.windowed_attn(self.instance_norm(out), prev_stage_feat, masks) + out
        
        out = self.ltc_attn(self.instance_norm(out), prev_stage_feat, masks) + out
        out = self.out_linear(out)
        out = self.dropout(out)
        out = out + inputs
        return out * masks


class LTCModule(nn.Module):
    def __init__(self,
                 num_layers: int,
                 input_dim: int,
                 model_dim: int,
                 num_classes: int,
                 attn_params: AttentionParams,
                 dilation_factor: int,
                 windowed_attn_w: int,
                 long_term_attn_g: int,
                 use_instance_norm: bool,
                 dropout_prob: float,
                 channel_dropout_prob: float):
        super(LTCModule, self).__init__()
        self.channel_dropout = nn.Dropout1d(channel_dropout_prob)
        self.input_proj = nn.Conv1d(input_dim, model_dim, kernel_size=1, bias=True)
        self.layers = nn.ModuleList([])
        for i in range(num_layers):
            self.layers.append(
                LTCBlock(
                    model_dim=model_dim,
                    dilation=dilation_factor**i,
                    windowed_attn_w=windowed_attn_w,
                    long_term_attn_g=long_term_attn_g,
                    attn_params=attn_params,
                    use_instance_norm=use_instance_norm,
                    dropout_prob=dropout_prob
                    )
            )
        self.out_proj = nn.Conv1d(model_dim, num_classes, kernel_size=1, bias=True)

    def forward(self, inputs: Tensor, masks: Tensor, prev_stage_feat: Tensor = None):
        inputs = self.channel_dropout(inputs)
        feature = self.input_proj(inputs)
        for layer in self.layers:
            feature = layer(feature, masks, prev_stage_feat)
        out = self.out_proj(feature) * masks
        return out, feature


class LTC(nn.Module):
    """
    Long-Term Context model
    """
    def __init__(
        self,
        input_dim: int = 768,
        num_classes: int = 7,
        # modules
        attn_cfg: TargetConf = None,
        ltc_cfg: TargetConf = None,
    ):                
        super(LTC, self).__init__()
        self.model_dim = ltc_cfg.model_dim
        self.num_stages = ltc_cfg.num_stages


        attn_params = AttentionParams(num_heads=attn_cfg.num_attn_heads,
                                      dropout=attn_cfg.dropout
                                      )

        self.stage1 = LTCModule(num_layers=ltc_cfg.num_layers,
                                input_dim=input_dim,
                                model_dim=ltc_cfg.model_dim,
                                num_classes=num_classes,
                                attn_params=attn_params,
                                dilation_factor=ltc_cfg.conv_dilation_factor,
                                windowed_attn_w=ltc_cfg.windowed_attn_w,
                                long_term_attn_g=ltc_cfg.long_term_attn_g,
                                use_instance_norm=ltc_cfg.use_instance_norm,
                                dropout_prob=ltc_cfg.dropout_prob,
                                channel_dropout_prob=ltc_cfg.channel_masking_prob
                                )

        reduced_dim = int(ltc_cfg.model_dim // ltc_cfg.dim_reduction)
        self.dim_reduction = nn.Conv1d(ltc_cfg.model_dim,
                                       reduced_dim,
                                       kernel_size=1,
                                       bias=True)
        self.stages = nn.ModuleList([])
        for s in range(1, ltc_cfg.num_stages):
            self.stages.append(
                LTCModule(num_layers=ltc_cfg.num_layers,
                          input_dim=num_classes,
                          model_dim=reduced_dim,
                          num_classes=num_classes,
                          attn_params=attn_params,
                          dilation_factor=ltc_cfg.conv_dilation_factor,
                          windowed_attn_w=ltc_cfg.windowed_attn_w,
                          long_term_attn_g=ltc_cfg.long_term_attn_g,
                          use_instance_norm=ltc_cfg.use_instance_norm,
                          dropout_prob=ltc_cfg.dropout_prob,
                          channel_dropout_prob=ltc_cfg.channel_masking_prob
                          )
            )
        
        self.avg_pooling = nn.AdaptiveAvgPool1d(1) # param: output size of last dimension.
        self.max_pooling = nn.AdaptiveMaxPool1d(20) # param: output size of last dimension.
        # project to 512 dim
        # self.proj_512 = nn.Conv1d(ltc_cfg.model_dim, 512, kernel_size=1, bias=True)

    def forward(self, inputs: Tensor, masks: Tensor) -> Tensor:
        """
        :param inputs: Tensor with shape [batch_size, input_dim_size, sequence_length]
        :param masks: Tensor with shape [batch_size, 1, sequence_length]
        :return: outputs: Tensor with shape [batch_size, num_classes, sequence_length]
        """
        # inputs = inputs.permute(0, 2, 1)
        print(f"[LTC] input shape: {inputs.shape}")
        print(f"[LTC] mask shape: {masks.shape}")

        out, feature = self.stage1(inputs, masks)
        print(f"[LTC] stage 1 output shape: {out.shape}")
        print(f"[LTC] stage 1 feature shape: {feature.shape}")
        output_list = [out]
        # max_feature = self.max_pooling(feature) # added by mboels
        # print(f"[LTC] stage 1 maxpooled feature shape: {max_feature.shape}")
        features_list = [feature] # added by mboels
        feature = self.dim_reduction(feature)
        print(f"[LTC] stage 1 feature shape after dim reduction: {feature.shape}")
        for i, stage in enumerate(self.stages):
            out, feature = stage(F.softmax(out, dim=1) * masks,
                                 prev_stage_feat=feature * masks,
                                 masks=masks)
            # max_feature = self.max_pooling(feature) # added by mboels
            # print(f"[LTC] stage maxpooled feature shape: {max_feature.shape}")
            print(f"[LTC] {i}th stage feature shape: {feature.shape}")
            output_list.append(out)
            features_list.append(feature)
        logits = torch.stack(output_list, dim=0)
        features = torch.cat(features_list, dim=1) # (B, stages*dim_reduction, mp_seq_len)
        print(f"[LTC] multi_stage logits shape: {logits.shape}")
        print(f"[LTC] feature out shape: {features.shape}")

        # feature = self.proj_512(feature)
        # print(f"[LTC] feature out shape after 512 proj: {feature.shape}")

        return logits, features

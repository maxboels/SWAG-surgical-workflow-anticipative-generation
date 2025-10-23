# Copyright (c) Facebook, Inc. and its affiliates.

"""
The overall base model.
"""
from typing import Dict, Tuple
import operator
import torch
import torch.nn as nn
import hydra
from omegaconf import OmegaConf

CLS_MAP_PREFIX = 'cls_map_'
PAST_LOGITS_PREFIX = 'past_'


class BaseModel(nn.Module):
    def __init__(self, model_cfg: OmegaConf, num_classes: Dict[str, int],
                 class_mappings: Dict[Tuple[str, str], torch.FloatTensor]):
        super().__init__()

        backbone_dim = 768
        self.mapper_to_inter = None
        if model_cfg.intermediate_featdim is None:
            model_cfg.intermediate_featdim = backbone_dim



        # LOAD MODEL
        if model_cfg.model_name=="supra":
            self.model = hydra.utils.instantiate(model_cfg.supra,
                in_features=model_cfg.intermediate_featdim,
                _recursive_=False)
        elif model_cfg.model_name=="skit":
            self.model = hydra.utils.instantiate(model_cfg.skit, _recursive_=False)
        elif model_cfg.model_name == "lstm":
            self.model = hydra.utils.instantiate(model_cfg.lstm,
                in_features=model_cfg.intermediate_featdim,
                _recursive_=False)
        else:
            raise ValueError(f"model_name {model_cfg.model_name} not supported")


        # Projection layer
        self.project_mlp = nn.Sequential()
        if model_cfg.project_dim_for_nce is not None:
            self.project_mlp = nn.Sequential(
                nn.Linear(temp_agg_output_dim, temp_agg_output_dim),
                nn.ReLU(inplace=True),
                nn.Linear(temp_agg_output_dim, model_cfg.project_dim_for_nce))
        # 2nd round of temporal aggregation, if needed
        #self.temporal_aggregator_after_future_pred = hydra.utils.instantiate(
        #    model_cfg.temporal_aggregator_after_future_pred,
        #    self.model.output_dim)
        # Dropout
        self.dropout = nn.Dropout(model_cfg.dropout)
        # Takes as input (B, C**) -> (B, num_classes)
        #cls_input_dim = self.temporal_aggregator_after_future_pred.output_dim
        # Make a separate classifier for each output
        self.classifiers = nn.ModuleDict()
        self.num_classes = num_classes
        cls_input_dim = 512
        for i, (cls_type, cls_dim) in enumerate(num_classes.items()):
            if model_cfg.use_cls_mappings and i > 0:
                # In this case, rely on the class mappings to generate the
                # other predictions, rather than creating a new linear layer
                break
            self.classifiers.update({
                cls_type:
                hydra.utils.instantiate(model_cfg.classifier,
                                        in_features=cls_input_dim,
                                        out_features=cls_dim)
            })

        self.regression_head = None
        #if model_cfg.add_regression_head:
        #    self.regression_head = nn.Linear(cls_input_dim, 1)
        # Init weights, as per the video resnets
        self._initialize_weights()
        # Set he BN momentum and eps here, Du uses a different value and its imp
        self._set_bn_params(model_cfg.bn.eps, model_cfg.bn.mom)
        self.cfg = model_cfg
        self.sigmoid = nn.Sigmoid()

    def _initialize_weights(self):
        # Copied over from
        # https://github.com/pytorch/vision/blob/75f5b57e680549d012b3fc01b356b2fb92658ea7/torchvision/models/video/resnet.py#L261
        # Making sure all layers get init to good video defaults
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _set_bn_params(self, bn_eps=1e-3, bn_mom=0.1):
        """
        Set the BN parameters to the defaults: Du's models were trained
            with 1e-3 and 0.9 for eps and momentum resp.
            Ref: https://github.com/facebookresearch/VMZ/blob/f4089e2164f67a98bc5bed4f97dc722bdbcd268e/lib/models/r3d_model.py#L208
        """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm3d):
                module.eps = bn_eps
                module.momentum = bn_mom

    def forward(self, video, current_gt, train_mode=True):
        # Mean pool the spatial dimensions = squeeze the spatial dims
        video_feats = torch.mean(video, [-1, -2])
        # Move the time dimension inside: B,C,T -> B,T,C
        video_feats = video_feats.permute((0, 2, 1))
        # Forward Pass
        outputs = self.model(video_feats, current_gt=current_gt, train_mode=train_mode)

        return outputs

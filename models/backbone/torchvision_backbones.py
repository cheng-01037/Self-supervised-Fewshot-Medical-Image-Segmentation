"""
Backbones supported by torchvison.
"""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

class TVDeeplabRes101Encoder(nn.Module):
    """
    FCN-Resnet101 backbone from torchvision deeplabv3
    No ASPP is used as we found emperically it hurts performance
    """
    def __init__(self, use_coco_init, aux_dim_keep = 64, use_aspp = False):
        super().__init__()
        _model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=use_coco_init, progress=True, num_classes=21, aux_loss=None)
        if use_coco_init:
            print("###### NETWORK: Using ms-coco initialization ######")
        else:
            print("###### NETWORK: Training from scratch ######")

        _model_list = list(_model.children())
        self.aux_dim_keep = aux_dim_keep
        self.backbone = _model_list[0]
        self.localconv = nn.Conv2d(2048, 256,kernel_size = 1, stride = 1, bias = False) # reduce feature map dimension
        self.asppconv = nn.Conv2d(256, 256,kernel_size = 1, bias = False)

        _aspp = _model_list[1][0]
        _conv256 = _model_list[1][1]
        self.aspp_out = nn.Sequential(*[_aspp, _conv256] )
        self.use_aspp = use_aspp

    def forward(self, x_in, low_level):
        """
        Args:
            low_level: whether returning aggregated low-level features in FCN
        """
        fts = self.backbone(x_in)
        if self.use_aspp:
            fts256 = self.aspp_out(fts['out'])
            high_level_fts = fts256
        else:
            fts2048 = fts['out']
            high_level_fts = self.localconv(fts2048)

        if low_level:
            low_level_fts = fts['aux'][:, : self.aux_dim_keep]
            return high_level_fts, low_level_fts
        else:
            return high_level_fts






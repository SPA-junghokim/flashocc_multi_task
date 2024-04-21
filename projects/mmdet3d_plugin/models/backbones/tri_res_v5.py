# Copyright (c) Phigent Robotics. All rights reserved.

import torch.utils.checkpoint as checkpoint
import torch
from torch import nn

from mmcv.cnn.bricks.conv_module import ConvModule
from mmdet.models.backbones.resnet import BasicBlock, Bottleneck
from mmdet3d.models import BACKBONES
from mmdet3d.models import builder

@BACKBONES.register_module()
class CustomTriResV5(nn.Module):
    def __init__(
            self,
            img_bev_encoder_backbone,
            img_bev_encoder_neck,
            grid_config,
            num_planes=3,
            out_enc=True):
        super(CustomTriResV5, self).__init__()
        self.create_grid_infos(**grid_config)
        self.num_planes = num_planes
        self.channels = img_bev_encoder_backbone['numC_input']
        
        self.encoder_backbones = nn.ModuleList()
        self.encoder_necks = nn.ModuleList()
        self.enc_linear = nn.ModuleList()
        self.dec_linear = nn.ModuleList()
        for i in range(num_planes):
            self.enc_linear.append(nn.Sequential(
                nn.Conv2d(self.channels * int(self.grid_size[-i-1]), self.channels, kernel_size=1),
                nn.BatchNorm2d(self.channels),
                nn.ReLU()))
            self.dec_linear.append(nn.Sequential(
                nn.Conv2d(self.channels, int(self.grid_size[-i-1]) * self.channels, kernel_size=1),
                nn.BatchNorm2d(int(self.grid_size[-i-1]) * self.channels),
                nn.ReLU()))
            self.encoder_backbones.append(
                builder.build_backbone(img_bev_encoder_backbone))
            self.encoder_necks.append(
                builder.build_neck(img_bev_encoder_neck))
        self.out = out_enc
        if out_enc:
            self.out_enc = nn.Sequential(
                nn.Conv3d(self.channels * 3, self.channels, kernel_size=1),
                nn.BatchNorm3d(self.channels),
                nn.ReLU()
            )
        
    def forward(self, x):
        """
        Args:
            x: (B, C=64, Dy, Dx)
        Returns:
            feats: List[
                (B, 2*C, Dy/2, Dx/2),
                (B, 4*C, Dy/4, Dx/4),
                (B, 8*C, Dy/8, Dx/8),
            ]
        """
        B, C, Z, H, W = x.shape
        
        feats = []
        for lid, (enc, dec, backbone, neck) in enumerate(zip(self.enc_linear,
                                                        self.dec_linear,
                                                        self.encoder_backbones, 
                                                        self.encoder_necks)):
            res = x.clone()
            feat = x.reshape(x.shape[0], -1, *x.shape[3:])
            feat = enc(feat)
            feat = neck(backbone(feat))
            feat = dec(feat).reshape(*res.shape) + res
            feats.append(feat)
            x = x.permute(0,1,3,4,2)

        feats[1] = feats[1].permute(0,1,4,2,3)
        feats[2] = feats[2].permute(0,1,3,4,2)
        out = torch.cat(feats, dim=1)
        if self.out:
            out = self.out_enc(out)
        
        return out
    
    def create_grid_infos(self, x, y, z, **kwargs):
        """Generate the grid information including the lower bound, interval,
        and size.

        Args:
            x (tuple(float)): Config of grid alone x axis in format of
                (lower_bound, upper_bound, interval).
            y (tuple(float)): Config of grid alone y axis in format of
                (lower_bound, upper_bound, interval).
            z (tuple(float)): Config of grid alone z axis in format of
                (lower_bound, upper_bound, interval).
            **kwargs: Container for other potential parameters
        """
        self.grid_lower_bound = torch.Tensor([cfg[0] for cfg in [x, y, z]])     # (min_x, min_y, min_z)
        self.grid_interval = torch.Tensor([cfg[2] for cfg in [x, y, z]])        # (dx, dy, dz)
        self.grid_size = torch.Tensor([(cfg[1] - cfg[0]) / cfg[2]
                                       for cfg in [x, y, z]])                   # (Dx, Dy, Dz)
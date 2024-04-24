# Copyright (c) Phigent Robotics. All rights reserved.

import torch.utils.checkpoint as checkpoint
import torch
from torch import nn

from mmcv.cnn.bricks.conv_module import ConvModule
from mmdet.models.backbones.resnet import BasicBlock, Bottleneck
from mmdet3d.models import BACKBONES
from mmdet3d.models import builder

@BACKBONES.register_module()
class CustomTriResV6(nn.Module):
    def __init__(
            self,
            img_bev_encoder_backbone,
            img_bev_encoder_neck,
            grid_config,
            num_planes=3):
        super(CustomTriResV6, self).__init__()
        self.create_grid_infos(**grid_config)
        self.num_planes = num_planes
        self.channels = img_bev_encoder_backbone['numC_input']
        
        self.encoder_backbones = nn.ModuleList()
        self.enc_linear = nn.ModuleList()
        for i in range(num_planes):
            self.enc_linear.append(nn.Sequential(
                nn.Conv2d(self.channels * int(self.grid_size[-i-1]), self.channels, kernel_size=1),
                nn.BatchNorm2d(self.channels),
                nn.ReLU()))
            self.encoder_backbones.append(
                builder.build_backbone(img_bev_encoder_backbone))
        
        self.encoder_neck = builder.build_neck(img_bev_encoder_neck)
        self.up = nn.Upsample(
            scale_factor=2, mode='trilinear', align_corners=True)
        self.out_enc = nn.Sequential(
            nn.Conv3d(self.channels, self.channels, kernel_size=1),
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
        res = x.clone()
        for lid, (enc, backbone) in enumerate(zip(self.enc_linear,
                                                  self.encoder_backbones)):
            feat = x.reshape(x.shape[0], -1, *x.shape[3:])
            feat = enc(feat)
            if lid==0:
                feat_1, feat_2, feat_3 = backbone(feat)
                feat_1 = feat_1.unsqueeze(2)
                feat_2 = feat_2.unsqueeze(2)
                feat_3 = feat_3.unsqueeze(2)
            elif lid==1:
                feat = backbone(feat)
                feat_1 = feat_1 + feat[0].unsqueeze(2).permute(0,1,4,2,3)
                feat_2 = feat_2 + feat[1].unsqueeze(2).permute(0,1,4,2,3)
                feat_3 = feat_3 + feat[2].unsqueeze(2).permute(0,1,4,2,3)
            else:
                feat = backbone(feat)
                feat_1 = feat_1 + feat[0].unsqueeze(2).permute(0,1,3,4,2)
                feat_2 = feat_2 + feat[1].unsqueeze(2).permute(0,1,3,4,2)
                feat_3 = feat_3 + feat[2].unsqueeze(2).permute(0,1,3,4,2)
            x = x.permute(0,1,3,4,2)

        out = self.encoder_neck([feat_1,feat_2,feat_3])
        out = self.out_enc(self.up(out))
        
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
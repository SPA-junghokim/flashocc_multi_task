# Copyright (c) Phigent Robotics. All rights reserved.

import torch.utils.checkpoint as checkpoint
import torch
from torch import nn

from mmcv.cnn.bricks.conv_module import ConvModule
from mmdet.models.backbones.resnet import BasicBlock, Bottleneck
from mmdet3d.models import BACKBONES
from mmdet3d.models import builder

@BACKBONES.register_module()
class CustomTriRes(nn.Module):
    def __init__(
            self,
            img_bev_encoder_backbone,
            img_bev_encoder_neck,
            grid_config,
            num_planes=3,
            residual=False,
            attn=False):
        super(CustomTriRes, self).__init__()
        self.create_grid_infos(**grid_config)
        self.num_planes = num_planes
        self.residual = residual
        self.attn = attn
        self.channels = img_bev_encoder_backbone['numC_input']
        
        self.encoder_backbones = nn.ModuleList()
        self.encoder_necks = nn.ModuleList()
        self.linears = nn.ModuleList()
        for i in range(num_planes):
            self.linears.append(
                nn.Linear(self.channels * int(self.grid_size[-i-1]), self.channels),
            )
            self.encoder_backbones.append(
                builder.build_backbone(img_bev_encoder_backbone))
            self.encoder_necks.append(
                builder.build_neck(img_bev_encoder_neck))
        self.out_enc = nn.Sequential(
            nn.Conv3d(self.channels*3, self.channels, kernel_size=1),
            nn.BatchNorm3d(self.channels),
            nn.ReLU()
        )
        
        if self.attn:
            self.pooling_layer = nn.Sequential(
                nn.Linear(self.channels, self.channels*2),
                nn.BatchNorm1d(self.channels*2),
                nn.ReLU(),
                nn.Linear(self.channels*2, self.channels),
            )
            self.attn_layer1 = nn.Linear(self.channels, self.channels * int(self.grid_size[2]))
            self.attn_layer2 = nn.Linear(self.channels, self.channels * int(self.grid_size[1]))
            self.attn_layer3 = nn.Linear(self.channels, self.channels * int(self.grid_size[0]))
        
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
        if self.residual:
            res = x.clone()
        feats = []
        B, C, Z, H, W = x.shape
        if self.attn:
            squeeze = x.reshape(B, C, -1)
            pool_feats = self.pooling_layer(torch.mean(squeeze, dim=-1)) # BCN -> BC -> BC 
        
        x_tmp = []
        x_tmp.append(x.permute(0,3,4,2,1).reshape(B,H,W,-1))
        x_tmp.append(x.permute(0,4,2,3,1).reshape(B,W,Z,-1))
        x_tmp.append(x.permute(0,2,3,4,1).reshape(B,Z,H,-1))

        for lid, (linear, backbone, neck) in enumerate(zip(self.linears, 
                                                    self.encoder_backbones, 
                                                    self.encoder_necks)):
            tmp = linear(x_tmp[lid]).permute(0,3,1,2).contiguous()
            tmp = backbone(tmp)
            feats.append(neck(tmp))

        if not self.attn:
            feats[0] = feats[0][:,:,None,:,:] # BCHW -> BC1HW
            feats[1] = feats[1][:,:,:,None,:].permute(0,1,4,3,2) # BCWZ -> BCW1Z -> BCZ1W
            feats[2] = feats[2][:,:,:,:,None] # BCZH -> BCZH1
        else:
            feats[0] = feats[0][:,:,None,:,:] * self.attn_layer1(pool_feats).reshape(B,C,Z,1,1).sigmoid()
            feats[1] = feats[1][:,:,:,None,:].permute(0,1,4,3,2) * self.attn_layer2(pool_feats).reshape(B,C,1,H,1).sigmoid()
            feats[2] = feats[2][:,:,:,:,None] * self.attn_layer3(pool_feats).reshape(B,C,1,1,W).sigmoid()
            
        fused_feats = torch.cat([feats[0], feats[1], feats[2]], dim=1)
        if self.residual:
            fused_feats = res + fused_feats
        out = self.out_enc(fused_feats)
        
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
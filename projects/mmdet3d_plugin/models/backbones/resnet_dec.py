# Copyright (c) Phigent Robotics. All rights reserved.

import torch.utils.checkpoint as checkpoint
import torch
from torch import nn

from mmcv.cnn.bricks.conv_module import ConvModule
from mmdet.models.backbones.resnet import BasicBlock, Bottleneck
from mmdet3d.models import BACKBONES

class BasicBlock3D(nn.Module):
    def __init__(self,
                 channels_in, channels_out, stride=1, downsample=None):
        super(BasicBlock3D, self).__init__()
        self.conv1 = ConvModule(
            channels_in,
            channels_out,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='BN3d', ),
            act_cfg=dict(type='ReLU',inplace=True))
        self.conv2 = ConvModule(
            channels_out,
            channels_out,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='BN3d', ),
            act_cfg=None)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + identity
        return self.relu(x)

class BasicBlock3D_dec(nn.Module):
    def __init__(self,
                 channels_in, channels_out, stride=1, downsample=None):
        super(BasicBlock3D_dec, self).__init__()
        
        gc_out = int(channels_out * 0.25)
        self.split_out = (channels_out - 3 * gc_out, gc_out, gc_out, gc_out)
        self.conv1 = nn.Conv3d(
            gc_out,
            gc_out,
            kernel_size=(1,1,1),
            stride=1,
            padding=(0,0,0),)
        self.conv1_0 = nn.Conv3d(
            gc_out,
            gc_out,
            kernel_size=(3,3,1),
            stride=1,
            padding=(1,1,0),)
        self.conv1_1 = nn.Conv3d(
            gc_out,
            gc_out,
            kernel_size=(3,1,3),
            stride=1,
            padding=(1,0,1),)
        self.conv1_2 = nn.Conv3d(
            gc_out,
            gc_out,
            kernel_size=(1,3,3),
            stride=1,
            padding=(0,1,1),)
        self.norm1 = nn.BatchNorm3d(channels_out)
        self.act1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv3d(
            gc_out,
            gc_out,
            kernel_size=(1,1,1),
            stride=1,
            padding=(0,0,0),)
        self.conv2_0 = nn.Conv3d(
            gc_out,
            gc_out,
            kernel_size=(3,3,1),
            stride=1,
            padding=(1,1,0),)
        self.conv2_1 = nn.Conv3d(
            gc_out,
            gc_out,
            kernel_size=(3,1,3),
            stride=1,
            padding=(1,0,1),)
        self.conv2_2 = nn.Conv3d(
            gc_out,
            gc_out,
            kernel_size=(1,3,3),
            stride=1,
            padding=(0,1,1),)
        self.norm2 = nn.BatchNorm3d(channels_out)
        self.act2 = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        identity = x
        x_id, x_0, x_1, x_2 = torch.split(x, self.split_out, dim=1)
        x = torch.cat([self.conv1(x_id), self.conv1_0(x_0), self.conv1_1(x_1), self.conv1_2(x_2)], dim=1)
        x = self.act1(self.norm1(x))

        x_id, x_0, x_1, x_2 = torch.split(x, self.split_out, dim=1)
        x = torch.cat([self.conv2(x_id), self.conv2_0(x_0), self.conv2_1(x_1), self.conv2_2(x_2)], dim=1)
        x = self.act2(self.norm2(x))

        x = x + identity
        return self.relu(x)


@BACKBONES.register_module()
class CustomResNet3D_dec(nn.Module):
    def __init__(
            self,
            numC_input,
            num_layer=[2, 2, 2],
            num_channels=None,
            stride=[2, 2, 2],
            backbone_output_ids=None,
            with_cp=False,
    ):
        super(CustomResNet3D_dec, self).__init__()
        # build backbone
        assert len(num_layer) == len(stride)
        num_channels = [numC_input * 2 ** (i + 1) for i in range(len(num_layer))] \
            if num_channels is None else num_channels
        self.backbone_output_ids = range(len(num_layer)) \
            if backbone_output_ids is None else backbone_output_ids
        layers = []
        curr_numC = numC_input
        for i in range(len(num_layer)):
            layer = [
                BasicBlock3D_dec(
                    curr_numC,
                    num_channels[i],
                    stride=stride[i],
                    downsample=ConvModule(
                        curr_numC,
                        num_channels[i],
                        kernel_size=3,
                        stride=stride[i],
                        padding=1,
                        bias=False,
                        conv_cfg=dict(type='Conv3d'),
                        norm_cfg=dict(type='BN3d', ),
                        act_cfg=None))
            ]
            curr_numC = num_channels[i]
            layer.extend([
                BasicBlock3D_dec(curr_numC, curr_numC)
                for _ in range(num_layer[i] - 1)
            ])
            layers.append(nn.Sequential(*layer))
        self.layers = nn.Sequential(*layers)

        self.with_cp = with_cp

    def forward(self, x):
        """
        Args:
            x: (B, C, Dz, Dy, Dx)
        Returns:
            feats: List[
                (B, C, Dz, Dy, Dx),
                (B, 2C, Dz/2, Dy/2, Dx/2),
                (B, 4C, Dz/4, Dy/4, Dx/4),
            ]
        """
        feats = []
        x_tmp = x
        for lid, layer in enumerate(self.layers):
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        return feats
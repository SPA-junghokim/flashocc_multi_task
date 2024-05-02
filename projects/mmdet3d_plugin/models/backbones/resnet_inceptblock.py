# Copyright (c) Phigent Robotics. All rights reserved.

import torch.utils.checkpoint as checkpoint
from torch import nn

from mmcv.cnn.bricks.conv_module import ConvModule
from mmdet.models.backbones.resnet import BasicBlock, Bottleneck
from mmdet3d.models import BACKBONES
from .inceptionnext import MetaNeXtBlock, MetaNeXtStage

class LayerNorm2d(nn.LayerNorm):
    """LayerNorm on channels for 2d images.

    Args:
        num_channels (int): The number of channels of the input tensor.
        eps (float): a value added to the denominator for numerical stability.
            Defaults to 1e-5.
        elementwise_affine (bool): a boolean value that when set to ``True``,
            this module has learnable per-element affine parameters initialized
            to ones (for weights) and zeros (for biases). Defaults to True.
    """

    def __init__(self, num_channels: int, **kwargs) -> None:
        super().__init__(num_channels, **kwargs)
        self.num_channels = self.normalized_shape[0]

    def forward(self, x):
        assert x.dim() == 4, 'LayerNorm2d only supports inputs with shape ' \
            f'(N, C, H, W), but got tensor with shape {x.shape}'
        return F.layer_norm(
            x.permute(0, 2, 3, 1).contiguous(), self.normalized_shape,
            self.weight, self.bias, self.eps).permute(0, 3, 1, 2).contiguous()


@BACKBONES.register_module()
class CustomResNet_inc(nn.Module):
    def __init__(
            self,
            numC_input,
            num_layer=[2, 2, 2],
            num_channels=None,
            stride=[2, 2, 2],
            backbone_output_ids=None,
            norm_cfg=dict(type='BN'),
            with_cp=False,
            block_type='Basic',
            mixer_kernel_size=11,
    ):
        super(CustomResNet_inc, self).__init__()
        # build backbone
        assert len(num_layer) == len(stride)
        num_channels = [numC_input*2**(i+1) for i in range(len(num_layer))] \
            if num_channels is None else num_channels
        self.backbone_output_ids = range(len(num_layer)) \
            if backbone_output_ids is None else backbone_output_ids
        layers = []
        if block_type == 'BottleNeck':
            curr_numC = numC_input
            for i in range(len(num_layer)):
                # 在第一个block中对输入进行downsample
                layer = [Bottleneck(inplanes=curr_numC, planes=num_channels[i]//4, stride=stride[i],
                                    downsample=nn.Conv2d(curr_numC, num_channels[i], 3, stride[i], 1),
                                    norm_cfg=norm_cfg)]
                curr_numC = num_channels[i]
                layer.extend([Bottleneck(inplanes=curr_numC, planes=num_channels[i]//4, stride=1,
                                         downsample=None, norm_cfg=norm_cfg) for _ in range(num_layer[i] - 1)])
                layers.append(nn.Sequential(*layer))
        elif block_type == 'Basic':
            curr_numC = numC_input
            for i in range(len(num_layer)):
                # 在第一个block中对输入进行downsample
                layer = [BasicBlock(inplanes=curr_numC, planes=num_channels[i], stride=stride[i],
                                    downsample=nn.Conv2d(curr_numC, num_channels[i], 3, stride[i], 1),
                                    norm_cfg=norm_cfg)]
                curr_numC = num_channels[i]
                layer.extend([MetaNeXtBlock(dim=curr_numC, drop_path=0.1, mlp_ratio=3, mixer_kernel_size=mixer_kernel_size) for _ in range(num_layer[i] - 1)])
                layers.append(nn.Sequential(*layer))
        elif block_type == 'Meta':
            curr_numC = numC_input
            for i in range(len(num_layer)):
                layer = MetaNeXtStage(
                    in_chs=curr_numC, out_chs=num_channels[i], ds_stride=2,
                    depth=2, drop_path_rates=[0., 0.], ls_init_value=1e-6, mlp_ratio=2
                )
                curr_numC = num_channels[i] 
                layers.append(layer)
        else:
            assert False

        self.layers = nn.Sequential(*layers)
        self.with_cp = with_cp

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

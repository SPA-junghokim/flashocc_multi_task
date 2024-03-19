from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F
from mmcv.runner import BaseModule, force_fp32

from mmdet3d.models.builder import HEADS


def sigmoid_xent_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    inputs = inputs.float()
    targets = targets.float()
    return F.binary_cross_entropy_with_logits(inputs, targets, reduction=reduction)


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "mean",
) -> torch.Tensor:
    inputs = inputs.float()
    targets = targets.float()
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss


class BEVGridTransform(nn.Module):
    def __init__(
        self,
        *,
        input_scope: List[Tuple[float, float, float]],
        output_scope: List[Tuple[float, float, float]],
        prescale_factor: float = 1,
    ) -> None:
        super().__init__()
        self.input_scope = input_scope
        self.output_scope = output_scope
        self.prescale_factor = prescale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.prescale_factor != 1:
            x = F.interpolate(
                x,
                scale_factor=self.prescale_factor,
                mode="bilinear",
                align_corners=False,
            )

        coords = []
        for (imin, imax, _), (omin, omax, ostep) in zip(
            self.input_scope, self.output_scope
        ):
            v = torch.arange(omin + ostep / 2, omax, ostep)
            v = (v - imin) / (imax - imin) * 2 - 1
            coords.append(v.to(x.device))

        u, v = torch.meshgrid(coords, indexing="ij")
        grid = torch.stack([v, u], dim=-1)
        grid = torch.stack([grid] * x.shape[0], dim=0)

        x = F.grid_sample(
            x,
            grid,
            mode="bilinear",
            align_corners=False,
        )
        return x


@HEADS.register_module()
class BEVSegmentationHead(nn.Module):
    def __init__(
        self,
        in_channels,
        classes,
        loss_type = 'focal',
        grid_transform=dict(
            input_scope= [[-51.2, 51.2, 0.8], [-51.2, 51.2, 0.8]],
            output_scope= [[-51.2, 51.2, 0.4], [-51.2, 51.2, 0.4]]
        ), # grid_transform is None -> nn.Identity()
        seperate_decoder=False,
        loss_weight=[20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
        show = False,
        
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.classes = classes
        self.loss_type = loss_type
        self.seperate_decoder = seperate_decoder
        self.loss_weight = loss_weight
        
        if grid_transform is not None:
            self.transform = BEVGridTransform(**grid_transform)
        else:
            self.transform = nn.Identity()
            
        
        if self.seperate_decoder:
            self.classifier = nn.ModuleList()
            for _ in range(len(classes)):
                self.classifier.append(nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(True),
                    nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(True),
                    nn.Conv2d(in_channels, 1, 1),
                ))
        else:
            self.classifier = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(True),
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(True),
                nn.Conv2d(in_channels, len(classes), 1),
            )
        self.show = show
        
    def forward(self, x):
        breakpoint()
        # x = self.transform(x).permute(0, 3, 2, 1)
        x = self.transform(x).permute(0, 1, 3, 2)
        if self.seperate_decoder:
            x_out = []
            for decoder_idx in range(len(self.classes)):
                x_out.append(self.classifier[decoder_idx](x))
            x = torch.concat(x_out, dim=1)
        else:
            x = self.classifier(x)
        return x

    @force_fp32()
    def loss(self, x, target, **kwargs):
        loss_dict = dict()
        
        if self.show:
            import random
            import cv2
            import numpy as np
            except_index = 9
            fig_size = (256, 256, 3)
            grid = np.array([[0.4, 0.4, 20]]) 
            canvas = np.zeros(fig_size)
            for i in range(len(self.classes)):
                if i == except_index:
                    continue
                canvas += target[0,i,:,:,None].detach().cpu().numpy().repeat(3, axis=2) * (255//len(self.classes)) * i
            cv2.imwrite(f'seg_visualize_temp/deb_map{random.randint(0,100000)}.png', canvas)
        
        target = target.flip(1)
        breakpoint()
        for index, name in enumerate(self.classes):
            if self.loss_type == "xent":
                loss_ = sigmoid_xent_loss(x[:, index], target[:, index]) * torch.tensor(self.loss_weight[index])
                loss_ = torch.nan_to_num(loss_)
            elif self.loss_type == "focal":
                loss_ = sigmoid_focal_loss(x[:, index], target[:, index]) * torch.tensor(self.loss_weight[index])
                loss_ = torch.nan_to_num(loss_)
            else:
                raise ValueError(f"unsupported loss: {self.loss_type}")
            loss_dict[f"loss_{name}_{self.loss_type}"] = loss_
        return loss_dict
        
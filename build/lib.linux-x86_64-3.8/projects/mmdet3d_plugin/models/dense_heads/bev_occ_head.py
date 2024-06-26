# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from torch import nn
import numpy as np
from mmdet3d.models.builder import HEADS, build_loss
from .lovasz_losses import lovasz_softmax
from torch.nn import functional as F

nusc_class_frequencies = np.array([
    944004,
    1897170,
    152386,
    2391677,
    16957802,
    724139,
    189027,
    2074468,
    413451,
    2384460,
    5916653,
    175883646,
    4275424,
    51393615,
    61411620,
    105975596,
    116424404,
    1892500630
])


@HEADS.register_module()
class BEVOCCHead3D(BaseModule):
    def __init__(self,
                 in_dim=32,
                 out_dim=32,
                 use_mask=True,
                 num_classes=18,
                 use_predicter=True,
                 class_balance=False,
                 class_wise=False,
                 loss_occ=None,
                 sololoss=True,
                 loss_weight=10,
                 weight=[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],
                 head_3dconv = False,
                 ):
        super(BEVOCCHead3D, self).__init__()
        self.out_dim = out_dim
        out_channels = out_dim if use_predicter else num_classes
        self.final_conv = ConvModule(
            in_dim,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            conv_cfg=dict(type='Conv3d')
        )
        self.use_predicter = use_predicter
        self.head_3dconv = head_3dconv
        if use_predicter:
            if self.head_3dconv:
                self.predicter = nn.Sequential(
                    nn.Conv3d(self.out_dim, self.out_dim*2, kernel_size=3, padding=1),
                    # nn.BatchNorm3d(self.out_dim*2),
                    nn.Softplus(),
                    nn.Conv3d(self.out_dim*2, num_classes, 1)
                )
            else:
                self.predicter = nn.Sequential(
                    nn.Linear(self.out_dim, self.out_dim*2),
                    nn.Softplus(),
                    nn.Linear(self.out_dim*2, num_classes),
                )

        self.num_classes = num_classes
        self.use_mask = use_mask
        self.class_balance = class_balance
        self.sololoss = sololoss
        
        if self.class_balance:
            class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies[:num_classes] + 0.001))
            self.weight = class_weights
            loss_occ['class_weight'] = class_weights        # ce loss
        else:
            self.weight = torch.Tensor(weight)
            
        self.loss_occ = build_loss(loss_occ)

        if self.sololoss:
            self.cross_entropy_loss = torch.nn.CrossEntropyLoss(weight=self.weight, ignore_index=255, reduction="mean")
            self.lovasz_softmax_loss = lovasz_softmax
            self.loss_weight=loss_weight
        else:
            self.loss_occ = build_loss(loss_occ)
        
    def forward(self, img_feats):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx)

        Returns:

        """
        # (B, C, Dz, Dy, Dx) --> (B, C, Dz, Dy, Dx) --> (B, Dx, Dy, Dz, C)
        if self.use_predicter:
            # (B, Dx, Dy, Dz, C) --> (B, Dx, Dy, Dz, 2*C) --> (B, Dx, Dy, Dz, n_cls)
            if self.head_3dconv:
                occ_pred = self.final_conv(img_feats)
                occ_pred = self.predicter(occ_pred)
                occ_pred = occ_pred.permute(0, 4, 3, 2, 1)
            else:
                occ_pred = self.final_conv(img_feats).permute(0, 4, 3, 2, 1)
                occ_pred = self.predicter(occ_pred)
            
        return occ_pred

    def loss(self, occ_pred, voxel_semantics, mask_camera, **kwargs):
        """
        Args:
            occ_pred: (B, Dx, Dy, Dz, n_cls)
            voxel_semantics: (B, Dx, Dy, Dz)
            mask_camera: (B, Dx, Dy, Dz)
        Returns:

        """
        loss = dict()
        voxel_semantics = voxel_semantics.long()
        if self.sololoss:
            occ_pred = occ_pred.permute(0,4,1,2,3)
            voxel_loss = self.cross_entropy_loss(occ_pred, voxel_semantics.long()) # x=[8, 17, 128, 128, 16], target = [8, 128, 128, 16]
            # lovasz_softmax_loss = self.lovasz_softmax_loss(F.softmax(x, dim=1), target, ignore=self.ignore_label)
            lovasz_softmax_loss = self.lovasz_softmax_loss(F.softmax(occ_pred, dim=1), voxel_semantics, ignore=255)

            loss['voxel_bev_loss'] = voxel_loss * self.loss_weight
            loss['lovasz_softmax_loss'] = lovasz_softmax_loss * self.loss_weight
        else:
            if self.use_mask:
                mask_camera = mask_camera.to(torch.int32)   # (B, Dx, Dy, Dz)
                # (B, Dx, Dy, Dz) --> (B*Dx*Dy*Dz, )
                voxel_semantics = voxel_semantics.reshape(-1)
                # (B, Dx, Dy, Dz, n_cls) --> (B*Dx*Dy*Dz, n_cls)
                preds = occ_pred.reshape(-1, self.num_classes)
                # (B, Dx, Dy, Dz) --> (B*Dx*Dy*Dz, )
                mask_camera = mask_camera.reshape(-1)
                num_total_samples = mask_camera.sum()
                loss_occ = self.loss_occ(
                    preds,      # (B*Dx*Dy*Dz, n_cls)
                    voxel_semantics,    # (B*Dx*Dy*Dz, )
                    mask_camera,        # (B*Dx*Dy*Dz, )
                    avg_factor=num_total_samples
                )
                loss['loss_occ'] = loss_occ
            else:
                voxel_semantics = voxel_semantics.reshape(-1)
                preds = occ_pred.reshape(-1, self.num_classes)
                loss_occ = self.loss_occ(preds, voxel_semantics,)
                loss['loss_occ'] = loss_occ
        return loss

    def get_occ(self, occ_pred, img_metas=None):
        """
        Args:
            occ_pred: (B, Dx, Dy, Dz, C)
            img_metas:

        Returns:
            List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        """
        occ_score = occ_pred.softmax(-1)    # (B, Dx, Dy, Dz, C)
        occ_res = occ_score.argmax(-1)      # (B, Dx, Dy, Dz)
        occ_res = occ_res.cpu().numpy().astype(np.uint8)     # (B, Dx, Dy, Dz)
        return list(occ_res)


@HEADS.register_module()
class BEVOCCHead2D(BaseModule):
    def __init__(self,
                 in_dim=256,
                 out_dim=256,
                 Dz=16,
                 use_mask=True,
                 num_classes=18,
                 use_predicter=True,
                 class_wise=False,
                 class_balance=False,
                 loss_occ=None,
                 sololoss=False,
                 weight=[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],
                 loss_weight=1,
                 z_embeding=False,
                 channel_down_for_3d=False,
                 ):
        super(BEVOCCHead2D, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.Dz = Dz
        out_channels = out_dim if use_predicter else num_classes * Dz
        self.final_conv = ConvModule(
            self.in_dim,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            conv_cfg=dict(type='Conv2d')
        )
        self.use_predicter = use_predicter
        if use_predicter:
            self.predicter = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim * 2),
                nn.Softplus(),
                nn.Linear(self.out_dim * 2, num_classes * Dz),
            )

        self.use_mask = use_mask
        self.num_classes = num_classes
        self.class_balance = class_balance
        
        self.channel_down_for_3d = channel_down_for_3d
        if self.channel_down_for_3d:
            self.channel_down_for_3d = nn.Linear(channel_down_for_3d, self.in_dim)
            
        
        if self.class_balance:
            class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies[:num_classes] + 0.001))
            self.weight = class_weights
            loss_occ['class_weight'] = class_weights        # ce loss
        else:
            self.weight = torch.Tensor(weight)
            
        self.sololoss = sololoss
        if self.sololoss:
            self.weight = torch.Tensor(weight)
            self.cross_entropy_loss = torch.nn.CrossEntropyLoss(weight=self.weight, ignore_index=255, reduction="mean")
            self.lovasz_softmax_loss = lovasz_softmax
            self.loss_weight=loss_weight
        else:
            self.loss_occ = build_loss(loss_occ)
        
        
        self.z_embeding = z_embeding
        if self.z_embeding:
            self.z_embeding = nn.Embedding(self.Dz, self.out_dim)
            self.predicter = nn.Sequential(
                nn.Conv3d(self.out_dim, self.out_dim//2, kernel_size=3, dilation=2, padding=2),
                nn.BatchNorm3d(self.out_dim//2),
                nn.Conv3d(self.out_dim//2, self.out_dim//4, kernel_size=3, padding=1),
                nn.BatchNorm3d(self.out_dim//4),
                nn.Conv3d(self.out_dim//4, num_classes, 1)
            )
        
    def forward(self, img_feats):
        """
        Args:
            img_feats: (B, C, Dy, Dx)

        Returns:

        """
        if self.channel_down_for_3d:
            B, C, Z, H, W = img_feats.shape
            img_feats = img_feats.reshape(B, C*Z, H, W).permute(0, 2, 3, 1)
            img_feats = self.channel_down_for_3d(img_feats).permute(0, 3, 1, 2)
            
        if self.z_embeding:
            occ_pred = self.final_conv(img_feats)
            occ_pred = occ_pred[..., None] + self.z_embeding.weight.permute(1, 0)[None, :, None, None, :]
            # occ_pred = self.predicter(occ_pred).permute(0,2,3,4,1) # 2024-03-28 Thur 22:46 : mIoU 24.45% mistake of permute, below is correct...
            occ_pred = self.predicter(occ_pred).permute(0,3,2,4,1)
        elif self.use_predicter:
            occ_pred = self.final_conv(img_feats).permute(0, 3, 2, 1)
            bs, Dx, Dy = occ_pred.shape[:3]
            # (B, Dx, Dy, C) --> (B, Dx, Dy, 2*C) --> (B, Dx, Dy, Dz*n_cls)
            occ_pred = self.predicter(occ_pred)
            occ_pred = occ_pred.view(bs, Dx, Dy, self.Dz, self.num_classes)

        return occ_pred

    def loss(self, occ_pred, voxel_semantics, mask_camera, **kwargs):
        """
        Args:
            occ_pred: (B, Dx, Dy, Dz, n_cls)
            voxel_semantics: (B, Dx, Dy, Dz)
            mask_camera: (B, Dx, Dy, Dz)
        Returns:

        """
        loss = dict()
        voxel_semantics = voxel_semantics.long()
        if self.sololoss:
            occ_pred = occ_pred.permute(0,4,1,2,3)
            voxel_loss = self.cross_entropy_loss(occ_pred, voxel_semantics.long()) # x=[8, 17, 128, 128, 16], target = [8, 128, 128, 16]
            # lovasz_softmax_loss = self.lovasz_softmax_loss(F.softmax(x, dim=1), target, ignore=self.ignore_label)
            lovasz_softmax_loss = self.lovasz_softmax_loss(F.softmax(occ_pred, dim=1), voxel_semantics, ignore=255)

            loss['voxel_bev_loss'] = voxel_loss * self.loss_weight
            loss['lovasz_softmax_loss'] = lovasz_softmax_loss * self.loss_weight
        else:
            if self.use_mask:
                mask_camera = mask_camera.to(torch.int32)   # (B, Dx, Dy, Dz)
                # (B, Dx, Dy, Dz) --> (B*Dx*Dy*Dz, )
                voxel_semantics = voxel_semantics.reshape(-1)
                # (B, Dx, Dy, Dz, n_cls) --> (B*Dx*Dy*Dz, n_cls)
                preds = occ_pred.reshape(-1, self.num_classes)
                # (B, Dx, Dy, Dz) --> (B*Dx*Dy*Dz, )
                mask_camera = mask_camera.reshape(-1)
                num_total_samples = mask_camera.sum()
                loss_occ = self.loss_occ(
                    preds,      # (B*Dx*Dy*Dz, n_cls)
                    voxel_semantics,    # (B*Dx*Dy*Dz, )
                    mask_camera,        # (B*Dx*Dy*Dz, )
                    avg_factor=num_total_samples
                )
                loss['loss_occ'] = loss_occ
            else:
                voxel_semantics = voxel_semantics.reshape(-1)
                preds = occ_pred.reshape(-1, self.num_classes)
                loss_occ = self.loss_occ(preds, voxel_semantics)
                loss['loss_occ'] = loss_occ
                
        return loss

    def get_occ(self, occ_pred, img_metas=None):
        """
        Args:
            occ_pred: (B, Dx, Dy, Dz, C)
            img_metas:

        Returns:
            List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        """
        occ_score = occ_pred.softmax(-1)    # (B, Dx, Dy, Dz, C)
        occ_res = occ_score.argmax(-1)      # (B, Dx, Dy, Dz)
        occ_res = occ_res.cpu().numpy().astype(np.uint8)     # (B, Dx, Dy, Dz)
        return list(occ_res)
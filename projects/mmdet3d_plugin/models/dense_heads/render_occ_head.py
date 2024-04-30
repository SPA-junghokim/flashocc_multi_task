# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from torch import nn
import numpy as np
from mmdet3d.models.builder import HEADS, build_loss, build_head
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
class RenderOCCHead2D(BaseModule):
    def __init__(self,
                 in_dim=256,
                 out_dim=256,
                 Dz=16,
                 use_mask=True,
                 num_classes=18,
                 use_predicter=False,
                 class_wise=False,
                 class_balance=False,
                 loss_occ=None,
                 sololoss=False,
                 weight=[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],
                 loss_weight=1,
                 channel_down_for_3d=False,
                 
                 use_3d_loss=True,
                 nerf_head=None,
                 last_no_softplus=False,
                 cnn_head=False,
                 cnn_soft_plus=False,
                 render_loss_weight=None,
                 lovasz_loss=False,
                 no_seperate=False,
                 ):
        super(RenderOCCHead2D, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.Dz = Dz
        self.final_conv = ConvModule(
            self.in_dim,
            out_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            conv_cfg=dict(type='Conv2d')
        )

        self.use_mask = use_mask
        self.num_classes = num_classes
        self.class_balance = class_balance
        self.loss_weight =loss_weight
        self.render_loss_weight = render_loss_weight if render_loss_weight is not None else loss_weight
        self.channel_down_for_3d = channel_down_for_3d
        if self.channel_down_for_3d:
            self.channel_down_for_3d = nn.Linear(channel_down_for_3d, self.in_dim)
        
        self.use_3d_loss = use_3d_loss
        
        if self.class_balance:
            class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies[:17] + 0.001)).float()
            class_weights = class_weights / class_weights.sum()
            self.class_weights = class_weights
            self.semantic_loss = nn.CrossEntropyLoss(
                weight=self.class_weights, reduction="mean"
            )
        else:
            self.semantic_loss = nn.CrossEntropyLoss(reduction="mean")
            
        self.loss_occ = build_loss(loss_occ)
        
        self.last_no_softplus = last_no_softplus
        self.cnn_head = cnn_head
        self.cnn_soft_plus = cnn_soft_plus
        
        if self.cnn_head:
            if self.cnn_soft_plus:
                self.semantic_mlp = nn.Sequential(
                    nn.Conv2d(self.out_dim, self.out_dim * 2, 3, padding= 1),
                    # nn.BatchNorm2d(self.out_dim* 2),
                    nn.Softplus(),
                    nn.Conv2d(self.out_dim* 2, (num_classes - 1) * Dz , 3, padding=1),
                )
                if last_no_softplus:
                    self.density_mlp = nn.Sequential(
                        nn.Conv2d(self.out_dim, self.out_dim * 2, 3, padding=1),
                        # nn.BatchNorm2d(self.out_dim* 2),
                        nn.Softplus(),
                        nn.Conv2d(self.out_dim* 2, 2 * Dz , 1,),
                    )
                else:
                    self.density_mlp = nn.Sequential(
                        nn.Conv2d(self.out_dim, self.out_dim * 2, 3, padding=1),
                        # nn.BatchNorm2d(self.out_dim* 2),
                        nn.Softplus(),
                        nn.Conv2d(self.out_dim* 2, 2 * Dz , 1,),
                        nn.Softplus(),
                    )
            else:
                self.semantic_mlp = nn.Sequential(
                    nn.Conv2d(self.out_dim, self.out_dim * 2, 3, padding= 1),
                    # nn.BatchNorm2d(self.out_dim* 2),
                    nn.ReLU(),
                    nn.Conv2d(self.out_dim* 2, (num_classes - 1) * Dz , 1,),
                )
                if last_no_softplus:
                    self.density_mlp = nn.Sequential(
                        nn.Conv2d(self.out_dim, self.out_dim * 2, 3, padding=1),
                        # nn.BatchNorm2d(self.out_dim* 2),
                        nn.ReLU(),
                        nn.Conv2d(self.out_dim* 2, 2 * Dz , 1),
                    )
                else:
                    self.density_mlp = nn.Sequential(
                        nn.Conv2d(self.out_dim, self.out_dim * 2, 3, padding=1),
                        # nn.BatchNorm2d(self.out_dim* 2),
                        nn.ReLU(),
                        nn.Conv2d(self.out_dim* 2, 2 * Dz , 1),
                        nn.ReLU(),
                    )
        else:
            if no_seperate:
                self.mlp_head = nn.Sequential(
                        nn.Linear(self.out_dim, self.out_dim * 2),
                        nn.Softplus(),
                        nn.Linear(self.out_dim * 2, (num_classes +1) * Dz),
                    )
            else:
                if last_no_softplus:
                    self.density_mlp = nn.Sequential(
                        nn.Linear(self.out_dim, self.out_dim * 2),
                        nn.Softplus(),
                        nn.Linear(self.out_dim * 2, 2 * Dz),
                    )
                else:
                    self.density_mlp = nn.Sequential(
                        nn.Linear(self.out_dim, self.out_dim * 2),
                        nn.Softplus(),
                        nn.Linear(self.out_dim * 2, 2 * Dz),
                        nn.Softplus(),
                    )
                self.semantic_mlp = nn.Sequential(
                    nn.Linear(self.out_dim, self.out_dim * 2),
                    nn.Softplus(),
                    nn.Linear(self.out_dim * 2, (num_classes - 1) * Dz),
                )
        
        self.no_seperate = no_seperate
        if nerf_head is not None:
            self.nerf_head = build_head(nerf_head)
        else:
            self.nerf_head = None
        self.lovasz_loss = lovasz_loss
        if self.lovasz_loss:
            self.lovasz_softmax_loss = lovasz_softmax
        
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
        if self.cnn_head:
            occ_pred = self.final_conv(img_feats)
            density_prob = self.density_mlp(occ_pred).permute(0, 3, 2, 1)
            semantic = self.semantic_mlp(occ_pred).permute(0, 3, 2, 1)

            bs, Dx, Dy = semantic.shape[:3]
            density_prob = density_prob.view(bs, Dx, Dy, self.Dz, 2)
            semantic = semantic.view(bs, Dx, Dy, self.Dz, self.num_classes-1)

        else:
            occ_pred = self.final_conv(img_feats).permute(0, 3, 2, 1)
            if self.no_seperate:
                occ_pred_out = self.mlp_head(occ_pred)
                
                bs, Dx, Dy = occ_pred_out.shape[:3]
                occ_pred_out = occ_pred_out.view(bs, Dx, Dy, self.Dz, self.num_classes+1).contiguous()
                
                density_prob = occ_pred_out[..., -2:]
                semantic = occ_pred_out[..., :-2]
            else:
                density_prob = self.density_mlp(occ_pred)
                semantic = self.semantic_mlp(occ_pred)
                
                bs, Dx, Dy = semantic.shape[:3]
                density_prob = density_prob.view(bs, Dx, Dy, self.Dz, 2)
                semantic = semantic.view(bs, Dx, Dy, self.Dz, self.num_classes-1)

        return [density_prob, semantic]


    def loss_3d(self, voxel_semantics, mask_camera, density_prob, semantic):
        voxel_semantics = voxel_semantics.reshape(-1)   # (B*Dx*Dy*Dz, )
        density_prob = density_prob.reshape(-1, 2)      # (B*Dx*Dy*Dz, 2)
        semantic = semantic.reshape(-1, self.num_classes - 1)
        density_target = (voxel_semantics == 17).long()
        semantic_mask = voxel_semantics != 17

        mask_camera = mask_camera.reshape(-1)
        num_total_samples = mask_camera.sum()
        # compute loss
        loss_geo = self.loss_occ(density_prob, density_target, mask_camera, avg_factor=num_total_samples)

        semantic_mask = torch.logical_and(semantic_mask, mask_camera)
        loss_sem = self.semantic_loss(semantic[semantic_mask], voxel_semantics[semantic_mask].long())

        loss_ = dict()
        loss_['loss_3d_geo'] = loss_geo
        loss_['loss_3d_sem'] = loss_sem
        return loss_

    def loss(self, occ_pred, voxel_semantics, mask_camera, **kwargs):
        """
        Args:
            occ_pred: (B, Dx, Dy, Dz, n_cls)
            voxel_semantics: (B, Dx, Dy, Dz)
            mask_camera: (B, Dx, Dy, Dz)
        Returns:

        """
        density_prob, semantic = occ_pred
        density = density_prob[..., 0]   # (B, Dx, Dy, Dz)
        
        loss = dict()
        
        if self.use_3d_loss:  # 3D loss
            voxel_semantics = voxel_semantics.long()
            mask_camera = mask_camera.to(torch.int32)   # (B, Dx, Dy, Dz)
            loss_occ = self.loss_3d(voxel_semantics, mask_camera, density_prob, semantic)
            for k,v in loss_occ.items():
                loss_occ[k] = self.loss_weight * v
                
            if self.lovasz_loss:
                voxel_semantics_no_empty = voxel_semantics.clone()
                voxel_semantics_no_empty[voxel_semantics == 17] = 255
                lovasz_softmax_loss = self.lovasz_softmax_loss(F.softmax(semantic.permute(0,4,1,2,3), dim=1), voxel_semantics_no_empty, ignore=255)
                loss['lovasz_softmax_loss'] = lovasz_softmax_loss * self.loss_weight
                
            loss.update(loss_occ)
        if self.nerf_head:  # 2D rendering loss
            loss_rendering = self.nerf_head(density, semantic, rays=kwargs['rays'], bda=kwargs['bda'])
            for k,v in loss_rendering.items():
                loss_rendering[k] = self.render_loss_weight * v
            loss.update(loss_rendering)
                
        return loss

    def get_occ(self, occ_pred, img_metas=None):
        """
        Args:
            occ_pred: (B, Dx, Dy, Dz, C)
            img_metas:

        Returns:
            List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        """
        density_prob, semantic = occ_pred
        no_empty_mask = (density_prob.argmax(dim=-1) == 0)
        semantic_res = semantic.argmax(-1)

        B, H, W, Z, C = semantic.shape
        occ = torch.ones((B, H, W, Z), dtype=semantic_res.dtype).to(semantic_res.device)
        occ = occ * (self.num_classes - 1)
        occ[no_empty_mask] = semantic_res[no_empty_mask]
        
        # occ_res = occ.squeeze(dim=0).cpu().numpy().astype(np.uint8)
        occ_res = occ.cpu().numpy().astype(np.uint8)
        return list(occ_res)
    
    


@HEADS.register_module()
class OCCHead3D_Sep(BaseModule):
    def __init__(self,
                 in_dim=256,
                 out_dim=256,
                 Dz=16,
                 use_mask=True,
                 num_classes=18,
                 class_wise=False,
                 class_balance=False,
                 loss_occ=None,
                 sololoss=False,
                 weight=[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],
                 loss_weight=1,
                 channel_down_for_3d=False,
                 render_loss_weight=None,
                 lovasz_loss=False,
                 ):
        super(OCCHead3D_Sep, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.Dz = Dz
        self.final_conv = ConvModule(
            self.in_dim,
            out_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            conv_cfg=dict(type='Conv3d')
        )

        self.use_mask = use_mask
        self.num_classes = num_classes
        self.class_balance = class_balance
        self.loss_weight =loss_weight
        self.render_loss_weight = render_loss_weight if render_loss_weight is not None else loss_weight
        
        if self.class_balance:
            class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies[:17] + 0.001)).float()
            class_weights = class_weights / class_weights.sum()
            self.class_weights = class_weights
            self.semantic_loss = nn.CrossEntropyLoss(
                weight=self.class_weights, reduction="mean"
            )
        else:
            self.semantic_loss = nn.CrossEntropyLoss(reduction="mean")
            
        self.loss_occ = build_loss(loss_occ)
            
        self.density_mlp = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim * 2),
            nn.Softplus(),
            nn.Linear(self.out_dim * 2, 2),
            nn.Softplus(),
        )
        self.semantic_mlp = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim * 2),
            nn.Softplus(),
            nn.Linear(self.out_dim * 2, (num_classes - 1)),
        )
        
        self.lovasz_loss = lovasz_loss
        if self.lovasz_loss:
            self.lovasz_softmax_loss = lovasz_softmax
        
    def forward(self, img_feats):
        """
        Args:
            img_feats: (B, C, Dy, Dx)

        Returns:

        """

        occ_pred = self.final_conv(img_feats).permute(0, 4, 3, 2, 1)

        density_prob = self.density_mlp(occ_pred)
        semantic = self.semantic_mlp(occ_pred)
        
        bs, Dx, Dy = semantic.shape[:3]
        density_prob = density_prob.view(bs, Dx, Dy, self.Dz, 2)
        semantic = semantic.view(bs, Dx, Dy, self.Dz, self.num_classes-1)

        return [density_prob, semantic]


    def loss_3d(self, voxel_semantics, mask_camera, density_prob, semantic):
        voxel_semantics = voxel_semantics.reshape(-1)   # (B*Dx*Dy*Dz, )
        density_prob = density_prob.reshape(-1, 2)      # (B*Dx*Dy*Dz, 2)
        semantic = semantic.reshape(-1, self.num_classes - 1)
        density_target = (voxel_semantics == 17).long()
        semantic_mask = voxel_semantics != 17

        mask_camera = mask_camera.reshape(-1)
        num_total_samples = mask_camera.sum()
        # compute loss
        loss_geo = self.loss_occ(density_prob, density_target, mask_camera, avg_factor=num_total_samples)

        semantic_mask = torch.logical_and(semantic_mask, mask_camera)
        loss_sem = self.semantic_loss(semantic[semantic_mask], voxel_semantics[semantic_mask].long())

        loss_ = dict()
        loss_['loss_3d_geo'] = loss_geo
        loss_['loss_3d_sem'] = loss_sem
        return loss_

    def loss(self, occ_pred, voxel_semantics, mask_camera, **kwargs):
        """
        Args:
            occ_pred: (B, Dx, Dy, Dz, n_cls)
            voxel_semantics: (B, Dx, Dy, Dz)
            mask_camera: (B, Dx, Dy, Dz)
        Returns:

        """
        density_prob, semantic = occ_pred
        density = density_prob[..., 0]   # (B, Dx, Dy, Dz)
        
        loss = dict()
        
        voxel_semantics = voxel_semantics.long()
        mask_camera = mask_camera.to(torch.int32)   # (B, Dx, Dy, Dz)
        loss_occ = self.loss_3d(voxel_semantics, mask_camera, density_prob, semantic)
        for k,v in loss_occ.items():
            loss_occ[k] = self.loss_weight * v
            
        if self.lovasz_loss:
            voxel_semantics_no_empty = voxel_semantics.clone()
            voxel_semantics_no_empty[voxel_semantics == 17] = 255
            lovasz_softmax_loss = self.lovasz_softmax_loss(F.softmax(semantic.permute(0,4,1,2,3), dim=1), voxel_semantics_no_empty, ignore=255)
            loss['lovasz_softmax_loss'] = lovasz_softmax_loss * self.loss_weight
            
        loss.update(loss_occ)
                
        return loss

    def get_occ(self, occ_pred, img_metas=None):
        """
        Args:
            occ_pred: (B, Dx, Dy, Dz, C)
            img_metas:

        Returns:
            List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        """
        density_prob, semantic = occ_pred
        no_empty_mask = (density_prob.argmax(dim=-1) == 0)
        semantic_res = semantic.argmax(-1)

        B, H, W, Z, C = semantic.shape
        occ = torch.ones((B, H, W, Z), dtype=semantic_res.dtype).to(semantic_res.device)
        occ = occ * (self.num_classes - 1)
        occ[no_empty_mask] = semantic_res[no_empty_mask]
        
        # occ_res = occ.squeeze(dim=0).cpu().numpy().astype(np.uint8)
        occ_res = occ.cpu().numpy().astype(np.uint8)
        return list(occ_res)
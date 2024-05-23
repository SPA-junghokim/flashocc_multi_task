# Copyright (c) Phigent Robotics. All rights reserved.
from ...ops import TRTBEVPoolv2
from .bevdet import BEVDet
from .bevdepth4d import BEVDepth4D
from .bevstereo4d import BEVStereo4D
from mmdet3d.models import DETECTORS
from mmdet3d.models.builder import build_head
import torch.nn.functional as F
import torch.nn as nn
import torch
from copy import deepcopy
from mmcv.runner import force_fp32
from mmdet3d.models import builder
import time
from torch.cuda.amp.autocast_mode import autocast

class voxelize_module(nn.Module):
    def __init__(
            self,
            after_voxelize_add = False,
            bev_z_list = [16, 8, 4, 2],
            bev_w_list = [200, 100, 50, 25],
            bev_h_list = [200, 100, 50, 25],
            num_scale = 4,
            in_dim=96,
            only_last_layer=False,
            vox_simple_reshape=False,
            voxelize_patent_Z=False,
            voxelize_patent_HW=False,
    ):
        super(voxelize_module, self).__init__()
        self.bev_h_list = bev_h_list
        self.bev_w_list = bev_w_list
        self.bev_z_list = bev_z_list
        self.num_scale = num_scale
        self.after_voxelize_add = after_voxelize_add
        
        self.sig = nn.Sigmoid()
        
        self.linear = nn.ModuleList()
        self.zembedding_linear = nn.ModuleList()
        self.avg_pool = nn.ModuleList()
        self.upsample = nn.ModuleList()
        self.only_last_layer = only_last_layer
        self.vox_simple_reshape = vox_simple_reshape
        
        self.voxelize_patent_Z = voxelize_patent_Z
        self.voxelize_patent_HW = voxelize_patent_HW
        
        if self.only_last_layer:
            if self.vox_simple_reshape:
                self.linear = nn.Sequential(
                    nn.Linear(in_dim, 2 * in_dim),
                    nn.ReLU(),
                    nn.Linear(2*in_dim, self.bev_z_list[0] * in_dim),
                )
            else:
                if self.voxelize_patent_Z:
                    self.linear = nn.Linear(in_dim, self.bev_z_list[0])
                elif self.voxelize_patent_HW:
                    self.zembedding_linear = nn.Linear(in_dim, self.bev_z_list[0]*in_dim)
                    self.avg_pool = nn.AvgPool2d((self.bev_h_list[0], self.bev_w_list[0]))
                else:
                    self.linear = nn.Linear(in_dim, self.bev_z_list[0])
                    self.zembedding_linear = nn.Linear(in_dim, self.bev_z_list[0]*in_dim)
                    self.avg_pool = nn.AvgPool2d((self.bev_h_list[0], self.bev_w_list[0]))
        else:
            for i in range(self.num_scale):
                self.linear.append(nn.Linear(in_dim, self.bev_z_list[i]))
                self.zembedding_linear.append(nn.Linear(in_dim, self.bev_z_list[i]*in_dim))
                self.avg_pool.append(nn.AvgPool2d((self.bev_h_list[i], self.bev_w_list[i])))
                if self.after_voxelize_add:
                    if i < self.num_scale -1:
                        self.upsample.append(nn.ConvTranspose3d(in_channels=in_dim, out_channels=in_dim, kernel_size=2, stride=2))


    def forward(self, x):
        bs = x[0].shape[0]
        
        if self.only_last_layer:
            if self.vox_simple_reshape:
                x_linear = self.linear(x[0].permute(0,2,3,1))
                B, H, W, C_Z = x_linear.shape
                x[0] = x_linear.reshape(B, H, W, self.bev_z_list[0], -1).permute(0,4,1,2,3)
            else:
                if self.voxelize_patent_Z:
                    attn_weight = self.sig(self.linear(x[0].permute(0,2,3,1)))[:,None]
                    x[0] = attn_weight * x[0][...,None]
                elif self.voxelize_patent_HW:
                    pooled_feat = self.avg_pool(x[0]).squeeze()
                    zembedding = self.sig(self.zembedding_linear(pooled_feat).reshape(bs, -1, self.bev_z_list[0]))
                    x[0] = x[0][...,None] * zembedding[:, :, None, None, :] # [[B, 256, 200, 200, 16]
                else:
                    attn_weight = self.sig(self.linear(x[0].permute(0,2,3,1)))[:,None]
                    pooled_feat = self.avg_pool(x[0]).squeeze()
                    zembedding = self.sig(self.zembedding_linear(pooled_feat).reshape(bs, -1, self.bev_z_list[0]))
                    x[0] = attn_weight * x[0][...,None]
                    x[0] = x[0] * zembedding[:, :, None, None, :] # [[B, 256, 200, 200, 16]
        else:
            for i in range(self.num_scale):
                attn_weight = self.sig(self.linear[i](x[i].permute(0,2,3,1)))[:,None]
                pooled_feat = self.avg_pool[i](x[i]).squeeze()
                zembedding = self.sig(self.zembedding_linear[i](pooled_feat).reshape(bs, -1, self.bev_z_list[i]))
                x[i] = attn_weight * x[i][...,None]
                x[i] = x[i] * zembedding[:, :, None, None, :] # [[B, 256, 200, 200, 16]
                
            if self.after_voxelize_add:
                for i in range(self.num_scale - 1):
                    x[self.num_scale-2-i] = x[self.num_scale-2-i] + self.upsample[i](x[self.num_scale-1-i])
            
            # x[0] [2, 48, 200, 200, 16]
            # x[1] [2, 48, 100, 100, 8]
            # x[2] [2, 48, 50, 50, 4]
            # x[3] [2, 48, 25, 25, 2]
        
        return x


@DETECTORS.register_module()
class BEVDetOCC_depthGT_occformer(BEVDepth4D):
    def __init__(self,
                 pts_bbox_head=None,
                 occ_head=None,
                 seg_head=None,
                 upsample=False,
                 down_sample_for_3d_pooling=None,
                 pc_range = [-40.0, -40.0, -1, 40.0, 40.0, 5.4],
                 grid_size = [200, 200, 16],
                 det_loss_weight=1,
                 occ_loss_weight=1,
                 seg_loss_weight=1,
                 img_bev_encoder=None,
                 img_bev_encoder_backbone=None,
                 occ_bev_encoder_backbone=None,
                 seg_bev_encoder_backbone=None,
                 img_bev_encoder_neck=None,
                 occ_bev_encoder_neck=None,
                 seg_bev_encoder_neck=None,
                 detection_backbone=False,
                 detection_neck=False,
                 
                 voxel_out_channels = 96,
                 after_voxelize_add = False,
                 imgfeat_32x88 = False,
                 only_last_layer=False,
                 vox_aux_loss_3d=False,
                 vox_aux_loss_3d_occ_head=None,
                 vox_simple_reshape=False,
                 aux_test=False,
                 
                 bev_neck_deform=False,
                 bev_deform_backbone=None,
                 bev_deform_neck = None,
                 
                 SA_loss=False,
                 BEVseg_loss_beforehead=False,
                 BEVseg_loss_after_pooling=False,
                 BEV_out_channel_beforehead=None,
                 BEV_out_channel_afterpooling=None,
                 BEVseg_loss_mode='softmax',
                 bevseg_loss_weight=3.0, 
                 
                 aux_bev2occ_head=None,                 
                 only_non_empty_voxel_dot = False,
                 
                 test_merge=False,
                 
                 voxelize_patent_Z=False,
                 voxelize_patent_HW=False,
                 time_check=False,
                 **kwargs):
        super(BEVDetOCC_depthGT_occformer, self).__init__(pts_bbox_head=pts_bbox_head, img_bev_encoder_backbone=img_bev_encoder_backbone,
                                             img_bev_encoder_neck=img_bev_encoder_neck,**kwargs)
        
        self.occ_head = occ_head
        self.seg_head = seg_head

        if pts_bbox_head == None:
            self.pts_bbox_head = None
        if self.occ_head is not None:
            self.occ_head = build_head(occ_head)
        if self.seg_head is not None:
            self.seg_head = build_head(seg_head)
            
        self.upsample = upsample
        self.down_sample_for_3d_pooling = down_sample_for_3d_pooling
        self.SA_loss = SA_loss
        self.BEVseg_loss_beforehead = BEVseg_loss_beforehead
        self.BEVseg_loss_after_pooling = BEVseg_loss_after_pooling
        self.bevseg_loss_weight = bevseg_loss_weight
        self.BEVseg_loss_mode = BEVseg_loss_mode
        
        if aux_bev2occ_head is not None:
            self.aux_bev2occ_head = build_head(aux_bev2occ_head)
        else:
            self.aux_bev2occ_head = None
            
        if self.BEVseg_loss_beforehead:
            self.BEV_out_channel_beforehead=BEV_out_channel_beforehead
            self.BEVseg = nn.Sequential(
                    nn.Conv2d(self.BEV_out_channel_beforehead, self.BEV_out_channel_beforehead * 2, kernel_size=3, padding=1),
                    nn.BatchNorm2d(self.BEV_out_channel_beforehead * 2),
                    nn.ReLU(),
                    nn.Conv2d(self.BEV_out_channel_beforehead * 2, self.BEV_out_channel_beforehead, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(self.BEV_out_channel_beforehead),
                    nn.ReLU(),
                    nn.Conv2d(self.BEV_out_channel_beforehead, 17, kernel_size=1, stride=1, padding=0)
                )
        if self.BEVseg_loss_after_pooling or (self.aux_bev2occ_head is not None):
            self.BEV_out_channel_afterpooling=BEV_out_channel_afterpooling
            self.BEVseg_after_pooling1 = nn.Sequential(
                    nn.Conv2d(self.BEV_out_channel_afterpooling, self.BEV_out_channel_afterpooling * 2, kernel_size=3, padding=1),
                    nn.BatchNorm2d(self.BEV_out_channel_afterpooling * 2),
                    nn.ReLU(),
                    nn.Conv2d(self.BEV_out_channel_afterpooling*2, self.BEV_out_channel_afterpooling * 2, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(self.BEV_out_channel_afterpooling * 2),
                    nn.ReLU(),
            )
            self.BEVseg_after_pooling2 = nn.Sequential(
                    nn.Conv2d(self.BEV_out_channel_afterpooling*2, self.BEV_out_channel_afterpooling * 2, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(self.BEV_out_channel_afterpooling * 2),
                    nn.ReLU(),
            )
            self.BEVseg_after_pooling3 = nn.Sequential(
                    nn.Conv2d(self.BEV_out_channel_afterpooling*2, self.BEV_out_channel_afterpooling * 2, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(self.BEV_out_channel_afterpooling * 2),
                    nn.ReLU(),
            )
            self.BEVseg_after_pooling4 = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    nn.Conv2d(self.BEV_out_channel_afterpooling*2, self.BEV_out_channel_afterpooling * 2, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(self.BEV_out_channel_afterpooling * 2),
                    nn.ReLU(),
            )
            self.BEVseg_after_pooling5 = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    nn.Conv2d(self.BEV_out_channel_afterpooling*2, self.BEV_out_channel_afterpooling * 2, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(self.BEV_out_channel_afterpooling * 2),
                    nn.ReLU(),
            )
            
            if self.BEVseg_loss_after_pooling:
                self.BEVseg_after_pooling_head = nn.Sequential(
                        nn.Conv2d(self.BEV_out_channel_afterpooling*2, self.BEV_out_channel_afterpooling * 2, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(self.BEV_out_channel_afterpooling * 2),
                        nn.ReLU(),
                        nn.Conv2d(self.BEV_out_channel_afterpooling*2, self.BEV_out_channel_afterpooling * 2, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(self.BEV_out_channel_afterpooling * 2),
                        nn.ReLU(),
                        nn.Conv2d(self.BEV_out_channel_afterpooling*2, 17, kernel_size=1, stride=1, padding=0),
                )
                
            if self.down_sample_for_3d_pooling is not None:
                self.down_sample_for_3d_pooling = \
                            nn.Conv2d(self.down_sample_for_3d_pooling[0],
                            self.down_sample_for_3d_pooling[1],
                            kernel_size=1,
                            padding=0,
                            stride=1)
                        
        self.pc_range = torch.tensor(pc_range)
        self.grid_size = torch.tensor(grid_size)
        self.det_loss_weight = det_loss_weight
        self.occ_loss_weight = occ_loss_weight
        self.seg_loss_weight = seg_loss_weight
        self.detection_backbone = detection_backbone
        self.detection_neck = detection_neck
        
        if img_bev_encoder is not None:
            self.img_bev_encoder = builder.build_backbone(img_bev_encoder)
            del self.img_bev_encoder_backbone
            del self.img_bev_encoder_neck
        else:
            self.img_bev_encoder = None

            
        if occ_bev_encoder_backbone is not None:
            self.occ_bev_encoder_backbone = builder.build_backbone(occ_bev_encoder_backbone)
        else:
            self.occ_bev_encoder_backbone = None
        if seg_bev_encoder_backbone is not None:
            self.seg_bev_encoder_backbone = builder.build_backbone(seg_bev_encoder_backbone)
        else:
            self.seg_bev_encoder_backbone = None
            
        if occ_bev_encoder_neck is not None:
            self.occ_bev_encoder_neck = builder.build_backbone(occ_bev_encoder_neck)
        else:
            self.occ_bev_encoder_neck = None
        if seg_bev_encoder_neck is not None:
            self.seg_bev_encoder_neck = builder.build_backbone(seg_bev_encoder_neck)
        else:
            self.seg_bev_encoder_neck = None
        self.imgfeat_32x88 = imgfeat_32x88
        if self.imgfeat_32x88:
            self.neck_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.reduce_conv_neck = nn.Conv2d(512 + kwargs['img_neck']['out_channels'] , kwargs['img_neck']['out_channels'], 1)
        
        self.voxelize_module = voxelize_module(
            after_voxelize_add = after_voxelize_add,
            in_dim=voxel_out_channels,
            only_last_layer=only_last_layer,
            vox_simple_reshape=vox_simple_reshape,
            voxelize_patent_Z=voxelize_patent_Z,
            voxelize_patent_HW=voxelize_patent_HW,)
        
            
        self.bev_neck_deform = bev_neck_deform
        if bev_neck_deform:
            self.img_bev_encoder_backbone = builder.build_neck(bev_deform_backbone)
            self.bev_deform_neck = builder.build_neck(bev_deform_neck)
            
        self.vox_aux_loss_3d = vox_aux_loss_3d
        if self.vox_aux_loss_3d:
            self.vox_aux_loss_3d_occ_head = build_head(vox_aux_loss_3d_occ_head)
        
            
        self.aux_test = aux_test
        
        self.only_non_empty_voxel_dot = only_non_empty_voxel_dot
        self.test_merge = test_merge
        
        self.time_check = time_check
        if self.time_check:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.time_list = []
        
    def extract_feat(self, points, img_inputs, img_metas, **kwargs):
        """Extract features from images and points."""
        """
        points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
        img_inputs:
                imgs:  (B, N_views, 3, H, W)        
                sensor2egos: (B, N_views, 4, 4)
                ego2globals: (B, N_views, 4, 4)
                intrins:     (B, N_views, 3, 3)
                post_rots:   (B, N_views, 3, 3)
                post_trans:  (B, N_views, 3)
                bda_rot:  (B, 3, 3)
        """
        img_feats, depth, trans_feat, bev_feat = self.extract_img_feat(img_inputs, img_metas, **kwargs)
        pts_feats = None
        return img_feats, pts_feats, depth, trans_feat, bev_feat


    def image_encoder(self, img, stereo=False):
        """
        Args:
            img: (B, N, 3, H, W)
            stereo: bool
        Returns:
            x: (B, N, C, fH, fW)
            stereo_feat: (B*N, C_stereo, fH_stereo, fW_stereo) / None
        """
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        x = self.img_backbone(imgs)
        stereo_feat = None
        if stereo:
            stereo_feat = x[0]
            x = x[1:]
        if self.with_img_neck:
            if self.imgfeat_32x88:
                neck_out = self.img_neck(x[1:])
                if type(neck_out) in [list, tuple]:
                    neck_out = neck_out[0]
                neck_out = self.neck_upsample(neck_out)
                x = self.reduce_conv_neck(torch.cat((x[0], neck_out), 1 ))
            else:
                x = self.img_neck(x)
                if type(x) in [list, tuple]:
                    x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        return x, stereo_feat
    
    
    def prepare_bev_feat(self, img, sensor2egos, ego2globals, intrin, post_rot, post_tran,
                         bda, mlp_input):
        """
        Args:
            imgs:  (B, N_views, 3, H, W)
            sensor2egos: (B, N_views, 4, 4)
            ego2globals: (B, N_views, 4, 4)
            intrins:     (B, N_views, 3, 3)
            post_rots:   (B, N_views, 3, 3)
            post_trans:  (B, N_views, 3)
            bda_rot:  (B, 3, 3)
            mlp_input:
        Returns:
            bev_feat: (B, C, Dy, Dx)
            depth: (B*N, D, fH, fW)
        """
        x, _ = self.image_encoder(img)      # x: (B, N, C, fH, fW)
        # bev_feat: (B, C * Dz(=1), Dy, Dx)
        # depth: (B * N, D, fH, fW)
        bev_feat, depth, trans_feat = self.img_view_transformer(
            [x, sensor2egos, ego2globals, intrin, post_rot, post_tran, bda, mlp_input])
        if self.pre_process:
            bev_feat = self.pre_process_net(bev_feat)[0]    # (B, C, Dy, Dx)
        return bev_feat, depth, trans_feat

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      voxel_semantics=None,
                      mask_camera=None,
                      gt_bboxes_ignore=None,
                      gt_seg_mask=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        img_feats, pts_feats, depth, trans_feat, bev_feat = self.extract_feat(
                points, img_inputs=img_inputs, img_metas=img_metas, **kwargs)

        gt_depth = kwargs['gt_depth']   # (B, N_views, img_H, img_W)
        
        losses = dict()
        if self.SA_loss:
            sa_gt_depth = kwargs['SA_gt_depth']
            sa_gt_semantic = kwargs['SA_gt_semantic']
            loss_depth = self.img_view_transformer.get_SA_loss(trans_feat, depth, sa_gt_depth, sa_gt_semantic)
        else:
            loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        losses.update(loss_depth)
        
        det_feats, occ_bev_feats, occ_vox_feats, seg_feats =  img_feats

        if self.pts_bbox_head is not None:
            bbox_outs = self.pts_bbox_head([det_feats])
            losses_pts = self.pts_bbox_head.loss(gt_bboxes_3d, gt_labels_3d, bbox_outs)
            loss_weight = {}
            for k, v in losses_pts.items():
                loss_weight[k] = v * self.det_loss_weight
            losses.update(loss_weight)
            
        if self.occ_head is not None:
            if 'non_vis_semantic_voxel' in list(kwargs.keys()):
                non_vis_semantic_voxel = kwargs['non_vis_semantic_voxel']     # (B, Dx, Dy, Dz)
            else:
                non_vis_semantic_voxel = [None] * mask_camera.shape[0]
            loss_occ = self.forward_occ_train(occ_bev_feats, occ_vox_feats, voxel_semantics, mask_camera, non_vis_semantic_voxel, img_inputs, depth)
            loss_weight = {}
            for k, v in loss_occ.items():
                loss_weight[k] = v * self.occ_loss_weight
            losses.update(loss_weight)
        
        if self.seg_head is not None:
            seg_out = self.seg_head(seg_feats)
            gt_seg_mask = gt_seg_mask.permute(0,3,1,2)
            losses_seg = self.seg_head.loss(seg_out, gt_seg_mask)
            loss_weight = {}
            for k, v in losses_seg.items():
                loss_weight[k] = v * self.seg_loss_weight
            losses.update(loss_weight)

        if self.BEVseg_loss_beforehead:
            bev_preds = self.BEVseg(occ_bev_feats[0]).permute(0,3,2,1)
            masked_semantics_gt = torch.where(mask_camera, voxel_semantics, torch.tensor(17).to(voxel_semantics))
            
            if self.BEVseg_loss_mode == 'softmax':
                bev_preds=bev_preds.softmax(-1)
                B,H,W,Z = masked_semantics_gt.shape
                bev_values = torch.full((B,H,W), 17, dtype=masked_semantics_gt.dtype, device=masked_semantics_gt.device)
                for z in range(Z):
                    mask = masked_semantics_gt[:,:,:,z] != 17 
                    bev_values[mask] = masked_semantics_gt[:,:,:,z][mask] 
                oh_bev_semantic = F.one_hot(bev_values.long(), num_classes=18).float()[..., :17]
                
            elif self.BEVseg_loss_mode == 'sigmoid':
                bev_preds = bev_preds.sigmoid()
                oh_voxel_semantics = F.one_hot(masked_semantics_gt.long(), num_classes=18).float()
                oh_bev_semantic = oh_voxel_semantics.sum(3).bool().float()[..., :17]
                
            bev_preds = bev_preds.reshape(-1,17)
            bev_labels = oh_bev_semantic.reshape(-1, 17)
            
            fg_mask = torch.max(bev_labels, dim=1).values > 0.0
            bev_labels = bev_labels[fg_mask]
            bev_preds = bev_preds[fg_mask]
            with autocast(enabled=False):
                bev_seg_loss = F.binary_cross_entropy(
                    bev_preds,
                    bev_labels,
                    reduction='none',
                ).sum() / max(1.0, fg_mask.sum())
            
            losses['loss_BEV_AUX_beforehead'] = bev_seg_loss * self.bevseg_loss_weight
            
        if self.BEVseg_loss_after_pooling or (self.aux_bev2occ_head is not None):
            bev_feat1 = self.BEVseg_after_pooling1(bev_feat)
            bev_feat2 = self.BEVseg_after_pooling2(bev_feat1)
            bev_feat3 = self.BEVseg_after_pooling3(bev_feat2)
            bev_feat4 = self.BEVseg_after_pooling4(bev_feat3)
            bev_feat5 = self.BEVseg_after_pooling5(bev_feat4+bev_feat2)
            if self.BEVseg_loss_after_pooling:
                bev_preds = self.BEVseg_after_pooling_head(bev_feat5+bev_feat1) # B, 128, 200, 200
                
                bev_preds = bev_preds.permute(0,3,2,1)
                masked_semantics_gt = torch.where(mask_camera, voxel_semantics, torch.tensor(17).to(voxel_semantics))
                
                if self.BEVseg_loss_mode == 'softmax':
                    bev_preds=bev_preds.softmax(-1)
                    B,H,W,Z = masked_semantics_gt.shape
                    bev_values = torch.full((B,H,W), 17, dtype=masked_semantics_gt.dtype, device=masked_semantics_gt.device)
                    for z in range(Z):
                        mask = masked_semantics_gt[:,:,:,z] != 17 
                        bev_values[mask] = masked_semantics_gt[:,:,:,z][mask] 
                    oh_bev_semantic = F.one_hot(bev_values.long(), num_classes=18).float()[..., :17]
                    
                elif self.BEVseg_loss_mode == 'sigmoid':
                    bev_preds = bev_preds.sigmoid()
                    oh_voxel_semantics = F.one_hot(masked_semantics_gt.long(), num_classes=18).float()
                    oh_bev_semantic = oh_voxel_semantics.sum(3).bool().float()[..., :17]
                    
                bev_preds = bev_preds.reshape(-1,17)
                bev_labels = oh_bev_semantic.reshape(-1, 17)
                
                fg_mask = torch.max(bev_labels, dim=1).values > 0.0
                bev_labels = bev_labels[fg_mask]
                bev_preds = bev_preds[fg_mask]
                with autocast(enabled=False):
                    bev_seg_loss = F.binary_cross_entropy(
                        bev_preds, 
                        bev_labels,
                        reduction='none',
                    ).sum() / max(1.0, fg_mask.sum())
                losses['loss_BEV_AUX_afterpooling'] = bev_seg_loss * self.bevseg_loss_weight


            if self.aux_bev2occ_head is not None:
                outs = self.aux_bev2occ_head(bev_feat5+bev_feat1)
                # assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
                kwargs['bda'] = img_inputs[-1]
                loss_aux_bev2occ = self.aux_bev2occ_head.loss(
                    outs,  # (B, Dx, Dy, Dz, n_cls)
                    voxel_semantics,  # (B, Dx, Dy, Dz)
                    mask_camera,  # (B, Dx, Dy, Dz)
                    **kwargs,
                )
                new_loss_aux_bev2occ = dict()
                for k, v in loss_aux_bev2occ.items():
                    new_loss_aux_bev2occ[k+'_BEV2OCC_aux'] = v
                if new_loss_aux_bev2occ is not None:
                    losses.update(new_loss_aux_bev2occ)        
            
            
        return losses

    
    def forward_occ_train(self, occ_bev_feats, occ_vox_feats, voxel_semantics, mask_camera, non_vis_semantic_voxel, img_inputs, depth, **kwargs):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
            voxel_semantics: (B, Dx, Dy, Dz)
            mask_camera: (B, Dx, Dy, Dz)
        Returns:
        """
        #img_feats = [4, 128, 200, 200]
        B = occ_bev_feats[0].shape[0]
        img_metas = [{"pc_range": self.pc_range, "occ_size":self.grid_size} for i in range(B)]
        
        new_loss_aux_3d, aux_occ_pred = None, None
        
        if self.vox_aux_loss_3d or self.only_non_empty_voxel_dot:
            aux_occ_pred = self.vox_aux_loss_3d_occ_head(occ_vox_feats[0].permute(0,1,4,2,3))
            if self.vox_aux_loss_3d:
                loss_aux_3d = self.vox_aux_loss_3d_occ_head.loss(aux_occ_pred, voxel_semantics, mask_camera,)
                new_loss_aux_3d = dict()
                for k, v in loss_aux_3d.items():
                    new_loss_aux_3d[k+'_aux3d'] = v
            
        loss_occ = self.occ_head.forward_train(occ_vox_feats, img_metas, voxel_semantics, mask_camera, non_vis_semantic_voxel, aux_occ_pred)

        # fig.savefig('first_fig.png')

        if new_loss_aux_3d is not None:
            loss_occ.update(new_loss_aux_3d)
            
        return loss_occ

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    voxel_semantics=None,
                    mask_lidar=None,
                    mask_camera=None,
                    gt_seg_mask=None,
                    **kwargs):
        
        if self.time_check:
            self.start_event.record()
            
        img_feats, _, depth, trans_feat, bev_feat = self.extract_feat(
            points, img_inputs=img, img_metas=img_metas, **kwargs)
        
        det_feats, occ_bev_feats, occ_vox_feats, seg_feats =  img_feats
        bbox_out, occ_out, seg_out = None, None, None
        if self.pts_bbox_head is not None:
            bbox_pts = self.simple_test_pts([det_feats], img_metas, rescale=rescale)
            bbox_out = [dict(pts_bbox=bbox_pts[0])]

        if self.occ_head is not None:
            occ_out = self.simple_test_occ(occ_bev_feats, occ_vox_feats, img_metas, depth, img)    # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
    
        if self.seg_head is not None:
            seg_out = self.seg_head(seg_feats)

        if self.time_check:
            self.end_event.record() 
            torch.cuda.synchronize()  
            cur_iter_time = self.start_event.elapsed_time(self.end_event)
            print(cur_iter_time)
            self.time_list.append(cur_iter_time)
            if len(self.time_list ) > 1000:
                print(sum(self.time_list[500:])/len(self.time_list[500:]))
                exit()
                
        return bbox_out, occ_out, voxel_semantics, mask_lidar, mask_camera, seg_out, gt_seg_mask

    def simple_test_occ(self, occ_bev_feats, occ_vox_feats, img_metas=None, depth=None, img_inputs=None):
        B = occ_bev_feats[0].shape[0]
        img_metas_occ = [{"pc_range": self.pc_range, "occ_size":self.grid_size} for i in range(B)]
        
        if self.aux_test:
            aux_occ_pred = self.vox_aux_loss_3d_occ_head(occ_vox_feats[0].permute(0,1,4,2,3))
            aux_occ_preds = self.vox_aux_loss_3d_occ_head.get_occ(aux_occ_pred, img_metas)      # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
            return aux_occ_preds
        
        occ_pred = None
        if self.only_non_empty_voxel_dot or self.test_merge:
            occ_pred = self.vox_aux_loss_3d_occ_head(occ_vox_feats[0].permute(0,1,4,2,3))

        occ_preds = self.occ_head.simple_test(occ_vox_feats, img_metas_occ, occ_pred)      # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        
        return occ_preds

    def forward_dummy(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        img_feats, pts_feats, depth, trans_feat, bev_feat = self.extract_feat(
            points, img_inputs=img_inputs, img_metas=img_metas, **kwargs)
        occ_bev_feature = img_feats[0]
        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)
        outs = self.occ_head(occ_bev_feature)
        return outs
    
    def extract_img_feat(self,
                         img_inputs,
                         img_metas,
                         pred_prev=False,
                         sequential=False,
                         **kwargs):
        """
        Args:
            img_inputs:
                imgs:  (B, N, 3, H, W)        # N = 6 * (N_history + 1)
                sensor2egos: (B, N, 4, 4)
                ego2globals: (B, N, 4, 4)
                intrins:     (B, N, 3, 3)
                post_rots:   (B, N, 3, 3)
                post_trans:  (B, N, 3)
                bda_rot:  (B, 3, 3)
            img_metas:
            **kwargs:
        Returns:
            x: [(B, C', H', W'), ]
            depth: (B*N_views, D, fH, fW)
        """
        if sequential:
            return self.extract_img_feat_sequential(img_inputs, kwargs['feat_prev'])
        imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, \
        bda, _ = self.prepare_inputs(img_inputs)
        

        """Extract features of images."""
        bev_feat_list = []
        depth_list = []
        trans_feat_list = []
        key_frame = True  # back propagation for key frame only

        for img, sensor2keyego, ego2global, intrin, post_rot, post_tran in zip(
                imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans):
            if key_frame or self.with_prev:
                if self.align_after_view_transfromation:
                    sensor2keyego, ego2global = sensor2keyegos[0], ego2globals[0]

                mlp_input = self.img_view_transformer.get_mlp_input(
                    sensor2keyegos[0], ego2globals[0], intrin, post_rot, post_tran, bda)    # (B, N_views, 27)

                inputs_curr = (img, sensor2keyego, ego2global, intrin, post_rot,
                               post_tran, bda, mlp_input)
                if key_frame:
                    # bev_feat: (B, C, Dy, Dx)
                    # depth: (B*N_views, D, fH, fW)
                    bev_feat, depth, trans_feat = self.prepare_bev_feat(*inputs_curr)
                    if self.down_sample_for_3d_pooling is not None:
                        bev_feat = self.down_sample_for_3d_pooling(bev_feat)
                else:
                    with torch.no_grad():
                        bev_feat, depth, trans_feat = self.prepare_bev_feat(*inputs_curr)
                        if self.down_sample_for_3d_pooling is not None:
                            bev_feat = self.down_sample_for_3d_pooling(bev_feat)
            else:
                # https://github.com/HuangJunJie2017/BEVDet/issues/275
                bev_feat = torch.zeros_like(bev_feat_list[0])
                depth = None
            bev_feat_list.append(bev_feat)
            depth_list.append(depth)
            trans_feat_list.append(trans_feat)
            key_frame = False

        # bev_feat_list: List[(B, C, Dy, Dx), (B, C, Dy, Dx), ...]
        # depth_list: List[(B*N_views, D, fH, fW), (B*N_views, D, fH, fW), ...]

        if pred_prev:
            assert self.align_after_view_transfromation
            assert sensor2keyegos[0].shape[0] == 1      # batch_size = 1
            feat_prev = torch.cat(bev_feat_list[1:], dim=0)
            # (1, N_views, 4, 4) --> (N_prev, N_views, 4, 4)
            ego2globals_curr = \
                ego2globals[0].repeat(self.num_frame - 1, 1, 1, 1)
            # (1, N_views, 4, 4) --> (N_prev, N_views, 4, 4)
            sensor2keyegos_curr = \
                sensor2keyegos[0].repeat(self.num_frame - 1, 1, 1, 1)
            ego2globals_prev = torch.cat(ego2globals[1:], dim=0)            # (N_prev, N_views, 4, 4)
            sensor2keyegos_prev = torch.cat(sensor2keyegos[1:], dim=0)      # (N_prev, N_views, 4, 4)
            bda_curr = bda.repeat(self.num_frame - 1, 1, 1)     # (N_prev, 3, 3)
            return feat_prev, [imgs[0],     # (1, N_views, 3, H, W)
                               sensor2keyegos_curr,     # (N_prev, N_views, 4, 4)
                               ego2globals_curr,        # (N_prev, N_views, 4, 4)
                               intrins[0],          # (1, N_views, 3, 3)
                               sensor2keyegos_prev,     # (N_prev, N_views, 4, 4)
                               ego2globals_prev,        # (N_prev, N_views, 4, 4)
                               post_rots[0],    # (1, N_views, 3, 3)
                               post_trans[0],   # (1, N_views, 3, )
                               bda_curr]        # (N_prev, 3, 3)

        if self.align_after_view_transfromation:
            for adj_id in range(1, self.num_frame):
                bev_feat_list[adj_id] = self.shift_feature(
                    bev_feat_list[adj_id],  # (B, C, Dy, Dx)
                    [sensor2keyegos[0],     # (B, N_views, 4, 4)
                     sensor2keyegos[adj_id]     # (B, N_views, 4, 4)
                    ],
                    bda     # (B, 3, 3)
                )   # (B, C, Dy, Dx)
        bev_feat = torch.cat(bev_feat_list, dim=1)      # (B, N_frames*C, Dy, Dx)
        
        det_bev, occ_bev, seg_bev = self.bev_encoder(bev_feat) # (B, 48, 200, 200) / (B, 48, 200, 200) x 4 / (B, 48, 200, 200) x 4
        # breakpoint()
        occ_bev_out = []
        for b in occ_bev:
            occ_bev_out.append(b.clone())
        occ_vox = self.voxelize_module(occ_bev) # (B, 48, 200, 200, 16) & (B, 48, 200, 200) x 3

        return [det_bev, occ_bev_out, occ_vox, seg_bev], depth, trans_feat_list[0], bev_feat

    @force_fp32()
    def bev_encoder(self, x):
        """
        Args:
            x: (B, C, Dy, Dx)
        Returns:
            x: (B, C', 2*Dy, 2*Dx)
        """
        if self.img_bev_encoder is not None:
            det_bev = self.img_bev_encoder(x)
            seg_bev = det_bev
            occ_bev = det_bev
        else:
            det_bev = self.img_bev_encoder_backbone(x)
            if self.occ_bev_encoder_backbone is not None:
                occ_bev = self.occ_bev_encoder_backbone(x)
            else:
                occ_bev = det_bev
                
            if self.seg_bev_encoder_backbone is not None:
                seg_bev = self.seg_bev_encoder_backbone(x)
            elif self.detection_backbone:
                seg_bev = occ_bev
            else:
                seg_bev = det_bev

            if self.bev_neck_deform:
                det_bev = self.bev_deform_neck(det_bev)
            det_bev = self.img_bev_encoder_neck(det_bev)
            
            if self.occ_bev_encoder_neck is not None:
                occ_bev = self.occ_bev_encoder_neck(occ_bev)
            else:
                occ_bev = det_bev
                
            if self.seg_bev_encoder_neck is not None:
                seg_bev = self.seg_bev_encoder_neck(seg_bev)
            elif self.detection_neck or self.detection_backbone:
                seg_bev = occ_bev
            else:
                seg_bev = det_bev

        if type(det_bev) in [list, tuple]:
            det_bev = det_bev[0]
        # if type(occ_bev) in [list, tuple]:
        #     occ_bev = occ_bev[0]
        if type(x) in [list, tuple]:
            seg_bev = seg_bev[0]
        return [det_bev, occ_bev, seg_bev]


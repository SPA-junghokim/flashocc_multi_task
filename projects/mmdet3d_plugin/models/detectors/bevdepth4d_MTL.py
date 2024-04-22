from ...ops import TRTBEVPoolv2
from .bevdet import BEVDet
from .bevdepth4d import BEVDepth4D
from mmdet3d.models import DETECTORS
from mmdet3d.models.builder import build_head
import torch.nn.functional as F
import torch.nn as nn
import torch
from mmcv.runner import force_fp32
from mmdet3d.models import builder
from mmcv.cnn import ConvModule

@DETECTORS.register_module()
class BEVDepth4D_MTL(BEVDepth4D):
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
                 imgfeat_32x88=False,
                 depth_attn=None,
                 frustum_depth_attr=False,
                 frustum_to_voxel=None,
                 frustum_depth_detach=False,
                 frustum_depth_residual=False,
                 pooling_head = False,
                 time_check=True,
                 **kwargs):
        super(BEVDepth4D_MTL, self).__init__(pts_bbox_head=pts_bbox_head, img_bev_encoder_backbone=img_bev_encoder_backbone,
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

        self.depth_attn = depth_attn
        self.frustum_depth_attr = frustum_depth_attr
        self.frustum_depth_detach = frustum_depth_detach
        self.frustum_depth_residual = frustum_depth_residual
        self.pooling_head = pooling_head
        if self.depth_attn is not None:
            self.frustum_to_voxel = builder.build_neck(frustum_to_voxel)
            self.depth_attn_downsample_conv = ConvModule( # 1x1 conv3d 가 빠른지 linear 가 빠른지 비교
                self.depth_attn[0],
                self.depth_attn[1],
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
                conv_cfg=dict(type='Conv3d')
            )
            
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
        img_feats, depth = self.extract_img_feat(img_inputs, img_metas, **kwargs)
        pts_feats = None
        return img_feats, pts_feats, depth


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
        bev_feat, depth = self.img_view_transformer(
            [x, sensor2egos, ego2globals, intrin, post_rot, post_tran, bda, mlp_input])
        if self.pre_process:
            bev_feat = self.pre_process_net(bev_feat)[0]    # (B, C, Dy, Dx)
        return bev_feat, depth

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
        img_feats, pts_feats, depth = self.extract_feat(
                points, img_inputs=img_inputs, img_metas=img_metas, **kwargs)

        gt_depth = kwargs['gt_depth']   # (B, N_views, img_H, img_W)
        
        losses = dict()
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        losses.update(loss_depth)
        # losses['loss_depth'] = loss_depth
        
        # Get box losses
        det_feats, occ_feats, seg_feats =  img_feats

        if self.pts_bbox_head is not None:
            bbox_outs = self.pts_bbox_head([det_feats])
            losses_pts = self.pts_bbox_head.loss(gt_bboxes_3d, gt_labels_3d, bbox_outs)
            loss_weight = {}
            for k, v in losses_pts.items():
                loss_weight[k] = v * self.det_loss_weight
            losses.update(loss_weight)
            
        if self.occ_head is not None:
            loss_occ = self.forward_occ_train(occ_feats, voxel_semantics, mask_camera, img_inputs, depth, **kwargs)
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
        
        return losses

    def forward_occ_train(self, img_feats, voxel_semantics, mask_camera, img_inputs, depth, **kwargs):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
            voxel_semantics: (B, Dx, Dy, Dz)
            mask_camera: (B, Dx, Dy, Dz)
        Returns:
        """
        if self.depth_attn:
            imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = img_inputs
            if self.frustum_depth_detach:
                depth_for_voxel = depth.detach()
            else:
                depth_for_voxel = depth
            B,N,C,H_,W_ = imgs.shape
            _,_,H,W = depth.shape
            bda_4x4 = torch.repeat_interleave(torch.eye(4)[None], 4, 0).to(imgs.device)
            bda_4x4[:,:2,:2] = bda[:,:2,:2]
            bda_4x4 = bda_4x4[:,None].repeat(1,N,1,1).reshape(B*N,4,4)
            ida_mat = torch.repeat_interleave(torch.eye(3)[None], B * N, dim=0).view(B, N, 3, 3).to(imgs[0].device)
            ida_mat[:,:,:3,:3] = post_rots
            ida_mat[:,:,:2,2] = post_trans[:,:,:2]
            trans_cam_to_img = torch.zeros(B, N, 3, 4).to(imgs.device)
            trans_cam_to_img[:,:,:3,:3] = ida_mat.matmul(intrins)
            trans_lidar_to_cam = torch.inverse(sensor2egos)
            image_shape = torch.tensor([H_,W_]).to(imgs.device).float()
            if self.frustum_depth_attr == 'trasnmittance':
                frustum_features = depth_for_voxel.cumsum(1).reshape(B,N,-1,H,W)
                canvas_ones = True
            elif self.frustum_depth_attr == 'reverse_transmittance':
                frustum_features = depth_for_voxel.cumsum(1).reshape(B,N,-1,H,W)
                frustum_features = 1 - frustum_features
                canvas_ones = False
            else:
                frustum_features = depth_for_voxel.reshape(B,N,-1,H,W)
                canvas_ones = False
            voxel_score = self.frustum_to_voxel(trans_lidar_to_cam, trans_cam_to_img, image_shape, bda_4x4, frustum_features, canvas_ones)
            
            img_feats_depthattn = img_feats[:,:,None] * voxel_score[:,None]
            if self.frustum_depth_residual:
                img_feats_depthattn = img_feats_depthattn + img_feats[:,:,None]
            img_feats = self.depth_attn_downsample_conv(img_feats_depthattn)
            if self.pooling_head:
                img_feats = img_feats.reshape(B,-1, self.grid_size[0],self.grid_size[1])
                
        outs = self.occ_head(img_feats)
        # assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        kwargs['bda'] = img_inputs[-1]
        loss_occ = self.occ_head.loss(
            outs,  # (B, Dx, Dy, Dz, n_cls)
            voxel_semantics,  # (B, Dx, Dy, Dz)
            mask_camera,  # (B, Dx, Dy, Dz)
            **kwargs,
        )
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
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        if self.time_check:
            self.start_event.record()
            
        img_feats, _, depth = self.extract_feat(
            points, img_inputs=img, img_metas=img_metas, **kwargs)
        
        det_feats, occ_feats, seg_feats =  img_feats
        bbox_out, occ_out, seg_out = None, None, None
        if self.pts_bbox_head is not None:
            bbox_pts = self.simple_test_pts([det_feats], img_metas, rescale=rescale)
            bbox_out = [dict(pts_bbox=bbox_pts[0])]
            
        if self.occ_head is not None:
            occ_out = self.simple_test_occ(occ_feats, img_metas, depth, img)    # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
    
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

    def simple_test_occ(self, img_feats, img_metas=None, depth=None, img_inputs=None):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
            img_metas:

        Returns:
            occ_preds: List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        """
        if self.depth_attn:
                
            imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = img_inputs
            if self.frustum_depth_detach:
                depth_for_voxel = depth.detach()
            else:
                depth_for_voxel = depth
            B,N,C,H_,W_ = imgs.shape
            _,_,H,W = depth.shape
            bda_4x4 = torch.repeat_interleave(torch.eye(4)[None], B, 0).to(imgs.device)
            bda_4x4[:,:2,:2] = bda[:,:2,:2]
            bda_4x4 = bda_4x4[:,None].repeat(1,N,1,1).reshape(B*N,4,4)
            ida_mat = torch.repeat_interleave(torch.eye(3)[None], B * N, dim=0).view(B, N, 3, 3).to(imgs[0].device)
            ida_mat[:,:,:3,:3] = post_rots
            ida_mat[:,:,:2,2] = post_trans[:,:,:2]
            trans_cam_to_img = torch.zeros(B, N, 3, 4).to(imgs.device)
            trans_cam_to_img[:,:,:3,:3] = ida_mat.matmul(intrins)
            trans_lidar_to_cam = torch.inverse(sensor2egos)
            image_shape = torch.tensor([H_,W_]).to(imgs.device).float()
            if self.frustum_depth_attr == 'trasnmittance':
                frustum_features = depth_for_voxel.cumsum(1).reshape(B,N,-1,H,W)
                canvas_ones = True
            elif self.frustum_depth_attr == 'reverse_transmittance':
                frustum_features = depth_for_voxel.cumsum(1).reshape(B,N,-1,H,W)
                frustum_features = 1 - frustum_features
                canvas_ones = False
            else:
                frustum_features = depth_for_voxel.reshape(B,N,-1,H,W)
                canvas_ones = False
            voxel_score = self.frustum_to_voxel(trans_lidar_to_cam, trans_cam_to_img, image_shape, bda_4x4, frustum_features, canvas_ones)
            
            img_feats_depthattn = img_feats[:,:,None] * voxel_score[:,None]
            if self.frustum_depth_residual:
                img_feats_depthattn = img_feats_depthattn + img_feats[:,:,None]
            img_feats = self.depth_attn_downsample_conv(img_feats_depthattn)
            if self.pooling_head:
                img_feats = img_feats.reshape(B,-1, self.grid_size[0],self.grid_size[1])
                    
        outs = self.occ_head(img_feats)
        occ_preds = self.occ_head.get_occ(outs, img_metas)      # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        return occ_preds

    def forward_dummy(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        img_feats, pts_feats, depth = self.extract_feat(
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
                    bev_feat, depth = self.prepare_bev_feat(*inputs_curr)
                    if self.down_sample_for_3d_pooling is not None:
                        bev_feat = self.down_sample_for_3d_pooling(bev_feat)
                else:
                    with torch.no_grad():
                        bev_feat, depth = self.prepare_bev_feat(*inputs_curr)
                        if self.down_sample_for_3d_pooling is not None:
                            bev_feat = self.down_sample_for_3d_pooling(bev_feat)
            else:
                # https://github.com/HuangJunJie2017/BEVDet/issues/275
                bev_feat = torch.zeros_like(bev_feat_list[0])
                depth = None
            bev_feat_list.append(bev_feat)
            depth_list.append(depth)
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
        
        x = self.bev_encoder(bev_feat)

        return x, depth_list[0]


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
        if type(occ_bev) in [list, tuple]:
            occ_bev = occ_bev[0]
        if type(x) in [list, tuple]:
            seg_bev = seg_bev[0]
        return [det_bev, occ_bev, seg_bev]


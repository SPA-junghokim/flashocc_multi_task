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
                 img_bev_encoder_backbone=None,
                 occ_bev_encoder_backbone=None,
                 seg_bev_encoder_backbone=None,
                 img_bev_encoder_neck=None,
                 occ_bev_encoder_neck=None,
                 seg_bev_encoder_neck=None,
                 detection_backbone=False,
                 detection_neck=False,
                 use_EADF=False,
                 use_FGD=False,
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


        self.use_EADF = use_EADF
        self.use_FGD = use_FGD

    def prepare_EADF(
            self,
            depth,
            step=7,
        ):
        B, N, C, H, W = depth.unsqueeze(dim=2).size()
        depth_tmp = depth.reshape(B*N, C, H, W)
        pad = int((step - 1) // 2)
        depth_tmp = F.pad(depth_tmp, [pad, pad, pad, pad], mode='constant', value=0)
        patches = depth_tmp.unfold(dimension=2, size=step, step=1)
        patches = patches.unfold(dimension=3, size=step, step=1)
        max_depth, _ = patches.reshape(B, N, C, H, W, -1).max(dim=-1)  # [2, 6, 1, 256, 704]

        step = float(step)
        shift_list = [[step / H, 0.0 / W], [-step / H, 0.0 / W], [0.0 / H, step / W], [0.0 / H, -step / W]]
        max_depth_tmp = max_depth.reshape(B*N, C, H, W)
        output_list = []
        for shift in shift_list:
            transform_matrix =torch.tensor([[1, 0, shift[0]],[0, 1, shift[1]]]).unsqueeze(0).repeat(B*N, 1, 1).cuda()
            grid = F.affine_grid(transform_matrix, max_depth_tmp.shape).float()
            output = F.grid_sample(max_depth_tmp, grid, mode='nearest').reshape(B, N, C, H, W)  #平移后图像
            output = max_depth - output
            output_mask = ((output == max_depth) == False)
            output = output * output_mask
            output_list.append(output)
        grad = torch.cat(output_list, dim=2)  # [2, 6, 4, 256, 704]
        depth_grad = torch.abs(grad).max(dim=2)[0].unsqueeze(2)

        return max_depth, depth_grad
    
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
        if self.use_FGD:
            img_feats, depth, pred_fine_depth = self.extract_img_feat(img_inputs, img_metas, **kwargs)
            pts_feats = None
            return img_feats, pts_feats, depth, pred_fine_depth

        else:
            img_feats, depth = self.extract_img_feat(img_inputs, img_metas, **kwargs)
            pts_feats = None
            return img_feats, pts_feats, depth

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
        bev_feat, depth, fine_depth = self.img_view_transformer(
            [x, sensor2egos, ego2globals, intrin, post_rot, post_tran, bda, mlp_input])
        if self.pre_process:
            bev_feat = self.pre_process_net(bev_feat)[0]    # (B, C, Dy, Dx)
        return bev_feat, depth, fine_depth

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
        if self.use_FGD:
            img_feats, pts_feats, depth, pred_fine_depth = self.extract_feat(
                points, img_inputs=img_inputs, img_metas=img_metas, **kwargs)
        else:
            img_feats, pts_feats, depth = self.extract_feat(
                points, img_inputs=img_inputs, img_metas=img_metas, **kwargs)

        gt_depth = kwargs['gt_depth']   # (B, N_views, img_H, img_W)
        
        
        losses = dict()
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        losses['loss_depth'] = loss_depth
        if self.use_FGD:
            FGD_loss = self.img_view_transformer.get_FGD_loss(gt_depth,pred_fine_depth)
            losses['FGD_loss'] = FGD_loss
        if self.use_EADF:
            dense_depth, depth_grad = self.prepare_EADF(gt_depth)
            EADF_loss = self.img_view_transformer.get_EADF_loss(dense_depth, depth_grad, pred_fine_depth)
            losses['loss_EADF'] = EADF_loss
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
            loss_occ = self.forward_occ_train(occ_feats, voxel_semantics, mask_camera)
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

    def forward_occ_train(self, img_feats, voxel_semantics, mask_camera):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
            voxel_semantics: (B, Dx, Dy, Dz)
            mask_camera: (B, Dx, Dy, Dz)
        Returns:
        """
        outs = self.occ_head(img_feats)
        # assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        loss_occ = self.occ_head.loss(
            outs,  # (B, Dx, Dy, Dz, n_cls)
            voxel_semantics,  # (B, Dx, Dy, Dz)
            mask_camera,  # (B, Dx, Dy, Dz)
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
        img_feats, _, _ = self.extract_feat(
            points, img_inputs=img, img_metas=img_metas, **kwargs)
        
        det_feats, occ_feats, seg_feats =  img_feats
        bbox_out, occ_out, seg_out = None, None, None
        if self.pts_bbox_head is not None:
            bbox_pts = self.simple_test_pts([det_feats], img_metas, rescale=rescale)
            bbox_out = [dict(pts_bbox=bbox_pts[0])]
            
        if self.occ_head is not None:
            occ_out = self.simple_test_occ(occ_feats, img_metas)    # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
    
        if self.seg_head is not None:
            seg_out = self.seg_head(seg_feats)

        return bbox_out, occ_out, voxel_semantics, mask_lidar, mask_camera, seg_out, gt_seg_mask

    def simple_test_occ(self, img_feats, img_metas=None):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
            img_metas:

        Returns:
            occ_preds: List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        """
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
        fine_depth_list = []
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
                    bev_feat, depth, fine_depth = self.prepare_bev_feat(*inputs_curr)
                else:
                    with torch.no_grad():
                        bev_feat, depth, fine_depth = self.prepare_bev_feat(*inputs_curr)
            else:
                # https://github.com/HuangJunJie2017/BEVDet/issues/275
                bev_feat = torch.zeros_like(bev_feat_list[0])
                depth = None
            bev_feat_list.append(bev_feat)
            depth_list.append(depth)
            fine_depth_list.append(fine_depth)
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
        
        if self.down_sample_for_3d_pooling is not None:
            bev_feat = self.down_sample_for_3d_pooling(bev_feat)
        x = self.bev_encoder(bev_feat)

        if self.use_FGD:
            return x, depth_list[0], fine_depth_list[0]
        else:
            return x, depth_list[0]


    @force_fp32()
    def bev_encoder(self, x):
        """
        Args:
            x: (B, C, Dy, Dx)
        Returns:
            x: (B, C', 2*Dy, 2*Dx)
        """
        
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


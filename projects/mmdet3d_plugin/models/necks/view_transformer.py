# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import numpy as np
from mmcv.runner import BaseModule, force_fp32
from mmcv.cnn import ConvModule, build_conv_layer
from mmdet3d.models.builder import NECKS
from ...ops import bev_pool_v2
from ..model_utils import DepthNet, CRN_DepthNet
from torch.cuda.amp.autocast_mode import autocast
import torch.nn.functional as F
from mmdet3d.models.builder import build_loss
from mmcv.cnn import build_conv_layer

from ...ops.average_voxel_pooling_v2 import average_voxel_pooling

# occ3d-nuscenes
nusc_class_frequencies = np.array([1163161, 2309034, 188743, 2997643, 20317180, 852476, 243808, 2457947, 
            497017, 2731022, 7224789, 214411435, 5565043, 63191967, 76098082, 128860031, 
            141625221, 2307405309])

@NECKS.register_module(force=True)
class LSSViewTransformer(BaseModule):
    r"""Lift-Splat-Shoot view transformer with BEVPoolv2 implementation.

    Please refer to the `paper <https://arxiv.org/abs/2008.05711>`_ and
        `paper <https://arxiv.org/abs/2211.17111>`

    Args:
        grid_config (dict): Config of grid alone each axis in format of
            (lower_bound, upper_bound, interval). axis in {x,y,z,depth}.
        input_size (tuple(int)): Size of input images in format of (height,
            width).
        downsample (int): Down sample factor from the input size to the feature
            size.
        in_channels (int): Channels of input feature.
        out_channels (int): Channels of transformed feature.
        accelerate (bool): Whether the view transformation is conducted with
            acceleration. Note: the intrinsic and extrinsic of cameras should
            be constant when 'accelerate' is set true.
        sid (bool): Whether to use Spacing Increasing Discretization (SID)
            depth distribution as `STS: Surround-view Temporal Stereo for
            Multi-view 3D Detection`.
        collapse_z (bool): Whether to collapse in z direction.
    """

    def __init__(
        self,
        grid_config,
        input_size,
        downsample=16,
        in_channels=512,
        out_channels=64,
        accelerate=False,
        sid=False,
        collapse_z=True,
        
    ):
        super(LSSViewTransformer, self).__init__()
        self.grid_config = grid_config
        self.downsample = downsample
        self.input_size = input_size
        self.create_grid_infos(**grid_config)
        self.sid = sid
        self.frustum = self.create_frustum(grid_config['depth'],
                                           input_size, downsample)      # (D, fH, fW, 3)  3:(u, v, d)
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.depth_net = nn.Conv2d(
            in_channels, self.D + self.out_channels, kernel_size=1, padding=0)
        self.accelerate = accelerate
        self.initial_flag = True
        self.collapse_z = collapse_z

        

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

    def create_frustum(self, depth_cfg, input_size, downsample):
        """Generate the frustum template for each image.

        Args:
            depth_cfg (tuple(float)): Config of grid alone depth axis in format
                of (lower_bound, upper_bound, interval).
            input_size (tuple(int)): Size of input images in format of (height,
                width).
            downsample (int): Down sample scale factor from the input size to
                the feature size.
        Returns:
            frustum: (D, fH, fW, 3)  3:(u, v, d)
        """
        H_in, W_in = input_size
        H_feat, W_feat = H_in // downsample, W_in // downsample
        d = torch.arange(*depth_cfg, dtype=torch.float)\
            .view(-1, 1, 1).expand(-1, H_feat, W_feat)      # (D, fH, fW)
        self.D = d.shape[0]
        if self.sid:
            d_sid = torch.arange(self.D).float()
            depth_cfg_t = torch.tensor(depth_cfg).float()
            d_sid = torch.exp(torch.log(depth_cfg_t[0]) + d_sid / (self.D-1) *
                              torch.log((depth_cfg_t[1]-1) / depth_cfg_t[0]))
            d = d_sid.view(-1, 1, 1).expand(-1, H_feat, W_feat)

        x = torch.linspace(0, W_in - 1, W_feat,  dtype=torch.float)\
            .view(1, 1, W_feat).expand(self.D, H_feat, W_feat)      # (D, fH, fW)
        y = torch.linspace(0, H_in - 1, H_feat,  dtype=torch.float)\
            .view(1, H_feat, 1).expand(self.D, H_feat, W_feat)      # (D, fH, fW)

        return torch.stack((x, y, d), -1)    # (D, fH, fW, 3)  3:(u, v, d)

    def get_lidar_coor(self, sensor2ego, ego2global, cam2imgs, post_rots, post_trans,
                       bda):
        """Calculate the locations of the frustum points in the lidar
        coordinate system.

        Args:
            rots (torch.Tensor): Rotation from camera coordinate system to
                lidar coordinate system in shape (B, N_cams, 3, 3).
            trans (torch.Tensor): Translation from camera coordinate system to
                lidar coordinate system in shape (B, N_cams, 3).
            cam2imgs (torch.Tensor): Camera intrinsic matrixes in shape
                (B, N_cams, 3, 3).
            post_rots (torch.Tensor): Rotation in camera coordinate system in
                shape (B, N_cams, 3, 3). It is derived from the image view
                augmentation.
            post_trans (torch.Tensor): Translation in camera coordinate system
                derived from image view augmentation in shape (B, N_cams, 3).

        Returns:
            torch.tensor: Point coordinates in shape
                (B, N_cams, D, ownsample, 3)
        """
        B, N, _, _ = sensor2ego.shape

        # post-transformation
        # B x N x D x H x W x 3
        points = self.frustum.to(sensor2ego) - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat(
            (points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), 5)
        combine = sensor2ego[:,:,:3,:3].matmul(torch.inverse(cam2imgs))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += sensor2ego[:,:,:3, 3].view(B, N, 1, 1, 1, 3)
        points = bda.view(B, 1, 1, 1, 1, 3,
                          3).matmul(points.unsqueeze(-1)).squeeze(-1)
        return points

    def get_ego_coor(self, sensor2ego, ego2global, cam2imgs, post_rots, post_trans,
                     bda):
        """Calculate the locations of the frustum points in the lidar
        coordinate system.

        Args:
            sensor2ego (torch.Tensor): Transformation from camera coordinate system to
                ego coordinate system in shape (B, N_cams, 4, 4).
            ego2global (torch.Tensor): Translation from ego coordinate system to
                global coordinate system in shape (B, N_cams, 4, 4).
            cam2imgs (torch.Tensor): Camera intrinsic matrixes in shape
                (B, N_cams, 3, 3).
            post_rots (torch.Tensor): Rotation in camera coordinate system in
                shape (B, N_cams, 3, 3). It is derived from the image view
                augmentation.
            post_trans (torch.Tensor): Translation in camera coordinate system
                derived from image view augmentation in shape (B, N_cams, 3).
            bda (torch.Tensor): Transformation in bev. (B, 3, 3)

        Returns:
            torch.tensor: Point coordinates in shape (B, N, D, fH, fW, 3)
        """
        B, N, _, _ = sensor2ego.shape

        # post-transformation
        # (D, fH, fW, 3) - (B, N, 1, 1, 1, 3) --> (B, N, D, fH, fW, 3)
        points = self.frustum.to(sensor2ego) - post_trans.view(B, N, 1, 1, 1, 3)
        # (B, N, 1, 1, 1, 3, 3) @ (B, N, D, fH, fW, 3, 1)  --> (B, N, D, fH, fW, 3, 1)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3)\
            .matmul(points.unsqueeze(-1))

        # cam_to_ego
        # (B, N_, D, fH, fW, 3, 1)  3: (du, dv, d)
        points = torch.cat(
            (points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), 5)
        # R_{c->e} @ K^-1
        combine = sensor2ego[:, :, :3, :3].matmul(torch.inverse(cam2imgs))
        # (B, N, 1, 1, 1, 3, 3) @ (B, N, D, fH, fW, 3, 1)  --> (B, N, D, fH, fW, 3, 1)
        # --> (B, N, D, fH, fW, 3)
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        # (B, N, D, fH, fW, 3) + (B, N, 1, 1, 1, 3) --> (B, N, D, fH, fW, 3)
        points += sensor2ego[:, :, :3, 3].view(B, N, 1, 1, 1, 3)

        # (B, 1, 1, 1, 3, 3) @ (B, N, D, fH, fW, 3, 1) --> (B, N, D, fH, fW, 3, 1)
        # --> (B, N, D, fH, fW, 3)
        points = bda.view(B, 1, 1, 1, 1, 3,
                          3).matmul(points.unsqueeze(-1)).squeeze(-1)
        return points

    def init_acceleration_v2(self, coor, kept):
        """Pre-compute the necessary information in acceleration including the
        index of points in the final feature.

        Args:
            coor (torch.tensor): Coordinate of points in lidar space in shape
                (B, N, D, H, W, 3).
            x (torch.tensor): Feature of points in shape
                (B, N_cams, D, H, W, C).
        """

        ranks_bev, ranks_depth, ranks_feat, \
            interval_starts, interval_lengths = \
            self.voxel_pooling_prepare_v2(coor, kept)
        # ranks_bev: (N_points, ),
        # ranks_depth: (N_points, ),
        # ranks_feat: (N_points, ),
        # interval_starts: (N_pillar, )
        # interval_lengths: (N_pillar, )

        self.ranks_bev = ranks_bev.int().contiguous()
        self.ranks_feat = ranks_feat.int().contiguous()
        self.ranks_depth = ranks_depth.int().contiguous()
        self.interval_starts = interval_starts.int().contiguous()
        self.interval_lengths = interval_lengths.int().contiguous()

    def voxel_pooling_v2(self, coor, depth, feat, kept):
        """
        Args:
            coor: (B, N, D, fH, fW, 3)
            depth: (B, N, D, fH, fW)
            feat: (B, N, C, fH, fW)
        Returns:
            bev_feat: (B, C*Dz(=1), Dy, Dx)
        """
        ranks_bev, ranks_depth, ranks_feat, \
            interval_starts, interval_lengths = \
            self.voxel_pooling_prepare_v2(coor, kept)
        # ranks_bev: (N_points, ),
        # ranks_depth: (N_points, ),
        # ranks_feat: (N_points, ),
        # interval_starts: (N_pillar, )
        # interval_lengths: (N_pillar, )
        if ranks_feat is None:
            print('warning ---> no points within the predefined '
                  'bev receptive field')
            dummy = torch.zeros(size=[
                feat.shape[0], feat.shape[2],
                int(self.grid_size[2]),
                int(self.grid_size[1]),
                int(self.grid_size[0])
            ]).to(feat)     # (B, C, Dz, Dy, Dx)
            dummy = torch.cat(dummy.unbind(dim=2), 1)   # (B, C*Dz, Dy, Dx)
            return dummy

        feat = feat.permute(0, 1, 3, 4, 2)      # (B, N, fH, fW, C)
        bev_feat_shape = (depth.shape[0], int(self.grid_size[2]),
                          int(self.grid_size[1]), int(self.grid_size[0]),
                          feat.shape[-1])       # (B, Dz, Dy, Dx, C)
        bev_feat = bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                               bev_feat_shape, interval_starts,
                               interval_lengths)    # (B, C, Dz, Dy, Dx)
        # collapse Z
        if self.collapse_z:
            bev_feat = torch.cat(bev_feat.unbind(dim=2), 1)     # (B, C*Dz, Dy, Dx)
        return bev_feat

    def voxel_pooling_prepare_v2(self, coor, kept):
        """Data preparation for voxel pooling.
        Args:
            coor (torch.tensor): Coordinate of points in the lidar space in
                shape (B, N, D, H, W, 3).
        Returns:
            tuple[torch.tensor]:
                ranks_bev: Rank of the voxel that a point is belong to in shape (N_points, ),
                    rank介于(0, B*Dx*Dy*Dz-1).
                ranks_depth: Reserved index of points in the depth space in shape (N_Points),
                    rank介于(0, B*N*D*fH*fW-1).
                ranks_feat: Reserved index of points in the feature space in shape (N_Points),
                    rank介于(0, B*N*fH*fW-1).
                interval_starts: (N_pillar, )
                interval_lengths: (N_pillar, )
        """
        B, N, D, H, W, _ = coor.shape
        num_points = B * N * D * H * W
        # record the index of selected points for acceleration purpose
        ranks_depth = torch.range(
            0, num_points - 1, dtype=torch.int, device=coor.device)    # (B*N*D*H*W, ), [0, 1, ..., B*N*D*fH*fW-1]
        ranks_feat = torch.range(
            0, num_points // D - 1, dtype=torch.int, device=coor.device)   # [0, 1, ...,B*N*fH*fW-1]
        ranks_feat = ranks_feat.reshape(B, N, 1, H, W)
        ranks_feat = ranks_feat.expand(B, N, D, H, W).flatten()     # (B*N*D*fH*fW, )

        # convert coordinate into the voxel space
        # ((B, N, D, fH, fW, 3) - (3, )) / (3, ) --> (B, N, D, fH, fW, 3)   3:(x, y, z)  grid coords.
        coor = ((coor - self.grid_lower_bound.to(coor)) /
                self.grid_interval.to(coor))
        coor = coor.long().view(num_points, 3)      # (B, N, D, fH, fW, 3) --> (B*N*D*fH*fW, 3)
        # (B, N*D*fH*fW) --> (B*N*D*fH*fW, 1)
        batch_idx = torch.range(0, B - 1).reshape(B, 1). \
            expand(B, num_points // B).reshape(num_points, 1).to(coor)
        coor = torch.cat((coor, batch_idx), 1)      # (B*N*D*fH*fW, 4)   4: (x, y, z, batch_id)

        if kept is not None:
            kept = kept.view(num_points)
            kept &= (coor[:, 0] >= 0) & (coor[:, 0] < self.grid_size[0]) & \
                (coor[:, 1] >= 0) & (coor[:, 1] < self.grid_size[1]) & \
                (coor[:, 2] >= 0) & (coor[:, 2] < self.grid_size[2])
        else:
            # filter out points that are outside box
            kept = (coor[:, 0] >= 0) & (coor[:, 0] < self.grid_size[0]) & \
                (coor[:, 1] >= 0) & (coor[:, 1] < self.grid_size[1]) & \
                (coor[:, 2] >= 0) & (coor[:, 2] < self.grid_size[2])
        if len(kept) == 0:
            return None, None, None, None, None

        # (N_points, 4), (N_points, ), (N_points, )
        coor, ranks_depth, ranks_feat = \
            coor[kept], ranks_depth[kept], ranks_feat[kept]

        # get tensors from the same voxel next to each other
        ranks_bev = coor[:, 3] * (
            self.grid_size[2] * self.grid_size[1] * self.grid_size[0])
        ranks_bev += coor[:, 2] * (self.grid_size[1] * self.grid_size[0])
        ranks_bev += coor[:, 1] * self.grid_size[0] + coor[:, 0]
        order = ranks_bev.argsort()
        # (N_points, ), (N_points, ), (N_points, )
        ranks_bev, ranks_depth, ranks_feat = \
            ranks_bev[order], ranks_depth[order], ranks_feat[order]

        kept = torch.ones(
            ranks_bev.shape[0], device=ranks_bev.device, dtype=torch.bool)
        kept[1:] = ranks_bev[1:] != ranks_bev[:-1]
        interval_starts = torch.where(kept)[0].int()
        if len(interval_starts) == 0:
            return None, None, None, None, None
        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = ranks_bev.shape[0] - interval_starts[-1]
        return ranks_bev.int().contiguous(), ranks_depth.int().contiguous(
        ), ranks_feat.int().contiguous(), interval_starts.int().contiguous(
        ), interval_lengths.int().contiguous()

    def pre_compute(self, input):
        if self.initial_flag:
            coor = self.get_ego_coor(*input[1:7])       # (B, N, D, fH, fW, 3)
            self.init_acceleration_v2(coor)
            self.initial_flag = False

    def view_transform_core(self, input, depth, tran_feat, kept):
        """
        Args:
            input (list(torch.tensor)):
                imgs:  (B, N, 3, H, W)        # N_views = 6 * (N_history + 1)
                sensor2egos: (B, N, 4, 4)
                ego2globals: (B, N, 4, 4)
                intrins:     (B, N, 3, 3)
                post_rots:   (B, N, 3, 3)
                post_trans:  (B, N, 3)
                bda_rot:  (B, 3, 3)
            depth:  (B*N, D, fH, fW)
            tran_feat: (B*N, C, fH, fW)
        Returns:
            bev_feat: (B, C*Dz(=1), Dy, Dx)
            depth: (B*N, D, fH, fW)
        """
        B, N, C, H, W = input[0].shape
        # Lift-Splat
        if self.accelerate:
            feat = tran_feat.view(B, N, self.out_channels, H, W)      # (B, N, C, fH, fW)
            feat = feat.permute(0, 1, 3, 4, 2)      # (B, N, fH, fW, C)
            depth = depth.view(B, N, self.D, H, W)      # (B, N, D, fH, fW)
            bev_feat_shape = (depth.shape[0], int(self.grid_size[2]),
                              int(self.grid_size[1]), int(self.grid_size[0]),
                              feat.shape[-1])   # (B, Dz, Dy, Dx, C)
            bev_feat = bev_pool_v2(depth, feat, self.ranks_depth,
                                   self.ranks_feat, self.ranks_bev,
                                   bev_feat_shape, self.interval_starts,
                                   self.interval_lengths, kept)   # (B, C, Dz, Dy, Dx)

            bev_feat = bev_feat.squeeze(2)      # (B, C, Dy, Dx)
        else:
            coor = self.get_ego_coor(*input[1:7])   # (B, N, D, fH, fW, 3)
            bev_feat = self.voxel_pooling_v2(
                coor, depth.view(B, N, self.D, H, W),
                tran_feat.view(B, N, self.out_channels, H, W), kept)      # (B, C*Dz(=1), Dy, Dx)
        return bev_feat, depth

    def view_transform(self, input, depth, tran_feat, kept):
        """
        Args:
            input (list(torch.tensor)):
                imgs:  (B, N, C, H, W)        # N_views = 6 * (N_history + 1)
                sensor2egos: (B, N, 4, 4)
                ego2globals: (B, N, 4, 4)
                intrins:     (B, N, 3, 3)
                post_rots:   (B, N, 3, 3)
                post_trans:  (B, N, 3)
                bda_rot:  (B, 3, 3)
            depth:  (B*N, D, fH, fW)
            tran_feat: (B*N, C, fH, fW)
        Returns:
            bev_feat: (B, C, Dy, Dx)
            depth: (B*N, D, fH, fW)
        """
        if self.accelerate:
            self.pre_compute(input)
        return self.view_transform_core(input, depth, tran_feat, kept)

    def forward(self, input):
        """Transform image-view feature into bird-eye-view feature.

        Args:
            input (list(torch.tensor)):
                imgs:  (B, N_views, 3, H, W)        # N_views = 6 * (N_history + 1)
                sensor2egos: (B, N_views, 4, 4)
                ego2globals: (B, N_views, 4, 4)
                intrins:     (B, N_views, 3, 3)
                post_rots:   (B, N_views, 3, 3)
                post_trans:  (B, N_views, 3)
                bda_rot:  (B, 3, 3)
        Returns:
            bev_feat: (B, C, Dy, Dx)
            depth: (B*N, D, fH, fW)
        """
        x = input[0]    # (B, N, C_in, fH, fW)
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)      # (B*N, C_in, fH, fW)

        # (B*N, C_in, fH, fW) --> (B*N, D+C, fH, fW)
        x = self.depth_net(x)
        depth_digit = x[:, :self.D, ...]    # (B*N, D, fH, fW)
        tran_feat = x[:, self.D:self.D + self.out_channels, ...]    # (B*N, C, fH, fW)

        depth = depth_digit.softmax(dim=1)

        if self.use_depth_threhold:
            kept = (depth >= self.depth_threshold)
        else:
            kept = None
        return self.view_transform(input, depth, tran_feat, kept)
    
    def get_mlp_input(self, rot, tran, intrin, post_rot, post_tran, bda):
        return None


@NECKS.register_module()
class LSSViewTransformerBEVDepth(LSSViewTransformer):
    def __init__(self, loss_depth_weight=3.0, 
                 loss_semantic_weight=3.0,
                 depthnet_cfg=dict(), 
                 virtual_depth=False, 
                 min_focal_length=800, 
                 virtual_depth_bin=180, 
                 min_ida_scale=0,
                 dpeht_render_loss=False, 
                 variance_focus=0.85, 
                 render_loss_depth_weight=1,
                 depth_loss_ce = True, 
                 depth_loss_focal=False, 
                 context_residual=False,
                 depth_render_sigmoid=False, 
                 segmentation_loss=False, 
                 loss_segmentation_weight=1, 
                 use_depth_threhold=False, 
                 depth_threshold=1,
                 LSS_Rendervalue=False,
                 balance_cls_weight=False,
                 PV32x88=False,
                 average_pool=False,
                 **kwargs):
        super(LSSViewTransformerBEVDepth, self).__init__(**kwargs)
        self.loss_depth_weight = loss_depth_weight
        self.loss_semantic_weight = loss_semantic_weight
        self.depth_channels = self.D
        self.virtual_depth=virtual_depth
        
        self.dpeht_render_loss = dpeht_render_loss
        self.segmentation_loss = segmentation_loss
        self.variance_focus = variance_focus
        self.render_loss_depth_weight = render_loss_depth_weight
        self.loss_segmentation_weight = loss_segmentation_weight
        self.depth_loss_ce = depth_loss_ce
        self.depth_loss_focal = depth_loss_focal
        self.use_depth_threhold = use_depth_threhold
        self.LSS_Rendervalue = LSS_Rendervalue
        self.balance_cls_weight = balance_cls_weight
        self.PV32x88 = PV32x88
        self.average_pool = average_pool

        if self.average_pool:
            self.x_bound = kwargs['grid_config']['x']
            self.y_bound = kwargs['grid_config']['y']
            self.z_bound = kwargs['grid_config']['z']
            self.register_buffer(
                'voxel_size',
                torch.Tensor([row[2] for row in [self.x_bound, self.y_bound, self.z_bound]]))
            self.register_buffer(
                'voxel_coord',
                torch.Tensor([
                    row[0] + row[2] / 2.0 for row in [self.x_bound, self.y_bound, self.z_bound]
                ]))
            self.register_buffer(
                'voxel_num',
                torch.LongTensor([(row[1] - row[0]) / row[2]
                                for row in [self.x_bound, self.y_bound, self.z_bound]]))
        
        if self.balance_cls_weight:
            self.class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies[:17] + 0.001)).float()
            zero_tensor = torch.zeros(1, dtype=self.class_weights.dtype, device=self.class_weights.device)
            self.class_weights = torch.cat((zero_tensor, self.class_weights),dim=0)
            self.class_weights = self.class_weights.reshape(1,1,1,1,18).contiguous()
        
        if self.depth_loss_focal:
            self.depth_focalloss = build_loss(dict(type="FocalLoss"))
            self.loss_depth_weight = 10
            
        if self.virtual_depth:
            self.depth_channels = virtual_depth_bin
            self.frustum_virtual = self.create_frustum_virtual(self.grid_config['depth'],self.input_size, self.downsample)
            
            min_ida_scale = 1 if min_ida_scale == 0 else min_ida_scale
            self.min_focal_length = min_focal_length * min_ida_scale
        
        self.depth_net = DepthNet(
            in_channels=self.in_channels,
            mid_channels=self.in_channels,
            context_channels=self.out_channels,
            depth_channels=self.depth_channels,
            virtual_depth=self.virtual_depth,
            **depthnet_cfg)
        
        if self.segmentation_loss:
            if self.PV32x88:
                self.class_predictor = nn.Sequential(
                        nn.ConvTranspose2d(self.out_channels, self.out_channels * 2, kernel_size=5, padding=2, stride=2, output_padding=1),
                        nn.BatchNorm2d(self.out_channels * 2),
                        nn.ReLU(),
                        nn.Conv2d(self.out_channels * 2 , self.out_channels * 2, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(self.out_channels * 2),
                        nn.ReLU(),
                        nn.Conv2d(self.out_channels * 2, self.out_channels * 2, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(self.out_channels * 2),
                        nn.ReLU(),
                        nn.Conv2d(self.out_channels * 2, 18, kernel_size=1, stride=1)
                        )
            else:
                self.class_predictor = nn.Sequential(
                        nn.Conv2d(self.out_channels , self.out_channels * 2, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(self.out_channels * 2),
                        nn.ReLU(),
                        nn.Conv2d(self.out_channels * 2, self.out_channels * 2, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(self.out_channels * 2),
                        nn.ReLU(),
                        nn.Conv2d(self.out_channels * 2, 18, kernel_size=1, stride=1)
                        )
            
        self.depth_render_sigmoid = depth_render_sigmoid
        self.LSS_Rendervalue = LSS_Rendervalue
        
        self.context_residual = context_residual
        self.depth_threshold = depth_threshold / self.D
        if self.context_residual:
            self.prepare_residual = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, padding=0)

    def create_frustum_virtual(self, depth_cfg, input_size, downsample):
        """Generate frustum"""
        # make grid in image plane
        ogfH, ogfW = input_size
        fH, fW = ogfH // downsample, ogfW // downsample
        d_coords = torch.arange(*depth_cfg, dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        d_coords = d_coords + depth_cfg[2] / 2.
        D, _, _ = d_coords.shape
        x_coords = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(
            1, 1, fW).expand(D, fH, fW)
        y_coords = torch.linspace(0, ogfH - 1, fH,
                                  dtype=torch.float).view(1, fH,
                                                          1).expand(D, fH, fW)
        paddings = torch.ones_like(d_coords)

        # D x H x W x 3
        frustum = torch.stack((x_coords, y_coords, d_coords, paddings), -1)
        return frustum.cuda()
    
    def get_mlp_input(self, sensor2ego, ego2global, intrin, post_rot, post_tran, bda):
        B, N, _, _ = sensor2ego.shape
        bda = bda.view(B, 1, 3, 3).repeat(1, N, 1, 1)
        if self.virtual_depth:
            mlp_input = torch.stack([
                intrin[:, :, 0, 0],
                intrin[:, :, 1, 1],
                intrin[:, :, 0, 2],
                intrin[:, :, 1, 2],
                post_rot[:, :, 0, 0],
                post_rot[:, :, 0, 1],
                post_tran[:, :, 0],
                post_rot[:, :, 1, 0],
                post_rot[:, :, 1, 1],
                post_tran[:, :, 1],],dim=-1,)
        else:
            mlp_input = torch.stack([
                intrin[:, :, 0, 0],
                intrin[:, :, 1, 1],
                intrin[:, :, 0, 2],
                intrin[:, :, 1, 2],
                post_rot[:, :, 0, 0],
                post_rot[:, :, 0, 1],
                post_tran[:, :, 0],
                post_rot[:, :, 1, 0],
                post_rot[:, :, 1, 1],
                post_tran[:, :, 1],
                bda[:, :, 0, 0],
                bda[:, :, 0, 1],
                bda[:, :, 1, 0],
                bda[:, :, 1, 1],
                bda[:, :, 2, 2],], dim=-1)
            sensor2ego = sensor2ego[:,:,:3,:].reshape(B, N, -1)
            mlp_input = torch.cat([mlp_input, sensor2ego], dim=-1)
        return mlp_input
    
    def get_image_scale(self, intrin_mat, ida_mat):
        img_mat = ida_mat.matmul(intrin_mat)
        fx = (img_mat[:, :, 0, 0] ** 2 + img_mat[:, :, 0, 1] ** 2).sqrt()
        fy = (img_mat[:, :, 1, 0] ** 2 + img_mat[:, :, 1, 1] ** 2).sqrt()
        image_scales = ((fx ** 2 + fy ** 2) / 2.).sqrt()
        return image_scales
    
    def depth_sampling(self, depth_feature, indices):
        b, c, h, w = depth_feature.shape
        indices = indices[:, :, None, None].repeat(1, 1, h, w)
        indices_floor = indices.floor()
        indices_ceil = indices_floor + 1
        max_index = indices_ceil.max().long()
        if max_index >= c:
            depth_feature = torch.cat([depth_feature, depth_feature.new_zeros(b, max_index - c + 1, h, w)], 1)
        sampled_depth_feature = (indices_ceil - indices) * torch.gather(depth_feature, 1, indices_floor.long()) + \
                                (indices - indices_floor) * torch.gather(depth_feature, 1, indices_ceil.long())
        return sampled_depth_feature
    def get_geometry_collapsed(self, sensor2ego, ego2global, cam2imgs, post_rots, post_trans, bda,
                               z_min=-5., z_max=3.):
        B, N, _, _ = sensor2ego.shape

        # post-transformation
        # B x N x D x H x W x 3
        points = self.frustum.to(sensor2ego) - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat(
            (points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), 5)
        combine = sensor2ego[:,:,:3,:3].matmul(torch.inverse(cam2imgs))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += sensor2ego[:,:,:3, 3].view(B, N, 1, 1, 1, 3)
        points = bda.view(B, 1, 1, 1, 1, 3,
                          3).matmul(points.unsqueeze(-1)).squeeze(-1)
        

        # combine = sensor2ego_mat.matmul(torch.inverse(intrin_mat)).double()
        # points = combine.view(batch_size, num_cams, 1, 1, 1, 4,
        #                       4).matmul(points).half()
        # if bda_mat is not None:
        #     bda_mat = bda_mat.unsqueeze(1).repeat(1, num_cams, 1, 1).view(
        #         batch_size, num_cams, 1, 1, 1, 4, 4)
        #     points = (bda_mat @ points).squeeze(-1)
        # else:
        #     points = points.squeeze(-1)

        points_out = points[:, :, :, 0:1, :, :3]
        points_valid_z = ((points[..., 2] > z_min) & (points[..., 2] < z_max))

        return points_out, points_valid_z
    
    def _split_batch_cam(self, feat, inv=False, num_cams=6):
        batch_size = feat.shape[0]
        if not inv:
            return feat.reshape(batch_size // num_cams, num_cams, *feat.shape[1:])
        else:
            return feat.reshape(batch_size * num_cams, *feat.shape[2:])
    def forward(self, input, stereo_metas=None):
        """
        Args:
            input (list(torch.tensor)):
                imgs:  (B, N_views, 3, H, W)        # N_views = 6 * (N_history + 1)
                sensor2egos: (B, N_views, 4, 4)
                ego2globals: (B, N_views, 4, 4)
                intrins:     (B, N_views, 3, 3)
                post_rots:   (B, N_views, 3, 3)
                post_trans:  (B, N_views, 3)
                bda_rot:  (B, 3, 3)
                mlp_input: (B, N_views, 27)
            stereo_metas:  None or dict{
                k2s_sensor: (B, N_views, 4, 4)
                intrins: (B, N_views, 3, 3)
                post_rots: (B, N_views, 3, 3)
                post_trans: (B, N_views, 3)
                frustum: (D, fH_stereo, fW_stereo, 3)  3:(u, v, d)
                cv_downsample: 4,
                downsample: self.img_view_transformer.downsample=16,
                grid_config: self.img_view_transformer.grid_config,
                cv_feat_list: [feat_prev_iv, stereo_feat]
            }
        Returns:
            bev_feat: (B, C, Dy, Dx)
            depth: (B*N, D, fH, fW)
        """
        (x, rots, trans, intrins, post_rots, post_trans, bda, mlp_input) = input[:8]
        B, N, C, H, W = x.shape

        x_input = x.view(B * N, C, H, W)      # (B*N_views, C, fH, fW)
        x_depth, middle_feat = self.depth_net(x_input, mlp_input, stereo_metas)      # (B*N_views, D+C_context, fH, fW)
        
        depth_digit = x_depth[:, :self.depth_channels, ...]    # (B*N_views, D, fH, fW)
        self.depth_feat = depth_digit # for focal loss
        
        if self.LSS_Rendervalue:
            if self.depth_render_sigmoid:
                self.transmittance = torch.exp(-(self.grid_config['depth'][2] * 2 * self.depth_feat.sigmoid()).cumsum(1))
                self.rendering_value = self.transmittance*(1-torch.exp(-self.grid_config['depth'][2] * 2 * self.depth_feat.sigmoid()))
            else:
                self.transmittance = torch.exp(-(self.grid_config['depth'][2] * 2 * self.depth_feat.softmax(dim=1)).cumsum(1))
                self.rendering_value = self.transmittance*(1-torch.exp(-self.grid_config['depth'][2] * 2 * self.depth_feat.softmax(dim=1)))
        
        
        tran_feat = x_depth[:, self.depth_channels:self.depth_channels + self.out_channels, ...]    # (B*N_views, C_context, fH, fW)
        
        if self.context_residual:
            x_for_residual = self.prepare_residual(x_input)
            tran_feat = tran_feat + x_for_residual
        
        if self.LSS_Rendervalue:
            depth = self.rendering_value
        else:
            depth = depth_digit.softmax(dim=1)  # (B*N_views, D, fH, fW)

        if self.use_depth_threhold:
            kept = (depth >= self.depth_threshold)
        else:
            kept = None

        if self.virtual_depth:
            ida = torch.repeat_interleave(torch.eye(3).unsqueeze(0), B * N, dim=0).view(B, N, 3, 3).to(rots.device)
            ida[:,:,:3,:3] = post_rots
            ida[:,:,:2,2] = post_trans[:,:,:2]
            
            visual_depth_feature = depth_digit[:, :self.depth_channels]
            image_scales = self.get_image_scale(intrins, ida)
            offset_per_meter = self.depth_channels / self.grid_config['depth'][1] * self.min_focal_length / image_scales.view(-1, 1)
            offset = self.frustum_virtual[:, 0, 0, 2].view(1, -1) * offset_per_meter
            
            real_depth_feature = self.depth_sampling(visual_depth_feature, offset)
            depth = real_depth_feature.softmax(1)
    
        if self.average_pool:
            img_feat_with_depth = depth.unsqueeze(1) * tran_feat.unsqueeze(2)
            geom_xyz, geom_xyz_valid = self.get_geometry_collapsed(
                rots,
                trans,
                intrins,
                post_rots, 
                post_trans,
                bda,
                z_min=-1.,
                z_max=5.4)
            
            geom_xyz_valid = self._split_batch_cam(geom_xyz_valid, inv=True).unsqueeze(1)
            img_feat_with_depth = (img_feat_with_depth * geom_xyz_valid).sum(3).unsqueeze(3)
            img_context = img_feat_with_depth
            img_context = self._split_batch_cam(img_context)
            img_context = img_context.permute(0, 1, 3, 4, 5, 2).contiguous()
            
            geom_xyz = ((geom_xyz - (self.voxel_coord - self.voxel_size / 2.0)) / self.voxel_size).int()
            geom_xyz[..., 2] = 0  # collapse z-axis
            geo_pos = torch.ones_like(geom_xyz)
            
            feature_map, _ = average_voxel_pooling(geom_xyz, img_context, geo_pos, self.voxel_num.cuda())      
            return feature_map.contiguous(), depth_digit.softmax(dim=1), middle_feat
        else:
            bev_feat, depth = self.view_transform(input, depth, tran_feat, kept)
            
            return bev_feat, depth, middle_feat

    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: (B, N_views, img_h, img_w)
        Output:
            gt_depths: (B*N_views*fH*fW, D)
        """
        B, N, H, W = gt_depths.shape
        # (B*N_views, fH, downsample, fW, downsample, 1)
        gt_depths = gt_depths.view(B * N,
                                   H // self.downsample, self.downsample,
                                   W // self.downsample, self.downsample,
                                   1)
        # (B*N_views, fH, fW, 1, downsample, downsample)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        # (B*N_views*fH*fW, downsample, downsample)
        gt_depths = gt_depths.view(-1, self.downsample * self.downsample)
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        # (B*N_views, fH, fW)
        gt_depths = gt_depths.view(B * N, H // self.downsample, W // self.downsample)

        if not self.sid:
            # (D - (min_dist - interval_dist)) / interval_dist
            # = (D - min_dist) / interval_dist + 1
            gt_depths = (gt_depths - (self.grid_config['depth'][0] - self.grid_config['depth'][2])) / self.grid_config['depth'][2]
        else:
            gt_depths = torch.log(gt_depths) - torch.log(
                torch.tensor(self.grid_config['depth'][0]).float())
            gt_depths = gt_depths * (self.D - 1) / torch.log(
                torch.tensor(self.grid_config['depth'][1] - 1.).float() /
                self.grid_config['depth'][0])
            gt_depths = gt_depths + 1.

        gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0), gt_depths, torch.zeros_like(gt_depths))     # (B*N_views, fH, fW)
        gt_depths_onehot = F.one_hot(gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:, 1:]   # (B*N_views*fH*fW, D)
        gt_depths = gt_depths - (self.grid_config['depth'][0] + self.grid_config['depth'][2])
        return gt_depths, gt_depths_onehot.float()
    
    
    def get_downsampled_gt_depth_and_semantic(self, gt_depths, gt_semantics):
        # remove point not in depth range
        gt_semantics[gt_depths < self.grid_config['depth'][0]] = 0
        gt_semantics[gt_depths > self.grid_config['depth'][1]] = 0
        gt_depths[gt_depths < self.grid_config['depth'][0]] = 0
        gt_depths[gt_depths > self.grid_config['depth'][1]] = 0

        B, N, H, W = gt_semantics.shape
        num_classes = 18
        one_hot = torch.nn.functional.one_hot(gt_semantics.to(torch.int64), num_classes=num_classes)
        
        if self.PV32x88:
            semantic_downsample = int(self.downsample / 2)
        else:
            semantic_downsample = self.downsample
            
        one_hot = one_hot.view(B, N, H // semantic_downsample, semantic_downsample, W // semantic_downsample, semantic_downsample, num_classes)
        class_counts = one_hot.sum(dim=(3, 5)).to(gt_semantics)
        class_counts[..., 0] = 0
        class_counts = class_counts.to(gt_semantics)
        if self.balance_cls_weight:
            class_counts = class_counts * self.class_weights
            
        _, most_frequent_classes = class_counts.max(dim=-1)
        gt_semantics = most_frequent_classes.view(B * N, H // semantic_downsample, W // semantic_downsample)
        # gt_semantics = F.one_hot(gt_semantics.long(), num_classes=18).view(-1, 18).float()
        gt_semantics = F.one_hot(gt_semantics.long(), num_classes=18).permute(0,3,1,2).float().contiguous()


        
        if self.PV32x88:
            B, N, H, W = gt_depths.shape
            gt_depths_32x88 = gt_depths.view(
                B * N,
                H // (self.downsample//2),
                self.downsample//2,
                W // (self.downsample//2),
                self.downsample//2,
                1,
            )
            gt_depths_32x88 = gt_depths_32x88.permute(0, 1, 3, 5, 2, 4).contiguous()
            gt_depths_32x88 = gt_depths_32x88.view(
                -1, (self.downsample//2) * (self.downsample//2))
            gt_depths_32x88_tmp = torch.where(gt_depths_32x88 == 0.0,
                                        1e5 * torch.ones_like(gt_depths_32x88),
                                        gt_depths_32x88)
            gt_depths_32x88 = torch.min(gt_depths_32x88_tmp, dim=-1).values
            gt_depths_32x88 = gt_depths_32x88.view(B * N, H // (self.downsample//2), W // (self.downsample//2))
            gt_depths_32x88 = (gt_depths_32x88 - (self.grid_config['depth'][0] - self.grid_config['depth'][2])) / self.grid_config['depth'][2]
            gt_depths_32x88 = torch.where((gt_depths_32x88 < self.D + 1) & (gt_depths_32x88 >= 0.0), gt_depths_32x88, torch.zeros_like(gt_depths_32x88))
            gt_depths_32x88_onehot = F.one_hot(gt_depths_32x88.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:, 1:].float()
        else:
            gt_depths_32x88_onehot = None

        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(
            B * N,
            H // self.downsample,
            self.downsample,
            W // self.downsample,
            self.downsample,
            1,
        )
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(
            -1, self.downsample * self.downsample)
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.downsample, W // self.downsample)
        gt_depths = (gt_depths - (self.grid_config['depth'][0] - self.grid_config['depth'][2])) / self.grid_config['depth'][2]
        gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0), gt_depths, torch.zeros_like(gt_depths))
        gt_depths_onehot = F.one_hot(gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:, 1:].float()
        gt_depths = gt_depths - (self.grid_config['depth'][0] + self.grid_config['depth'][2])
        
        return gt_depths, gt_depths_onehot, gt_semantics, gt_depths_32x88_onehot
    
    @force_fp32()
    def get_SA_loss(self, semantic_preds, depth_preds, sa_gt_depth, sa_gt_semantic):
        depth_loss_dict = dict()
        depth_labels_value, depth_labels, semantic_labels_PV, depth_labels_32x88 = self.get_downsampled_gt_depth_and_semantic(sa_gt_depth, sa_gt_semantic)
        
        if self.dpeht_render_loss:
            C,num_bins,feature_h,feature_w= depth_preds.shape
            depth_bin_linspace = torch.linspace(start=self.grid_config['depth'][0]*2, end=self.grid_config['depth'][1]*2, steps=num_bins)
            frustum_distance_bin = depth_bin_linspace.repeat(C,feature_h,feature_w, 1).permute(0, 3, 1, 2).contiguous().to(self.depth_feat)

            if self.depth_render_sigmoid:
                transmittance = torch.exp(-(self.grid_config['depth'][2] * 2 * self.depth_feat.sigmoid()).cumsum(1))
                depth_pred_rendered = (transmittance*(1-torch.exp(-self.grid_config['depth'][2] * 2 * self.depth_feat.sigmoid()))*frustum_distance_bin).sum(1)
            else:
                transmittance = torch.exp(-(self.grid_config['depth'][2] * 2 * self.depth_feat.softmax(dim=1)).cumsum(1))
                depth_pred_rendered = (transmittance*(1-torch.exp(-self.grid_config['depth'][2] * 2 * self.depth_feat.softmax(dim=1)))*frustum_distance_bin).sum(1)
            # transmittance = (self.grid_config['depth'][2] * depth_preds).cumsum(1)
            # depth_pred_rendered = (transmittance*(1-torch.exp(-self.grid_config['depth'][2] * depth_preds))).sum(1)
            
            fg_mask = depth_labels_value > 0.0
            log_d = torch.log(depth_pred_rendered[fg_mask]) - torch.log(depth_labels_value[fg_mask])
            depth_render_loss = torch.sqrt((log_d ** 2).mean() - self.variance_focus * (log_d.mean() ** 2))
            depth_render_loss = depth_render_loss * self.render_loss_depth_weight
            depth_loss_dict['loss_depth_render'] = depth_render_loss
            
        if self.segmentation_loss:
            context_feature = self.class_predictor(semantic_preds) # 24,256,16,44
            if self.PV32x88:
                B,C,W,H = sa_gt_depth.shape
                semantic_downsample = int(self.downsample / 2)
                W = int(W / self.downsample)
                H = int(H / self.downsample)
                
                PV_fg_mask = torch.max(depth_labels_32x88, dim=1).values > 0.0
            else:
                PV_fg_mask = torch.max(depth_labels, dim=1).values > 0.0
            context_feature = context_feature.softmax(dim=1).permute(0, 2, 3, 1).contiguous().view(-1, 18)
            semantic_labels = semantic_labels_PV.permute(0, 2, 3, 1).contiguous().view(-1, 18).to(context_feature)
            semantic_pred = context_feature[PV_fg_mask]
            semantic_labels = semantic_labels[PV_fg_mask]
            with autocast(enabled=False):
                segmentation_loss = F.binary_cross_entropy(
                    semantic_pred,
                    semantic_labels,
                    reduction='none',
                ).sum() / max(1.0, PV_fg_mask.sum())
            depth_loss_dict['loss_segmentation'] = self.loss_segmentation_weight * segmentation_loss
            
        # depth_labels_value, depth_labels = self.get_downsampled_gt_depth(gt_depth)
        if self.depth_loss_ce:
            if self.depth_loss_focal:
                depth_preds = self.depth_feat
                depth_labels_value = depth_labels_value.view(-1)
                fg_mask = depth_labels_value > 0.0
                depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(-1, self.D)
                depth_labels_value = depth_labels_value[fg_mask]
                depth_preds = depth_preds[fg_mask]
                depth_loss = self.depth_focalloss(depth_preds, depth_labels_value.long())
            else:
                # (B*N_views, D, fH, fW) --> (B*N_views, fH, fW, D) --> (B*N_views*fH*fW, D)
                depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(-1, self.D)
                fg_mask = torch.max(depth_labels, dim=1).values > 0.0
                depth_labels = depth_labels[fg_mask]
                depth_preds = depth_preds[fg_mask]
                with autocast(enabled=False):
                    depth_loss = F.binary_cross_entropy(
                        depth_preds,
                        depth_labels,
                        reduction='none',
                    ).sum() / max(1.0, fg_mask.sum())
            depth_loss_dict['loss_depth'] = self.loss_depth_weight * depth_loss
        
        return depth_loss_dict, semantic_labels_PV, PV_fg_mask
    
    
    @force_fp32()
    def get_depth_loss(self, gt_depth, depth_preds):
        """
            Args:
                depth_labels: (B, N_views, img_h, img_w)
                depth_preds: (B*N_views, D, fH, fW)
            Returns:
        """
        depth_loss_dict = dict()
        depth_labels_value, depth_labels = self.get_downsampled_gt_depth(gt_depth)      # (B*N_views*fH*fW, D)
        
        if self.dpeht_render_loss:
            C, num_bins, feature_h, feature_w= depth_preds.shape
            depth_bin_linspace = torch.arange(self.D)
            frustum_distance_bin = depth_bin_linspace.repeat(C, feature_h, feature_w, 1).permute(0, 3, 1, 2).contiguous().to(self.depth_feat)
            # transmittance = (self.grid_config['depth'][2] * depth_preds).cumsum(1)
            # depth_pred_rendered = (transmittance*(1-torch.exp(-self.grid_config['depth'][2] * depth_preds))).sum(1)
            if self.LSS_Rendervalue:
                transmittance=self.transmittance
                depth_pred_rendered=(self.rendering_value*frustum_distance_bin).sum(1)
            else:
                if self.depth_render_sigmoid:
                    transmittance = torch.exp(-(self.grid_config['depth'][2] * 2 * self.depth_feat.sigmoid()).cumsum(1))
                    depth_pred_rendered = (transmittance*(1-torch.exp(-self.grid_config['depth'][2] * 2 * self.depth_feat.sigmoid()))*frustum_distance_bin).sum(1)
                else:
                    transmittance = torch.exp(-(self.grid_config['depth'][2] * 2 * self.depth_feat.softmax(dim=1)).cumsum(1))
                    depth_pred_rendered = (transmittance*(1-torch.exp(-self.grid_config['depth'][2] * 2 * self.depth_feat.softmax(dim=1)))*frustum_distance_bin).sum(1)
            
            fg_mask = depth_labels_value > 0.0
            log_d = torch.log(depth_pred_rendered[fg_mask]) - torch.log(depth_labels_value[fg_mask])
            depth_render_loss = torch.sqrt((log_d ** 2).mean() - self.variance_focus * (log_d.mean() ** 2))
            depth_render_loss = depth_render_loss * self.render_loss_depth_weight
            depth_loss_dict['loss_depth_render'] = depth_render_loss
            
        if self.depth_loss_ce:
            if self.depth_loss_focal:
                depth_preds = self.depth_feat
                depth_labels_value = depth_labels_value.view(-1)
                fg_mask = depth_labels_value > 0.0
                depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(-1, self.D)
                depth_labels_value = depth_labels_value[fg_mask]
                depth_preds = depth_preds[fg_mask]
                depth_loss = self.depth_focalloss(depth_preds, depth_labels_value.long())
            else:
                # (B*N_views, D, fH, fW) --> (B*N_views, fH, fW, D) --> (B*N_views*fH*fW, D)
                depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(-1, self.D)
                fg_mask = torch.max(depth_labels, dim=1).values > 0.0
                depth_labels = depth_labels[fg_mask]
                depth_preds = depth_preds[fg_mask]
                with autocast(enabled=False):
                    depth_loss = F.binary_cross_entropy(
                        depth_preds,
                        depth_labels,
                        reduction='none',
                    ).sum() / max(1.0, fg_mask.sum())
            depth_loss_dict['loss_depth'] = self.loss_depth_weight * depth_loss

        return depth_loss_dict




@NECKS.register_module()
class CRN_LSS(LSSViewTransformer):
    def __init__(self, loss_depth_weight=3.0, loss_semantic_weight=3.0, depthnet_cfg=dict(), virtual_depth=False, 
                 dpeht_render_loss=False, variance_focus=0.85, render_loss_depth_weight=1,
                 depth_loss_ce = True, depth_loss_focal=False, context_residual=False,
                 depth_render_sigmoid=False, segmentation_loss=False, loss_segmentation_weight=1, use_depth_threhold=False, 
                 depth_threshold=1,LSS_Rendervalue=False, depth_ASPP=False,
                 **kwargs):
        super(CRN_LSS, self).__init__(**kwargs)
        self.loss_depth_weight = loss_depth_weight
        self.loss_semantic_weight = loss_semantic_weight
        self.depth_channels = self.D
        self.virtual_depth=virtual_depth
        
        self.dpeht_render_loss = dpeht_render_loss
        self.segmentation_loss = segmentation_loss
        self.variance_focus = variance_focus
        self.render_loss_depth_weight = render_loss_depth_weight
        self.loss_segmentation_weight = loss_segmentation_weight
        self.depth_loss_ce = depth_loss_ce
        self.depth_loss_focal = depth_loss_focal
        self.use_depth_threhold = use_depth_threhold
        self.LSS_Rendervalue = LSS_Rendervalue
        self.depth_ASPP = depth_ASPP
        
        if self.depth_loss_focal:
            self.depth_focalloss = build_loss(dict(type="FocalLoss"))
            self.loss_depth_weight = 10    

        if self.segmentation_loss:
            if self.PV32x88:
                self.class_predictor = self.class_predictor = nn.Sequential(
                        nn.Conv2d(self.out_channels , self.out_channels * 2, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(self.out_channels * 2),
                        nn.ReLU(),
                        nn.Conv2d(self.out_channels * 2, self.out_channels * 2, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(self.out_channels * 2),
                        nn.ReLU(),
                        nn.ConvTranspose2d(self.out_channels * 2, 18, kernel_size=3, stride=2, padding=1)
                        )
            else:
                self.class_predictor = nn.Sequential(
                        nn.Conv2d(self.out_channels , self.out_channels * 2, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(self.out_channels * 2),
                        nn.ReLU(),
                        nn.Conv2d(self.out_channels * 2, self.out_channels * 2, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(self.out_channels * 2),
                        nn.ReLU(),
                        nn.Conv2d(self.out_channels * 2, 18, kernel_size=1, stride=1)
                        )
        
        if self.depth_ASPP:
            self.depth_net = DepthNet(
                in_channels=self.in_channels,
                mid_channels=self.in_channels,
                context_channels=self.out_channels,
                depth_channels=self.depth_channels,
                **depthnet_cfg)
        else:
            self.depth_net = CRN_DepthNet(
                in_channels=self.in_channels,
                mid_channels=self.in_channels,
                context_channels=self.out_channels,
                depth_channels=self.depth_channels,
                )
        # self.register_buffer(
        #     'voxel_size',
        #     torch.Tensor([row[2] for row in [x_bound, y_bound, z_bound]]))
        # self.register_buffer(
        #     'voxel_coord',
        #     torch.Tensor([
        #         row[0] + row[2] / 2.0 for row in [x_bound, y_bound, z_bound]
        #     ]))

        self.loss_predict_depth_grad = build_loss(dict(type="FocalLoss"))
        self.loss_predict_upsampledepth = build_loss(dict(type="FocalLoss"))
        self.depth_render_sigmoid=depth_render_sigmoid
        self.context_residual = context_residual
        self.depth_threshold = depth_threshold / self.D
        if self.context_residual:
            self.prepare_residual = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, padding=0)
        
        self.x_bound = kwargs['grid_config']['x']
        self.y_bound = kwargs['grid_config']['y']
        self.z_bound = kwargs['grid_config']['z']
        self.register_buffer(
            'voxel_size',
            torch.Tensor([row[2] for row in [self.x_bound, self.y_bound, self.z_bound]]))
        self.register_buffer(
            'voxel_coord',
            torch.Tensor([
                row[0] + row[2] / 2.0 for row in [self.x_bound, self.y_bound, self.z_bound]
            ]))
        self.register_buffer(
            'voxel_num',
            torch.LongTensor([(row[1] - row[0]) / row[2]
                              for row in [self.x_bound, self.y_bound, self.z_bound]]))
    
    def get_geometry_collapsed(self, sensor2ego, ego2global, cam2imgs, post_rots, post_trans, bda,
                               z_min=-5., z_max=3.):
        B, N, _, _ = sensor2ego.shape

        # post-transformation
        # B x N x D x H x W x 3
        points = self.frustum.to(sensor2ego) - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat(
            (points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), 5)
        combine = sensor2ego[:,:,:3,:3].matmul(torch.inverse(cam2imgs))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += sensor2ego[:,:,:3, 3].view(B, N, 1, 1, 1, 3)
        points = bda.view(B, 1, 1, 1, 1, 3,
                          3).matmul(points.unsqueeze(-1)).squeeze(-1)
        

        # combine = sensor2ego_mat.matmul(torch.inverse(intrin_mat)).double()
        # points = combine.view(batch_size, num_cams, 1, 1, 1, 4,
        #                       4).matmul(points).half()
        # if bda_mat is not None:
        #     bda_mat = bda_mat.unsqueeze(1).repeat(1, num_cams, 1, 1).view(
        #         batch_size, num_cams, 1, 1, 1, 4, 4)
        #     points = (bda_mat @ points).squeeze(-1)
        # else:
        #     points = points.squeeze(-1)

        points_out = points[:, :, :, 0:1, :, :3]
        points_valid_z = ((points[..., 2] > z_min) & (points[..., 2] < z_max))

        return points_out, points_valid_z
    
    def _split_batch_cam(self, feat, inv=False, num_cams=6):
        batch_size = feat.shape[0]
        if not inv:
            return feat.reshape(batch_size // num_cams, num_cams, *feat.shape[1:])
        else:
            return feat.reshape(batch_size * num_cams, *feat.shape[2:])
        
    def forward(self, input, stereo_metas=None):
        """
        Args:
            input (list(torch.tensor)):
                imgs:  (B, N_views, 3, H, W)        # N_views = 6 * (N_history + 1)
                sensor2egos: (B, N_views, 4, 4)
                ego2globals: (B, N_views, 4, 4)
                intrins:     (B, N_views, 3, 3)
                post_rots:   (B, N_views, 3, 3)
                post_trans:  (B, N_views, 3)
                bda_rot:  (B, 3, 3)
                mlp_input: (B, N_views, 27)
            stereo_metas:  None or dict{
                k2s_sensor: (B, N_views, 4, 4)
                intrins: (B, N_views, 3, 3)
                post_rots: (B, N_views, 3, 3)
                post_trans: (B, N_views, 3)
                frustum: (D, fH_stereo, fW_stereo, 3)  3:(u, v, d)
                cv_downsample: 4,
                downsample: self.img_view_transformer.downsample=16,
                grid_config: self.img_view_transformer.grid_config,
                cv_feat_list: [feat_prev_iv, stereo_feat]
            }
        Returns:
            bev_feat: (B, C, Dy, Dx)
            depth: (B*N, D, fH, fW)
        """
        (x, rots, trans, intrins, post_rots, post_trans, bda, mlp_input) = input[:8]
        B, N, C, H, W = x.shape

        x_input = x.view(B * N, C, H, W)      # (B*N_views, C, fH, fW)
        x_depth, middle_feat = self.depth_net(x_input, mlp_input)      # (B*N_views, D+C_context, fH, fW)
        
        depth_digit = x_depth[:, :self.depth_channels, ...]    # (B*N_views, D, fH, fW)
        self.depth_feat = depth_digit # for focal loss

        if self.LSS_Rendervalue:
            if self.depth_render_sigmoid:
                self.transmittance = torch.exp(-(self.grid_config['depth'][2] * 2 * self.depth_feat.sigmoid()).cumsum(1))
                self.rendering_value = self.transmittance*(1-torch.exp(-self.grid_config['depth'][2] * 2 * self.depth_feat.sigmoid()))
            else:
                self.transmittance = torch.exp(-(self.grid_config['depth'][2] * 2 * self.depth_feat.softmax(dim=1)).cumsum(1))
                self.rendering_value = self.transmittance*(1-torch.exp(-self.grid_config['depth'][2] * 2 * self.depth_feat.softmax(dim=1)))
        

        tran_feat = x_depth[:, self.depth_channels:self.depth_channels + self.out_channels, ...]    # (B*N_views, C_context, fH, fW)
            
        depth = depth_digit.softmax(dim=1)  # (B*N_views, D, fH, fW)
        img_feat_with_depth = depth.unsqueeze(1) * tran_feat.unsqueeze(2)
        
        ida = torch.zeros(B, N, 4, 4)
        ida[:, :, :3, :3] = post_rots
        ida[:, :, :3, 3] = post_trans
        ida[:, :, 3, 3] = 1
        

        geom_xyz, geom_xyz_valid = self.get_geometry_collapsed(
            rots,
            trans,
            intrins,
            post_rots, 
            post_trans,
            bda,
            z_min=-1.,
            z_max=5.4)
        
        geom_xyz_valid = self._split_batch_cam(geom_xyz_valid, inv=True).unsqueeze(1)
        img_feat_with_depth = (img_feat_with_depth * geom_xyz_valid).sum(3).unsqueeze(3)
        img_context = img_feat_with_depth
        img_context = self._split_batch_cam(img_context)
        img_context = img_context.permute(0, 1, 3, 4, 5, 2).contiguous()
        
        geom_xyz = ((geom_xyz - (self.voxel_coord - self.voxel_size / 2.0)) /
                    self.voxel_size).int()
        geom_xyz[..., 2] = 0  # collapse z-axis
        geo_pos = torch.ones_like(geom_xyz)
        
        feature_map, _ = average_voxel_pooling(geom_xyz, img_context, geo_pos, self.voxel_num.cuda())
        
        # bev_feat, depth = self.view_transform(input, depth, tran_feat, kept)
        
        return feature_map.contiguous(), depth_digit.softmax(dim=1), middle_feat


    def get_image_scale(self, intrin_mat, ida_mat):
        img_mat = ida_mat.matmul(intrin_mat)
        fx = (img_mat[:, :, 0, 0] ** 2 + img_mat[:, :, 0, 1] ** 2).sqrt()
        fy = (img_mat[:, :, 1, 0] ** 2 + img_mat[:, :, 1, 1] ** 2).sqrt()
        image_scales = ((fx ** 2 + fy ** 2) / 2.).sqrt()
        return image_scales
    
    def depth_sampling(self, depth_feature, indices):
        b, c, h, w = depth_feature.shape
        indices = indices[:, :, None, None].repeat(1, 1, h, w)
        indices_floor = indices.floor()
        indices_ceil = indices_floor + 1
        max_index = indices_ceil.max().long()
        if max_index >= c:
            depth_feature = torch.cat([depth_feature, depth_feature.new_zeros(b, max_index - c + 1, h, w)], 1)
        sampled_depth_feature = (indices_ceil - indices) * torch.gather(depth_feature, 1, indices_floor.long()) + \
                                (indices - indices_floor) * torch.gather(depth_feature, 1, indices_ceil.long())
        return sampled_depth_feature
    
    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: (B, N_views, img_h, img_w)
        Output:
            gt_depths: (B*N_views*fH*fW, D)
        """
        B, N, H, W = gt_depths.shape
        # (B*N_views, fH, downsample, fW, downsample, 1)
        gt_depths = gt_depths.view(B * N,
                                   H // self.downsample, self.downsample,
                                   W // self.downsample, self.downsample,
                                   1)
        # (B*N_views, fH, fW, 1, downsample, downsample)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        # (B*N_views*fH*fW, downsample, downsample)
        gt_depths = gt_depths.view(-1, self.downsample * self.downsample)
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        # (B*N_views, fH, fW)
        gt_depths = gt_depths.view(B * N, H // self.downsample, W // self.downsample)

        if not self.sid:
            # (D - (min_dist - interval_dist)) / interval_dist
            # = (D - min_dist) / interval_dist + 1
            gt_depths = (gt_depths - (self.grid_config['depth'][0] - self.grid_config['depth'][2])) / self.grid_config['depth'][2]
        else:
            gt_depths = torch.log(gt_depths) - torch.log(
                torch.tensor(self.grid_config['depth'][0]).float())
            gt_depths = gt_depths * (self.D - 1) / torch.log(
                torch.tensor(self.grid_config['depth'][1] - 1.).float() /
                self.grid_config['depth'][0])
            gt_depths = gt_depths + 1.

        gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0), gt_depths, torch.zeros_like(gt_depths))     # (B*N_views, fH, fW)
        gt_depths_onehot = F.one_hot(gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:, 1:]   # (B*N_views*fH*fW, D)
        gt_depths = gt_depths - (self.grid_config['depth'][0] + self.grid_config['depth'][2])
        return gt_depths, gt_depths_onehot.float()
    
    def get_downsampled_gt_depth_and_semantic(self, gt_depths, gt_semantics):
        # remove point not in depth range
        gt_semantics[gt_depths < self.grid_config['depth'][0]] = 0
        gt_semantics[gt_depths > self.grid_config['depth'][1]] = 0
        gt_depths[gt_depths < self.grid_config['depth'][0]] = 0
        gt_depths[gt_depths > self.grid_config['depth'][1]] = 0

        B, N, H, W = gt_semantics.shape
        num_classes = 18
        one_hot = torch.nn.functional.one_hot(gt_semantics.to(torch.int64), num_classes=num_classes)
        if self.PV32x88:
            semantic_downsample = int(self.downsample / 2)
        else:
            semantic_downsample = self.downsample
        one_hot = one_hot.view(B, N, H // self.downsample, self.downsample, W // self.downsample, self.downsample, num_classes)
        class_counts = one_hot.sum(dim=(3, 5))
        class_counts[..., 0] = 0
        _, most_frequent_classes = class_counts.max(dim=-1)
        gt_semantics = most_frequent_classes.view(B * N, H // self.downsample, W // self.downsample)
        # gt_semantics = F.one_hot(gt_semantics.long(), num_classes=18).view(-1, 18).float()
        gt_semantics = F.one_hot(gt_semantics.long(), num_classes=18).permute(0,3,1,2).float().contiguous()

        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(
            B * N,
            H // self.downsample,
            self.downsample,
            W // self.downsample,
            self.downsample,
            1,
        )
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(
            -1, self.downsample * self.downsample)
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.downsample, W // self.downsample)
        gt_depths = (gt_depths - (self.grid_config['depth'][0] - self.grid_config['depth'][2])) / self.grid_config['depth'][2]
        gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0), gt_depths, torch.zeros_like(gt_depths))
        gt_depths_onehot = F.one_hot(gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:, 1:].float()
        gt_depths = gt_depths - (self.grid_config['depth'][0] + self.grid_config['depth'][2])
        return gt_depths, gt_depths_onehot, gt_semantics
            
    def get_mlp_input(self, sensor2ego, ego2global, intrin, post_rot, post_tran, bda):
        B, N, _, _ = sensor2ego.shape
        bda = bda.view(B, 1, 3, 3).repeat(1, N, 1, 1)
        if self.virtual_depth:
            mlp_input = torch.stack([
                intrin[:, :, 0, 0],
                intrin[:, :, 1, 1],
                intrin[:, :, 0, 2],
                intrin[:, :, 1, 2],
                post_rot[:, :, 0, 0],
                post_rot[:, :, 0, 1],
                post_tran[:, :, 0],
                post_rot[:, :, 1, 0],
                post_rot[:, :, 1, 1],
                post_tran[:, :, 1],],dim=-1,)
        else:
            mlp_input = torch.stack([
                intrin[:, :, 0, 0],
                intrin[:, :, 1, 1],
                intrin[:, :, 0, 2],
                intrin[:, :, 1, 2],
                post_rot[:, :, 0, 0],
                post_rot[:, :, 0, 1],
                post_tran[:, :, 0],
                post_rot[:, :, 1, 0],
                post_rot[:, :, 1, 1],
                post_tran[:, :, 1],
                bda[:, :, 0, 0],
                bda[:, :, 0, 1],
                bda[:, :, 1, 0],
                bda[:, :, 1, 1],
                bda[:, :, 2, 2],], dim=-1)
            sensor2ego = sensor2ego[:,:,:3,:].reshape(B, N, -1)
            mlp_input = torch.cat([mlp_input, sensor2ego], dim=-1)
        return mlp_input
    
    @force_fp32()
    def get_SA_loss(self, semantic_preds, depth_preds, sa_gt_depth, sa_gt_semantic):
        depth_loss_dict = dict()
        depth_labels_value, depth_labels, semantic_labels = self.get_downsampled_gt_depth_and_semantic(sa_gt_depth, sa_gt_semantic)
        
        if self.dpeht_render_loss:
            C,num_bins,feature_h,feature_w= depth_preds.shape
            depth_bin_linspace = torch.linspace(start=self.grid_config['depth'][0]*2, end=self.grid_config['depth'][1]*2, steps=num_bins)
            frustum_distance_bin = depth_bin_linspace.repeat(C,feature_h,feature_w, 1).permute(0, 3, 1, 2).contiguous().to(self.depth_feat)

            if self.depth_render_sigmoid:
                transmittance = torch.exp(-(self.grid_config['depth'][2] * 2 * self.depth_feat.sigmoid()).cumsum(1))
                depth_pred_rendered = (transmittance*(1-torch.exp(-self.grid_config['depth'][2] * 2 * self.depth_feat.sigmoid()))*frustum_distance_bin).sum(1)
            else:
                transmittance = torch.exp(-(self.grid_config['depth'][2] * 2 * self.depth_feat.softmax(dim=1)).cumsum(1))
                depth_pred_rendered = (transmittance*(1-torch.exp(-self.grid_config['depth'][2] * 2 * self.depth_feat.softmax(dim=1)))*frustum_distance_bin).sum(1)
            # transmittance = (self.grid_config['depth'][2] * depth_preds).cumsum(1)
            # depth_pred_rendered = (transmittance*(1-torch.exp(-self.grid_config['depth'][2] * depth_preds))).sum(1)
            
            fg_mask = depth_labels_value > 0.0
            log_d = torch.log(depth_pred_rendered[fg_mask]) - torch.log(depth_labels_value[fg_mask])
            depth_render_loss = torch.sqrt((log_d ** 2).mean() - self.variance_focus * (log_d.mean() ** 2))
            depth_render_loss = depth_render_loss * self.render_loss_depth_weight
            depth_loss_dict['loss_depth_render'] = depth_render_loss
            
        if self.segmentation_loss:
            context_feature = self.class_predictor(semantic_preds)
            # context_with_depth = torch.einsum('bnik,bmik->bnmik', self.depth_feat.sigmoid(), context_feature)
            fg_mask = torch.max(depth_labels, dim=1).values > 0.0
            context_feature = context_feature.softmax(dim=1).permute(0, 2, 3, 1).contiguous().view(-1, 18)
            semantic_labels = semantic_labels.permute(0, 2, 3, 1).contiguous().view(-1, 18)
            semantic_pred = context_feature[fg_mask]
            semantic_labels = semantic_labels[fg_mask]
            breakpoint()
            with autocast(enabled=False):
                segmentation_loss = F.binary_cross_entropy(
                    semantic_pred,
                    semantic_labels,
                    reduction='none',
                ).sum() / max(1.0, fg_mask.sum())
            depth_loss_dict['loss_segmentation'] = self.loss_segmentation_weight * segmentation_loss
            
            
        # depth_labels_value, depth_labels = self.get_downsampled_gt_depth(gt_depth)
        if self.depth_loss_ce:
            if self.depth_loss_focal:
                depth_preds = self.depth_feat
                depth_labels_value = depth_labels_value.view(-1)
                fg_mask = depth_labels_value > 0.0
                depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(-1, self.D)
                depth_labels_value = depth_labels_value[fg_mask]
                depth_preds = depth_preds[fg_mask]
                depth_loss = self.depth_focalloss(depth_preds, depth_labels_value.long())
            else:
                # (B*N_views, D, fH, fW) --> (B*N_views, fH, fW, D) --> (B*N_views*fH*fW, D)
                depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(-1, self.D)
                fg_mask = torch.max(depth_labels, dim=1).values > 0.0
                depth_labels = depth_labels[fg_mask]
                depth_preds = depth_preds[fg_mask]
                with autocast(enabled=False):
                    depth_loss = F.binary_cross_entropy(
                        depth_preds,
                        depth_labels,
                        reduction='none',
                    ).sum() / max(1.0, fg_mask.sum())
            depth_loss_dict['loss_depth'] = self.loss_depth_weight * depth_loss
        
        return depth_loss_dict
    
    @force_fp32()
    def get_depth_loss(self, gt_depth, depth_preds):
        """
            Args:
                depth_labels: (B, N_views, img_h, img_w)
                depth_preds: (B*N_views, D, fH, fW)
            Returns:
        """
        depth_loss_dict = dict()
        depth_labels_value, depth_labels = self.get_downsampled_gt_depth(gt_depth)      # (B*N_views*fH*fW, D)
        
        if self.dpeht_render_loss:
            C, num_bins, feature_h, feature_w= depth_preds.shape
            depth_bin_linspace = torch.arange(self.D)
            frustum_distance_bin = depth_bin_linspace.repeat(C, feature_h, feature_w, 1).permute(0, 3, 1, 2).contiguous().to(self.depth_feat)
            # transmittance = (self.grid_config['depth'][2] * depth_preds).cumsum(1)
            # depth_pred_rendered = (transmittance*(1-torch.exp(-self.grid_config['depth'][2] * depth_preds))).sum(1)
            if self.LSS_Rendervalue:
                transmittance=self.transmittance
                depth_pred_rendered=(self.rendering_value*frustum_distance_bin).sum(1)
            else:
                if self.depth_render_sigmoid:
                    transmittance = torch.exp(-(self.grid_config['depth'][2] * 2 * self.depth_feat.sigmoid()).cumsum(1))
                    depth_pred_rendered = (transmittance*(1-torch.exp(-self.grid_config['depth'][2] * 2 * self.depth_feat.sigmoid()))*frustum_distance_bin).sum(1)
                else:
                    transmittance = torch.exp(-(self.grid_config['depth'][2] * 2 * self.depth_feat.softmax(dim=1)).cumsum(1))
                    depth_pred_rendered = (transmittance*(1-torch.exp(-self.grid_config['depth'][2] * 2 * self.depth_feat.softmax(dim=1)))*frustum_distance_bin).sum(1)
            
            fg_mask = depth_labels_value > 0.0
            log_d = torch.log(depth_pred_rendered[fg_mask]) - torch.log(depth_labels_value[fg_mask])
            depth_render_loss = torch.sqrt((log_d ** 2).mean() - self.variance_focus * (log_d.mean() ** 2))
            depth_render_loss = depth_render_loss * self.render_loss_depth_weight
            depth_loss_dict['loss_depth_render'] = depth_render_loss
            
        if self.depth_loss_ce:
            if self.depth_loss_focal:
                depth_preds = self.depth_feat
                depth_labels_value = depth_labels_value.view(-1)
                fg_mask = depth_labels_value > 0.0
                depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(-1, self.D)
                depth_labels_value = depth_labels_value[fg_mask]
                depth_preds = depth_preds[fg_mask]
                depth_loss = self.depth_focalloss(depth_preds, depth_labels_value.long())
            else:
                # (B*N_views, D, fH, fW) --> (B*N_views, fH, fW, D) --> (B*N_views*fH*fW, D)
                depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(-1, self.D)
                fg_mask = torch.max(depth_labels, dim=1).values > 0.0
                depth_labels = depth_labels[fg_mask]
                depth_preds = depth_preds[fg_mask]
                with autocast(enabled=False):
                    depth_loss = F.binary_cross_entropy(
                        depth_preds,
                        depth_labels,
                        reduction='none',
                    ).sum() / max(1.0, fg_mask.sum())
            depth_loss_dict['loss_depth'] = self.loss_depth_weight * depth_loss
            
            
        return depth_loss_dict
    
@NECKS.register_module()
class LSSViewTransformerBEVStereo(LSSViewTransformerBEVDepth):
    def __init__(self,  **kwargs):
        super(LSSViewTransformerBEVStereo, self).__init__(**kwargs)
        # (D, fH_stereo, fW_stereo, 3)  3:(u, v, d)
        self.cv_frustum = self.create_frustum(kwargs['grid_config']['depth'],
                                              kwargs['input_size'],
                                              downsample=4)



from projects.mmdet3d_plugin.models.necks.view_transformer import LSSViewTransformerBEVDepth

import torch
from torch import nn
from torch.cuda.amp import autocast
from mmdet3d.models.builder import NECKS

class HoriConv(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels, cat_dim=0):
        """HoriConv that reduce the image feature
            in height dimension and refine it.

        Args:
            in_channels (int): in_channels
            mid_channels (int): mid_channels
            out_channels (int): output channels
            cat_dim (int, optional): channels of position
                embedding. Defaults to 0.
        """
        super().__init__()

        self.merger = nn.Sequential(
            nn.Conv2d(in_channels + cat_dim,
                      in_channels,
                      kernel_size=1,
                      bias=True),
            nn.Sigmoid(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True),
        )

        self.reduce_conv = nn.Sequential(
            nn.Conv1d(
                in_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.conv1 = nn.Sequential(
            nn.Conv1d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.out_conv = nn.Sequential(
            nn.Conv1d(
                mid_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    @autocast(False)
    def forward(self, x, pe=None):
        # [N,C,H,W]
        if pe is not None:
            x = self.merger(torch.cat([x, pe], 1))
        else:
            x = self.merger(x)
        x = x.max(2)[0]
        x = self.reduce_conv(x)
        x = self.conv1(x) + x
        x = self.conv2(x) + x
        x = self.out_conv(x)
        return x


class DepthReducer(nn.Module):

    def __init__(self, img_channels, mid_channels):
        """Module that compresses the predicted
            categorical depth in height dimension

        Args:
            img_channels (int): in_channels
            mid_channels (int): mid_channels
        """
        super().__init__()
        self.vertical_weighter = nn.Sequential(
            nn.Conv2d(img_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, kernel_size=3, stride=1, padding=1),
        )

    @autocast(False)
    def forward(self, feat, depth):
        vert_weight = self.vertical_weighter(feat).softmax(2)  # [N,1,H,W]
        depth = (depth * vert_weight).sum(2)
        return depth


@NECKS.register_module()
class MatrixVT(LSSViewTransformerBEVDepth):
    def __init__(self, matrixVT_grid_config, **kwargs):
        super(MatrixVT, self).__init__(**kwargs)
        self.matrixVT_grid_config = matrixVT_grid_config
        self.x_bound = self.matrixVT_grid_config['x']
        self.y_bound = self.matrixVT_grid_config['y']
        self.z_bound = self.matrixVT_grid_config['z']
        self.register_buffer(
            'voxel_num',
            torch.LongTensor([(row[1] - row[0]) / row[2] for row in [self.x_bound, self.y_bound, self.z_bound]]))
        self.register_buffer('bev_anchors',
                             self.create_bev_anchors(self.x_bound, self.y_bound))
        self.register_buffer(
            'voxel_coord',
            torch.Tensor([
                row[0] + row[2] / 2.0 for row in [self.x_bound, self.y_bound, self.z_bound]
            ]))
        self.register_buffer(
            'voxel_size',
            torch.Tensor([row[2] for row in [self.x_bound, self.y_bound, self.z_bound]]))
        self.horiconv = HoriConv(self.out_channels, 512,
                                 self.out_channels)
        self.depth_reducer = DepthReducer(self.out_channels,
                                          self.out_channels)
        self.static_mat = None
    def create_bev_anchors(self, x_bound, y_bound, ds_rate=1):
        """Create anchors in BEV space

        Args:
            x_bound (list): xbound in meters [start, end, step]
            y_bound (list): ybound in meters [start, end, step]
            ds_rate (iint, optional): downsample rate. Defaults to 1.

        Returns:
            anchors: anchors in [W, H, 2]
        """
        x_coords = ((torch.linspace(
            x_bound[0],
            x_bound[1] - x_bound[2] * ds_rate,
            self.voxel_num[0] // ds_rate,
            dtype=torch.float,
        ) + x_bound[2] * ds_rate / 2).view(self.voxel_num[0] // ds_rate,
                                           1).expand(
                                               self.voxel_num[0] // ds_rate,
                                               self.voxel_num[1] // ds_rate))
        y_coords = ((torch.linspace(
            y_bound[0],
            y_bound[1] - y_bound[2] * ds_rate,
            self.voxel_num[1] // ds_rate,
            dtype=torch.float,
        ) + y_bound[2] * ds_rate / 2).view(
            1,
            self.voxel_num[1] // ds_rate).expand(self.voxel_num[0] // ds_rate,
                                                 self.voxel_num[1] // ds_rate))

        anchors = torch.stack([x_coords, y_coords]).permute(1, 2, 0).contiguous()
        return anchors
    
    # def create_frustum(self):
    #     """Generate frustum"""
    #     # make grid in image plane
    #     ogfH, ogfW = self.final_dim
    #     fH, fW = ogfH // self.downsample_factor, ogfW // self.downsample_factor
    #     d_coords = torch.arange(*self.d_bound,
    #                             dtype=torch.float).view(-1, 1,
    #                                                     1).expand(-1, fH, fW)
    #     D, _, _ = d_coords.shape
    #     x_coords = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(
    #         1, 1, fW).expand(D, fH, fW)
    #     y_coords = torch.linspace(0, ogfH - 1, fH,
    #                               dtype=torch.float).view(1, fH,
    #                                                       1).expand(D, fH, fW)
    #     paddings = torch.ones_like(d_coords)

    #     # D x H x W x 3
    #     frustum = torch.stack((x_coords, y_coords, d_coords, paddings), -1)
    #     return frustum
    
    def get_geometry(self, sensor2ego_mat, intrin_mat, ida_mat, bda_mat):
        """Transfer points from camera coord to ego coord.

        Args:
            rots(Tensor): Rotation matrix from camera to ego.
            trans(Tensor): Translation matrix from camera to ego.
            intrins(Tensor): Intrinsic matrix.
            post_rots_ida(Tensor): Rotation matrix for ida.
            post_trans_ida(Tensor): Translation matrix for ida
            post_rot_bda(Tensor): Rotation matrix for bda.

        Returns:
            Tensors: points ego coord.
        """
        batch_size, num_cams, _, _ = sensor2ego_mat.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum
        ida_mat = ida_mat.view(batch_size, num_cams, 1, 1, 1, 4, 4).contiguous() 
        points = ida_mat.inverse().matmul(points.unsqueeze(-1))
        # cam_to_ego
        points = torch.cat(
            (points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
             points[:, :, :, :, :, 2:]), 5)

        combine = sensor2ego_mat.matmul(torch.inverse(intrin_mat))
        points = combine.view(batch_size, num_cams, 1, 1, 1, 4,
                              4).matmul(points).contiguous() 
        if bda_mat is not None:
            bda_mat = bda_mat.unsqueeze(1).repeat(1, num_cams, 1, 1).view(
                batch_size, num_cams, 1, 1, 1, 4, 4).contiguous() 
            points = (bda_mat @ points).squeeze(-1)
        else:
            points = points.squeeze(-1)
        return points[..., :3]
    
    def get_proj_mat(self, inputs, mats_dict=None):
        """Create the Ring Matrix and Ray Matrix

        Args:
            mats_dict (dict, optional): dictionary that
                contains intrin- and extrin- parameters.
            Defaults to None.

        Returns:
            tuple: Ring Matrix in [B, D, L, L] and Ray Matrix in [B, W, L, L]
        """
        if self.static_mat is not None:
            return self.static_mat

        bev_size = int(self.voxel_num[0])  # only consider square BEV
        geom_sep = self.get_lidar_coor(*inputs[1:7])
        geom_sep = (geom_sep - (self.voxel_coord - self.voxel_size / 2.0)) / self.voxel_size
        geom_sep = geom_sep.mean(3).permute(0, 1, 3, 2, 4).contiguous()  # B,Ncam,W,D,2
        B, Nc, W, D, _ = geom_sep.shape
        geom_sep = geom_sep.long().view(B, Nc * W, D, -1).contiguous()[..., :2]

        invalid1 = torch.logical_or((geom_sep < 0)[..., 0], (geom_sep < 0)[...,
                                                                           1])
        invalid2 = torch.logical_or((geom_sep > (bev_size - 1))[..., 0],
                                    (geom_sep > (bev_size - 1))[..., 1])
        geom_sep[(invalid1 | invalid2)] = int(bev_size / 2)
        geom_idx = geom_sep[..., 1] * bev_size + geom_sep[..., 0]

        geom_uni = self.bev_anchors[None].repeat([B, 1, 1, 1])  # B,128,128,2
        B, L, L, _ = geom_uni.shape

        circle_map = geom_uni.new_zeros((B, D, L * L))

        ray_map = geom_uni.new_zeros((B, Nc * W, L * L))
        for b in range(B):
            for dir in range(Nc * W):
                ray_map[b, dir, geom_idx[b, dir]] += 1
            for d in range(D):
                circle_map[b, d, geom_idx[b, :, d]] += 1
        null_point = int((bev_size / 2) * (bev_size + 1))
        circle_map[..., null_point] = 0
        ray_map[..., null_point] = 0
        circle_map = circle_map.view(B, D, L * L)
        ray_map = ray_map.view(B, -1, L * L)
        circle_map /= circle_map.max(1)[0].clip(min=1)[:, None]
        ray_map /= ray_map.max(1)[0].clip(min=1)[:, None]

        return circle_map, ray_map

    @autocast(False)
    def reduce_and_project(self, feature, depth, inputs, mats_dict):
        """reduce the feature and depth in height
            dimension and make BEV feature

        Args:
            feature (Tensor): image feature in [B, C, H, W]
            depth (Tensor): Depth Prediction in [B, D, H, W]
            mats_dict (dict): dictionary that contains intrin-
                and extrin- parameters

        Returns:
            Tensor: BEV feature in B, C, L, L
        """
        # [N,112,H,W], [N,256,H,W]
        depth = self.depth_reducer(feature, depth)

        B = mats_dict['intrin_mats'].shape[0]

        # N, C, H, W = feature.shape
        # feature=feature.reshape(N,C*H,W)
        feature = self.horiconv(feature)
        # feature = feature.max(2)[0]
        # [N.112,W], [N,C,W]
        depth = depth.permute(0, 2, 1).contiguous().reshape(B, -1, self.D)
        feature = feature.permute(0, 2, 1).contiguous().reshape(B, -1, self.out_channels)
        circle_map, ray_map = self.get_proj_mat(inputs, mats_dict)

        proj_mat = depth.matmul(circle_map)
        proj_mat = (proj_mat * ray_map).permute(0, 2, 1).contiguous()
        img_feat_with_depth = proj_mat.matmul(feature)
        img_feat_with_depth = img_feat_with_depth.permute(0, 2, 1).contiguous().reshape(B, -1, *self.voxel_num[:2])

        return img_feat_with_depth

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


        mats_dict (dict):
            sensor2ego_mats(Tensor): Transformation matrix from
                camera to ego with shape of (B, num_sweeps,
                num_cameras, 4, 4).
            intrin_mats(Tensor): Intrinsic matrix with shape
                of (B, num_sweeps, num_cameras, 4, 4).
            ida_mats(Tensor): Transformation matrix for ida with
                shape of (B, num_sweeps, num_cameras, 4, 4).
            sensor2sensor_mats(Tensor): Transformation matrix
                from key frame camera to sweep frame camera with
                shape of (B, num_sweeps, num_cameras, 4, 4).
            bda_mat(Tensor): Rotation matrix for bda with shape
                of (B, 4, 4).
        """
        (x, rots, trans, intrins, post_rots, post_trans, bda, mlp_input) = input[:8]
        B, N, C, H, W = x.shape

        ida = torch.zeros(B, N, 4, 4)
        ida[:, :, :3, :3] = post_rots
        ida[:, :, :3, 3] = post_trans
        ida[:, :, 3, 3] = 1
        inputs = input[:8]
        mats_dict={'sensor2ego_mats':rots.unsqueeze(dim=1).contiguous() ,'ida_mats':ida.unsqueeze(dim=1).contiguous() ,'intrin_mats':intrins.unsqueeze(dim=1).contiguous() ,'bda_mat':bda}

        x = x.view(B * N, C, H, W).contiguous()      # (B*N_views, C, fH, fW)
        
        x = self.depth_net(x, mlp_input, stereo_metas)      # (B*N_views, D+C_context, fH, fW)

        with autocast(enabled=False):
            depth_digit = x[:, :self.D, ...]    # (B*N_views, D, fH, fW)
            tran_feat = x[:, self.D:self.D + self.out_channels, ...]    # (B*N_views, C_context, fH, fW)
            depth = depth_digit.softmax(dim=1)  # (B*N_views, D, fH, fW)

            bev_feat = self.reduce_and_project(tran_feat, depth, inputs, mats_dict)  # [b*n, c, d, w]


        # bev_feat, depth = self.view_transform(input, depth, tran_feat)
        return bev_feat, depth
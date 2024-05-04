import torch
import torch.nn as nn
from datetime import datetime

from .frustum_grid_generator import FrustumGridGenerator
from .sampler import Sampler
from mmdet3d.models import NECKS
from mmcv.runner import force_fp32


@NECKS.register_module()
class FrustumToVoxel(nn.Module):

    def __init__(self, grid_config, pc_range, mode='bilinear', padding_mode='zeros',
                 depth_mode="UD", ):
        """
        Initializes module to transform frustum features to voxel features via 3D transformation and sampling
        Args:
            model_cfg: EasyDict, Module configuration
            grid_size: [X, Y, Z], Voxel grid size
            pc_range: [x_min, y_min, z_min, x_max, y_max, z_max], Voxelization point cloud range (m)
            disc_cfg: EasyDict, Depth discretiziation configuration
        """
        super().__init__()
        
        num_bins = (grid_config['depth'][1] - grid_config['depth'][0])//grid_config['depth'][2]
        depth_min = grid_config['depth'][0]
        depth_max = grid_config['depth'][1]
        self.grid_size = [(grid_config['x'][1]-grid_config['x'][0])/grid_config['x'][2],
                          (grid_config['y'][1]-grid_config['y'][0])/grid_config['y'][2],
                          (grid_config['z'][1]-grid_config['z'][0])/grid_config['z'][2]
                          ]
        
        self.pc_range = pc_range
        self.grid_generator = FrustumGridGenerator(grid_size=self.grid_size,
                                                   pc_range=pc_range,
                                                   num_bins=num_bins,
                                                   depth_mode=depth_mode,
                                                   depth_min=depth_min,
                                                   depth_max=depth_max,
                                                   )
        self.sampler = Sampler(mode, padding_mode)

    def forward(self, trans_lidar_to_cam, trans_cam_to_img, image_shape, bda_4x4, frustum_features, frustum_depth_attr):
        """
        Generates voxel features via 3D transformation and sampling
        Args:
            frustum_features: (B, C, D, H_image, W_image), Image frustum features
            lidar_to_cam: (B, 4, 4), LiDAR to camera frame transformation
            cam_to_img: (B, 3, 4), Camera projection matrix
            image_shape: (B, 2), Image shape [H, W]
        Returns:
            voxel_features: (B, C, Z, Y, X), Image voxel features
        """
        
        B, N, _, _ = trans_lidar_to_cam.shape
        grid = self.grid_generator(bda_4x4=bda_4x4,lidar_to_cam=trans_lidar_to_cam.reshape(B*N,*trans_lidar_to_cam.shape[2:]),
                                   cam_to_img=trans_cam_to_img.reshape(B*N,*trans_cam_to_img.shape[2:]),
                                   image_shape=image_shape,
                                   )  # (B, X, Y, Z, 3)

        # Sample frustum volume to generate voxel volume
        
        voxel_features = self.sampler(input_features=frustum_features.reshape(B*N, *frustum_features.shape[2:])[:,None], grid=grid)  # (B, C, X, Y, Z) # v2
        # voxel_features = self.sampler(input_features=frustum_features.reshape(B*N, *frustum_features.shape[2:])[:,None].permute(0,1,4,3,2), grid=grid)  # (B, C, X, Y, Z) #v1
        voxel_features = voxel_features.squeeze(1).reshape(B,N,*voxel_features.shape[-3:])

        voxel_non_zero = (voxel_features!=0).sum(1)
        if frustum_depth_attr:
            voxel_score = torch.ones_like(voxel_non_zero).to(voxel_features)
        else:
            voxel_score = torch.zeros_like(voxel_non_zero).to(voxel_features)
        voxel_non_bool = voxel_non_zero.bool()
        voxel_score[voxel_non_bool] = voxel_features.sum(1)[voxel_non_bool]/voxel_non_zero[voxel_non_bool]
        voxel_score = torch.nan_to_num(voxel_score)
        return voxel_score

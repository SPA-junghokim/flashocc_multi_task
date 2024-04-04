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

    def forward(self, trans_lidar_to_cam, trans_cam_to_img, image_shape, bda_4x4, frustum_features):
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
        breakpoint()
        
        voxel_features = self.sampler(input_features=frustum_features.reshape(B*N, *frustum_features.shape[2:])[:,None], grid=grid)  # (B, C, X, Y, Z)
        voxel_features = self.sampler(input_features=frustum_features.reshape(B*N, *frustum_features.shape[2:])[:,None].permute(0,1,4,3,2), grid=grid)  # (B, C, X, Y, Z)
        # voxel_features = self.sampler(input_features=frustum_features.reshape(B*N, *frustum_features.shape[2:])[:,None].permute(0,1,3,4,2), grid=grid)  # (B, C, X, Y, Z)
        # voxel_features = self.sampler(input_features=frustum_features.reshape(B*N, *frustum_features.shape[2:])[:,None].permute(0,1,3,4,2), grid=grid/self.downsample)  # (B, C, X, Y, Z)
        voxel_features = voxel_features.squeeze(1).reshape(B,N,*voxel_features.shape[-3:])
        # voxel_features = voxel_features.permute(0, 1, 4, 3, 2)
        # (B, C, X, Y, Z) -> (B, C, Z, Y, X)
        import matplotlib.pyplot as plt
        cam_id = 0
        tensor = (voxel_features[0][cam_id].sum(0).bool()).float().cpu()
        tensor = tensor/tensor.max()*255
        plt.imshow(tensor, cmap='gray')
        plt.axis('off')  # 축 제거
        plt.savefig(f'tensor_image{cam_id}.png', bbox_inches='tight', pad_inches=0)  # 이미지 저장
        
        cam_id = 1
        tensor = (voxel_features[0][cam_id].sum(0).bool()).float().cpu()
        tensor = tensor/tensor.max()*255
        plt.imshow(tensor, cmap='gray')
        plt.axis('off')  # 축 제거
        plt.savefig(f'tensor_image{cam_id}.png', bbox_inches='tight', pad_inches=0)  # 이미지 저장

        cam_id = 2
        tensor = (voxel_features[0][cam_id].sum(0).bool()).float().cpu()
        tensor = tensor/tensor.max()*255
        plt.imshow(tensor, cmap='gray')
        plt.axis('off')  # 축 제거
        plt.savefig(f'tensor_image{cam_id}.png', bbox_inches='tight', pad_inches=0)  # 이미지 저장

        cam_id = 3
        tensor = (voxel_features[0][cam_id].sum(0).bool()).float().cpu()
        tensor = tensor/tensor.max()*255
        plt.imshow(tensor, cmap='gray')
        plt.axis('off')  # 축 제거
        plt.savefig(f'tensor_image{cam_id}.png', bbox_inches='tight', pad_inches=0)  # 이미지 저장

        cam_id = 4
        tensor = (voxel_features[0][cam_id].sum(0).bool()).float().cpu()
        tensor = tensor/tensor.max()*255
        plt.imshow(tensor, cmap='gray')
        plt.axis('off')  # 축 제거
        plt.savefig(f'tensor_image{cam_id}.png', bbox_inches='tight', pad_inches=0)  # 이미지 저장

        cam_id = 5
        tensor = (voxel_features[0][cam_id].sum(0).bool()).float().cpu()
        tensor = tensor/tensor.max()*255
        plt.imshow(tensor, cmap='gray')
        plt.axis('off')  # 축 제거
        plt.savefig(f'tensor_image{cam_id}.png', bbox_inches='tight', pad_inches=0)  # 이미지 저장

        tensor0 = (voxel_features[0][0])
        tensor1 = (voxel_features[0][1])
        (tensor0!=tensor1).sum()
        # voxel_features = voxel_features.permute(0, 1, 4, 3, 2)

        return voxel_features

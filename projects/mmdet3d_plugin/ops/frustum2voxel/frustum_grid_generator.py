import torch
import torch.nn as nn

try:
    from kornia.utils.grid import create_meshgrid3d
    from kornia.geometry.linalg import transform_points
except Exception as e:
    # Note: Kornia team will fix this import issue to try to allow the usage of lower torch versions.
    # print('Warning: kornia is not installed correctly, please ignore this warning if you do not use CaDDN. Otherwise, it is recommended to use torch version greater than 1.2 to use kornia properly.')
    pass

from .utils_frustum_to_voxel import project_to_image, normalize_coords, bin_depths

from datetime import datetime
from mmcv.runner import force_fp32


class FrustumGridGenerator(nn.Module):

    def __init__(self, grid_size, pc_range, num_bins=88, depth_mode="UD", depth_min=1.0, depth_max=45.0):
        """
        Initializes Grid Generator for frustum features
        Args:
            grid_size: [X, Y, Z], Voxel grid size
            pc_range: [x_min, y_min, z_min, x_max, y_max, z_max], Voxelization point cloud range (m)
        """
        super().__init__()
        try:
            import kornia
        except Exception as e:
            # Note: Kornia team will fix this import issue to try to allow the usage of lower torch versions.
            print('Error: kornia is not installed correctly, please ignore this warning if you do not use CaDDN. '
                  'Otherwise, it is recommended to use torch version greater than 1.2 to use kornia properly.')
            exit(-1)

        self.dtype = torch.float32
        self.grid_size = torch.as_tensor(grid_size, dtype=self.dtype)
        self.pc_range = pc_range
        self.out_of_bounds_val = -2
        
        self.depth_mode = depth_mode
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.num_bins = num_bins

        # Calculate voxel size
        pc_range = torch.as_tensor(pc_range).reshape(2, 3)
        self.pc_min = pc_range[0]
        self.pc_max = pc_range[1]
        self.voxel_size = (self.pc_max - self.pc_min) / self.grid_size

        # Create voxel grid
        self.width, self.height, self.depth = self.grid_size.int()
        # self.voxel_grid = create_meshgrid3d(depth=self.depth,height=self.height,width=self.width,normalized_coordinates=False)
        # self.voxel_grid = create_meshgrid3d(depth=self.width,height=self.depth,width=self.height,normalized_coordinates=False)
                
        
        xs = torch.linspace(0.5, self.width - 0.5, self.width, 
                            dtype=torch.float32).view(1, 1, self.width).expand(self.depth, self.height, self.width)
        ys = torch.linspace(0.5, self.height - 0.5, self.height, 
                            dtype=torch.float32).view(-1, self.height, 1).expand(self.depth, self.height, self.width)
        zs = torch.linspace(0.5, self.depth - 0.5, self.depth, 
                            dtype=torch.float32).view(-1, 1, 1).expand(self.depth, self.height, self.width)
        self.voxel_grid = torch.stack((xs, ys, zs), -1)
        # base_grid = stack(torch_meshgrid([xs, ys, zs], indexing="ij"), dim=-1)  # DxWxHx3
        # self.voxel_grid = self.voxel_grid.permute(0, 1, 3, 2, 4)  # XZY-> XYZ
        
        # Add offsets to center of voxel
        # self.voxel_grid += 0.5
        self.grid_to_lidar = self.grid_to_lidar_unproject(pc_min=self.pc_min,
                                                          voxel_size=self.voxel_size)

        
    def grid_to_lidar_unproject(self, pc_min, voxel_size):
        """
        Calculate grid to LiDAR unprojection for each plane
        Args:
            pc_min: [x_min, y_min, z_min], Minimum of point cloud range (m)
            voxel_size: [x, y, z], Size of each voxel (m)
        Returns:
            unproject: (4, 4), Voxel grid to LiDAR unprojection matrix
        """
        x_size, y_size, z_size = voxel_size
        x_min, y_min, z_min = pc_min
        unproject = torch.tensor([[x_size, 0, 0, x_min],
                                  [0, y_size, 0, y_min],
                                  [0,  0, z_size, z_min],
                                  [0,  0, 0, 1]],
                                 dtype=self.dtype)  # (4, 4)
        return unproject

    def transform_grid(self, voxel_grid, grid_to_lidar, lidar_to_cam, cam_to_img, bda_4x4):
        """
        Transforms voxel sampling grid into frustum sampling grid
        Args:
            grid: (B, X, Y, Z, 3), Voxel sampling grid
            grid_to_lidar: (4, 4), Voxel grid to LiDAR unprojection matrix
            lidar_to_cam: (B, 4, 4), LiDAR to camera frame transformation
            cam_to_img: (B, 3, 4), Camera projection matrix
        Returns:
            frustum_grid: (B, X, Y, Z, 3), Frustum sampling grid
        """
        B = lidar_to_cam.shape[0]

        # Create transformation matricies
        V_G = grid_to_lidar[None].repeat(B,1,1) # Voxel Grid -> LiDAR (4, 4)
        C_V = lidar_to_cam  # LiDAR -> Camera (B, 4, 4)
        # C_V = torch.inverse(lidar_to_cam)  # LiDAR -> Camera (B, 4, 4)
        I_C = cam_to_img  # Camera -> Image (B, 3, 4)
        # trans = C_V @ bda_4x4 @ V_G
        trans = C_V @ bda_4x4 @ V_G
        # Reshape to match dimensions
        V_G_ = V_G.reshape(B, 1, 1, 4, 4)
        trans = trans.reshape(B, 1, 1, 4, 4)
        voxel_grid = voxel_grid.repeat_interleave(repeats=B, dim=0)

        # Transform to camera frame
        camera_grid = transform_points(trans_01=trans, points_1=voxel_grid)
        
        # camera_grid_ = transform_points(trans_01=trans, points_1=voxel_grid)
        # camera_grid_ = transform_points(trans_01=V_G_, points_1=voxel_grid)
        # camera_grid_ = camera_grid_.reshape(4,6,200,200,16,3)
        # import matplotlib.pyplot as plt
        # cam_id = 0
        # tensor = camera_grid_[0][cam_id,...,0,0].float().cpu()
        # tensor = tensor/tensor.max()*255
        # plt.imshow(tensor, cmap='gray')
        # plt.axis('off')  # 축 제거
        # plt.savefig(f'tensor_image{cam_id}_0.png', bbox_inches='tight', pad_inches=0)  # 이미지 저장
        
        # tensor = camera_grid_[0][cam_id,...,0,1].float().cpu()
        # tensor = tensor/tensor.max()*255
        # plt.imshow(tensor, cmap='gray')
        # plt.axis('off')  # 축 제거
        # plt.savefig(f'tensor_image{cam_id}_1.png', bbox_inches='tight', pad_inches=0)  # 이미지 저장
        
        # from kornia.geometry.conversions import convert_points_to_homogeneous, convert_points_from_homogeneous
        # h_voxel_grid = convert_points_to_homogeneous(voxel_grid)
        # shape_inp = list(h_voxel_grid.shape)
        # h_voxel_grid = h_voxel_grid.reshape(-1, h_voxel_grid.shape[-2], h_voxel_grid.shape[-1])
        # V_G_ = V_G_.reshape(-1, V_G_.shape[-2], V_G_.shape[-1])
        # V_G_ = torch.repeat_interleave(V_G_, repeats=h_voxel_grid.shape[0] // V_G_.shape[0], dim=0)
        # V_G_ = V_G_.transpose(1,2)
        # camera_grid_h = h_voxel_grid@V_G_
        # camera_grid = convert_points_from_homogeneous(camera_grid_h)  # BxNxD
        # shape_inp[-2] = camera_grid.shape[-2]
        # shape_inp[-1] = camera_grid.shape[-1]
        # camera_grid = camera_grid.reshape(shape_inp)

        # Project to image
        I_C = I_C.reshape(B, 1, 1, 3, 4)
        image_grid, image_depths = project_to_image(project=I_C, points=camera_grid)

        # Convert depths to depth bins
        image_depths = bin_depths(depth_map=image_depths, mode=self.depth_mode, depth_min=self.depth_min, depth_max=self.depth_max, num_bins=self.num_bins)

        # Stack to form frustum grid
        image_depths = image_depths.unsqueeze(-1)
        frustum_grid = torch.cat((image_grid, image_depths), dim=-1)
        return frustum_grid

    def forward(self, bda_4x4, lidar_to_cam, cam_to_img, image_shape):
    # def trans(self, lidar_to_cam, cam_to_img, image_shape):
        """
        Generates sampling grid for frustum features
        Args:
            lidar_to_cam: (B, 4, 4), LiDAR to camera frame transformation
            cam_to_img: (B, 3, 4), Camera projection matrix
            image_shape: (B, 2), Image shape [H, W]
        Returns:
            frustum_grid (B, X, Y, Z, 3), Sampling grids for frustum features
        """

        frustum_grid = self.transform_grid(voxel_grid=self.voxel_grid.to(lidar_to_cam.device)[None],
                                           grid_to_lidar=self.grid_to_lidar.to(lidar_to_cam.device),
                                           lidar_to_cam=lidar_to_cam,
                                           cam_to_img=cam_to_img,
                                           bda_4x4=bda_4x4,)

        # Normalize grid
        # image_shape, _ = torch.max(image_shape, dim=0)
        image_depth = torch.tensor([self.num_bins],device=image_shape.device,dtype=image_shape.dtype)
        frustum_shape = torch.cat((image_depth, image_shape))
        frustum_grid = normalize_coords(coords=frustum_grid, shape=frustum_shape)

        # Replace any NaNs or infinites with out of bounds
        mask = ~torch.isfinite(frustum_grid)
        frustum_grid[mask] = self.out_of_bounds_val

        return frustum_grid


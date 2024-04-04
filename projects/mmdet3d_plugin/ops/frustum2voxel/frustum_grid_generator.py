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
        xs = torch.linspace(0.5, self.width - 0.5, self.width, 
                            dtype=torch.float32).view(1, 1, self.width).expand(self.depth, self.height, self.width)
        ys = torch.linspace(0.5, self.height - 0.5, self.height, 
                            dtype=torch.float32).view(-1, self.height, 1).expand(self.depth, self.height, self.width)
        zs = torch.linspace(0.5, self.depth - 0.5, self.depth, 
                            dtype=torch.float32).view(-1, 1, 1).expand(self.depth, self.height, self.width)
        self.voxel_grid = torch.stack((xs, ys, zs), -1)
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

        V_G = grid_to_lidar[None].repeat(B,1,1) # Voxel Grid -> LiDAR (4, 4)
        C_V = lidar_to_cam  # LiDAR -> Camera (B, 4, 4)
        I_C = cam_to_img  # Camera -> Image (B, 3, 4)
        trans = C_V @ bda_4x4 @ V_G
        V_G_ = V_G.reshape(B, 1, 1, 4, 4)
        trans = trans.reshape(B, 1, 1, 4, 4)
        voxel_grid = voxel_grid.repeat_interleave(repeats=B, dim=0)

        camera_grid = transform_points(trans_01=trans, points_1=voxel_grid)
        I_C = I_C.reshape(B, 1, 1, 3, 4)
        image_grid, image_depths = project_to_image(project=I_C, points=camera_grid)
        image_depths = bin_depths(depth_map=image_depths, mode=self.depth_mode, depth_min=self.depth_min, depth_max=self.depth_max, num_bins=self.num_bins)
        image_depths = image_depths.unsqueeze(-1)
        frustum_grid = torch.cat((image_grid, image_depths), dim=-1)
        return frustum_grid

    def forward(self, bda_4x4, lidar_to_cam, cam_to_img, image_shape):
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
        image_depth = torch.tensor([self.num_bins],device=image_shape.device,dtype=image_shape.dtype)
        frustum_shape = torch.cat((image_depth, image_shape))
        frustum_grid = normalize_coords(coords=frustum_grid, shape=frustum_shape)

        mask = ~torch.isfinite(frustum_grid)
        frustum_grid[mask] = self.out_of_bounds_val

        return frustum_grid


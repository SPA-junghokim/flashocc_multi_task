# Copyright (c) OpenMMLab. All rights reserved.
import os
import mmcv
import torch
import cv2
import numpy as np
from tqdm import tqdm

from mmdet3d.datasets import DATASETS
from .nuscenes_dataset_bevdet import NuScenesDatasetBEVDet as NuScenesDataset
from ..core.evaluation.occ_metrics import Metric_mIoU, Metric_FScore

colors_map = np.array(
    [
        [0,   0,   0, 255],  # 0 undefined
        [255, 158, 0, 255],  # 1 car  orange
        [0, 0, 230, 255],    # 2 pedestrian  Blue
        [47, 79, 79, 255],   # 3 sign  Darkslategrey
        [220, 20, 60, 255],  # 4 CYCLIST  Crimson
        [255, 69, 0, 255],   # 5 traiffic_light  Orangered
        [255, 140, 0, 255],  # 6 pole  Darkorange
        [233, 150, 70, 255], # 7 construction_cone  Darksalmon
        [255, 61, 99, 255],  # 8 bycycle  Red
        [112, 128, 144, 255],# 9 motorcycle  Slategrey
        [222, 184, 135, 255],# 10 building Burlywood
        [0, 175, 0, 255],    # 11 vegetation  Green
        [165, 42, 42, 255],  # 12 trunk  nuTonomy green
        [0, 207, 191, 255],  # 13 curb, road, lane_marker, other_ground
        [75, 0, 75, 255], # 14 walkable, sidewalk
        [255, 0, 0, 255], # 15 unobsrvd
        [0, 0, 0, 0],  # 16 undefined
        [0, 0, 0, 0],  # 16 undefined
    ])


@DATASETS.register_module()
class NuScenesDatasetOccpancy(NuScenesDataset):
    def __init__(self, segmentation=False, **kwargs):
        super(NuScenesDatasetOccpancy, self).__init__(**kwargs)
        self.segmentation = segmentation
    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        input_dict = super(NuScenesDatasetOccpancy, self).get_data_info(index)
        # standard protocol modified from SECOND.Pytorch
        # input_dict['occ_gt_path'] = os.path.join(self.data_root, self.data_infos[index]['occ_path'])
        input_dict['occ_gt_path'] = self.data_infos[index]['occ_path']
        
        if self.segmentation:
            info = self.data_infos[index]
            input_dict.update(dict(
                    location=info['log']['location'],
                ))
            input_dict['global_to_curr_lidar_rt'] = torch.FloatTensor(nuscenes_get_rt_matrix(
                    self.data_infos[index], self.data_infos[index],
                    "global", "lidar"))
        return input_dict

    def evaluate(self, occ_results, runner=None, show_dir=None, **eval_kwargs):
        self.occ_eval_metrics = Metric_mIoU(
            num_classes=18,
            use_lidar_mask=False,
            use_image_mask=True)

        print('\nStarting Evaluation...')
        for index, occ_pred in enumerate(tqdm(occ_results)):
            # occ_pred: (Dx, Dy, Dz)
            info = self.data_infos[index]
            # occ_gt = np.load(os.path.join(self.data_root, info['occ_path'], 'labels.npz'))
            occ_gt = np.load(os.path.join(info['occ_path'], 'labels.npz'))
            gt_semantics = occ_gt['semantics']      # (Dx, Dy, Dz)
            mask_lidar = occ_gt['mask_lidar'].astype(bool)      # (Dx, Dy, Dz)
            mask_camera = occ_gt['mask_camera'].astype(bool)    # (Dx, Dy, Dz)
            # occ_pred = occ_pred
            self.occ_eval_metrics.add_batch(
                occ_pred,   # (Dx, Dy, Dz)
                gt_semantics,   # (Dx, Dy, Dz)
                mask_lidar,     # (Dx, Dy, Dz)
                mask_camera     # (Dx, Dy, Dz)
            )

            # if index % 100 == 0 and show_dir is not None:
            #     gt_vis = self.vis_occ(gt_semantics)
            #     pred_vis = self.vis_occ(occ_pred)
            #     mmcv.imwrite(np.concatenate([gt_vis, pred_vis], axis=1),
            #                  os.path.join(show_dir + "%d.jpg"%index))

            if show_dir is not None:
                mmcv.mkdir_or_exist(show_dir)
                # scene_name = info['scene_name']
                scene_name = [tem for tem in info['occ_path'].split('/') if 'scene-' in tem][0]
                sample_token = info['token']
                mmcv.mkdir_or_exist(os.path.join(show_dir, scene_name, sample_token))
                save_path = os.path.join(show_dir, scene_name, sample_token, 'pred.npz')
                np.savez_compressed(save_path, pred=occ_pred, gt=occ_gt, sample_token=sample_token)

        return self.occ_eval_metrics.count_miou()

    def vis_occ(self, semantics):
        # simple visualization of result in BEV
        semantics_valid = np.logical_not(semantics == 17)
        d = np.arange(16).reshape(1, 1, 16)
        d = np.repeat(d, 200, axis=0)
        d = np.repeat(d, 200, axis=1).astype(np.float32)
        d = d * semantics_valid
        selected = np.argmax(d, axis=2)

        selected_torch = torch.from_numpy(selected)
        semantics_torch = torch.from_numpy(semantics)

        occ_bev_torch = torch.gather(semantics_torch, dim=2,
                                     index=selected_torch.unsqueeze(-1))
        occ_bev = occ_bev_torch.numpy()

        occ_bev = occ_bev.flatten().astype(np.int32)
        occ_bev_vis = colors_map[occ_bev].astype(np.uint8)
        occ_bev_vis = occ_bev_vis.reshape(200, 200, 4)[::-1, ::-1, :3]
        occ_bev_vis = cv2.resize(occ_bev_vis,(400,400))
        return occ_bev_vis
    
    
import numpy as np
from pyquaternion import Quaternion

def nuscenes_get_rt_matrix(
    src_sample,
    dest_sample,
    src_mod,
    dest_mod):
    
    """
    CAM_FRONT_XYD indicates going from 2d image coords + depth
        Note that image coords need to multiplied with said depths first to bring it into 2d hom coords.
    CAM_FRONT indicates going from camera coordinates xyz
    
    Method is: whatever the input is, transform to global first.
    """
    possible_mods = ['CAM_FRONT_XYD', 
                     'CAM_FRONT_RIGHT_XYD', 
                     'CAM_FRONT_LEFT_XYD', 
                     'CAM_BACK_XYD', 
                     'CAM_BACK_LEFT_XYD', 
                     'CAM_BACK_RIGHT_XYD',
                     'CAM_FRONT', 
                     'CAM_FRONT_RIGHT', 
                     'CAM_FRONT_LEFT', 
                     'CAM_BACK', 
                     'CAM_BACK_LEFT', 
                     'CAM_BACK_RIGHT',
                     'lidar',
                     'ego',
                     'global']

    assert src_mod in possible_mods and dest_mod in possible_mods
    
    src_lidar_to_ego = np.eye(4, 4)
    src_lidar_to_ego[:3, :3] = Quaternion(src_sample['lidar2ego_rotation']).rotation_matrix
    src_lidar_to_ego[:3, 3] = np.array(src_sample['lidar2ego_translation'])
    
    src_ego_to_global = np.eye(4, 4)
    src_ego_to_global[:3, :3] = Quaternion(src_sample['ego2global_rotation']).rotation_matrix
    src_ego_to_global[:3, 3] = np.array(src_sample['ego2global_translation'])
    
    dest_lidar_to_ego = np.eye(4, 4)
    dest_lidar_to_ego[:3, :3] = Quaternion(dest_sample['lidar2ego_rotation']).rotation_matrix
    dest_lidar_to_ego[:3, 3] = np.array(dest_sample['lidar2ego_translation'])
    
    dest_ego_to_global = np.eye(4, 4)
    dest_ego_to_global[:3, :3] = Quaternion(dest_sample['ego2global_rotation']).rotation_matrix
    dest_ego_to_global[:3, 3] = np.array(dest_sample['ego2global_translation'])
    
    src_mod_to_global = None
    dest_global_to_mod = None
    
    if src_mod == "global":
        src_mod_to_global = np.eye(4, 4)
    elif src_mod == "ego":
        src_mod_to_global = src_ego_to_global
    elif src_mod == "lidar":
        src_mod_to_global = src_ego_to_global @ src_lidar_to_ego
    elif "CAM" in src_mod:
        src_sample_cam = src_sample['cams'][src_mod.replace("_XYD", "")]
        
        src_cam_to_lidar = np.eye(4, 4)
        src_cam_to_lidar[:3, :3] = src_sample_cam['sensor2lidar_rotation']
        src_cam_to_lidar[:3, 3] = src_sample_cam['sensor2lidar_translation']
        
        src_cam_intrinsics = np.eye(4, 4)
        src_cam_intrinsics[:3, :3] = src_sample_cam['cam_intrinsic']
        
        if "XYD" not in src_mod:
            src_mod_to_global = (src_ego_to_global @ src_lidar_to_ego @ 
                                 src_cam_to_lidar)
        else:
            src_mod_to_global = (src_ego_to_global @ src_lidar_to_ego @ 
                                 src_cam_to_lidar @ np.linalg.inv(src_cam_intrinsics))
            
            
    
    if dest_mod == "global":
        dest_global_to_mod = np.eye(4, 4)
    elif dest_mod == "ego":
        dest_global_to_mod = np.linalg.inv(dest_ego_to_global)
    elif dest_mod == "lidar":
        dest_global_to_mod = np.linalg.inv(dest_ego_to_global @ dest_lidar_to_ego)
    elif "CAM" in dest_mod:
        dest_sample_cam = dest_sample['cams'][dest_mod.replace("_XYD", "")]
        
        dest_cam_to_lidar = np.eye(4, 4)
        dest_cam_to_lidar[:3, :3] = dest_sample_cam['sensor2lidar_rotation']
        dest_cam_to_lidar[:3, 3] = dest_sample_cam['sensor2lidar_translation']
        
        dest_cam_intrinsics = np.eye(4, 4)
        dest_cam_intrinsics[:3, :3] = dest_sample_cam['cam_intrinsic']
        
        if "XYD" not in dest_mod:
            dest_global_to_mod = np.linalg.inv(dest_ego_to_global @ dest_lidar_to_ego @ 
                                               dest_cam_to_lidar)
        else:
            dest_global_to_mod = np.linalg.inv(dest_ego_to_global @ dest_lidar_to_ego @ 
                                               dest_cam_to_lidar @ np.linalg.inv(dest_cam_intrinsics))
    
    return dest_global_to_mod @ src_mod_to_global
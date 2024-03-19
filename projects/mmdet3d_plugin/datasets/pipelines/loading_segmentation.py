import torch
import os
import numpy as np
from PIL import Image
import cv2
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import Box
from mmdet.datasets.builder import PIPELINES
# from nuscenes.map_expansion.map_api import NuScenesMap
from mmdet.datasets.pipelines import RandomFlip
import math
from .map_api import NuScenesMap

@PIPELINES.register_module()
class GenSegGT(object):
    def __init__(self,
                 root_path,
                 grid_config,
                 patch_size=[102.4, 102.4],
                 canvas_size=(256, 256), # canvas size must be tuple
                 map_classes=['drivable_area', 'ped_crossing', 'walkway', 'stop_line', 'carpark_area', 'divider'],
                 vehicle_labels=[0,1,2,3,4,6,7],
                 show = False,
                 ):
        self.nusc_maps = {
            'boston-seaport': NuScenesMap(dataroot=root_path, map_name='boston-seaport'),
            'singapore-hollandvillage': NuScenesMap(dataroot=root_path, map_name='singapore-hollandvillage'),
            'singapore-onenorth': NuScenesMap(dataroot=root_path, map_name='singapore-onenorth'),
            'singapore-queenstown': NuScenesMap(dataroot=root_path, map_name='singapore-queenstown'),
        }
        self.patch_size = patch_size
        self.canvas_size = canvas_size
        self.map_classes = map_classes

        self.mappings = {}
        for name in self.map_classes:
            if name == "drivable_area*":
                self.mappings[name] = ["road_segment", "lane"]
            elif name == "divider":
                self.mappings[name] = ["road_divider", "lane_divider"]
            else:
                self.mappings[name] = [name]

        layer_names = []
        for name in self.mappings:
            layer_names.extend(self.mappings[name])
        self.layer_names = list(set(layer_names))
        
        self.grid_config = grid_config
        self.vehicle_labels = vehicle_labels

        self.show = show

    def __call__(self, input_dict):
        if 'pcd_rotation' in input_dict:
            aug_r_mat = torch.eye(4) # rot_mat_T
            aug_r_mat[:3, :3] = input_dict['pcd_rotation'].T # rot_mat_T
        else:
            aug_r_mat = torch.eye(4) # rot_mat_T

        g2l_mat = input_dict['global_to_curr_lidar_rt']
        loc = input_dict['location']
        nusc_map = self.nusc_maps[loc]

        g2a_mat = aug_r_mat @ g2l_mat
        a2g_mat = torch.inverse(g2a_mat).numpy()

        a2g_t = a2g_mat[:2, -1]
        # self.canvas_size = (512,512)
        patch_box = (a2g_t[0], a2g_t[1], self.patch_size[0], self.patch_size[1])
        
        a2g_r = a2g_mat[:3, :3]
        v = np.dot(a2g_r, np.array([1, 0, 0]))
        yaw = np.arctan2(v[1], v[0])
        patch_angle = yaw / np.pi * 180

        map_mask = nusc_map.get_map_mask(patch_box, patch_angle, self.layer_names, canvas_size=self.canvas_size)
        
        if 'pcd_horizontal_flip' in input_dict:
            if input_dict['pcd_horizontal_flip']:
                map_mask = np.flip(map_mask, axis=1)
        if 'pcd_vertical_flip' in input_dict:
            if input_dict['pcd_vertical_flip']:
                map_mask = np.flip(map_mask, axis=2)
        # map_mask : 7, 256, 256 
        seg_map_mask = map_mask.transpose((1,2,0)) # --> 256, 256, 7 (x,y)
        seg_map_mask = seg_map_mask.astype(np.bool)
        num_classes = len(self.map_classes)
        labels = np.zeros((*self.canvas_size, num_classes), dtype=np.long)
        for k, name in enumerate(self.map_classes):
            for layer_name in self.mappings[name]:
                index = self.layer_names.index(layer_name)
                labels[seg_map_mask[:, :, index], k] = 1
                
        input_dict['gt_seg_mask'] = labels.astype(np.float32)
        
        if self.show:
        # if True:
            import random
            except_index = 9
            # fig_size = (512, 512, 3)
            fig_size = (256, 256, 3)
            # grid = np.array([[0.2, 0.2, 20]]) # 512,512
            grid = np.array([[0.4, 0.4, 20]]) # 128,128

            canvas = np.zeros(fig_size)
            points_xyz = input_dict['points'][:,:3].tensor.numpy()
            points_xyz = (points_xyz - np.array([[-51.2, -51.2, -10]])) / grid
            points_xyz = points_xyz.astype(np.int)
            valid_mask = (points_xyz[:,0] >= 0) & (points_xyz[:,0] < fig_size[0]) & (points_xyz[:,1] >= 0) & (points_xyz[:,1] < fig_size[1]) & (points_xyz[:,2] >= 0) & (points_xyz[:,2] < 1)
            points_xyz = points_xyz[valid_mask]
            points_xyz = np.unique(points_xyz, axis=0)
            # canvas[points_xyz[:,0], points_xyz[:,1], :] = 255
            canvas[points_xyz[:,1], points_xyz[:,0], :] = 255
            
            # for i in range(num_classes):
            #     if i == except_index:
            #         continue
            #     canvas += labels[:,:,i,None].repeat(3, axis=2) * (255//num_classes) * i
            canvas += labels[:,:,0,None].repeat(3, axis=2) * (255//num_classes) * 0
            canvas += labels[:,:,1,None].repeat(3, axis=2) * (255//num_classes) * 1
            canvas += labels[:,:,2,None].repeat(3, axis=2) * (255//num_classes) * 2
            canvas += labels[:,:,3,None].repeat(3, axis=2) * (255//num_classes) * 3
            canvas += labels[:,:,4,None].repeat(3, axis=2) * (255//num_classes) * 4
            canvas += labels[:,:,5,None].repeat(3, axis=2) * (255//num_classes) * 5
            
            # cv2.imwrite(f'deb_map{random.randint(0,100000)}.png', canvas)
            cv2.imwrite(f'deb_map.png', canvas)
            breakpoint()

        return input_dict

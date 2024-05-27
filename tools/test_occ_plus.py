# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

import mmdet
# from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from mmdet.datasets import replace_ImageToTensor
# from projects.mmdet3d_plugin.core.hook.MeanIOUHook import MeanIoU

import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Callable, List, Optional
import torch.distributed as dist
import time
import numpy as np

import os
from PIL import Image
from pyquaternion import Quaternion
import math
from mayavi import mlab
from tqdm import tqdm
from mmcv.engine import collect_results_cpu
if mmdet.__version__ > '2.23.0':
    # If mmdet version > 2.23.0, setup_multi_processes would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import setup_multi_processes
else:
    from mmdet3d.utils import setup_multi_processes

try:
    # If mmdet version > 2.23.0, compat_cfg would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import pickle as pkl
import argparse
import time
import torch
import sys, platform
from sklearn.neighbors import KDTree
from termcolor import colored
from pathlib import Path
from copy import deepcopy
from functools import reduce

np.seterr(divide='ignore', invalid='ignore')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def pcolor(string, color, on_color=None, attrs=None):
    """
    Produces a colored string for printing

    Parameters
    ----------
    string : str
        String that will be colored
    color : str
        Color to use
    on_color : str
        Background color to use
    attrs : list of str
        Different attributes for the string

    Returns
    -------
    string: str
        Colored string
    """
    return colored(string, color, on_color, attrs)


def getCellCoordinates(points, voxelSize):
    return (points / voxelSize).astype(np.int)


def getNumUniqueCells(cells):
    M = cells.max() + 1
    return np.unique(cells[:, 0] + M * cells[:, 1] + M ** 2 * cells[:, 2]).shape[0]


class Metric_mIoU():
    def __init__(self,
                 save_dir='.',
                 num_classes=18,
                 use_lidar_mask=False,
                 use_image_mask=False,
                 ):
        self.class_names = ['others','barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
                            'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
                            'driveable_surface', 'other_flat', 'sidewalk',
                            'terrain', 'manmade', 'vegetation','free']
        self.save_dir = save_dir
        self.use_lidar_mask = use_lidar_mask
        self.use_image_mask = use_image_mask
        self.num_classes = num_classes

        self.point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
        self.occupancy_size = [0.4, 0.4, 0.4]
        self.voxel_size = 0.4
        self.occ_xdim = int((self.point_cloud_range[3] - self.point_cloud_range[0]) / self.occupancy_size[0])
        self.occ_ydim = int((self.point_cloud_range[4] - self.point_cloud_range[1]) / self.occupancy_size[1])
        self.occ_zdim = int((self.point_cloud_range[5] - self.point_cloud_range[2]) / self.occupancy_size[2])
        self.voxel_num = self.occ_xdim * self.occ_ydim * self.occ_zdim
        self.hist = np.zeros((self.num_classes, self.num_classes))
        self.cnt = 0

    def hist_info(self, n_cl, pred, gt):
        """
        build confusion matrix
        # empty classes:0
        non-empty class: 0-16
        free voxel class: 17

        Args:
            n_cl (int): num_classes_occupancy
            pred (1-d array): pred_occupancy_label, (N_valid, )
            gt (1-d array): gt_occupancu_label, (N_valid, )

        Returns:
            tuple:(hist, correctly number_predicted_labels, num_labelled_sample)
        """
        assert pred.shape == gt.shape
        # pred = pred.cpu().numpy() 
        # gt = gt.cpu().numpy()
        k = (gt >= 0) & (gt < n_cl)  # exclude 255
        labeled = np.sum(k)     # N_total
        correct = np.sum((pred[k] == gt[k]))    # N_correct

        return (
            np.bincount(
                n_cl * gt[k].astype(int) + pred[k].astype(int), minlength=n_cl ** 2
            ).reshape(n_cl, n_cl),    # (N_cls, N_cls),
            correct,    # N_correct
            labeled,    # N_total
        )

    def per_class_iu(self, hist):

        return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

    def compute_mIoU(self, pred, label, n_classes):
        """
        Args:
            pred: (N_valid, )
            label: (N_valid, )
            n_classes: int=18

        Returns:

        """
        hist = np.zeros((n_classes, n_classes))     # (N_cls, N_cls)
        new_hist, correct, labeled = self.hist_info(n_classes, pred.flatten(), label.flatten())
        hist += new_hist    # (N_cls, N_cls)
        mIoUs = self.per_class_iu(hist)
        # for ind_class in range(n_classes):
        #     print(str(round(mIoUs[ind_class] * 100, 2)))
        # print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))
        return round(np.nanmean(mIoUs) * 100, 2), hist

    def add_batch(self, semantics_pred, semantics_gt, mask_lidar, mask_camera):
        """
        Args:
            semantics_pred: (Dx, Dy, Dz, n_cls)
            semantics_gt: (Dx, Dy, Dz)
            mask_lidar: (Dx, Dy, Dz)
            mask_camera: (Dx, Dy, Dz)

        Returns:

        """
        self.cnt += 1
        if self.use_image_mask:
            masked_semantics_gt = semantics_gt[mask_camera]     # (N_valid, )
            masked_semantics_pred = semantics_pred[mask_camera]     # (N_valid, )
        elif self.use_lidar_mask:
            masked_semantics_gt = semantics_gt[mask_lidar]
            masked_semantics_pred = semantics_pred[mask_lidar]
        else:
            masked_semantics_gt = semantics_gt
            masked_semantics_pred = semantics_pred

            # # pred = np.random.randint(low=0, high=17, size=masked_semantics.shape)
        _, _hist = self.compute_mIoU(masked_semantics_pred, masked_semantics_gt, self.num_classes)
        self.hist += _hist    # (N_cls, N_cls)  列对应每个gt类别，行对应每个预测类别, 这样只有对角线位置上的预测是准确的.

    def count_miou(self):
        mIoU = self.per_class_iu(self.hist)
        # assert cnt == num_samples, 'some samples are not included in the miou calculation'
        print(f'===> per class IoU of {self.cnt} samples:')
        for ind_class in range(self.num_classes-1):
            print(f'===> {self.class_names[ind_class]} - IoU = ' + str(round(mIoU[ind_class] * 100, 2)))

        print(f'===> mIoU of {self.cnt} samples: ' + str(round(np.nanmean(mIoU[:self.num_classes-1]) * 100, 2)))
        # print(f'===> sample-wise averaged mIoU of {cnt} samples: ' + str(round(np.nanmean(mIoU_avg), 2)))

        eval_res = dict()
        # eval_res['class_name'] = self.class_names
        eval_res['mIoU'] = mIoU
        # eval_res['cnt'] = self.cnt
        return eval_res
class MeanIoU:

    def __init__(self,
                 class_indices,
                 ignore_label: int=0,
                 # label_str,
                 names: list=None,
                 # empty_class: int
                 ):
        self.class_indices = class_indices
        self.num_classes = len(class_indices)
        self.ignore_label = ignore_label
        # self.label_str = label_str
        if names is None:
            self.names = ['noise','barrier','bicycle','bus','car','construction_vehicle','motorcycle','pedestrian','traffic_cone','trailer','truck',
'driveable_surface','other_flat','sidewalk','terrain','manmade','vegetation','empty']
        else:
            self.names = names

    def reset(self) -> None:
        self.total_seen = torch.zeros(self.num_classes).cuda()
        self.total_correct = torch.zeros(self.num_classes).cuda()
        self.total_positive = torch.zeros(self.num_classes).cuda()

    def _after_step(self, outputs, targets):
        outputs = outputs[targets != self.ignore_label]
        targets = targets[targets != self.ignore_label]

        for i, c in enumerate(self.class_indices):
            self.total_seen[i] += torch.sum(targets == c).item()
            self.total_correct[i] += torch.sum((targets == c)
                                               & (outputs == c)).item()
            self.total_positive[i] += torch.sum(outputs == c).item()
        # print("total_seen:{} \n total_correct:{} \n total_positive:{}".format(self.total_seen, self.total_correct, self.total_positive))

    def _after_epoch(self):
        dist.all_reduce(self.total_seen)
        dist.all_reduce(self.total_correct)
        dist.all_reduce(self.total_positive)

        ious = []

        for i in range(self.num_classes):
            if self.total_seen[i] == 0:
                ious.append(1)
            else:
                cur_iou = self.total_correct[i] / (self.total_seen[i]
                                                   + self.total_positive[i]
                                                   - self.total_correct[i])
                ious.append(cur_iou.item())

        miou = np.mean(ious)
        print(f'Validation per class iou {self.names}:')
        for iou, name in zip(ious, self.names):
            print('%s : %.2f%%' % (name, iou * 100))
        print('%s : %.2f%%' % ('mIOU', miou * 100))

        # return miou * 100
def get_grid_coords(dims, resolution):
    """
    :param dims: the dimensions of the grid [x, y, z] (i.e. [256, 256, 32])
    :return coords_grid: is the center coords of voxels in the grid
    """
    g_xx = np.arange(0, dims[0] + 1) #128
    g_yy = np.arange(0, dims[1] + 1) #128
    g_zz = np.arange(0, dims[2] + 1) #10

    resolution = np.array([0.8,0.8,0.5])
    # Obtaining the grid with coords...
    xx, yy, zz = np.meshgrid(g_xx[:-1], g_yy[:-1], g_zz[:-1])
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    coords_grid = coords_grid.astype(np.float32)

    coords_grid = (coords_grid * resolution) + resolution / 2

    temp = np.copy(coords_grid)
    temp[:, 0] = coords_grid[:, 1]
    temp[:, 1] = coords_grid[:, 0]
    coords_grid = np.copy(temp)

    return coords_grid

def draw_vox(
    voxels,
    save_path,
    idx,
    voxel_size=0.8,
    state='GT',
):
    # Compute the voxels coordinates
    grid_coords = get_grid_coords([voxels.shape[0], voxels.shape[1], voxels.shape[2]], voxel_size)
    # Attach the predicted class to every voxel
    grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T
    grid_voxels = grid_coords[(grid_coords[:, 3] > 0) & (grid_coords[:, 3] < 255)]

    # figure = mlab.figure(size=(1400, 1400), bgcolor=(1, 1, 1))
    figure = mlab.figure(size=(2000, 2000), bgcolor=(1, 1, 1))
    # voxel_size= np.array(voxel_size)
    plt_plot = mlab.points3d(
        grid_voxels[:, 0],
        grid_voxels[:, 1],
        grid_voxels[:, 2],
        grid_voxels[:, 3],
        # grid_voxels[:, 3] * grid_voxels[:, 2] if isbinary else grid_voxels[:, 3],
        colormap="viridis",
        scale_factor=voxel_size - 0.05 * voxel_size,
        mode="cube",
        opacity=1.0,
        vmin=1,
        vmax=19,
    )
    
    colormap16 = {
        'undefined': (0, 0, 0),  # Black.
        "barrier": (112, 128, 144),  # Slategrey
        "bicycle": (220, 20, 60),  # Crimson
        "bus": (255, 127, 80),  # Coral
        "car": (255, 158, 0),  # Orange
        "const.veh.": (233, 150, 70),  # Darksalmon
        "motorcycle": (255, 61, 99),  # Red
        "pedestrian.adult": (0, 0, 230),  # Blue
        "trafficcone": (47, 79, 79),  # Darkslategrey
        "trailer": (255, 140, 0),  # Darkorange
        "truck": (255, 99, 71),  # Tomato
        "drive.surf.": (0, 207, 191),  # nuTonomy green
        "other flat": (175, 0, 75),
        "sidewalk": (75, 0, 75),
        "terrain": (112, 180, 60),
        "manmade": (222, 184, 135),  # Burlywood
        "vegetation": (0, 175, 0),  # Green
        # "empty":(255,255,255)
    }
    
    colors = np.array(list(colormap16.values())).astype(np.uint8)
    alpha = np.ones((colors.shape[0], 1), dtype=np.uint8) * 255
    # alpha[-1] = 0
    colors = np.hstack([colors, alpha])
    plt_plot.glyph.scale_mode = "scale_by_vector"
    plt_plot.module_manager.scalar_lut_manager.lut.table = colors
    plt_plot.module_manager.scalar_lut_manager.data_range = [0, 16]
    
    
    
    visualize_keys = ['DRIVING_VIEW', 'BIRD_EYE_VIEW']
    save_folder = save_path

    scene = figure.scene
        
    for i in range(2):
        # bird-eye-view and facing front 
        if i == 0:
            # scene.camera.position = [  0.75131739 + 51.2, -35.08337438 + 51.2,  16.71378558 + 10] 
            # scene.camera.focal_point = [  0.75131739 + 51.2, -34.21734897 + 51.2,  16.21378558 + 9.5]  #0, -0.87, 0.5
            # scene.camera.view_angle = 60.0
            # scene.camera.view_up = [0.0, 0.0, 1.0]
            # scene.camera.clipping_range = [0.01, 300.]
            # scene.camera.compute_view_plane_normal()
            # scene.render()

            scene.camera.position = [93.865227065337, -154.73045984466188, 189.29087284239284]
            scene.camera.focal_point = [85.01427243978316, 72.3696856898386, -4.51856593876564]
            scene.camera.view_angle = 30.0
            scene.camera.view_up = [0.00868395245614043, 0.6493241548338916, 0.7604621824384118]
            scene.camera.clipping_range = [165.0577077629964, 468.5786967476918]
            scene.camera.compute_view_plane_normal()
            scene.render()

        
        # bird-eye-view
        else:
            scene.camera.position = [80.49809298917242, 79.99999786354601, 441.3100633609743]
            scene.camera.focal_point = [80.49809298917242, 79.99999786354601, 4.000000059604645]
            scene.camera.view_angle = 30.0
            scene.camera.view_up = [0.0, 1.0, 0.0]
            scene.camera.clipping_range = [409.3985444802312, 473.49973312589145]
            scene.camera.compute_view_plane_normal()
            scene.render()
        save_folder_to = os.path.join(save_folder, visualize_keys[i])
        os.makedirs(save_folder_to, exist_ok=True)
        save_file = os.path.join(save_folder_to, f'{idx:06d}.png')
        mlab.savefig(save_file,size=(1200,1300))
    
    mlab.close()
        # scene.render()
        # if sample_num < args.sample_num - 1:
        #     mlab.close()
        #     return 0
        # breakpoint()


def multi_gpu_test(model: nn.Module,
                   data_loader: DataLoader,
                   tmpdir: Optional[str] = None,
                   gpu_collect: bool = False,
                   class_names: list = None):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting
    ``gpu_collect=True``, it encodes results to gpu tensors and use gpu
    communication for results collection. On cpu mode it saves the results on
    different gpus to ``tmpdir`` and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    
    #for vis
    mlab.options.offscreen = True
    
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.

    CalMeanIou_vox = MeanIoU(class_indices=range(len(class_names)), names=class_names)
    CalMeanIou_vox.reset()
    
    map_classes = ['drivable_area', 'ped_crossing', 'walkway', 'stop_line', 'carpark_area', 'divider']
               
    MAP_PALETTE = {
        "drivable_area": (166, 206, 227),
        "road_segment": (31, 120, 180),
        "road_block": (178, 223, 138),
        "lane": (51, 160, 44),
        "ped_crossing": (251, 154, 153),
        "walkway": (227, 26, 28),
        "stop_line": (253, 191, 111),
        "carpark_area": (255, 127, 0),
        "road_divider": (202, 178, 214),
        "lane_divider": (106, 61, 154),
        "divider": (106, 61, 154),
    }
    total_intersect = 0.0
    total_union = 0
    total_fp = 0.0
    total_fn = 0.0
    
    occ_eval_metrics = Metric_mIoU(
            num_classes=18,
            use_lidar_mask=False,
            use_image_mask=True)
    for i, data in enumerate(data_loader):
        
        with torch.no_grad():
            result, voxel_out, gt_semantics, mask_lidar, mask_camera, seg_out, gt_seg_mask =\
                model(return_loss=False, return_vox_results=True, **data) #return_vox_results=True,
            # calculate mIOU
                        
            if voxel_out is not None:
                # if len(voxel_out.shape) == len(gt_voxel_bev[0].shape): #[B, X, Y, Z]
                #     voxel_out = torch.round(voxel_out).int()
                # else:
                #     voxel_out = torch.argmax(voxel_out, dim=1) #[B, C, X, Y, Z]
                for count in range(len(data["img_metas"])):
                    occ_eval_metrics.add_batch(
                        voxel_out[count],   # (Dx, Dy, Dz)
                        gt_semantics[0][count].cpu().numpy().astype(np.uint8),   # (Dx, Dy, Dz)
                        mask_lidar[0][count].cpu().numpy().astype(np.bool),     # (Dx, Dy, Dz)
                        mask_camera[0][count].cpu().numpy().astype(np.bool)     # (Dx, Dy, Dz)
                    )
                    # CalMeanIou_vox._after_step(
                    #     voxel_out[count].flatten(),
                    #     gt_voxel_bev[count].flatten())
            # use out_dir
            if seg_out is not None:
                thresholds = torch.tensor([0.20, 0.25, 0.30, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.5, 0.55, 0.6, 0.65])

                num_classes = seg_out.shape[1]
                num_thr = len(thresholds)
                tp = torch.zeros(num_classes, num_thr).cuda()
                fp = torch.zeros(num_classes, num_thr).cuda()
                fn = torch.zeros(num_classes, num_thr).cuda()
                
                f_lane = seg_out.sigmoid()
                
                f_lane = f_lane.squeeze(0).view(num_classes, -1)
                gt_seg_mask_for_metric = gt_seg_mask[0].flip(2).permute(0,3,1,2)
                
                # gt_seg_mask_for_metric = gt_seg_mask[0].permute(0,3,1,2)
                gt_seg_mask_for_metric = gt_seg_mask_for_metric.squeeze(0).view(6,-1) # (6, -1)보다 (num_classes, -1)이 맞는듯
                
                preds = f_lane[:,:,None] >= thresholds.cuda()
                gts = gt_seg_mask_for_metric[:,:,None].repeat(1, 1, num_thr).bool()

                tp += (preds & gts).sum(dim=1)
                fp += (preds & ~gts).sum(dim=1)
                fn += (~preds & gts).sum(dim=1)
                total_intersect += tp
                total_union += tp+fp+fn+1e-7
                total_fp += fp
                total_fn += fn
                
                show = False
                pred_only = False
                seg_save = False
                final_thr_list = [
                                  torch.tensor([0.52, 0.41, 0.44, 0.41, 0.39, 0.41]).unsqueeze(1).cuda(),
                                ]
                seg_save_dir = 'seg_save_temp'
                os.makedirs(seg_save_dir, exist_ok=True)
                if show:
                    cur_pred = []
                    cur_save_dir = os.path.join(seg_save_dir, 'total')
                    os.makedirs(cur_save_dir, exist_ok=True)
                    b, c, h, w = seg_out.shape
                    gt=torch.zeros(h, w,3)
                    gt+=255
                    pres = gt_seg_mask_for_metric.view(6, h, w)
                    
                    for map_idx, map_name in enumerate(map_classes):
                        gt[...,0][pres[map_idx]==1]=MAP_PALETTE[map_name][0]
                        gt[...,1][pres[map_idx]==1]=MAP_PALETTE[map_name][1]
                        gt[...,2][pres[map_idx]==1]=MAP_PALETTE[map_name][2]
                    gt=gt.cpu().numpy()
                    gt = gt.transpose(1,0,2)
                    gt = gt[::-1]
                    gt = gt[100-90:100+90, 100-90:100+90, :]
                    
                    for thre_idx, final_thr in enumerate(final_thr_list):
                        cur_save_dir_ = os.path.join(seg_save_dir, str(thre_idx))
                        os.makedirs(cur_save_dir_, exist_ok=True)
                        pre=torch.zeros(h, w,3)
                        pre+=255
                        f_lane_show = f_lane >= final_thr
                        f_lane_show = f_lane_show.view(6, h, w)
                        for map_idx, map_name in enumerate(map_classes):
                            pre[...,0][f_lane_show[map_idx]==1]=MAP_PALETTE[map_name][0]
                            pre[...,1][f_lane_show[map_idx]==1]=MAP_PALETTE[map_name][1]
                            pre[...,2][f_lane_show[map_idx]==1]=MAP_PALETTE[map_name][2]
                        pre=pre.cpu().numpy()      
                        
                        pre = pre.transpose(1,0,2)
                        pre = pre[::-1]
                        pre = pre[100-90:100+90, 100-90:100+90,  :]
                        cur_pred.append(pre)
                        # if True:
                        #     imgss=np.concatenate(cur_pred,axis=1)
                        #     imgss = cv2.cvtColor(pre, cv2.COLOR_BGR2RGB)      
                        #     cv2.imwrite(os.path.join(cur_save_dir_, f'{thre_idx}_{i:04d}_{data["img_metas"][0].data[0][0]["sample_idx"]}.png'), imgss)
                    if seg_save:
                        # cur_save_dir_ = os.path.join(seg_save_dir, 'gt')
                        # os.makedirs(cur_save_dir_, exist_ok=True)
                        # imgss=np.concatenate(cur_pred,axis=1)
                        # imgss = cv2.cvtColor(pre, cv2.COLOR_BGR2RGB)      
                        # cv2.imwrite(os.path.join(cur_save_dir_, f'gt_{i:04d}_{data["img_metas"][0].data[0][0]["sample_idx"]}.png'), imgss)
                        
                        # cur_pred.append(gt)
                        # imgss=np.concatenate(cur_pred,axis=1)
                        # imgss = cv2.cvtColor(imgss, cv2.COLOR_BGR2RGB)      
                        # cv2.imwrite(os.path.join(cur_save_dir, f'{thre_idx}_{i:04d}_{data["img_metas"][0].data[0][0]["sample_idx"]}.png'), imgss)
                        cur_save_dir_ = os.path.join(seg_save_dir, 'gt')
                        os.makedirs(cur_save_dir_, exist_ok=True)
                        imgss=np.concatenate(gt,axis=1)
                        imgss = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)      
                        cv2.imwrite(os.path.join(cur_save_dir_, f'gt_{i:04d}_{data["img_metas"][0].data[0][0]["sample_idx"]}.png'), imgss)
                        
                        cur_pred.append(gt)
                        imgss=np.concatenate(pre,axis=1)
                        imgss = cv2.cvtColor(pre, cv2.COLOR_BGR2RGB)      
                        cv2.imwrite(os.path.join(cur_save_dir, f'{thre_idx}_{i:04d}_{data["img_metas"][0].data[0][0]["sample_idx"]}.png'), imgss)
                        
        if result is not None:
            results.extend(result)
        batch_size = 1

        #for vis

        # voxel_pred = voxel_pred.softmax(dim=1).permute(0,2,3,4,1)
        # voxel_pred = torch.gather(voxel_pred, -1, voxel_out.unsqueeze(-1).repeat(1,1,1,1, voxel_pred.shape[-1]))[...,0]
        # voxel_out[voxel_pred < 0.5] = 0
        # voxel_out[gt_voxel_bev[0]==0] = 0

        # voxel_size = 0.8
        # save_root = 'occ_vis'
        # file_name = 'depth_occ_lovasz_non_visible_ignore_3dpooling_depthrenderloss3var85_only'
        # gt_root = 'GT'
        # pred_root = 'pred'
        # os.makedirs(save_root, exist_ok=True)
        # os.makedirs(f'{save_root}/{file_name}/{gt_root}', exist_ok=True)
        # os.makedirs(f'{save_root}/{file_name}/{pred_root}', exist_ok=True)
        # gt = gt_semantics[0][0]
        # pred = voxel_out[0]
        
        # gt[gt==17]=0
        # pred[pred==17]=0
        
        # # pred = pred[::-1,:,:]
        # # draw_vox(voxels=np.rot90(gt.cpu().numpy(),1,axes=(0,1)), save_path=f'{save_root}/{file_name}/{gt_root}', idx=i)
        # draw_vox(voxels=pred.transpose(1,0,2)[::-1,:,:], save_path=f'{save_root}/{file_name}/{pred_root}', idx=i)

        #GT시계반대방향90도 회전
        #Pred
        
        if rank == 0:
            batch_size_all = batch_size * world_size
            if batch_size_all + prog_bar.completed > len(dataset):
                batch_size_all = len(dataset) - prog_bar.completed
            for _ in range(batch_size_all):
                prog_bar.update()
                
    
    if voxel_out is not None:
        print(occ_eval_metrics.count_miou())
    
    if seg_out is not None:
        iou = total_intersect / total_union * 100
        print("\n\n\n Map Segmentation Result:")
        print("classes: ['drivable_area', 'ped_crossing', 'walkway', 'stop_line', 'carpark_area', 'divider']\n")
        for t in range(num_thr):
            print(f'\nIOU@{thresholds[t]:.2f}   : ',end='')
            for m_ in range(len(map_classes)):
                print(f'{iou[m_][t]:.2f}',end='\t')
        max_idx = iou.argmax(1)

        print('\n\n\n')
        max_iou_list = []
        for t in range(len(map_classes)):
            interval='\t\t' if len(map_classes[t]) < 8 else "\t"
            print(f'IOU@Max {map_classes[t]}{interval} - {thresholds[max_idx[t]]:.2f}:{iou[t][max_idx[t]]:.2f}%')
            max_iou_list.append(iou[t][max_idx[t]])
        print(f'mIoU : \t\t\t {sum(max_iou_list)/len(max_iou_list):.2f}%')
        # print(f'\nIOU@Max  dri_thr - {thresholds[max_idx[0]]:.2f}:{iou[0][max_idx[0]]:.2f}%')
        # print(f'IOU@Max  lan_thr - {thresholds[max_idx[1]]:.2f}:{iou[1][max_idx[1]]:.2f}%')
        # print(f'IOU@Max  veh_thr - {thresholds[max_idx[2]]:.2f}:{iou[2][max_idx[2]]:.2f}%')
        vis = ((2*total_fp+total_fn) /total_intersect) * 100
        vis = thresholds[vis.argmin(1)]

        print(f'\nVis@thr  ', end="")
        for t in range(len(map_classes)):
            print(f'{map_classes[t]} - {vis[t]:.2f}', end=", ")
        print("\n\n\n")

    # collect results from all ranks
    if result is None:
        exit()
    # if gpu_collect:
    #     result_from_ranks = collect_results_gpu(results, len(dataset))
    # else:
    result_from_ranks = collect_results_cpu(results, len(dataset), tmpdir)
        
    return result_from_ranks



def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--no-aavt',
        action='store_true',
        help='Do not align after view transformer.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg = compat_cfg(cfg)

    # set multi-process settings
    setup_multi_processes(cfg)
    
    # import modules from string list.
    # if cfg.get('custom_imports', None):
    #     from mmcv.utils import import_modules_from_strings
    #     import_modules_from_strings(**cfg['custom_imports'])

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed testing. Use the first GPU '
                      'in `gpu_ids` now.')
    else:
        cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=distributed, shuffle=False)

    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)
    breakpoint()
    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    if not args.no_aavt:
        if '4D' in cfg.model.type:
            cfg.model.align_after_view_transfromation=True
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE

    class_names = ['others','barrier','bicycle','bus','car','construction_vehicle','motorcycle','pedestrian','traffic_cone','trailer','truck','driveable_surface','other_flat','sidewalk','terrain','manmade','vegetation']
    
    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir, class_names=class_names)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect, class_names)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(dataset.evaluate(outputs, jsonfile_prefix=args.show_dir, **eval_kwargs))


if __name__ == '__main__':
    main()
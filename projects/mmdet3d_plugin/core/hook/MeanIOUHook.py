import torch
import torch.distributed as dist
import numpy as np
from mmcv.runner.hooks.evaluation import DistEvalHook
import os.path as osp
import warnings
from math import inf
from typing import Callable, List, Optional

from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils.data import DataLoader
import torch.nn as nn
import time
import mmcv
import pickle
import tempfile
from mmcv.runner import get_dist_info
from mmcv.engine import collect_results_cpu, collect_results_gpu
import mmcv
import torch
from mmcv.image import tensor2imgs
from os import path as osp

from mmdet3d.models import Base3DDetector
import copy
import cv2
import os

from ..evaluation.occ_metrics import Metric_mIoU, Metric_FScore
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

def IOU (intputs, targets, eps=1e-6):
    intputs = intputs.bool()
    targets = targets.bool()
    inter = (intputs & targets).sum(-1)
    union = (intputs | targets).sum(-1)
    # iou = (numerator + eps) / (denominator + eps - numerator)
    return inter.cpu(),union.cpu()

def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3,
                    class_names: list = None):
    """Test model with single gpu.

    This method tests model with single gpu and gives the 'show' option.
    By setting ``show=True``, it saves the visualization results under
    ``out_dir``.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        show (bool): Whether to save viualization results.
            Default: True.
        out_dir (str): The path to save visualization results.
            Default: None.

    Returns:
        list[dict]: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    CalMeanIou_vox = MeanIoU(class_indices=range(len(class_names)), names=class_names)
    CalMeanIou_vox.reset()
    
    total_intersect = 0.0
    total_union = 0
    
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            # result = model(return_loss=False, rescale=True, **data)
            result, voxel_out, voxel_semantics, mask_lidar, mask_camera, seg_out, gt_seg_mask =\
                model(return_loss=False, return_vox_results=True, **data) #return_vox_results=True,
            # calculate mIOU
            if voxel_out is not None:
                if len(voxel_out.shape) == len(gt_voxel_bev[0].shape): #[B, X, Y, Z]
                    voxel_out = torch.round(voxel_out).int()
                else:
                    voxel_out = torch.argmax(voxel_out, dim=1) #[B, C, X, Y, Z]
                for count in range(len(data["img_metas"])):
                    CalMeanIou_vox._after_step(
                        voxel_out[count].flatten(),
                        gt_voxel_bev[count].flatten())
            # use out_dir
            if seg_out is not None:
                                
                f_lane=seg_out.sigmoid()
                f_lane[f_lane>=0.43]=1
                f_lane[f_lane<0.43]=0
                f_lane=f_lane.view(3,-1)

                gt_seg_mask = gt_seg_mask[0].permute(0,3,1,2)
                gt_seg_mask=gt_seg_mask.view(3,-1) 
                        
                inter,union=IOU(f_lane,gt_seg_mask)
                ret_iou=inter/union
                ret_iou=[ret_iou[0].item(),ret_iou[1].item(),ret_iou[2].item()]
                total_intersect += inter
                total_union += union # calculate mIOU
            
                
            if i % 10 == 0:
                np.save(osp.join(out_dir, 'pred_{}.npy'.format(data['img_metas'][0].data[0][0]['sample_idx'])),np.array(voxel_out.cpu()))
                np.save(osp.join(out_dir, 'gt_{}.npy'.format(data['img_metas'][0].data[0][0]['sample_idx'])),np.array(gt_voxel_bev[0].cpu()))

        if show:
            # Visualize the results of MMDetection3D model
            # 'show_results' is MMdetection3D visualization API
            models_3d = (Base3DDetector)
            if isinstance(model.module, models_3d):
                model.module.show_results(data, result, out_dir=out_dir)
            # Visualize the results of MMDetection model
            # 'show_result' is MMdetection visualization API
            else:
                batch_size = len(result)
                if batch_size == 1 and isinstance(data['img'][0],
                                                  torch.Tensor):
                    img_tensor = data['img'][0]
                else:
                    img_tensor = data['img'][0].data[0]
                img_metas = data['img_metas'][0].data[0]
                imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
                assert len(imgs) == len(img_metas)

                for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                    h, w, _ = img_meta['img_shape']
                    img_show = img[:h, :w, :]

                    ori_h, ori_w = img_meta['ori_shape'][:-1]
                    img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                    if out_dir:
                        out_file = osp.join(out_dir, img_meta['ori_filename'])
                    else:
                        out_file = None

                    model.module.show_result(
                        img_show,
                        result[i],
                        show=show,
                        out_file=out_file,
                        score_thr=show_score_thr)
        if result is not None:
            results.extend(result)

        if result is not None:
            batch_size = len(result)
        elif voxel_out is not None:
            batch_size = voxel_out.shape[0]
        elif seg_out is not None:
            batch_size = seg_out.shape[0]
            
        for _ in range(batch_size):
            prog_bar.update()
            
    if voxel_out is not None:
        CalMeanIou_vox._after_epoch()
    
    if seg_out is not None:
        print("\n\n\n Map Segmentation Result:")
        print(f"segementation mIoU': {total_intersect / total_union} %")
        print("\n\n\n")
        
    if result is None:
        exit()
    
    return results



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
    if gpu_collect:
        result_from_ranks = collect_results_gpu(results, len(dataset))
    else:
        result_from_ranks = collect_results_cpu(results, len(dataset), tmpdir)
        
    return result_from_ranks


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

class DistODOccEvalHook(DistEvalHook):
    """Distributed evaluation hook.

    This hook will regularly perform evaluation in a given interval when
    performing in distributed environment.

    Args:
        dataloader (DataLoader): A PyTorch dataloader, whose dataset has
            implemented ``evaluate`` function.
        start (int | None, optional): Evaluation starting epoch. It enables
            evaluation before the training starts if ``start`` <= the resuming
            epoch. If None, whether to evaluate is merely decided by
            ``interval``. Default: None.
        interval (int): Evaluation interval. Default: 1.
        by_epoch (bool): Determine perform evaluation by epoch or by iteration.
            If set to True, it will perform by epoch. Otherwise, by iteration.
            default: True.
        save_best (str, optional): If a metric is specified, it would measure
            the best checkpoint during evaluation. The information about best
            checkpoint would be saved in ``runner.meta['hook_msgs']`` to keep
            best score value and best checkpoint path, which will be also
            loaded when resume checkpoint. Options are the evaluation metrics
            on the test dataset. e.g., ``bbox_mAP``, ``segm_mAP`` for bbox
            detection and instance segmentation. ``AR@100`` for proposal
            recall. If ``save_best`` is ``auto``, the first key of the returned
            ``OrderedDict`` result will be used. Default: None.
        rule (str | None, optional): Comparison rule for best score. If set to
            None, it will infer a reasonable rule. Keys such as 'acc', 'top'
            .etc will be inferred by 'greater' rule. Keys contain 'loss' will
            be inferred by 'less' rule. Options are 'greater', 'less', None.
            Default: None.
        test_fn (callable, optional): test a model with samples from a
            dataloader in a multi-gpu manner, and return the test results. If
            ``None``, the default test function ``mmcv.engine.multi_gpu_test``
            will be used. (default: ``None``)
        tmpdir (str | None): Temporary directory to save the results of all
            processes. Default: None.
        gpu_collect (bool): Whether to use gpu or cpu to collect results.
            Default: False.
        broadcast_bn_buffer (bool): Whether to broadcast the
            buffer(running_mean and running_var) of rank 0 to other rank
            before evaluation. Default: True.
        out_dir (str, optional): The root directory to save checkpoints. If not
            specified, `runner.work_dir` will be used by default. If specified,
            the `out_dir` will be the concatenation of `out_dir` and the last
            level directory of `runner.work_dir`.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details. Default: None.
        **eval_kwargs: Evaluation arguments fed into the evaluate function of
            the dataset.
    """

    def __init__(self,
                 dataloader: DataLoader,
                 start: Optional[int] = None,
                 interval: int = 1,
                 by_epoch: bool = True,
                 save_best: Optional[str] = None,
                 rule: Optional[str] = None,
                 test_fn: Optional[Callable] = None,
                 greater_keys: Optional[List[str]] = None,
                 less_keys: Optional[List[str]] = None,
                 broadcast_bn_buffer: bool = True,
                 tmpdir: Optional[str] = None,
                 gpu_collect: bool = False,
                 out_dir: Optional[str] = None,
                 file_client_args: Optional[dict] = None,
                 num_classes: int = 10,
                 # class_indices: Optional[list] = None,
                 **eval_kwargs):

        if test_fn is None:
            test_fn = multi_gpu_test

        super().__init__(
            dataloader,
            start=start,
            interval=interval,
            by_epoch=by_epoch,
            save_best=save_best,
            rule=rule,
            test_fn=test_fn,
            greater_keys=greater_keys,
            less_keys=less_keys,
            out_dir=out_dir,
            file_client_args=file_client_args,
            **eval_kwargs)

        self.broadcast_bn_buffer = broadcast_bn_buffer
        self.tmpdir = tmpdir
        self.gpu_collect = gpu_collect

        self.num_classes = num_classes
        self.class_indices = range(num_classes)

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        results = self.test_fn(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print('\n')
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            key_score = self.evaluate(runner, results) #bbox list
            # the key_score may be `None` so it needs to skip the action to
            # save the best checkpoint
            if self.save_best and key_score:
                self._save_ckpt(runner, key_score)

            #mIOU
            total_seen = torch.zeros(self.num_classes).cuda()
            total_correct = torch.zeros(self.num_classes).cuda()
            total_positive = torch.zeros(self.num_classes).cuda()

            targets = results.flatten()
            outputs = results.flatten()

            for i, c in enumerate(self.class_indices):
                total_seen[i] += torch.sum(targets == c).item()
                total_correct[i] += torch.sum((targets == c)
                                                   & (outputs == c)).item()
                total_positive[i] += torch.sum(outputs == c).item()


            dist.all_reduce(total_seen)
            dist.all_reduce(total_correct)
            dist.all_reduce(total_positive)

            ious = []

            for i in range(self.num_classes):
                if total_seen[i] == 0:
                    ious.append(1)
                else:
                    cur_iou = total_correct[i] / (total_seen[i] + total_positive[i] - total_correct[i])
                    ious.append(cur_iou.item())

            miou = np.mean(ious)
            runner.log_buffer.output['mIOU'] = miou
            runner.log_buffer.ready = True

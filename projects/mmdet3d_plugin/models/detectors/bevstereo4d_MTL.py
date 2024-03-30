from ...ops import TRTBEVPoolv2
from .bevdet import BEVDet
from .bevstereo4d import BEVStereo4D
from mmdet3d.models import DETECTORS
from mmdet3d.models.builder import build_head
import torch.nn.functional as F
import torch.nn as nn
import torch
from mmcv.runner import force_fp32
from mmdet3d.models import builder
from mmdet.models.backbones.resnet import ResNet
@DETECTORS.register_module()
class BEVStereo4D_MTL(BEVStereo4D):
    def __init__(self,
                pts_bbox_head=None,
                occ_head=None,
                seg_head=None,
                upsample=False,
                down_sample_for_3d_pooling=None,
                pc_range = [-40.0, -40.0, -1, 40.0, 40.0, 5.4],
                grid_size = [200, 200, 16],
                det_loss_weight=1,
                occ_loss_weight=1,
                seg_loss_weight=1,
                img_bev_encoder_backbone=None,
                occ_bev_encoder_backbone=None,
                seg_bev_encoder_backbone=None,
                img_bev_encoder_neck=None,
                occ_bev_encoder_neck=None,
                seg_bev_encoder_neck=None,
                detection_backbone=False,
                detection_neck=False,
                **kwargs):
        super(BEVStereo4D_MTL, self).__init__(pts_bbox_head=pts_bbox_head, img_bev_encoder_backbone=img_bev_encoder_backbone,
                                            img_bev_encoder_neck=img_bev_encoder_neck,**kwargs)
    
        self.occ_head = occ_head
        self.seg_head = seg_head

        if pts_bbox_head == None:
            self.pts_bbox_head = None
        if self.occ_head is not None:
            self.occ_head = build_head(occ_head)
        if self.seg_head is not None:
            self.seg_head = build_head(seg_head)
            
        self.upsample = upsample
        self.down_sample_for_3d_pooling = down_sample_for_3d_pooling        
        
        if self.down_sample_for_3d_pooling is not None:
            self.down_sample_for_3d_pooling = \
                        nn.Conv2d(self.down_sample_for_3d_pooling[0],
                        self.down_sample_for_3d_pooling[1],
                        kernel_size=1,
                        padding=0,
                        stride=1)
                        
        self.pc_range = torch.tensor(pc_range)
        self.grid_size = torch.tensor(grid_size)
        self.det_loss_weight = det_loss_weight
        self.occ_loss_weight = occ_loss_weight
        self.seg_loss_weight = seg_loss_weight
        self.detection_backbone = detection_backbone
        self.detection_neck = detection_neck
        
        
        if occ_bev_encoder_backbone is not None:
            self.occ_bev_encoder_backbone = builder.build_backbone(occ_bev_encoder_backbone)
        else:
            self.occ_bev_encoder_backbone = None
        if seg_bev_encoder_backbone is not None:
            self.seg_bev_encoder_backbone = builder.build_backbone(seg_bev_encoder_backbone)
        else:
            self.seg_bev_encoder_backbone = None
            
        if occ_bev_encoder_neck is not None:
            self.occ_bev_encoder_neck = builder.build_backbone(occ_bev_encoder_neck)
        else:
            self.occ_bev_encoder_neck = None
        if seg_bev_encoder_neck is not None:
            self.seg_bev_encoder_neck = builder.build_backbone(seg_bev_encoder_neck)
        else:
            self.seg_bev_encoder_neck = None





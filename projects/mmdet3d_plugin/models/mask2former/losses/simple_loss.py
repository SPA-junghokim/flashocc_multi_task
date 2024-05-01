import torch
import torch.nn as nn
from mmdet.models.builder import LOSSES
import torch.nn.functional as F
from mmdet.models.losses import FocalLoss, weight_reduce_loss

@LOSSES.register_module()
class SimpleLoss(torch.nn.Module):
    def __init__(self, pos_weight, loss_weight):
        super(SimpleLoss, self).__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]))
        self.loss_weight = loss_weight

    def forward(self, ypred, ytgt):
        # import ipdb;ipdb.set_trace()
        loss = self.loss_fn(ypred, ytgt)
        return loss*self.loss_weight
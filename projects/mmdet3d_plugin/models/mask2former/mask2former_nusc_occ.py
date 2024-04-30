# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, Conv3d, caffe2_xavier_init
from mmcv.cnn.bricks.transformer import (build_positional_encoding,
                                         build_transformer_layer_sequence)
from mmcv.runner import ModuleList, force_fp32
from mmdet.core import build_assigner, build_sampler, reduce_mean, multi_apply
from mmdet.models.builder import HEADS, build_loss

from .base.mmdet_utils import (get_nusc_lidarseg_point_coords,
                               get_nusc_lidarseg_point_coords_dn,
                          preprocess_occupancy_gt, point_sample_3d)

from .base.anchor_free_head import AnchorFreeHead
from .base.maskformer_head import MaskFormerHead
from projects.mmdet3d_plugin.utils import per_class_iu, fast_hist_crop
from projects.mmdet3d_plugin.models.dense_heads.lovasz_losses import lovasz_softmax_occ
import pdb

# Mask2former for 3D Occupancy Segmentation on nuScenes dataset
@HEADS.register_module()
class Mask2FormerNuscOccHead(MaskFormerHead):
    """Implements the Mask2Former head.

    See `Masked-attention Mask Transformer for Universal Image
    Segmentation <https://arxiv.org/pdf/2112.01527>`_ for details.

    Args:
        in_channels (list[int]): Number of channels in the input feature map.
        feat_channels (int): Number of channels for features.
        out_channels (int): Number of channels for output.
        num_things_classes (int): Number of things.
        num_stuff_classes (int): Number of stuff.
        num_queries (int): Number of query in Transformer decoder.
        pixel_decoder (:obj:`mmcv.ConfigDict` | dict): Config for pixel
            decoder. Defaults to None.
        enforce_decoder_input_project (bool, optional): Whether to add
            a layer to change the embed_dim of tranformer encoder in
            pixel decoder to the embed_dim of transformer decoder.
            Defaults to False.
        transformer_decoder (:obj:`mmcv.ConfigDict` | dict): Config for
            transformer decoder. Defaults to None.
        positional_encoding (:obj:`mmcv.ConfigDict` | dict): Config for
            transformer decoder position encoding. Defaults to None.
        loss_cls (:obj:`mmcv.ConfigDict` | dict): Config of the classification
            loss. Defaults to None.
        loss_mask (:obj:`mmcv.ConfigDict` | dict): Config of the mask loss.
            Defaults to None.
        loss_dice (:obj:`mmcv.ConfigDict` | dict): Config of the dice loss.
            Defaults to None.
        train_cfg (:obj:`mmcv.ConfigDict` | dict): Training config of
            Mask2Former head.
        test_cfg (:obj:`mmcv.ConfigDict` | dict): Testing config of
            Mask2Former head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 feat_channels,
                 out_channels,
                 num_occupancy_classes=20,
                 num_queries=100,
                 num_transformer_feat_level=3,
                 enforce_decoder_input_project=False,
                 transformer_decoder=None,
                 positional_encoding=None,
                 pooling_attn_mask=True,
                 point_cloud_range=None,
                 padding_mode='border',
                 loss_cls=None,
                 loss_mask=None,
                 loss_dice=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 loss_flatten = False,
                 loss_lovasz=True,
                 lovasz_loss_weight = 1,
                 lovasz_flatten=True,
                 consider_visible_mask = True,
                 learned_pos_embed = False,
                 scalar=5,
                 dn_label_noise_ratio=0.2,
                 dn_enable=False,
                 dn_query_init_type='class',
                 pooling_attn_mask_dn = True,
                 noise_type='gt',
                 dn_mask_noise_scale = 0.2,
                 mask_size=[200,200],
                 point_sample_for_dn= False,
                 only_non_empty_voxel_dot=False,
                 **kwargs):
        super(AnchorFreeHead, self).__init__(init_cfg)
        
        self.num_occupancy_classes = num_occupancy_classes
        self.num_classes = self.num_occupancy_classes
        self.num_queries = num_queries
        self.point_cloud_range = point_cloud_range
        
        ''' Transformer Decoder Related '''
        # number of multi-scale features for masked attention
        self.num_transformer_feat_level = num_transformer_feat_level
        self.num_heads = transformer_decoder.transformerlayers.attn_cfgs.num_heads
        self.num_transformer_decoder_layers = transformer_decoder.num_layers
        
        self.transformer_decoder = build_transformer_layer_sequence(
            transformer_decoder)
        self.decoder_embed_dims = self.transformer_decoder.embed_dims
        
        self.decoder_input_projs = ModuleList()
        # from low resolution to high resolution, align the channel of input features
        for _ in range(num_transformer_feat_level):
            if (self.decoder_embed_dims != feat_channels
                    or enforce_decoder_input_project):
                self.decoder_input_projs.append(
                    Conv3d(
                        feat_channels, self.decoder_embed_dims, kernel_size=1))
            else:
                self.decoder_input_projs.append(nn.Identity())
                
        self.decoder_positional_encoding = build_positional_encoding(positional_encoding)
        if dn_enable == False and self.num_transformer_feat_level != 0:
            self.query_embed = nn.Embedding(self.num_queries, feat_channels)
            
        self.query_feat = nn.Embedding(self.num_queries, feat_channels)
        # from low resolution to high resolution
        if self.num_transformer_feat_level != 0:
            self.level_embed = nn.Embedding(self.num_transformer_feat_level, feat_channels)

        ''' Pixel Decoder Related, skipped '''
        self.cls_embed = nn.Linear(feat_channels, self.num_classes + 1)
        self.mask_embed = nn.Sequential(
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, out_channels))

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        if train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            self.sampler = build_sampler(self.train_cfg.sampler, context=self)
            self.num_points = self.train_cfg.get('num_points', 12544)
            self.oversample_ratio = self.train_cfg.get('oversample_ratio', 3.0)
            self.importance_sample_ratio = self.train_cfg.get(
                'importance_sample_ratio', 0.75)
        
        self.loss_cls = build_loss(loss_cls)
        self.loss_mask = build_loss(loss_mask)
        
        self.pooling_attn_mask = pooling_attn_mask
        self.class_weight = loss_cls.class_weight
        
        # align_corners
        self.align_corners = True
        self.padding_mode = padding_mode
        self.loss_flatten = loss_flatten
        self.voxel_coord = None
        self.loss_lovasz = loss_lovasz
        self.loss_dice = build_loss(loss_dice)
        
        if self.loss_lovasz:
            self.loss_lovasz = lovasz_softmax_occ
            self.lovasz_loss_weight = lovasz_loss_weight
            self.lovasz_flatten = lovasz_flatten
            
        self.consider_visible_mask = consider_visible_mask
        
        self.learned_pos_embed = learned_pos_embed
        if self.num_transformer_feat_level != 0:
            if self.learned_pos_embed:
                self.learned_pos_embed  = nn.Sequential(
                    nn.Conv3d(out_channels, out_channels,1),
                    nn.ReLU(),
                    nn.Conv3d(out_channels, out_channels,1),
                )
            
        
        # dn
        self.scalar=scalar
        self.dn_label_noise_ratio=dn_label_noise_ratio
        self.dn_enable = dn_enable
        self.out_channels = out_channels
        self.feat_channels = feat_channels
        self.dn_query_init_type = dn_query_init_type
        if self.dn_query_init_type == 'class' and self.dn_enable:
            self.label_enc = nn.Embedding(self.num_classes, self.feat_channels)
        self.pooling_attn_mask_dn = pooling_attn_mask_dn
        self.noise_type = noise_type
        self.dn_mask_noise_scale = dn_mask_noise_scale
        self.mask_size = mask_size
        self.point_sample_for_dn = point_sample_for_dn
        self.index_for_debug = 0
        self.index_for_debug_list = []
        self.only_non_empty_voxel_dot = only_non_empty_voxel_dot
        
    def init_weights(self):
        for m in self.decoder_input_projs:
            if isinstance(m, Conv2d):
                caffe2_xavier_init(m, bias=0)
        
        if hasattr(self, "pixel_decoder"):
            self.pixel_decoder.init_weights()

        if self.num_transformer_decoder_layers != 0:
            for p in self.transformer_decoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_normal_(p)
    
    def get_targets(self, cls_scores_list, mask_preds_list, gt_labels_list,
                    gt_masks_list, gt_binary_list, img_metas):
        """Compute classification and mask targets for all images for a decoder
        layer.

        Args:
            cls_scores_list (list[Tensor]): Mask score logits from a single
                decoder layer for all images. Each with shape (num_queries,
                cls_out_channels).
            mask_preds_list (list[Tensor]): Mask logits from a single decoder
                layer for all images. Each with shape (num_queries, h, w).
            gt_labels_list (list[Tensor]): Ground truth class indices for all
                images. Each with shape (n, ), n is the sum of number of stuff
                type and number of instance in a image.
            gt_masks_list (list[Tensor]): Ground truth mask for each image,
                each with shape (n, h, w).
            img_metas (list[dict]): List of image meta information.

        Returns:
            tuple[list[Tensor]]: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels of all images.\
                    Each with shape (num_queries, ).
                - label_weights_list (list[Tensor]): Label weights\
                    of all images. Each with shape (num_queries, ).
                - mask_targets_list (list[Tensor]): Mask targets of\
                    all images. Each with shape (num_queries, h, w).
                - mask_weights_list (list[Tensor]): Mask weights of\
                    all images. Each with shape (num_queries, ).
                - num_total_pos (int): Number of positive samples in\
                    all images.
                - num_total_neg (int): Number of negative samples in\
                    all images.
        """
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         pos_inds_list,
         neg_inds_list) = multi_apply(self._get_target_single, cls_scores_list, mask_preds_list, 
                    gt_labels_list, gt_masks_list, gt_binary_list, img_metas)

        return (labels_list, label_weights_list, mask_targets_list,
                mask_weights_list)
    def _get_target_single(self, cls_score, mask_pred, gt_labels, gt_masks, gt_binary,
                    img_metas):
        """Compute classification and mask targets for one image.

        Args:
            cls_score (Tensor): Mask score logits from a single decoder layer
                for one image. Shape (num_queries, cls_out_channels).
            mask_pred (Tensor): Mask logits for a single decoder layer for one
                image. Shape (num_queries, x, y, z).
            gt_labels (Tensor): Ground truth class indices for one image with
                shape (num_gts, ).
            gt_masks (Tensor): Ground truth mask for each image, each with
                shape (num_gts, x, y, z).
            img_metas (dict): Image informtation.

        Returns:
            tuple[Tensor]: A tuple containing the following for one image.

                - labels (Tensor): Labels of each image. \
                    shape (num_queries, ).
                - label_weights (Tensor): Label weights of each image. \
                    shape (num_queries, ).
                - mask_targets (Tensor): Mask targets of each image. \
                    shape (num_queries, h, w).
                - mask_weights (Tensor): Mask weights of each image. \
                    shape (num_queries, ).
                - pos_inds (Tensor): Sampled positive indices for each \
                    image.
                - neg_inds (Tensor): Sampled negative indices for each \
                    image.
        """
        # sample points
        num_queries = cls_score.shape[0]
        num_gts = gt_labels.shape[0]
        gt_labels = gt_labels.long()
        
        num_rand = self.num_points 
        
        if self.voxel_coord is None:
            W,H,Z = gt_binary.shape
            x_range = torch.linspace(0, 1, W)
            y_range = torch.linspace(0, 1, H)
            z_range = torch.linspace(0, 1, Z)
            x, y, z = torch.meshgrid(x_range, y_range, z_range)
            self.voxel_coord = torch.stack((x, y, z), dim=-1).reshape(-1, 3).to(cls_score.device)
        point_coords = self.voxel_coord[gt_binary.flatten()]
        
        num_lidarseg = min(self.num_points // 2, point_coords.shape[0])
        if num_lidarseg < point_coords.shape[0]:
            point_coords = point_coords[torch.randperm(point_coords.shape[0])[:num_lidarseg]]
        
        num_rand = self.num_points - num_lidarseg
        rand_point_coords = torch.rand((num_rand, 3), device=cls_score.device)
        point_coords = torch.cat((point_coords, rand_point_coords), dim=0)
        point_coords = point_coords[..., [2, 1, 0]]
        
        # since there are out-of-range lidar points, the padding_mode is set to border
        mask_points_pred = point_sample_3d(mask_pred.unsqueeze(1),  
            point_coords.repeat(num_queries, 1, 1), padding_mode=self.padding_mode).squeeze(1) # torch.Size([100, 50176])
        
        gt_points_masks = point_sample_3d(gt_masks.unsqueeze(1).float(), 
            point_coords.repeat(num_gts, 1, 1), padding_mode=self.padding_mode).squeeze(1) # torch.Size([100, 50176])
        # mask_points_pred = mask_pred.flatten(1)
        # gt_points_masks = gt_masks.flatten(1)
        
        if torch.isnan(cls_score).sum():
            self.index_for_debug_list.append(self.index_for_debug)
            print(self.index_for_debug_list)
            return None, None, None, None, None, None
        assign_result = self.assigner.assign(cls_score, mask_points_pred,
                                             gt_labels, gt_points_masks,
                                             img_metas)
        
        sampling_result = self.sampler.sample(assign_result, mask_pred,
                                              gt_masks)
        
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        
        # label target
        labels = gt_labels.new_full((self.num_queries, ), self.num_classes, dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = labels.new_ones(self.num_queries).type_as(cls_score)
        class_weights_tensor = torch.tensor(self.class_weight).type_as(cls_score)
        # import time
        # time.sleep(0.05)
        # print()

        # mask target
        mask_targets = gt_masks[sampling_result.pos_assigned_gt_inds]
        mask_weights = mask_pred.new_zeros((self.num_queries, ))
        mask_weights[pos_inds] = class_weights_tensor[labels[pos_inds]]

        return (labels, label_weights, mask_targets, mask_weights, pos_inds, neg_inds)
    
    
    def prepare_for_dn_loss(self, dn_args):
        dn_cls_pred_list = dn_args['cls_pred_list']
        dn_mask_pred_list = dn_args['mask_pred_list']
        
        known_labels = dn_args["known_labels"]
        known_masks = dn_args["known_masks"]
        map_known_indices = dn_args["map_known_indices"].long()
        bid = dn_args["known_bid"].long()
        known_indice = dn_args["known_indice"].long()
        num_tgt = known_indice.numel()

        if len(dn_cls_pred_list) > 0:
            for idx in range(len(dn_cls_pred_list)):
                dn_cls_pred_list[idx] = dn_cls_pred_list[idx][(bid, map_known_indices)]
                dn_mask_pred_list[idx] = dn_mask_pred_list[idx][(bid, map_known_indices)]

        return known_labels, known_masks, dn_cls_pred_list, dn_mask_pred_list, num_tgt
    
    @force_fp32(apply_to=('all_cls_scores', 'all_mask_preds'))
    def loss(self, all_cls_scores, all_mask_preds, gt_labels_list, 
             gt_masks_list, gt_binary_list, gt_occ, mask_camera, img_metas, dn_args):
        """Loss function.

        Args:
            all_cls_scores (Tensor): Classification scores for all decoder
                layers with shape (num_decoder, batch_size, num_queries,
                cls_out_channels). Note `cls_out_channels` should includes
                background.
            all_mask_preds (Tensor): Mask scores for all decoder layers with
                shape (num_decoder, batch_size, num_queries, h, w).
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (n, ). n is the sum of number of stuff type
                and number of instance in a image.
            gt_masks_list (list[Tensor]): Ground truth mask for each image with
                shape (n, h, w).
            img_metas (list[dict]): List of image meta information.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_dec_layers = len(all_cls_scores)
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_masks_list = [gt_masks_list for _ in range(num_dec_layers)]
        all_gt_binary_list = [gt_binary_list for _ in range(num_dec_layers)]
        all_mask_camera = [mask_camera for _ in range(num_dec_layers)]
        
        img_metas_list = [img_metas for _ in range(num_dec_layers)]
        losses_cls, losses_mask, losses_loavsz, losses_dice, all_point_coords = multi_apply(
            self.loss_single, all_cls_scores, all_mask_preds,
            all_gt_labels_list, all_gt_masks_list, all_gt_binary_list, all_mask_camera, img_metas_list)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_mask'] = losses_mask[-1]
        loss_dict['loss_dice'] = losses_dice[-1]
        if self.loss_lovasz:
            loss_dict['loss_loavsz'] = losses_loavsz[-1]

        num_dec_layer = 0
        for loss_cls_i, loss_mask_i, losses_loavsz_i, loss_dice_i in zip(losses_cls[:-1], losses_mask[:-1], losses_loavsz[:-1], losses_dice[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_mask'] = loss_mask_i
            loss_dict[f'd{num_dec_layer}.loss_dice'] = loss_dice_i           
            if self.loss_lovasz:
                loss_dict[f'd{num_dec_layer}.loss_loavsz'] = losses_loavsz_i 
            num_dec_layer += 1
            
        if self.dn_enable and self.training:
            known_labels, known_masks, dn_cls_pred_list, dn_mask_pred_list, num_tgt = self.prepare_for_dn_loss(dn_args)
            
            dn_gt_labels_list = [known_labels.long() for _ in range(num_dec_layers)]
            all_known_gt_masks_list = [known_masks for _ in range(num_dec_layers)]
            all_num_tgts_list = [num_tgt for _ in range(num_dec_layers)]
            
            losses_cls_dn, losses_mask_dn, losses_loavsz_dn, losses_dice_dn = multi_apply(
                self.dn_loss_single, dn_cls_pred_list, dn_mask_pred_list,
                dn_gt_labels_list, all_known_gt_masks_list, all_gt_masks_list, all_gt_binary_list, all_mask_camera,
                all_num_tgts_list, all_point_coords,all_gt_labels_list)
            
            loss_dict['loss_cls_dn'] = losses_cls_dn[-1]
            loss_dict['loss_mask_dn'] = losses_mask_dn[-1]
            loss_dict['loss_dice_dn'] = losses_dice_dn[-1]
            if self.loss_lovasz:
                loss_dict['loss_loavsz_dn'] = losses_loavsz_dn[-1]
            num_dec_layer = 0
            for idx in range(len(losses_cls_dn[:-1])):
                loss_dict[f'd{num_dec_layer}.loss_cls_dn'] = losses_cls_dn[:-1][idx]
                loss_dict[f'd{num_dec_layer}.loss_mask_dn'] = losses_mask_dn[:-1][idx]
                loss_dict[f'd{num_dec_layer}.loss_dice_dn'] = losses_dice_dn[:-1][idx]
                if self.loss_lovasz:
                    loss_dict[f'd{num_dec_layer}.loss_loavsz_dn'] = losses_loavsz_dn[:-1][idx]
                num_dec_layer += 1
                
        return loss_dict

    def dn_loss_single(self, cls_scores, mask_preds, gt_labels_list,
                    known_gt_masks_list, gt_masks_list, gt_binary_list, mask_camera, num_tgts_list, point_coords,batch_gt_labels_list):
        num_total_pos = cls_scores.new_tensor([num_tgts_list])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1.0).item()
        
        label_weights = torch.ones_like(gt_labels_list)
        class_weight = cls_scores.new_tensor(self.class_weight)
        loss_cls = self.loss_cls(cls_scores,gt_labels_list,label_weights,avg_factor=class_weight[gt_labels_list].sum(),)


        class_weights_tensor = torch.tensor(self.class_weight).type_as(cls_scores)
        mask_targets = known_gt_masks_list
        mask_weights = class_weights_tensor[gt_labels_list]
        
        mask_preds = mask_preds[mask_weights > 0] 
        mask_weights = mask_weights[mask_weights > 0] 
        
        if mask_targets.shape[0] == 0:
            loss_mask = mask_preds.sum()
            loss_lovasz = mask_preds.sum()
            loss_dice = mask_preds.sum()
            if self.loss_lovasz == False: loss_lovasz=None
            return loss_cls, loss_mask, loss_lovasz, loss_dice

        if self.point_sample_for_dn:
            
            batch_size = len(gt_masks_list)
            gt_idx_in_batch = []
            start_idx = [0]
            for i in range(batch_size):
                gt_idx_in_batch.append(start_idx[-1] + torch.arange(gt_masks_list[i].shape[0]))
                start_idx.append(start_idx[-1] + gt_masks_list[i].shape[0])
            
            new_gt_idx = []
            for gt_idx in gt_idx_in_batch:
                temp_idx_list = []
                for i in range(self.scalar):
                    temp_idx_list.append(gt_idx + start_idx[-1] * i)
                new_gt_idx.append(torch.cat(temp_idx_list))

            if self.loss_flatten:
                mask_point_preds = mask_preds.flatten(1)
                mask_point_targets = mask_targets.flatten(1)
            else:
                with torch.no_grad():
                    point_coords = get_nusc_lidarseg_point_coords_dn(mask_preds.unsqueeze(1), 
                        gt_binary_list, self.voxel_coord, self.num_points, self.oversample_ratio, 
                        self.importance_sample_ratio, self.point_cloud_range, mask_camera, new_gt_idx,
                        padding_mode=self.padding_mode, consider_visible_mask=self.consider_visible_mask)
                point_coords = point_coords[..., [2, 1, 0]]
        else:
            point_coords = point_coords.repeat(self.scalar,1,1)
        mask_point_preds = point_sample_3d(mask_preds.unsqueeze(1), point_coords, padding_mode=self.padding_mode).squeeze(1)
        mask_point_targets = point_sample_3d(known_gt_masks_list.unsqueeze(1).float(), point_coords, padding_mode=self.padding_mode).squeeze(1)
        
        # the weighted version
        num_total_mask_weights = reduce_mean(mask_weights.sum())        
                
        loss_dice = self.loss_dice(mask_point_preds, mask_point_targets, weight=mask_weights, avg_factor=num_total_mask_weights)
        if self.loss_lovasz:
            batch_num_classes=len(gt_labels_list)/self.scalar
            loss_lovasz = 0
            for scaler_idx in range(self.scalar):
                for i in range(len(batch_gt_labels_list)):
                    if i == 0: start = int(scaler_idx * batch_num_classes)
                    else: start += len(batch_gt_labels_list[i-1])
                    batch_num_classes=len(batch_gt_labels_list[i])
                    
                    prepared_pred = torch.zeros((mask_point_targets.shape[1],18),device=torch.device('cuda'))
                    batch_gt_class = torch.argmax(mask_point_targets[start:start+batch_num_classes,:].permute(1,0), axis = 1).long()
                    
                    batch_mask_preds = mask_point_preds[start:start+batch_num_classes,:].permute(1,0).sigmoid()

                    for idx, j in enumerate(batch_gt_labels_list[i]):
                        prepared_pred[:,j]=batch_mask_preds[:,idx]
                    batch_gt_class = batch_gt_labels_list[i][batch_gt_class]
                    if i == 0 and scaler_idx == 0:
                        pred = prepared_pred
                        flattened_gt_occ = batch_gt_class
                    else:
                        pred = torch.cat((pred,prepared_pred),0)
                        flattened_gt_occ = torch.cat((flattened_gt_occ,batch_gt_class),0)
            loss_lovasz = self.loss_lovasz(pred, flattened_gt_occ, ignore=255, flattened = self.lovasz_flatten) * self.lovasz_loss_weight
        
        mask_point_preds = mask_point_preds.reshape(-1)
        mask_point_targets = mask_point_targets.reshape(-1)
        loss_mask = self.loss_mask(mask_point_preds,mask_point_targets,avg_factor=num_total_mask_weights * self.num_points,)
        
        if self.loss_lovasz == False: loss_lovasz=None
        return loss_cls, loss_mask, loss_lovasz, loss_dice


    def loss_single(self, cls_scores, mask_preds, gt_labels_list,
                    gt_masks_list, gt_binary_list, mask_camera, img_metas):
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]
        
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         ) = self.get_targets(cls_scores_list, mask_preds_list, gt_labels_list, 
                    gt_masks_list, gt_binary_list, img_metas)

        # gt_labels_list : [tensor([ 2,  7, 10, 11, 12, 13, 14, 15, 16, 17], device='cuda:0'), tensor([ 0,  4, 11, 13, 14, 15, 16, 17], device='cuda:0')]
        
        # shape (batch_size, num_queries)
        # no_none_list = []
        
        # new_labels_list = []
        # new_label_weights_list = []
        # new_mask_targets_list = []
        # new_mask_weights_list = []
        # new_gt_labels_list = []
        # new_gt_binary_list = []
        # for i in range(len(labels_list)):
        #     if labels_list[i] is not None:
        #         new_labels_list.append(labels_list[i])
        #         new_label_weights_list.append(label_weights_list[i])
        #         new_mask_targets_list.append(mask_targets_list[i])
        #         new_mask_weights_list.append(mask_weights_list[i])
                
        #         new_gt_labels_list.append(gt_labels_list[i])
        #         new_gt_binary_list.append(gt_binary_list[i])
        #         no_none_list.append(i)
        #     else:
        #         self.index_for_debug_list.append(self.index_for_debug)
        #         print(self.index_for_debug_list)
        # labels_list = new_labels_list
        # label_weights_list = new_label_weights_list
        # mask_targets_list = new_mask_targets_list
        # mask_weights_list = new_mask_weights_list     
        # gt_labels_list = new_gt_labels_list
        # gt_binary_list = new_gt_binary_list

        labels = torch.stack(labels_list, dim=0)
        # shape (batch_size, num_queries)
        label_weights = torch.stack(label_weights_list, dim=0)
        # shape (num_total_gts, h, w)
        mask_targets = torch.cat(mask_targets_list, dim=0)
        # shape (batch_size, num_queries)
        mask_weights = torch.stack(mask_weights_list, dim=0)

        # classfication loss
        # shape (batch_size * num_queries, )
        # cls_scores = cls_scores[no_none_list]
        cls_scores = cls_scores.flatten(0, 1)
        labels = labels.flatten(0, 1)
        label_weights = label_weights.flatten(0, 1)
        class_weight = cls_scores.new_tensor(self.class_weight)

        loss_cls = self.loss_cls(
            cls_scores,
            labels,
            label_weights,
            avg_factor=class_weight[labels].sum(),
        )
        # extract positive ones
        # shape (batch_size, num_queries, h, w) -> (num_total_gts, h, w)
        # mask_preds = [1, 100, 200, 200, 16] / [2, 100, 200, 200, 16]
        # mask_weights = [1, 100] / [2, 100]
        batch_size = len(gt_labels_list)
        batch_wise_cls = []
        for i in range(batch_size):
            batch_wise_cls.append(gt_labels_list[i])
        
        # mask_preds = mask_preds[no_none_list]
        mask_preds = mask_preds[mask_weights > 0] # [14, 200, 200, 16] / [18, 200, 200, 16]
        mask_weights = mask_weights[mask_weights > 0] # 14 100개 중 14개만 1 / 18
        
        if mask_targets.shape[0] == 0:
            loss_mask = mask_preds.sum()
            loss_lovasz = mask_preds.sum()
            loss_dice = mask_preds.sum()
            if self.loss_lovasz == False: loss_lovasz=None
            return loss_cls, loss_mask, loss_lovasz, loss_dice, None

        ''' 
        randomly sample K points for supervision, which can largely improve the 
        efficiency and preserve the performance. oversample_ratio = 3.0, importance_sample_ratio = 0.75
        '''
        
        if self.loss_flatten:
            mask_point_preds = mask_preds.flatten(1)
            mask_point_targets = mask_targets.flatten(1)
        else:
            with torch.no_grad():
                point_coords = get_nusc_lidarseg_point_coords(mask_preds.unsqueeze(1), 
                    gt_labels_list, gt_binary_list, self.voxel_coord, self.num_points, self.oversample_ratio, 
                    self.importance_sample_ratio, self.point_cloud_range, mask_camera, padding_mode=self.padding_mode, consider_visible_mask=self.consider_visible_mask)
            point_coords = point_coords[..., [2, 1, 0]]
            mask_point_preds = point_sample_3d(mask_preds.unsqueeze(1), point_coords, padding_mode=self.padding_mode).squeeze(1)
            # dice loss
            mask_point_targets = point_sample_3d(mask_targets.unsqueeze(1).float(), point_coords, padding_mode=self.padding_mode).squeeze(1)
        num_total_mask_weights = reduce_mean(mask_weights.sum())

        loss_dice = self.loss_dice(mask_point_preds, mask_point_targets, weight=mask_weights, avg_factor=num_total_mask_weights)
        if self.loss_lovasz:
            for i in range(len(gt_labels_list)):
                if i == 0: start = 0
                else: start += len(gt_labels_list[i-1])
                batch_num_classes=len(gt_labels_list[i])
                
                prepared_pred = torch.zeros((mask_point_targets.shape[1],18),device=torch.device('cuda'))
                batch_gt_class = torch.argmax(mask_point_targets[start:start+batch_num_classes,:].permute(1,0), axis = 1).long()
                batch_mask_preds = mask_point_preds[start:start+batch_num_classes,:].permute(1,0).sigmoid()

                for idx, j in enumerate(gt_labels_list[i]):
                    prepared_pred[:,j]=batch_mask_preds[:,idx]
                batch_gt_class = gt_labels_list[i][batch_gt_class]
                if i == 0:
                    pred = prepared_pred
                    flattened_gt_occ = batch_gt_class
                else:
                    pred = torch.cat((pred,prepared_pred),0)
                    flattened_gt_occ = torch.cat((flattened_gt_occ,batch_gt_class),0)
                    
            loss_lovasz = self.loss_lovasz(pred, flattened_gt_occ, ignore=255, flattened = self.lovasz_flatten) * self.lovasz_loss_weight
        
        # mask loss
        mask_point_preds = mask_point_preds.reshape(-1)
        mask_point_targets = mask_point_targets.reshape(-1)

        loss_mask = self.loss_mask(
            mask_point_preds,
            mask_point_targets,
            avg_factor=num_total_mask_weights * self.num_points,
        )


        if self.loss_lovasz == False: loss_lovasz=None
        return loss_cls, loss_mask, loss_lovasz, loss_dice, point_coords


    def forward_head_only_non_empty_vox(self, decoder_out, mask_feature, attn_mask_target_size, occ_pred):

        B, W, H, Z, _ = occ_pred.shape
        Q, _ = decoder_out.shape
        non_empty_flag = occ_pred.argmax(-1) != 17
        non_zero = torch.nonzero(non_empty_flag) # N,4
        non_zero_feat = mask_feature.permute(0,2,3,4,1)[non_empty_flag]
        
        decoder_out = self.transformer_decoder.post_norm(decoder_out)
        cls_pred = self.cls_embed(decoder_out.unsqueeze(0).repeat((B,1, 1)))
        mask_embed = self.mask_embed(decoder_out)
        
        non_zero_mask_pred = torch.einsum('qc,xc->xq', mask_embed, non_zero_feat)
        
        mask_pred = torch.zeros(B, Q, W, H, Z).to(non_zero_mask_pred)  # mask_pred를 모두 0으로 초기화
        mask_pred[non_zero[:, 0], :, non_zero[:, 1], non_zero[:, 2], non_zero[:, 3]] = non_zero_mask_pred

        if self.num_transformer_decoder_layers != 0:
            if self.pooling_attn_mask:
                attn_mask = F.adaptive_max_pool3d(mask_pred.float(), attn_mask_target_size)
            else:
                attn_mask = F.interpolate(mask_pred, attn_mask_target_size, mode='trilinear', align_corners=self.align_corners)
            attn_mask = attn_mask.flatten(2).detach() # detach the gradients back to mask_pred
            attn_mask = attn_mask.sigmoid() < 0.5
            attn_mask = attn_mask.unsqueeze(1).repeat((1, self.num_heads, 1, 1)).flatten(0, 1)
        else:
            attn_mask = None 
            
        return cls_pred, mask_pred, attn_mask


    def forward_head(self, decoder_out, mask_feature, attn_mask_target_size):
        """Forward for head part which is called after every decoder layer.

        Args:
            decoder_out (Tensor): in shape (num_queries, batch_size, c).
            mask_feature (Tensor): in shape (batch_size, c, h, w).
            attn_mask_target_size (tuple[int, int]): target attention
                mask size.

        Returns:
            tuple: A tuple contain three elements.

            - cls_pred (Tensor): Classification scores in shape \
                (batch_size, num_queries, cls_out_channels). \
                Note `cls_out_channels` should includes background.
            - mask_pred (Tensor): Mask scores in shape \
                (batch_size, num_queries, x, y, z).
            - attn_mask (Tensor): Attention mask in shape \
                (batch_size * num_heads, num_queries, h, w).
        """
        decoder_out = self.transformer_decoder.post_norm(decoder_out)
        decoder_out = decoder_out.transpose(0, 1)
        # shape (batch_size, num_queries, c)
        cls_pred = self.cls_embed(decoder_out)
        # shape (batch_size, num_queries, c)
        mask_embed = self.mask_embed(decoder_out)
        # shape (batch_size, num_queries, h, w)
        mask_pred = torch.einsum('bqc,bcxyz->bqxyz', mask_embed, mask_feature)

        ''' 对于一些样本数量较少的类别来说，经过 trilinear 插值 + 0.5 阈值，正样本直接消失 '''

        if self.num_transformer_decoder_layers != 0:
            if self.pooling_attn_mask:
                # however, using max-pooling can save more positive samples, which is quite important for rare classes
                attn_mask = F.adaptive_max_pool3d(mask_pred.float(), attn_mask_target_size)
            else:
                # by default, we use trilinear interp for downsampling
                attn_mask = F.interpolate(mask_pred, attn_mask_target_size, mode='trilinear', align_corners=self.align_corners)
            
            # merge the dims of [x, y, z]
            attn_mask = attn_mask.flatten(2).detach() # detach the gradients back to mask_pred
            attn_mask = attn_mask.sigmoid() < 0.5
            
            # repeat for the num_head axis, (batch_size, num_queries, num_seq) -> (batch_size * num_head, num_queries, num_seq)
            attn_mask = attn_mask.unsqueeze(1).repeat((1, self.num_heads, 1, 1)).flatten(0, 1)

        else:
            attn_mask = None 
        return cls_pred, mask_pred, attn_mask

    def preprocess_gt(self, gt_occ, img_metas):
        num_class_list = [self.num_occupancy_classes] * len(img_metas)
        targets = multi_apply(preprocess_occupancy_gt, gt_occ, num_class_list, img_metas)
        
        labels, masks, binary_mask = targets
        return labels, masks, binary_mask

    def forward_train(self,
            voxel_feats,
            img_metas,
            gt_occ,
            mask_camera,
            non_vis_semantic_voxel,
            occ_pred,
            **kwargs,
        ):
        gt_labels, gt_masks, gt_binaries = self.preprocess_gt(gt_occ, img_metas)
        all_cls_scores, all_mask_preds, dn_args = self(voxel_feats, img_metas, occ_pred, targets=dict(gt_labels=gt_labels,
                                                                                   gt_masks=gt_masks,
                                                                                   gt_binaries= gt_binaries,))

        # loss
        losses = self.loss(all_cls_scores, all_mask_preds, gt_labels, gt_masks, gt_binaries, gt_occ, mask_camera, img_metas, dn_args)

        return losses

    def prepare_for_dn(self, targets, size_list, query_feat):
        target_mask = targets['gt_masks']
        mask_list = [target_mask[i] for i in range(len(target_mask))]
                
        num_mask = [len(b) for b in mask_list]
        single_pad = max_num = max(num_mask)
        if self.scalar >= 100:
            self.scalar = self.scalar // max_num
        if max_num == 0 or self.scalar==0:
            return None

        known = [torch.ones_like(t, device=query_feat.device) for t in targets['gt_labels']]
        unmask_bbox = unmask_label = torch.cat(known)
        known_indice = torch.nonzero(unmask_label + unmask_bbox)
        known_indice = known_indice.view(-1)
        known_indice = known_indice.repeat(self.scalar).view(-1)

        dn_pad_size = self.scalar * max_num
        dn_args = dict()
        dn_args['max_num'] = max_num
        dn_args['dn_pad_size'] = dn_pad_size
        bs = len(mask_list)

        padding = torch.zeros([bs, dn_pad_size, self.feat_channels]).cuda()
        padding_mask = torch.ones([bs, dn_pad_size, size_list[0]*size_list[1]*size_list[2]]).cuda().bool()
        if self.pooling_attn_mask_dn:
            # however, using max-pooling can save more positive samples, which is quite important for rare classes
            masks = torch.cat([target_mask[i].float().unsqueeze(1) for i in range(len(target_mask)) if len(target_mask[i])>0])
            masks = (F.adaptive_max_pool3d(masks.float(), size_list))
        else:
            # by default, we use trilinear interp for downsampling
            # masks = torch.cat([F.interpolate(target_mask[i].float().unsqueeze(1), size=size_list, mode="area").flatten(1)<=1e-8 for i in range(len(target_mask)) if len(target_mask[i])>0]).repeat(self.scalar, 1)
            masks = torch.cat([F.interpolate(target_mask[i].float().unsqueeze(1), size=size_list, mode="area") for i in range(len(target_mask)) if len(target_mask[i])>0])

        masks = masks.flatten(1).repeat(self.scalar, 1) < 1e-8

        if self.noise_type=='point':
            areas= (~masks).sum(1)
            noise_ratio=areas*self.dn_mask_noise_scale/(size_list[0]*size_list[1]*size_list[2])
            delta_mask=torch.rand_like(masks,dtype=torch.float)<noise_ratio[:,None]
            masks=torch.logical_xor(masks,delta_mask)

        labels=torch.cat([t for t in targets['gt_labels']])
        known_labels = labels.repeat(self.scalar, 1).view(-1)
        known_masks = torch.cat(mask_list, dim=0).repeat(self.scalar,1,1,1)
        
        dn_label_noise_ratio = self.dn_label_noise_ratio
        knwon_labels_expand = known_labels.clone()
        if dn_label_noise_ratio > 0:
            prob = torch.rand_like(knwon_labels_expand.float())
            chosen_indice = prob < dn_label_noise_ratio
            new_label = torch.randint_like(knwon_labels_expand[chosen_indice], 0,
                                           self.num_classes)  # randomly put a new one here
            # gt_labels_expand.scatter_(0, chosen_indice, new_label)
            knwon_labels_expand[chosen_indice] = new_label
        batch_idx = torch.cat([torch.full_like(t.long(), i) for i, t in enumerate(targets['gt_labels'])])
        known_bid = batch_idx.repeat(self.scalar, 1).view(-1)
        # map_known_indice = torch.tensor([]).to('cuda')
        # import pdb;pdb.set_trace()
        map_known_indices = torch.cat([torch.tensor(range(num)) for num in num_mask])  # [1,2, 1,2,3]
        # print(map_known_indices)
        # print(boxes)
        map_known_indices = torch.cat([map_known_indices + single_pad * i for i in range(self.scalar)]).long().cuda()

        # for i in range(bs):
        if self.dn_query_init_type == 'class':
            noised_known_features = self.label_enc(knwon_labels_expand)
        else:
            noised_known_features = torch.zeros((knwon_labels_expand.shape[0], self.feat_channels))
        padding[(known_bid, map_known_indices)] = noised_known_features
        res = torch.cat([padding.transpose(0, 1), query_feat], dim=0)

        padding_mask[(known_bid, map_known_indices)]= masks
        padding_mask=padding_mask.unsqueeze(1).repeat([1,self.num_heads,1,1])


        tgt_size = dn_pad_size + self.num_queries
        self_attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
        # attn_mask = attn_mask.to('cuda')
        # match query cannot see the reconstruct
        self_attn_mask[dn_pad_size:, :dn_pad_size] = True
        # reconstruct cannot see each other
        for i in range(self.scalar):
            self_attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):dn_pad_size] = True
            self_attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
            
        ###################
        # outputs_class, outputs_mask, attn_mask_ = self.forward_prediction_heads(res, mask_features,attn_mask_target_size=size_list[0])
        # attn_mask_=attn_mask_.view([bs,8,-1,attn_mask_.shape[-1]])
        # attn_mask_[:,:,:-self.num_queries]=padding_mask
        # attn_mask_=attn_mask_.flatten(0,1)
        ###################

        dn_args['known_bid'] = known_bid
        dn_args['map_known_indices'] = map_known_indices
        dn_args['known_labels'] = known_labels
        dn_args['known_masks'] = known_masks
        dn_args['known_indice'] = known_indice
        
        return known_bid, map_known_indices, res, self_attn_mask, dn_args, padding_mask, dn_pad_size
        # pass

    def gen_mask_dn(self, targets, size_list, known_bid, map_known_indices):
        target_mask = targets['gt_masks']
        mask_list = [target_mask[i] for i in range(len(target_mask))]
        num_mask = [len(b) for b in mask_list]
        max_num = max(num_mask)
        if max_num == 0 or self.scalar == 0:
            return None
        if self.scalar >= 100:
            self.scalar = self.scalar // max_num
        dn_pad_size = self.scalar * max_num
        bs = len(mask_list)

        padding_mask = torch.ones([bs, dn_pad_size, size_list[0] * size_list[1] * size_list[2]]).cuda().bool()
        if self.pooling_attn_mask_dn:
            masks = torch.cat([target_mask[i].float().unsqueeze(1) for i in range(len(target_mask)) if len(target_mask[i])>0])
            masks = (F.adaptive_max_pool3d(masks.float(), size_list))
        else:
            masks = torch.cat([F.interpolate(target_mask[i].float().unsqueeze(1), size=size_list, mode="area")
                    for i in range(len(target_mask)) if len(target_mask[i])>0])
        masks = masks.flatten(1).repeat(self.scalar, 1) < 1e-8
        if self.noise_type=='point':
            areas= (~masks).sum(1)
            noise_ratio=areas*self.dn_mask_noise_scale/(size_list[0]*size_list[1]*size_list[2])
            delta_mask=torch.rand_like(masks,dtype=torch.float)<noise_ratio[:,None]
            masks=torch.logical_xor(masks,delta_mask)

        padding_mask[(known_bid, map_known_indices)] = masks
        padding_mask = padding_mask.unsqueeze(1).repeat([1, self.num_heads, 1, 1])
        
        return padding_mask

    def forward(self, 
            voxel_feats,
            img_metas,
            occ_pred,
            targets=None,
            **kwargs,
        ):
        """Forward function.

        Args:
            feats (list[Tensor]): Multi scale Features from the
                upstream network, each is a 5D-tensor (B, C, X, Y, Z).
            img_metas (list[dict]): List of image information.

        Returns:
            tuple: A tuple contains two elements.

            - cls_pred_list (list[Tensor)]: Classification logits \
                for each decoder layer. Each is a 3D-tensor with shape \
                (batch_size, num_queries, cls_out_channels). \
                Note `cls_out_channels` should includes background.
            - mask_pred_list (list[Tensor]): Mask logits for each \
                decoder layer. Each with shape (batch_size, num_queries, \
                 X, Y, Z).
        """
        
        # len(voxel_feats) : 4
        # voxel_feats[0].shape : torch.Size([1, 192, 128, 128, 16])
        # voxel_feats[1].shape : torch.Size([1, 192, 64, 64, 8])
        # voxel_feats[2].shape : torch.Size([1, 192, 32, 32, 4])
        # voxel_feats[3].shape : torch.Size([1, 192, 16, 16, 2])
        # img_metas[0] : {'pc_range': array([-51.2, -51.2,...2,   3. ]), 'occ_size': array([256, 256,  32])}
        
        batch_size = len(img_metas)
        mask_features = voxel_feats[0].permute(0,1,3,2,4)
        multi_scale_memorys = voxel_feats[:0:-1]
        for i in range(len(multi_scale_memorys)):
            if multi_scale_memorys[i].dim()==5:
                multi_scale_memorys[i] = multi_scale_memorys[i].permute(0,1,3,2,4)
        
        decoder_inputs = []
        decoder_positional_encodings = []
        size_list = []
        for i in range(self.num_transformer_feat_level):
            size_list.append(multi_scale_memorys[i].shape[-3:])
            ''' with flatten features '''
            # projection for input features
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            # shape (batch_size, c, x, y, z) -> (x * y * z, batch_size, c)
            decoder_input = decoder_input.flatten(2).permute(2, 0, 1)
            ''' with level embeddings '''
            level_embed = self.level_embed.weight[i].view(1, 1, -1) # [1, 1, 48]
            decoder_input = decoder_input + level_embed
            ''' with positional encodings '''
            mask = decoder_input.new_zeros((batch_size, ) + multi_scale_memorys[i].shape[-3:], dtype=torch.bool)
            decoder_positional_encoding = self.decoder_positional_encoding(mask)
            if self.learned_pos_embed:
                decoder_positional_encoding = self.learned_pos_embed(decoder_positional_encoding.contiguous())
            decoder_positional_encoding = decoder_positional_encoding.flatten(2).permute(2, 0, 1)
            
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)
            
        if self.only_non_empty_voxel_dot:
            query_feat = self.query_feat.weight
        else:
            query_feat = self.query_feat.weight.unsqueeze(1).repeat((1, batch_size, 1))

        if self.dn_enable == False and self.num_transformer_feat_level != 0:
            query_embed = self.query_embed.weight.unsqueeze(1).repeat((1, batch_size, 1))
        else:
            query_embed = None
        dn_args = {}
        if self.dn_enable and self.training:
            if self.noise_type == 'gt' or self.noise_type == 'point':
                if len(size_list) == 0:
                    size_list.append(multi_scale_memorys[-1].shape[-3:])
                known_bid, map_known_indices, query_feat, self_attn_mask, dn_args, padding_mask, dn_pad_size  = self.prepare_for_dn(targets, size_list[0], query_feat)
            else:
                known_bid, map_known_indices, query_feat, self_attn_mask, dn_args, padding_mask_3level, dn_pad_size  = self.prepare_for_dn_shift(targets, size_list, query_feat)
        else:
            self_attn_mask = None
            dn_pad_size = 0
        ''' directly deocde the learnable queries, as simple proposals '''
        cls_pred_list = []
        mask_pred_list = []
        
        if self.only_non_empty_voxel_dot:
            cls_pred, mask_pred, attn_mask = self.forward_head_only_non_empty_vox(query_feat, 
                        mask_features, multi_scale_memorys[0].shape[-3:], occ_pred)
        else:
            cls_pred, mask_pred, attn_mask = self.forward_head(query_feat, 
                        mask_features, multi_scale_memorys[0].shape[-3:])

        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)
        
        if self.num_transformer_decoder_layers != 0:
            B, C, W, H, Z = mask_features.shape
            attn_mask_target_size = multi_scale_memorys[0].shape[-3:]
            
            if self.pooling_attn_mask:
                attn_mask = F.adaptive_max_pool3d(mask_pred.float(), attn_mask_target_size)
            else:
                attn_mask = F.interpolate(mask_pred, attn_mask_target_size, mode='trilinear', align_corners=self.align_corners)
            attn_mask = attn_mask.flatten(2).detach() # detach the gradients back to mask_pred
            attn_mask = attn_mask.sigmoid() < 0.5
            attn_mask = attn_mask.unsqueeze(1).repeat((1, self.num_heads, 1, 1)).flatten(0, 1)
            if self.dn_enable and self.training:
                attn_mask=attn_mask.view([batch_size,self.num_heads,-1,attn_mask.shape[-1]])
                if self.noise_type == 'gt' or self.noise_type == 'point':
                    attn_mask[:,:,:dn_pad_size]=padding_mask
                else:
                    attn_mask[:,:,:dn_pad_size]=padding_mask_3level[0]
                attn_mask=attn_mask.flatten(0,1)
            
        self.index_for_debug += 1
        query_list= []
        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # cross_attn + self_attn
            layer = self.transformer_decoder.layers[i]
            attn_masks = [attn_mask, None]
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                attn_masks=attn_masks,
                self_attn_masks = self_attn_mask,
                query_key_padding_mask=None,
                key_padding_mask=None)
            
            cls_pred, mask_pred, attn_mask = self.forward_head(
                query_feat, mask_features, 
                multi_scale_memorys[(i + 1) % self.num_transformer_feat_level].shape[-3:])
            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)
            query_list.append(query_feat.clone())
            
            if self.dn_enable and self.training:
                if self.noise_type == 'gt' or self.noise_type == 'point':
                    padding_mask = self.gen_mask_dn(targets, size_list[(i + 1) % self.num_transformer_decoder_layers], known_bid, map_known_indices)
                else:
                    padding_mask = padding_mask_3level[(i + 1) % self.num_transformer_decoder_layers]
                attn_mask=attn_mask.view([batch_size,self.num_heads,-1,attn_mask.shape[-1]])
                attn_mask[:,:,:dn_pad_size]=padding_mask
                attn_mask=attn_mask.flatten(0,1)
        
        cls_pred_list_ = []
        mask_pred_list_ = []
        dn_cls_pred_list = []
        dn_mask_pred_list = []
        for i in range(len(cls_pred_list)):
            # dn_pad_size = dn_args['dn_pad_size']
            cls_pred_list_.append(cls_pred_list[i][:, dn_pad_size:])
            mask_pred_list_.append(mask_pred_list[i][:, dn_pad_size:])
            dn_cls_pred_list.append(cls_pred_list[i][:, :dn_pad_size])
            dn_mask_pred_list.append(mask_pred_list[i][:, :dn_pad_size])
        
        dn_args['cls_pred_list'] = dn_cls_pred_list
        dn_args['mask_pred_list'] = dn_mask_pred_list
        return cls_pred_list_, mask_pred_list_, dn_args

    def format_results(self, mask_cls_results, mask_pred_results):
        mask_cls = F.softmax(mask_cls_results, dim=-1)[..., :-1] # [1b, 100q, 18c]
        mask_pred = mask_pred_results.sigmoid() # [1b, 100q, 200x, 200y, 16z]
        output_voxels = torch.einsum("bqc, bqxyz->bcxyz", mask_cls, mask_pred) # 100 -> 18
        return output_voxels

    def simple_test(self, 
            voxel_feats,
            img_metas,
            occ_pred,
            **kwargs,
        ):
        """Test without augmentaton.

        Args:
            feats (list[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple: A tuple contains two tensors.

            - mask_cls_results (Tensor): Mask classification logits,\
                shape (batch_size, num_queries, cls_out_channels).
                Note `cls_out_channels` should includes background.
            - mask_pred_results (Tensor): Mask logits, shape \
                (batch_size, num_queries, h, w).
        """
        all_cls_scores, all_mask_preds, _ = self(voxel_feats, img_metas, occ_pred)
        mask_cls_results = all_cls_scores[-1]
        mask_pred_results = all_mask_preds[-1]
        
        # rescale mask prediction
        # mask_pred_results = F.interpolate(
        #     mask_pred_results,
        #     size=tuple(img_metas[0]['occ_size']),
        #     mode='trilinear',
        #     align_corners=self.align_corners,
        # )
        
        output_voxels = self.format_results(mask_cls_results, mask_pred_results)
        occ_score = output_voxels.permute(0,2,3,4,1).softmax(-1)    # (B, Dx, Dy, Dz, C)
        occ_res = occ_score.argmax(-1)      # (B, Dx, Dy, Dz)
        occ_res = occ_res.cpu().numpy().astype(np.uint8)     # (B, Dx, Dy, Dz)

        return list(occ_res)

_base_ = ['../../../mmdetection3d/configs/_base_/datasets/nus-3d.py',
          '../../../mmdetection3d/configs/_base_/default_runtime.py']

plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams':
    6,
    'input_size': (256, 704),
    'src_size': (900, 1600),

    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

grid_config = {
    'x': [-40, 40, 0.4],
    'y': [-40, 40, 0.4],
    'z': [-1, 5.4, 6.4],
    'depth': [1.0, 45.0, 0.5],
}

grid_config_3dpool = {
    'x': [-40, 40, 0.4],
    'y': [-40, 40, 0.4],
    'z': [-1, 5.4, 6.4],
    'depth': [1.0, 45.0, 0.5],
}
learning_map = {
                1: 0,   5: 0,   7: 0,   8: 0,
                10: 0,  11: 0,  13: 0,  19: 0,
                20: 0,  0: 0,   29: 0,  31: 0,
                9: 1,   14: 2,  15: 3,  16: 3,
                17: 4,  18: 5,  21: 6,  2: 7,
                3: 7,   4: 7,   6: 7,   12: 8,
                22: 9,  23: 10, 24: 11, 25: 12,
                26: 13, 27: 14, 28: 15, 30: 16,
}

voxel_size = [0.1, 0.1, 0.2]
grid_size = [200, 200, 16]
numC_Trans = 64
numC_Trans_pool = 64

depth_categories = 88
num_class = 18
voxel_out_channels = 48
mask2former_num_queries = 100
mask2former_feat_channel = voxel_out_channels
mask2former_output_channel = voxel_out_channels
mask2former_pos_channel = mask2former_feat_channel / 3 # divided by ndim
mask2former_pos_channel_bev = mask2former_feat_channel / 2 # divided by ndim
# mask2former_num_heads = voxel_out_channels // 32
mask2former_num_heads = 8

multi_adj_frame_id_cfg = (1, 1, 1)

if len(range(*multi_adj_frame_id_cfg)) == 0:
    numC_Trans_cat = 0
else:
    numC_Trans_cat = numC_Trans
    
model = dict(
    align_after_view_transfromation=False,
    num_adj=len(range(*multi_adj_frame_id_cfg)),
    type='BEVDetOCC_depthGT_occformer_BEVaux',
    pc_range = point_cloud_range,
    grid_size = grid_size,
    voxel_out_channels = voxel_out_channels,
    only_last_layer=True,
    vox_simple_reshape=True,
    vox_aux_loss_3d=True,
    BEV_aux_channel=numC_Trans_pool,
    vox_aux_loss_3d_occ_head=dict(
        type='BEVOCCHead3D',
        in_dim=voxel_out_channels,
        out_dim=32,
        use_mask=True,
        num_classes=18,
        use_predicter=True,
        class_wise=False,
        loss_occ=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            ignore_index=255,
            loss_weight=1.0
        ),
        sololoss=True,
        loss_weight=10.,
    ),
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch',
        pretrained='torchvision://resnet50',
    ),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[1024, 2048],
        out_channels=256,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    img_view_transformer=dict(
        type='LSSViewTransformerBEVDepth',
        grid_config=grid_config,
        input_size=data_config['input_size'],
        in_channels=256,
        out_channels=numC_Trans_pool,
        sid=False,
        collapse_z=True,
        downsample=16,
        depthnet_cfg=dict(use_dcn=False, aspp_mid_channels=96),
        segmentation_loss=True,
        PV32x88=True
        ),
    # down_sample_for_3d_pooling=[numC_Trans*grid_size[2], numC_Trans],
    img_bev_encoder_backbone=dict(
        type='CustomResNet',
        numC_input=numC_Trans + numC_Trans_cat,
        num_channels=[numC_Trans * 2, numC_Trans * 4, numC_Trans * 8]),
    
    img_bev_encoder_neck=dict(
        type='Custom_FPN_LSS',
        only_largest_voxel_feature_used=True,
        catconv_in_channels1=numC_Trans * 8 + numC_Trans * 4,
        catconv_in_channels2=numC_Trans * 2 + voxel_out_channels * 2,
        outconv_in_channels1=numC_Trans * 8,
        outconv_in_channels2=voxel_out_channels * 2,
        outconv_in_channels3=voxel_out_channels * 2,
        out_channels=voxel_out_channels,
        input_feature_index=(0, 1, 2),
        ),
    
    # down_sample_for_3d_pooling=[numC_Trans, voxel_out_channels*2],
    # bev_neck_deform=True,
    # bev_deform_backbone = dict(
    #     type='SimpleBEVEncoder',
    #     in_channels=voxel_out_channels*2,
    #     ),
    # bev_deform_neck=dict(
    #     type="MSDeformAttnPixelDecoder",
    #     num_outs=4,
    #     in_channels=[voxel_out_channels*2, voxel_out_channels*2, voxel_out_channels*2, voxel_out_channels*2],
    #     strides=[1, 2, 4, 8],
    #     norm_cfg=dict(type="GN", num_groups=16),
    #     act_cfg=dict(type="ReLU"),
    #     feat_channels = voxel_out_channels*2,
    #     out_channels = voxel_out_channels*2,
    #     encoder=dict(  # DeformableDetrTransformerEncoder
    #         num_layers=6,
    #         layer_cfg=dict(  # DeformableDetrTransformerEncoderLayer
    #             self_attn_cfg=dict(embed_dims=voxel_out_channels*2, num_heads=8, num_levels=3, num_points=4, dropout=0.0,
    #                                 batch_first=True),  # MultiScaleDeformableAttention
    #             ffn_cfg=dict(embed_dims=voxel_out_channels*2, feedforward_channels=voxel_out_channels*8, num_fcs=2, ffn_drop=0.0,
    #                             act_cfg=dict(type="ReLU", inplace=True)),
    #         ),
    #     ),
    #     num_cp=0,
    #     positional_encoding=dict(num_feats=voxel_out_channels, normalize=True),
    # ),
    # img_bev_encoder_neck=dict(
    #     type='Custom_FPN_LSS',
    #     only_largest_voxel_feature_used=True,
    #     catconv_in_channels1=voxel_out_channels * 2 + voxel_out_channels * 2,
    #     catconv_in_channels2=voxel_out_channels * 2 + voxel_out_channels * 2,
    #     outconv_in_channels1=voxel_out_channels * 2,
    #     outconv_in_channels2=voxel_out_channels * 2,
    #     outconv_in_channels3=voxel_out_channels * 2,
    #     out_channels=voxel_out_channels,
    #     input_feature_index=(0, 1, 2),
    #     ),

    BEV1=dict(
        type='BEVOCCHead3D',
        in_dim=numC_Trans_pool,
        out_dim=32,
        use_mask=True,
        num_classes=18,
        use_predicter=True,
        class_wise=False,
        loss_occ=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            ignore_index=255,
            loss_weight=1.0
        ),
        sololoss=True,
        loss_weight=10.,
    ),
    occ_head=dict(
        type='Mask2FormerNuscOccHead',
        feat_channels=mask2former_feat_channel,
        out_channels=mask2former_output_channel,
        num_queries=mask2former_num_queries,
        num_occupancy_classes=num_class,
        pooling_attn_mask=True,
        sample_weight_gamma=0.25,
        num_transformer_feat_level=0,
        # using stand-alone pixel decoder
        positional_encoding=dict(
            type='SinePositionalEncoding3D', num_feats=mask2former_pos_channel, normalize=True),
        # using the original transformer decoder
        transformer_decoder=dict(
            type='DetrTransformerDecoder_custom',
            return_intermediate=True,
            num_layers=0,
            transformerlayers=dict(
                type='DetrTransformerDecoderLayer',
                attn_cfgs=dict(
                    type='MultiheadAttention',
                    embed_dims=mask2former_feat_channel,
                    num_heads=mask2former_num_heads,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=False),
                ffn_cfgs=dict(
                    embed_dims=mask2former_feat_channel,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True),
                feedforward_channels=mask2former_feat_channel * 8,
                operation_order=('cross_attn', 'norm', 'self_attn', 'norm',
                                 'ffn', 'norm')),
            init_cfg=None),
        # loss settings
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0] * num_class + [0.1]),
        loss_mask=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0),
        
        point_cloud_range=point_cloud_range,
        train_cfg=dict(
            num_points=12544*3,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            assigner=dict(
                type='MaskHungarianAssigner',
                cls_cost=dict(type='ClassificationCost', weight=2.0),
                mask_cost=dict(
                    type='CrossEntropyLossCost', weight=5.0, use_sigmoid=True),
                dice_cost=dict(
                    type='DiceCost', weight=5.0, pred_act=True, eps=1.0)),
                sampler=dict(type='MaskPseudoSampler'),
            ),
        test_cfg=dict(
                semantic_on=True,
                panoptic_on=False,
                instance_on=False),
        loss_lovasz=True,
        lovasz_loss_weight=1,
        lovasz_flatten=True,
        consider_visible_mask = True,
        learned_pos_embed=True,
    ),
    after_voxelize_add = True,
    det_loss_weight = 1,
    occ_loss_weight = 1,
    seg_loss_weight = 1.,
    SA_loss=True,
    BEV2OCC_3Dhead=True,
)

# Data
dataset_type = 'NuScenesDatasetOccpancy'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

bda_aug_conf = dict(
    rot_lim=(-0., 0.),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5
)

train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        # load_point_label=True,
        data_config=data_config,
        sequential=True),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=True),
    dict(type='LoadOccGTFromFile', ignore_nonvisible=True),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='LoadLidarsegFromFile',
        grid_config=grid_config,
        occupancy_root="./data/nuscenes/pc_panoptic/",
        learning_map=learning_map,
        label_from='panoptic',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='Collect3D', keys=['img_inputs', 'gt_depth', 'voxel_semantics',
                                'mask_lidar', 'mask_camera', 'SA_gt_depth', 'SA_gt_semantic'])
]

test_pipeline = [
    dict(type='PrepareImageInputs', data_config=data_config, sequential=False),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=False),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='LoadOccGTFromFile'),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img_inputs', 'voxel_semantics',
                                'mask_lidar', 'mask_camera'])
        ])
]


input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

share_data_config = dict(
    type=dataset_type,
    data_root=data_root,
    classes=class_names,
    modality=input_modality,
    stereo=False,
    filter_empty_gt=False,
    # img_info_prototype='bevdet4d',
    img_info_prototype='bevdet4d',
    multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,
)

test_data_config = dict(
    pipeline=test_pipeline,
    # ann_file=data_root + 'data10_seg.pkl')
    ann_file=data_root + 'bevdetv2-nuscenes_infos_val_seg.pkl')

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        data_root=data_root,
        ann_file=data_root + 'bevdetv2-nuscenes_infos_train_seg.pkl',
        # ann_file=data_root + 'data10_seg.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR',
        ),
    val=test_data_config,
    test=test_data_config
    )

for key in ['val', 'train', 'test']:
    data[key].update(share_data_config)

# Optimizer
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[12, ])
runner = dict(type='EpochBasedRunner', max_epochs=12)

custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
    ),
]

# load_from = "ckpts/bevdet-r50-cbgs.pth"
# fp16 = dict(loss_scale='dynamic')
evaluation = dict(interval=3, start=12, pipeline=test_pipeline)
checkpoint_config = dict(interval=3, max_keep_ckpts=5)


log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

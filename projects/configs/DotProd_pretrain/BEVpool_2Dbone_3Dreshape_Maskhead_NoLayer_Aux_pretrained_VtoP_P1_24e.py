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
    
coord_index = False # default : True

model = dict(
    align_after_view_transfromation=False,
    num_adj=len(range(*multi_adj_frame_id_cfg)),
    type='BEVDetOCC_depthGT_occformer_pretrain',
    pc_range = point_cloud_range,
    grid_size = grid_size,
    voxel_out_channels = voxel_out_channels,
    only_last_layer=True,
    vox_simple_reshape=True,
    grid_config=grid_config,
    global_VtoP=True,
    pretrain_weight=3.,
    coord_index=coord_index,
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
    
    after_voxelize_add = True,
    det_loss_weight = 1,
    occ_loss_weight = 1,
    seg_loss_weight = 1.,
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
    dict(type='PointToMultiViewDepth', downsample=1, 
         grid_config=grid_config,
         preprocess_for_pretrain=True,
         coord_index=coord_index,
        #  num_samples=2000,
         ),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D', keys=['img_inputs', 'gt_depth', 'voxel_semantics',
                                'mask_lidar', 'mask_camera','points', 'points_img_list',
                                'coor_list', 'norm_points_list'
                                ])
    # dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config, preprocess_for_pretrain=False),
    # dict(type='DefaultFormatBundle3D', class_names=class_names),
    # dict(
    #     type='Collect3D', keys=['img_inputs', 'gt_depth', 'voxel_semantics',
    #                             'mask_lidar', 'mask_camera','points', 'points_img_list',
    #                             ])
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
    samples_per_gpu=1,
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
    step=[24, ])
runner = dict(type='EpochBasedRunner', max_epochs=24)

custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
    ),
]

# load_from = "ckpts/bevdet-r50-cbgs.pth"
# fp16 = dict(loss_scale='dynamic')
evaluation = dict(interval=1, start=24, pipeline=test_pipeline)
checkpoint_config = dict(interval=3, max_keep_ckpts=10)


log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])


# load_from='./work_dirs/DotProd/BEVpool_2Dbone_3Dreshape_Maskhead_NoLayer_Aux_PVAux/epoch_24_ema.pth'
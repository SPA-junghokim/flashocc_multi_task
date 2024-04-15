point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
dataset_type = 'NuScenesDatasetOccpancy'
data_root = 'data/nuscenes/'
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=dict(
            cams=[
                'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(256, 704),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.0),
        sequential=True),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=dict(
            rot_lim=(-0.0, 0.0),
            scale_lim=(1.0, 1.0),
            flip_dx_ratio=0.5,
            flip_dy_ratio=0.5),
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        is_train=True),
    dict(type='LoadOccGTFromFile', ignore_nonvisible=True),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(
        type='PointToMultiViewDepth',
        downsample=1,
        grid_config=dict(
            x=[-40, 40, 0.4],
            y=[-40, 40, 0.4],
            z=[-1, 5.4, 0.4],
            depth=[1.0, 45.0, 0.5])),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='Collect3D',
        keys=[
            'img_inputs', 'gt_depth', 'voxel_semantics', 'mask_lidar',
            'mask_camera'
        ])
]
test_pipeline = [
    dict(
        type='PrepareImageInputs',
        data_config=dict(
            cams=[
                'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(256, 704),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.0),
        sequential=False),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=dict(
            rot_lim=(-0.0, 0.0),
            scale_lim=(1.0, 1.0),
            flip_dx_ratio=0.5,
            flip_dy_ratio=0.5),
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        is_train=False),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(type='LoadOccGTFromFile'),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ],
                with_label=False),
            dict(
                type='Collect3D',
                keys=[
                    'points', 'img_inputs', 'voxel_semantics', 'mask_lidar',
                    'mask_camera'
                ])
        ])
]
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        file_client_args=dict(backend='disk')),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ],
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='NuScenesDatasetOccpancy',
        data_root='data/nuscenes/',
        ann_file='data/nuscenes/bevdetv2-nuscenes_infos_train_seg.pkl',
        pipeline=[
            dict(
                type='PrepareImageInputs',
                is_train=True,
                data_config=dict(
                    cams=[
                        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
                    ],
                    Ncams=6,
                    input_size=(256, 704),
                    src_size=(900, 1600),
                    resize=(-0.06, 0.11),
                    rot=(-5.4, 5.4),
                    flip=True,
                    crop_h=(0.0, 0.0),
                    resize_test=0.0),
                sequential=True),
            dict(
                type='LoadAnnotationsBEVDepth',
                bda_aug_conf=dict(
                    rot_lim=(-0.0, 0.0),
                    scale_lim=(1.0, 1.0),
                    flip_dx_ratio=0.5,
                    flip_dy_ratio=0.5),
                classes=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ],
                is_train=True),
            dict(type='LoadOccGTFromFile', ignore_nonvisible=True),
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='PointToMultiViewDepth',
                downsample=1,
                grid_config=dict(
                    x=[-40, 40, 0.4],
                    y=[-40, 40, 0.4],
                    z=[-1, 5.4, 0.4],
                    depth=[1.0, 45.0, 0.5])),
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ]),
            dict(
                type='Collect3D',
                keys=[
                    'img_inputs', 'gt_depth', 'voxel_semantics', 'mask_lidar',
                    'mask_camera'
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=False,
        box_type_3d='LiDAR',
        use_valid_flag=True,
        stereo=False,
        filter_empty_gt=False,
        img_info_prototype='bevdet4d',
        multi_adj_frame_id_cfg=(1, 1, 1)),
    val=dict(
        type='NuScenesDatasetOccpancy',
        data_root='data/nuscenes/',
        ann_file='data/nuscenes/bevdetv2-nuscenes_infos_val_seg.pkl',
        pipeline=[
            dict(
                type='PrepareImageInputs',
                data_config=dict(
                    cams=[
                        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
                    ],
                    Ncams=6,
                    input_size=(256, 704),
                    src_size=(900, 1600),
                    resize=(-0.06, 0.11),
                    rot=(-5.4, 5.4),
                    flip=True,
                    crop_h=(0.0, 0.0),
                    resize_test=0.0),
                sequential=False),
            dict(
                type='LoadAnnotationsBEVDepth',
                bda_aug_conf=dict(
                    rot_lim=(-0.0, 0.0),
                    scale_lim=(1.0, 1.0),
                    flip_dx_ratio=0.5,
                    flip_dy_ratio=0.5),
                classes=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ],
                is_train=False),
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(type='LoadOccGTFromFile'),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        with_label=False),
                    dict(
                        type='Collect3D',
                        keys=[
                            'points', 'img_inputs', 'voxel_semantics',
                            'mask_lidar', 'mask_camera'
                        ])
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR',
        stereo=False,
        filter_empty_gt=False,
        img_info_prototype='bevdet4d',
        multi_adj_frame_id_cfg=(1, 1, 1)),
    test=dict(
        type='NuScenesDatasetOccpancy',
        data_root='data/nuscenes/',
        ann_file='data/nuscenes/bevdetv2-nuscenes_infos_val_seg.pkl',
        pipeline=[
            dict(
                type='PrepareImageInputs',
                data_config=dict(
                    cams=[
                        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
                    ],
                    Ncams=6,
                    input_size=(256, 704),
                    src_size=(900, 1600),
                    resize=(-0.06, 0.11),
                    rot=(-5.4, 5.4),
                    flip=True,
                    crop_h=(0.0, 0.0),
                    resize_test=0.0),
                sequential=False),
            dict(
                type='LoadAnnotationsBEVDepth',
                bda_aug_conf=dict(
                    rot_lim=(-0.0, 0.0),
                    scale_lim=(1.0, 1.0),
                    flip_dx_ratio=0.5,
                    flip_dy_ratio=0.5),
                classes=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ],
                is_train=False),
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(type='LoadOccGTFromFile'),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        with_label=False),
                    dict(
                        type='Collect3D',
                        keys=[
                            'points', 'img_inputs', 'voxel_semantics',
                            'mask_lidar', 'mask_camera'
                        ])
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR',
        stereo=False,
        filter_empty_gt=False,
        img_info_prototype='bevdet4d',
        multi_adj_frame_id_cfg=(1, 1, 1)))
evaluation = dict(
    interval=1,
    pipeline=[
        dict(
            type='PrepareImageInputs',
            data_config=dict(
                cams=[
                    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                    'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
                ],
                Ncams=6,
                input_size=(256, 704),
                src_size=(900, 1600),
                resize=(-0.06, 0.11),
                rot=(-5.4, 5.4),
                flip=True,
                crop_h=(0.0, 0.0),
                resize_test=0.0),
            sequential=False),
        dict(
            type='LoadAnnotationsBEVDepth',
            bda_aug_conf=dict(
                rot_lim=(-0.0, 0.0),
                scale_lim=(1.0, 1.0),
                flip_dx_ratio=0.5,
                flip_dy_ratio=0.5),
            classes=[
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                'traffic_cone'
            ],
            is_train=False),
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=5,
            use_dim=5,
            file_client_args=dict(backend='disk')),
        dict(type='LoadOccGTFromFile'),
        dict(
            type='MultiScaleFlipAug3D',
            img_scale=(1333, 800),
            pts_scale_ratio=1,
            flip=False,
            transforms=[
                dict(
                    type='DefaultFormatBundle3D',
                    class_names=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ],
                    with_label=False),
                dict(
                    type='Collect3D',
                    keys=[
                        'points', 'img_inputs', 'voxel_semantics',
                        'mask_lidar', 'mask_camera'
                    ])
            ])
    ],
    start=24)
checkpoint_config = dict(interval=1, max_keep_ckpts=5)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './debug'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
data_config = dict(
    cams=[
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    Ncams=6,
    input_size=(256, 704),
    src_size=(900, 1600),
    resize=(-0.06, 0.11),
    rot=(-5.4, 5.4),
    flip=True,
    crop_h=(0.0, 0.0),
    resize_test=0.0)
grid_config = dict(
    x=[-40, 40, 0.4],
    y=[-40, 40, 0.4],
    z=[-1, 5.4, 0.4],
    depth=[1.0, 45.0, 0.5])
voxel_size = [0.1, 0.1, 0.2]
numC_Trans = 32
multi_adj_frame_id_cfg = (1, 1, 1)
numC_Trans_cat = 0
model = dict(
    type='BEVDepth4D_MTL',
    align_after_view_transfromation=False,
    num_adj=0,
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
        pretrained='torchvision://resnet50'),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[1024, 2048],
        out_channels=256,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    img_view_transformer=dict(
        type='LSSViewTransformerBEVDepth',
        grid_config=dict(
            x=[-40, 40, 0.4],
            y=[-40, 40, 0.4],
            z=[-1, 5.4, 0.4],
            depth=[1.0, 45.0, 0.5]),
        input_size=(256, 704),
        in_channels=256,
        out_channels=32,
        sid=False,
        collapse_z=False,
        downsample=16,
        depthnet_cfg=dict(use_dcn=False, aspp_mid_channels=96)),
    img_bev_encoder_backbone=dict(
        type='CustomResNet3D',
        numC_input=32,
        num_layer=[1, 2, 4],
        with_cp=False,
        num_channels=[32, 64, 128],
        stride=[1, 2, 2],
        backbone_output_ids=[0, 1, 2]),
    img_bev_encoder_neck=dict(
        type='LSSFPN3D', in_channels=224, out_channels=32),
    img_bev_encoder=dict(
        type='CustomTriResV3',
        img_bev_encoder_backbone=dict(
            type='CustomResNet', numC_input=32, num_channels=[32, 64, 128]),
        img_bev_encoder_neck=dict(
            type='FPN_LSS', in_channels=160, out_channels=32),
        grid_config=dict(
            x=[-40, 40, 0.4],
            y=[-40, 40, 0.4],
            z=[-1, 5.4, 0.4],
            depth=[1.0, 45.0, 0.5])),
    occ_head=dict(
        type='BEVOCCHead2D',
        channel_down_for_3d=512,
        in_dim=256,
        out_dim=256,
        Dz=16,
        use_mask=True,
        num_classes=18,
        use_predicter=True,
        class_wise=False,
        loss_occ=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            ignore_index=255,
            loss_weight=1.0),
        sololoss=True,
        loss_weight=10),
    det_loss_weight=1,
    occ_loss_weight=1,
    seg_loss_weight=1.0)
bda_aug_conf = dict(
    rot_lim=(-0.0, 0.0),
    scale_lim=(1.0, 1.0),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)
share_data_config = dict(
    type='NuScenesDatasetOccpancy',
    data_root='data/nuscenes/',
    classes=[
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ],
    modality=dict(
        use_lidar=False,
        use_camera=True,
        use_radar=False,
        use_map=False,
        use_external=False),
    stereo=False,
    filter_empty_gt=False,
    img_info_prototype='bevdet4d',
    multi_adj_frame_id_cfg=(1, 1, 1))
test_data_config = dict(
    pipeline=[
        dict(
            type='PrepareImageInputs',
            data_config=dict(
                cams=[
                    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                    'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
                ],
                Ncams=6,
                input_size=(256, 704),
                src_size=(900, 1600),
                resize=(-0.06, 0.11),
                rot=(-5.4, 5.4),
                flip=True,
                crop_h=(0.0, 0.0),
                resize_test=0.0),
            sequential=False),
        dict(
            type='LoadAnnotationsBEVDepth',
            bda_aug_conf=dict(
                rot_lim=(-0.0, 0.0),
                scale_lim=(1.0, 1.0),
                flip_dx_ratio=0.5,
                flip_dy_ratio=0.5),
            classes=[
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                'traffic_cone'
            ],
            is_train=False),
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=5,
            use_dim=5,
            file_client_args=dict(backend='disk')),
        dict(type='LoadOccGTFromFile'),
        dict(
            type='MultiScaleFlipAug3D',
            img_scale=(1333, 800),
            pts_scale_ratio=1,
            flip=False,
            transforms=[
                dict(
                    type='DefaultFormatBundle3D',
                    class_names=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ],
                    with_label=False),
                dict(
                    type='Collect3D',
                    keys=[
                        'points', 'img_inputs', 'voxel_semantics',
                        'mask_lidar', 'mask_camera'
                    ])
            ])
    ],
    ann_file='data/nuscenes/bevdetv2-nuscenes_infos_val_seg.pkl',
    type='NuScenesDatasetOccpancy',
    data_root='data/nuscenes/',
    classes=[
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ],
    modality=dict(
        use_lidar=False,
        use_camera=True,
        use_radar=False,
        use_map=False,
        use_external=False),
    stereo=False,
    filter_empty_gt=False,
    img_info_prototype='bevdet4d',
    multi_adj_frame_id_cfg=(1, 1, 1))
key = 'test'
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[24])
runner = dict(type='EpochBasedRunner', max_epochs=24)
custom_hooks = [
    dict(type='MEGVIIEMAHook', init_updates=10560, priority='NORMAL')
]
gpu_ids = range(0, 1)

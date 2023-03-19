# dataset settings
dataset_type = 'ADE20KDataset'
data_root = 'data/ade/ADEChallengeData2016'
split_root = '/home/chenjiaqi/pj/mmsegmentation/'
meta_keys = ('filename', 'ori_filename', 'ori_shape',
             'img_shape', 'pad_shape', 'scale_factor',
             'flip', 'flip_direction', 'img_norm_cfg',
             'dataset_name', 'dataset_type')
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
test_img_scale = (512, 512)

# ----------------------------------------
# Train dataset name      #  N      #   T
# ----------------------------------------
# ade20k_150_train,       # 20k,    6 times
# bdd_train,              # 7k,     17 times
# cityscapes_19_train,    # 3k,     40 times
# coco_panoptic_133_train,# 118k,   1 times
# idd_39_train,           # 7k,     17 times
# mapillary_public_65_train, # 18k, 7 times
# sunrgbd_37_train        # 5k,     24 times
# ----------------------------------------

times = {
    'ade20k_150': 6,
     'bdd': 17,
     'cityscapes_19': 40,
     'coco_panoptic_133': 1,
     'idd_39': 17,
     'mapillary_public_65': 7,
     'sunrgbd_37': 24
}

dataset_names = {
    'ade20k_150': 'ade20k-150-relabeled',
    'bdd': 'bdd-relabeled',
    'cityscapes_19': 'cityscapes-19-relabeled',
    'coco_panoptic_133': 'coco-panoptic-133-relabeled',
    'idd_39': 'idd-39-relabeled',
    'mapillary_public_65': 'mapillary-public65-relabeled',
    'sunrgbd_37': 'sunrgbd-37-relabeled',
    'camvid_11': 'camvid-11',
    'kitti_19': 'kitti-19',
    'pascal_context_60': 'pascal-context-60',
    'scannet_20': 'scannet-20',
    'voc2012': 'voc2012',
    'wilddash_19': 'wilddash-19'
}

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'], meta_keys=meta_keys),
        ])
]
ade20k_150_train = dict(
    type='RepeatDataset',
    times=times['ade20k_150'],
    dataset=dict(
        type='MSegDataset',
        dataset_name=dataset_names['ade20k_150'],
        data_root='data/mseg_dataset/ADE20K/ADEChallengeData2016',
        img_dir='images/training',
        ann_dir='annotations_semseg150_relabeled/training',
        img_suffix='.jpg',
        seg_map_suffix='.png',
        split=split_root + 'mseg-api/mseg/dataset_lists/ade20k-150-relabeled/list/train_mmseg.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', reduce_zero_label=False),
            dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
            dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
            dict(type='ToUniversalLabel', dataset_name=dataset_names['ade20k_150'], use_naive_taxonomy=False),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'], meta_keys=meta_keys)
        ]
    )
)
bdd_train = dict(
        type='RepeatDataset',
        times=times['bdd'],
        dataset=dict(
        type='MSegDataset',
        dataset_name=dataset_names['bdd'],
        data_root='data/mseg_dataset/BDD/bdd100k',
        img_dir='seg/images/train',
        ann_dir='seg_relabeled/labels/train',
        img_suffix='.jpg',
        seg_map_suffix='_train_id.png',
        split=split_root + 'mseg-api/mseg/dataset_lists/bdd-relabeled/list/train_mmseg.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', reduce_zero_label=False),
            dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
            dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
            dict(type='ToUniversalLabel', dataset_name=dataset_names['bdd'], use_naive_taxonomy=False),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'], meta_keys=meta_keys)
        ]
))
cityscapes_19_train = dict(
        type='RepeatDataset',
        times=times['cityscapes_19'],
        dataset=dict(
        type='MSegDataset',
        dataset_name=dataset_names['cityscapes_19'],
        data_root='data/mseg_dataset/Cityscapes',
        img_dir='leftImg8bit/train',
        ann_dir='gtFine_19cls_relabeled/train',
        img_suffix='_leftImg8bit.png',
        seg_map_suffix='_gtFine_labelIds.png',
        split=split_root + 'mseg-api/mseg/dataset_lists/cityscapes-19-relabeled/list/train_mmseg.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', reduce_zero_label=False),
            dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
            dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
            dict(type='ToUniversalLabel', dataset_name=dataset_names['cityscapes_19'], use_naive_taxonomy=False),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'], meta_keys=meta_keys)
        ]
))
coco_panoptic_133_train = dict(
        type='RepeatDataset',
        times=times['coco_panoptic_133'],
        dataset=dict(
        type='MSegDataset',
        dataset_name=dataset_names['coco_panoptic_133'],
        data_root='data/mseg_dataset/COCOPanoptic',
        img_dir='images/train2017',
        ann_dir='semantic_relabeled133/train2017',
        img_suffix='.jpg',
        seg_map_suffix='.png',
        split=split_root + 'mseg-api/mseg/dataset_lists/coco-panoptic-133-relabeled/list/train_mmseg.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', reduce_zero_label=False),
            dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
            dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
            dict(type='ToUniversalLabel', dataset_name=dataset_names['coco_panoptic_133'], use_naive_taxonomy=False),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'], meta_keys=meta_keys)
        ]
))
idd_39_train = dict(
        type='RepeatDataset',
        times=times['idd_39'],
        dataset= dict(type='MSegDataset',
        dataset_name=dataset_names['idd_39'],
        data_root='data/mseg_dataset/IDD/IDD_Segmentation',
        img_dir='leftImg8bit/train',
        ann_dir='gtFine39_relabeled/train',
        img_suffix='_leftImg8bit.png',
        seg_map_suffix='_gtFine_labelids.png',
        split=split_root + 'mseg-api/mseg/dataset_lists/idd-39-relabeled/list/train_mmseg.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', reduce_zero_label=False),
            dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
            dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
            dict(type='ToUniversalLabel', dataset_name=dataset_names['idd_39'], use_naive_taxonomy=False),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'], meta_keys=meta_keys)
        ]
))
mapillary_public_65_train = dict(
        type='RepeatDataset',
        times=times['mapillary_public_65'],
        dataset=dict(type='MSegDataset',
        dataset_name=dataset_names['mapillary_public_65'],
        data_root='data/mseg_dataset/MapillaryVistasPublic',
        img_dir='training/images',
        ann_dir='training_semseg65_relabeled/labels',
        img_suffix='.jpg',
        seg_map_suffix='.png',
        split=split_root + 'mseg-api/mseg/dataset_lists/mapillary-public65-relabeled/list/train_mmseg.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', reduce_zero_label=False),
            dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
            dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
            dict(type='ToUniversalLabel', dataset_name=dataset_names['mapillary_public_65'], use_naive_taxonomy=False),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'], meta_keys=meta_keys)
        ]
))

sunrgbd_37_train = dict(
        type='RepeatDataset',
        times=times['sunrgbd_37'],
        dataset=dict(
        type='MSegDataset',
        dataset_name=dataset_names['sunrgbd_37'],
        data_root='data/mseg_dataset/SUNRGBD',
        img_dir='image/train',
        ann_dir='semseg-relabeled37/train',
        img_prefix='img-',
        seg_map_prefix='00',
        img_suffix='.jpg',
        seg_map_suffix='.png',
        split=split_root + '/mseg-api/mseg/dataset_lists/sunrgbd-37-relabeled/list/train_mmseg.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', reduce_zero_label=False),
            dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
            dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
            dict(type='ToUniversalLabel', dataset_name=dataset_names['sunrgbd_37'], use_naive_taxonomy=False),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'], meta_keys=meta_keys)
        ]
))

ade20k_150_val = dict(
    type='MSegDataset',
    dataset_name=dataset_names['ade20k_150'],
    data_root='data/mseg_dataset/ADE20K/ADEChallengeData2016',
    img_dir='images/validation',
    ann_dir='annotations_semseg150_relabeled/validation',
    img_suffix='.jpg',
    seg_map_suffix='.png',
    split=split_root + 'mseg-api/mseg/dataset_lists/ade20k-150-relabeled/list/val_mmseg.txt',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', reduce_zero_label=False),
        dict(type='ToUniversalLabel', dataset_name=dataset_names['ade20k_150'], use_naive_taxonomy=False),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(2048, 512),
            # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img', 'gt_semantic_seg'], meta_keys=meta_keys),
            ])
    ]
)

bdd_val = dict(
    type='MSegDataset',
    dataset_name=dataset_names['bdd'],
    data_root='data/mseg_dataset/BDD/bdd100k',
    img_dir='seg/images/val',
    ann_dir='seg_relabeled/labels/val',
    img_suffix='.jpg',
    seg_map_suffix='_train_id.png',
    split=split_root + 'mseg-api/mseg/dataset_lists/bdd-relabeled/list/val_mmseg.txt',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', reduce_zero_label=False),
        dict(type='ToUniversalLabel', dataset_name=dataset_names['bdd'], use_naive_taxonomy=False),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(2048, 512),
            # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img', 'gt_semantic_seg'], meta_keys=meta_keys),
            ])
    ]
)
cityscapes_19_val = dict(
    type='MSegDataset',
    dataset_name=dataset_names['wilddash_19'],
    data_root='data/mseg_dataset/Cityscapes',
    img_dir='leftImg8bit/val',
    ann_dir='gtFine_19cls/val',
    img_suffix='_leftImg8bit.png',
    seg_map_suffix='_gtFine_labelIds.png',
    split=split_root + 'mseg-api/mseg/dataset_lists/cityscapes-19-relabeled/list/val_mmseg.txt',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', reduce_zero_label=False),
        dict(type='ToUniversalLabel', dataset_name=dataset_names['cityscapes_19'], use_naive_taxonomy=False),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(2048, 512),
            # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img', 'gt_semantic_seg'], meta_keys=meta_keys),
            ])
    ]
)
coco_panoptic_133_val=dict(
    type='MSegDataset',
    dataset_name=dataset_names['coco_panoptic_133'],
    data_root='data/mseg_dataset/COCOPanoptic',
    img_dir='images/val2017',
    ann_dir='semantic_relabeled133/val2017',
    img_suffix='.jpg',
    seg_map_suffix='.png',
    split=split_root + 'mseg-api/mseg/dataset_lists/coco-panoptic-133-relabeled/list/val_mmseg.txt',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', reduce_zero_label=False),
        dict(type='ToUniversalLabel', dataset_name=dataset_names['coco_panoptic_133'], use_naive_taxonomy=False),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(2048, 512),
            # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img', 'gt_semantic_seg'], meta_keys=meta_keys),
            ])
    ]
)

idd_39_val=dict(
    type='MSegDataset',
    dataset_name=dataset_names['idd_39'],
    data_root='data/mseg_dataset/IDD/IDD_Segmentation',
    img_dir='leftImg8bit/val',
    ann_dir='gtFine39_relabeled/val',
    img_suffix='_leftImg8bit.png',
    seg_map_suffix='_gtFine_labelids.png',
    split=split_root + 'mseg-api/mseg/dataset_lists/idd-39-relabeled/list/val_mmseg.txt',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', reduce_zero_label=False),
        dict(type='ToUniversalLabel', dataset_name=dataset_names['idd_39'], use_naive_taxonomy=False),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(2048, 512),
            # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img', 'gt_semantic_seg'], meta_keys=meta_keys),
            ])
    ]
)

mapillary_public_65_val=dict(
    type='MSegDataset',
    dataset_name=dataset_names['mapillary_public_65'],
    data_root='data/mseg_dataset/MapillaryVistasPublic',
    img_dir='validation/images',
    ann_dir='validation_semseg65_relabeled/labels',
    img_suffix='.jpg',
    seg_map_suffix='.png',
    split=split_root + 'mseg-api/mseg/dataset_lists/mapillary-public65-relabeled/list/val_mmseg.txt',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', reduce_zero_label=False),
        dict(type='ToUniversalLabel', dataset_name=dataset_names['mapillary_public_65'], use_naive_taxonomy=False),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(2048, 512),
            # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img', 'gt_semantic_seg'], meta_keys=meta_keys),
            ])
    ]
)

sunrgbd_37_val=dict(
    type='MSegDataset',
    dataset_name=dataset_names['sunrgbd_37'],
    data_root='data/mseg_dataset/SUNRGBD',
    img_dir='image/test',
    ann_dir='semseg-relabeled37/test',
    img_prefix='img-',
    seg_map_prefix='00',
    img_suffix='.jpg',
    seg_map_suffix='.png',
    split=split_root + '/mseg-api/mseg/dataset_lists/sunrgbd-37-relabeled/list/val_mmseg.txt',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', reduce_zero_label=False),
        dict(type='ToUniversalLabel', dataset_name=dataset_names['sunrgbd_37'], use_naive_taxonomy=False),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(2048, 512),
            # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img', 'gt_semantic_seg'], meta_keys=meta_keys),
            ])
    ]
)

camvid_11_test=dict(
    type='MSegDataset',
    dataset_name=dataset_names['camvid_11'],
    data_root='data/mseg_dataset/Camvid',
    img_dir='701_StillsRaw_full',
    ann_dir='semseg11',
    img_suffix='.png',
    seg_map_suffix='_L.png',
    split=split_root + '/mseg-api/mseg/dataset_lists/camvid-11/list/val_mmseg.txt',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', reduce_zero_label=False),
        # dict(type='ToUniversalLabel', dataset_name=dataset_names['camvid_11'], use_naive_taxonomy=False),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(2048, 512),
            # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img', 'gt_semantic_seg'], meta_keys=meta_keys),
            ])
    ]
)

kitti_19_test=dict(
    type='MSegDataset',
    dataset_name=dataset_names['kitti_19'],
    data_root='data/mseg_dataset/KITTI/',
    img_dir='training/image_2',
    ann_dir='training/label19',
    img_suffix='.png',
    seg_map_suffix='.png',
    split=split_root + '/mseg-api/mseg/dataset_lists/kitti-19/list/val_mmseg.txt',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', reduce_zero_label=False),
        # dict(type='ToUniversalLabel', dataset_name=dataset_names['kitti_19'], use_naive_taxonomy=False),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(2048, 512),
            # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img', 'gt_semantic_seg'], meta_keys=meta_keys),
            ])
    ]
)

pascal_context_60_test=dict(
    type='MSegDataset',
    dataset_name=dataset_names['pascal_context_60'],
    data_root='data/mseg_dataset/PASCAL_Context',
    img_dir='JPEGImages',
    ann_dir='Segmentation_GT_60cls',
    img_suffix='.jpg',
    seg_map_suffix='.png',
    split=split_root + '/mseg-api/mseg/dataset_lists/pascal-context-60/list/val_mmseg.txt',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', reduce_zero_label=False),
        # dict(type='Resize', img_scale=test_img_scale, ratio_range=(0.5, 2.0)),
        # dict(type='ToUniversalLabel', dataset_name=dataset_names['pascal_context_60'], use_naive_taxonomy=False),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(2048, 512),
            # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img', 'gt_semantic_seg'], meta_keys=meta_keys),
            ])
    ]
)

scannet_20_test=dict(
    type='MSegDataset',
    dataset_name=dataset_names['scannet_20'],
    data_root='data/mseg_dataset/ScanNet/scannet_frames_25k',
    img_dir='',
    ann_dir='',
    img_suffix='.jpg',
    seg_map_suffix='.png',
    split=split_root + '/mseg-api/mseg/dataset_lists/scannet-20/list/val_mmseg.txt',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', reduce_zero_label=False),
        # dict(type='Resize', img_scale=test_img_scale, ratio_range=(0.5, 2.0)),
        # dict(type='ToUniversalLabel', dataset_name=dataset_names['scannet_20'], use_naive_taxonomy=False),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(2048, 512),
            # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img', 'gt_semantic_seg'], meta_keys=meta_keys),
            ])
    ]
)

voc2012_test=dict(
    type='MSegDataset',
    dataset_name=dataset_names['voc2012'],
    data_root='data/mseg_dataset/PASCAL_VOC_2012',
    img_dir='JPEGImages',
    ann_dir='SegmentationClassAug',
    img_suffix='.jpg',
    seg_map_suffix='.png',
    split=split_root + '/mseg-api/mseg/dataset_lists/voc2012/list/val_mmseg.txt',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', reduce_zero_label=False),
        # dict(type='Resize', img_scale=test_img_scale, ratio_range=(0.5, 2.0)),
        # dict(type='ToUniversalLabel', dataset_name=dataset_names['voc2012'], use_naive_taxonomy=False),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(2048, 512),
            # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img', 'gt_semantic_seg'], meta_keys=meta_keys),
            ])
    ]
)

wilddash_19_test=dict(
    type='MSegDataset',
    dataset_name=dataset_names['wilddash_19'],
    data_root='data/mseg_dataset/WildDash',
    img_dir='wd_val_01',
    ann_dir='wd_val_19class',
    img_suffix='_100000.png',
    seg_map_suffix='_100000_labelIds.png',
    split=split_root + '/mseg-api/mseg/dataset_lists/wilddash-19/list/val_mmseg.txt',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', reduce_zero_label=False),
        # dict(type='Resize', img_scale=test_img_scale, ratio_range=(0.5, 2.0)),
        # dict(type='ToUniversalLabel', dataset_name=dataset_names['wilddash_19'], use_naive_taxonomy=False),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(2048, 512),
            # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img', 'gt_semantic_seg'], meta_keys=meta_keys),
            ])
    ]
)

train_data_list = [
    ade20k_150_train,           # 20k,      6 times
    bdd_train,                  # 7k,       17 times
    cityscapes_19_train,        # 3k,       40 times
    coco_panoptic_133_train,    # 118k,     1 times
    idd_39_train,               # 7k,       17 times
    mapillary_public_65_train,  # 18k,      7 times
    sunrgbd_37_train,           # 5k,       24 times
]

val_data_list = [
    ade20k_150_val,
    bdd_val,
    cityscapes_19_val,
    coco_panoptic_133_val,
    idd_39_val,
    mapillary_public_65_val,
    sunrgbd_37_val,
]

test_data_list = [          # 12215
    # cityscapes_19_val,
    voc2012_test,           # 1449, 74.18,
    pascal_context_60_test, # 5105, 71.96,
    camvid_11_test,         # 101,  73.97,   74.0
    wilddash_19_test,       # 70,   59.0,    61.55
    kitti_19_test,          # 50,   61.55,   67.91
    scannet_20_test         # 5440, 45.62,   46.49
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=train_data_list,
    val=test_data_list[2],
    test=test_data_list
    # test=test_data_list[3]
)

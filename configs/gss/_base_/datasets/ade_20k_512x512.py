data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='ADE20KDataset',
        data_root='data/ade/ADEChallengeData2016',
        img_dir='images/training',
        ann_dir='annotations/training',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', reduce_zero_label=True),
            dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
            dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        type='ADE20KDataset',
        data_root='data/ade/ADEChallengeData2016',
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', reduce_zero_label=True),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
                ])
        ]),
    test=dict(
        type='ADE20KDataset',
        data_root='data/ade/ADEChallengeData2016',
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', reduce_zero_label=True),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
                ])
        ]))
_base_ = ['vqseg_agg_swin_large_patch4_window12_768x768_pretrain_384x384_22K_300e_cityscapes.py']
runner = dict(type='EpochBasedRunner', max_epochs=300)

norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
img_size = (768, 768)

model = dict(
    decode_head=dict(
        in_channels=[192, 384, 768, 1536],
        in_index=[0, 1, 2, 3],
        channels=384))

optimizer = dict(lr=1e-3 * 8 / 24)

# deal with OOM
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    # orginal test config
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            # dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img', 'gt_semantic_seg']),
        ])
]

data = dict(
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline),
    samples_per_gpu=1)

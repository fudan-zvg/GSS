_base_ = [
    '../_base_/models/mask_vq_seg_agg_v2_1.py', '../_base_/datasets/cityscapes_for_vqseg_768x768.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_300e_swin.py'
]

checkpoint_file = 'ckp/swin_large_patch4_window12_384_22k_20220412-6580f57d.pth'
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        pretrain_img_size=384,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        patch_norm=True),
    decode_head=dict(num_classes=19))
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (768, 768)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2049, 1025), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
data = dict(
    train=dict(pipeline=train_pipeline),samples_per_gpu=5)

optimizer = dict(lr=1.5e-3)
lr_config = dict(min_lr_ratio=1e-2,)


_base_ = [
    '../_base_/datasets/ade_20k_512x512.py',
    '../_base_/models/gss-ft-w_swin-l.py',
    '../_base_/default_runtime.py'
]

model=dict(
    backbone=dict(init_cfg=None),
    decode_head=dict(post_swin_depth=2))
data = dict(samples_per_gpu=2)

# load_from = 'work_dirs/bigseg_ade20k_conns_swin_160k_025_dim_768_2048_wo_cpm_bs4x8_transformer/iter_32000.pth'

optimizer = dict(
    type='AdamW',
    lr=0.0015,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
optimizer_config = dict()
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=8000)
evaluation = dict(interval=8000, metric='mIoU', pre_eval=True)
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
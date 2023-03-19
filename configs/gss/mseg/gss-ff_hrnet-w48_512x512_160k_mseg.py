_base_ = [
    '../_base_/datasets/mseg_512x512.py',
    '../_base_/models/gss-ff_hrnet-w48.py',
    '../_base_/default_runtime.py'
]

optimizer = dict(
    type='AdamW',
    lr=0.00012,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
optimizer_config = dict()
runner = dict(type='IterBasedRunner', max_iters=160000)
checkpoint_config = dict(by_epoch=False, interval=4000)
evaluation = dict(interval=8000, metric='mIoU', pre_eval=True)
checkpoint_file = '/home/chenjiaqi/pj/mmsegmentation/ckp/swin_large_patch4_window12_384_22k_20220412-6580f57d.pth'

lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
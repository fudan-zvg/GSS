_base_ = [
    '../_base_/models/uvim_stage_2.py', '../_base_/datasets/ade20k_for_vqseg.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=1e-3,
    betas=(0.9, 0.999),
    weight_decay=1e-6,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

lr_config = dict(
    # _delete_=True,
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=4000,
    warmup_ratio=1e-6,
    min_lr_ratio=1e-3,
    by_epoch=False,
    warmup_by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=1)
optimizer_config = dict(grad_clip=dict(max_norm=1, norm_type=2))
runner = dict(type='IterBasedRunner', max_iters=160000)
checkpoint_config = dict(by_epoch=False, interval=16000)
evaluation = dict(interval=160000, metric='mIoU', pre_eval=True)
load_from = 'work_dirs/uvim_ade20k_stage1_80k_vit_swin_clip/iter_80000.pth'





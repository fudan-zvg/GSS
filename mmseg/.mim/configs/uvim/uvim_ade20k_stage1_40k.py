_base_ = [
    '../_base_/models/uvim_stage_1.py', '../_base_/datasets/ade20k_for_vqseg.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=4e-4,
    betas=(0.9, 0.999),
    weight_decay=4e-5,
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
data = dict(samples_per_gpu=2)

runner = dict(max_iters=40000)





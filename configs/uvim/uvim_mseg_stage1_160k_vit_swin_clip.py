_base_ = [
    '../_base_/models/uvim_stage_1.py', '../_base_/datasets/mseg_for_vqseg.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

model=dict(
    decode_head=dict(
        type='VQVAEHeadTransformer',
        oracle_depth=6,
        base_model_depth=12,
        num_heads=12,
        num_classes=194,
        patch_size=16,
        embed_dims=768,
        mlp_ratio=4,
        dict_size=4096,
        codeword_dim=768,
    ),
    # test_cfg=dict(mode='slide', crop_size=256, stride=170)
)

data = dict(samples_per_gpu=1)
runner = dict(type='IterBasedRunner', max_iters=160000)
checkpoint_config = dict(by_epoch=False, interval=10000)
evaluation = dict(interval=170000, metric='mIoU', pre_eval=True)
optimizer_config = dict(grad_clip=dict(max_norm=1, norm_type=2))
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





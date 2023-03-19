_base_ = [
    '../_base_/models/mask_vqseg_segformer_mit-b0.py',
    '../_base_/datasets/cityscapes_1024x1024.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

checkpoint = './ckp/mit_b1_20220624-02e5a6a1.pth'  # noqa

model = dict(
    backbone=dict(init_cfg=dict(type='Pretrained', checkpoint=checkpoint), embed_dims=64),
    test_cfg=dict(mode='whole'),
    decode_head=dict(in_channels=[128, 64, 320, 512]))

# optimizer
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=1.5e-3,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

optimizer_config = dict(grad_clip=dict(max_norm=5.0))

# optimizer = dict(
#     _delete_=True,
#     type='AdamW',
#     lr=0.00006,
#     betas=(0.9, 0.999),
#     weight_decay=0.01,
#     paramwise_cfg=dict(
#         custom_keys={
#             'pos_block': dict(decay_mult=0.),
#             'norm': dict(decay_mult=0.),
#             'head': dict(lr_mult=10.)
#         }))


lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False,
    min_lr_ratio=1e-3,
    warmup='linear',
    warmup_ratio=1e-3,
    warmup_iters=1500,
    warmup_by_epoch=False)

# lr_config = dict(
#     _delete_=True,
#     policy='poly',
#     warmup='linear',
#     warmup_iters=1500,
#     warmup_ratio=1e-6,
#     power=1.0,
#     min_lr=0.0,
#     by_epoch=False)


data = dict(samples_per_gpu=2, workers_per_gpu=2)

# data = dict(samples_per_gpu=1, workers_per_gpu=1)


_base_ = 'bigseg_nyudepth_swin_160k_tf_loading_corr.py'

model=dict(
    decode_head=dict(
        indice_channel_index=1,
        pixel_channel_index=0,
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_name='loss_ce', use_sigmoid=False, loss_weight=1),
            dict(type='SmothL1Loss', loss_name='loss_pixel', loss_weight=1)
        ]
))

optimizer = dict(lr=1e-4)

lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-3,
    by_epoch=False
)

data = dict(samples_per_gpu=4)
checkpoint_config = dict(by_epoch=False, interval=4000)
runner = dict(max_iters=160000)
evaluation = dict(interval=2000, metric='mIoU', pre_eval=True)



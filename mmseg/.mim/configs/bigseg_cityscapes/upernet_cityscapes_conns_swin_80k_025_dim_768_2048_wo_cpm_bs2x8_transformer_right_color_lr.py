_base_=['bigseg_cityscapes_conns_swin_160k_025_dim_768_2048_wo_cpm_bs2x8_transformer_right_color_lr.py']
norm_cfg = dict(type='BN2d', requires_grad=True)
checkpoint_config = dict(by_epoch=False, interval=8000)
evaluation = dict(interval=8000, metric='mIoU', pre_eval=True)
runner = dict(max_iters=80000)
# model=dict(test_cfg=dict(mode='slide', crop_size=(769, 769), stride=(513, 513)))

model=dict(
    decode_head=dict(
        _delete_=True,
        type='UPerHead',
        in_channels=[192, 384, 768, 1536],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=dict(type='BN2d', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # test_cfg=dict(mode='slide', crop_size=(768, 768), stride=(512, 512))
)

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=6e-05,
    betas=(0.9, 0.999),
    weight_decay=0.01
)
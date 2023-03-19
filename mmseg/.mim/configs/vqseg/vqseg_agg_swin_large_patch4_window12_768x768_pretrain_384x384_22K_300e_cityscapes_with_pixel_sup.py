_base_ = ['vqseg_agg_swin_large_patch4_window12_768x768_pretrain_384x384_22K_300e_cityscapes.py']
runner = dict(type='EpochBasedRunner', max_epochs=300)
norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    auxiliary_head=[
        dict(
            type='SegformerHead',
            in_channels=[192, 384, 768, 1536],
            in_index=[0, 1, 2, 3],
            channels=256,
            dropout_ratio=0.1,
            num_classes=19,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
    ],
)
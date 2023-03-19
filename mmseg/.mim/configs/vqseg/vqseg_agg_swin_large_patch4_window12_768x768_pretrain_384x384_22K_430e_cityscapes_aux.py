_base_ = ['vqseg_agg_swin_large_patch4_window12_768x768_pretrain_384x384_22K_300e_cityscapes.py']
runner = dict(type='EpochBasedRunner', max_epochs=430)
img_size = (768, 768)
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
auxiliary_head=dict(
        type='VQSegAggHead',
        in_channels=[768],
        in_index=[2],
        channels=768,
        num_classes=19,
        dropout_ratio=0.1,
        d_vae_type='dalle',
        task_type='seg',
        img_size=img_size,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
)
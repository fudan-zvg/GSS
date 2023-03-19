_base_=['bigseg_cityscapes_conns_swin_160k_025_dim_768_2048_wo_cpm_bs4x8_transformer.py']


model = dict(
    backbone=dict(type='SwinTransformerMask',
        mask_ratio = 0.75),
    decode_head=dict(
        type = 'BigSegAggHeadWoCPMTransformerMask',
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_name='loss_ce', use_sigmoid=False, loss_weight=0.8),
            dict(type='CrossEntropyLoss', loss_name='loss_ce_pixel', use_sigmoid=False, loss_weight=0.2),
            dict(type='SmothL1Loss', loss_name='loss_recon', loss_weight=0.2),
            # dict(type='CrossEntropyLoss', loss_name='loss_ce_dense', use_sigmoid=False, loss_weight=0.0),
        ]
    ),
)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2
)
runner = dict(max_iters=80000)
optimizer = dict(lr=0.00006 * 2)
checkpoint_config = dict(by_epoch=False, interval=16000)
evaluation = dict(interval=8000, metric='mIoU', pre_eval=True)
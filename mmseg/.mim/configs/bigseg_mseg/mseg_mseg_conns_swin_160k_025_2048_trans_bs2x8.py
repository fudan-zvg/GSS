_base_=['bigseg_mseg_conns_swin_160k.py']

model = dict(
    type='MultiDomainEncoderDecoder',
    backbone=dict(with_cp=False),
    decode_head=dict(
        _delete_=True,
        type='MSegHead',
        input_transform='multiple_select',
        in_channels=[192, 384, 768, 1536],
        in_index=[0, 1, 2, 3],
        channels=192 + 384 + 768 + 1536,
        num_classes=194,
))
checkpoint_config = dict(by_epoch=False, interval=8000)
evaluation = dict(interval=8000, metric='mIoU', pre_eval=True)

runner = dict(max_iters=160000)

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006 * 2,
    betas=(0.9, 0.999),
    weight_decay=0.01)

data = dict(samples_per_gpu=2,
            workers_per_gpu=4,)
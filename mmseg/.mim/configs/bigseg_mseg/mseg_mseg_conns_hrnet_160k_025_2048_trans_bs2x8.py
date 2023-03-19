_base_=['bigseg_mseg_conns_swin_160k.py']
model = dict(
    type='MultiDomainEncoderDecoder',
    pretrained='open-mmlab://msra/hrnetv2_w48',
    backbone=dict(
        _delete_=True,
        type='HRNet',
        norm_cfg=dict(type='BN2d', requires_grad=True),
        norm_eval=False,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4,),
                num_channels=(64,)),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(48, 96)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(48, 96, 192)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(48, 96, 192, 384)))),
    decode_head=dict(
        _delete_=True,
        type='MSegHead',
        input_transform='multiple_select',
        in_channels=[48, 96, 192, 384],
        in_index=[0, 1, 2, 3],
        channels=48 + 96 + 192 + 384,
        num_classes=194,
))
checkpoint_config = dict(by_epoch=False, interval=8000)
evaluation = dict(interval=8000, metric='mIoU', pre_eval=True)
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006 * 2,
    betas=(0.9, 0.999),
    weight_decay=0.01)

runner = dict(max_iters=160000)


data = dict(samples_per_gpu=2,
            workers_per_gpu=4,)
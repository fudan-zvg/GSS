_base_=['bigseg_mseg_conns_swin_160k_025_2048_trans.py']
norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    pretrained='open-mmlab://msra/hrnetv2_w48',
    backbone=dict(
        _delete_=True,
        type='HRNet',
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
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
        type='BigSegAggHeadWoCPMTransformerSingleFusion',
        in_channels=[96, 48, 192, 384],
        swin_depth=6
))
# checkpoint_config = dict(by_epoch=False, interval=8000)
# evaluation = dict(interval=40000, metric='mIoU', pre_eval=True)
checkpoint_config = dict(by_epoch=False, interval=4000)
evaluation = dict(interval=8000, metric='mIoU', pre_eval=True)

# runner = dict(max_iters=160000)
data = dict(samples_per_gpu=2)
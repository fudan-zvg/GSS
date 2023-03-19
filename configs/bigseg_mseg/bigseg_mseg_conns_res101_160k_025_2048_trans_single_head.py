_base_=['bigseg_mseg_conns_swin_160k_025_2048_trans.py']
norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(
        _delete_=True,
        type='ResNetV1c',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='BigSegAggHeadWoCPMTransformerSingleFusion',
        in_channels=[512, 256, 1024, 2048],
))
# checkpoint_config = dict(by_epoch=False, interval=8000)
# evaluation = dict(interval=40000, metric='mIoU', pre_eval=True)
checkpoint_config = dict(by_epoch=False, interval=4000)
evaluation = dict(interval=8000, metric='mIoU', pre_eval=True)

# runner = dict(max_iters=160000)
data = dict(samples_per_gpu=2)
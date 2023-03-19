_base_=['bigseg_ade20k_conns_swin_160k_025_dim_2048_wo_cpm_bs4x8.py']
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
        type='PostSwinBigSegAggHeadWoCPMTransformer',
        in_channels=[512, 256, 1024, 2048],
        indice_seg_channel=512,
        indice_cls_channel=2048,
        swin_depth=2,
        post_seg_channel=128,  # keep same with the Transformer in original decode_head
        post_swin_num_head=4,
        post_swin_depth=2,
        post_swin_window_size=7,
))
# eval-only
checkpoint_config = dict(by_epoch=False, interval=4000)
evaluation = dict(interval=8000, metric='mIoU', pre_eval=True)
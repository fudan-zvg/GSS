_base_=['bigseg_ade20k_conns_swin_160k_025_dim_2048_wo_cpm_bs4x8.py']
checkpoint_file = 'ckp/swin_base_patch4_window12_384_22k_20220317-e5c09f74.pth'  # noqa

model = dict(
    backbone=dict(
        pretrain_img_size=384,
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=12,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file)
    ),
    decode_head=dict(
        type='BigSegAggHeadWoCPMTransformer',
        in_channels=[256, 128, 512, 1024],
        indice_seg_channel=512,
        indice_cls_channel=2048,
        swin_depth=4,
))
# checkpoint_config = dict(by_epoch=False, interval=16000)
evaluation = dict(interval=16001, metric='mIoU', pre_eval=True)
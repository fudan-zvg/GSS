_base_=['bigseg_ade20k_conns_swin_160k_025_dim_2048.py']

checkpoint_file = 'ckp/swin_large_patch4_window7_224_22k_20220412-aeecf2aa.pth'
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        window_size=7,
        with_cp=False,
        pretrain_img_size=224,
    ),
    decode_head=dict(
        type='BigSegAggHeadWoCPM',
        num_classes=150))

data = dict(samples_per_gpu=5)
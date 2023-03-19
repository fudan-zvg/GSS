_base_ = [
    'upernet_swin_large_patch4_window7_512x512_'
    'pretrain_224x224_22K_160k_ade20k.py'
]
# checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_large_patch4_window12_384_22k_20220412-6580f57d.pth'  # noqa
checkpoint_file = 'ckp/swin_large_patch4_window12_384_22k_20220412-6580f57d.pth'

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='ckp/swin_large_patch4_window12_384_22k_20220412-6580f57d.pth'),
        pretrain_img_size=384,
        window_size=12))

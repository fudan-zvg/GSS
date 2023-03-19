_base_ = ['vqseg_swin_large_patch4_window12_768x768_pretrain_384x384_22K_300e_cityscapes.py']

model = dict(
    decode_head=dict(
        # type='VQSegHead',
        in_channels=[384, 384, 384, 384],
        channels=sum([384, 384, 384, 384]),
        input_transform='resize_concat',
        d_vae_type='dalle',
        task_type='seg',
        in_index=[1, 0, 2, 3], # resize all feature map to 1/8 scale (the scale of in_index[0] feature)
        num_classes=19,
        align_corners=False))
optimizer = dict(lr=1e-4)
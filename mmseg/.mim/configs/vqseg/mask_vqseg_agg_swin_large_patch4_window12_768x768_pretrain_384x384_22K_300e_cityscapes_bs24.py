_base_ = ['mask_vqseg_agg_swin_large_patch4_window12_768x768_pretrain_384x384_22K_300e_cityscapes.py']

data = dict(samples_per_gpu=3)
optimizer = dict(lr=1e-3)
lr_config = dict(min_lr_ratio=1e-3,)


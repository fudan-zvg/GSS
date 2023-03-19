_base_ = ['mask_vqseg_agg_swin_large_patch4_window12_768x768_pretrain_384x384_22K_300e_cityscapes.py']
runner = dict(type='EpochBasedRunner', max_epochs=500)
data = dict(samples_per_gpu=5)
optimizer = dict(lr=1.5e-3)
lr_config = dict(min_lr_ratio=1e-3,warmup_iters=30)

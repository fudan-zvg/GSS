_base_ = ['vqseg_agg_swin_large_patch4_window12_768x768_pretrain_384x384_22K_300e_cityscapes.py']
runner = dict(type='EpochBasedRunner', max_epochs=860)
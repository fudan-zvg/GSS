_base_ = ['mask_vqseg_agg_swin_large_patch4_window12_768x768_pretrain_384x384_22K_300e_cityscapes.py']

optimizer = dict(lr=1e-3 * 5.0 / 3.0)
_base_ = ['mask_vqseg_agg_swin_large_patch4_window12_768x768_pretrain_384x384_22K_300e_cityscapes_relax_e08.py']

model = dict(
    decode_head=dict(
        loss_decode=dict(
            type='FocalLoss', use_sigmoid=True, loss_weight=1.0)))

# sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000)

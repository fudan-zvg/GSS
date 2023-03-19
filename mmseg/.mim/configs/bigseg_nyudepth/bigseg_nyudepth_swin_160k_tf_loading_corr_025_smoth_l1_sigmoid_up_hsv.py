_base_ = 'bigseg_nyudepth_swin_160k_tf_loading_corr_025_smoth_l1_sigmoid.py'

model=dict(
    decode_head=dict(
        type='BigSegAggHeadWoCPMTransformerDepthUpHSV',
        indice_channel_index=1,
        pixel_channel_index=0,
    ))

data = dict(samples_per_gpu=2)
_base_=['bigseg_ade20k_conns_swin_80k_025_dim_2048_wo_cpm_bs4x8.py']
model = dict(decode_head=dict(
    type='BigSegAggHeadWoCPMTransformer',
    indice_seg_channel=768,
    indice_cls_channel=2048,
))

# log_config = dict(interval=1)
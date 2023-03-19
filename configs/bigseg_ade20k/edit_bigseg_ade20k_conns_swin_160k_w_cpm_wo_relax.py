_base_ = ['edit_bigseg_ade20k_conns_swin_160k.py']

model=dict(decode_head=dict(type='BigSegAggHeadHungerEdit'))

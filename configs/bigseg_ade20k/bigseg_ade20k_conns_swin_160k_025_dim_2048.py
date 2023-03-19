_base_ = ['bigseg_ade20k_conns_swin_160k_025.py']

model=dict(decode_head=dict(indice_seg_channel=2048))
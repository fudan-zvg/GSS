_base_ = ['bigseg_ade20k_conns_swin_160k_wo_cpm.py']

data = dict(samples_per_gpu=1)
model=dict(decode_head=dict(type='EditColor'))
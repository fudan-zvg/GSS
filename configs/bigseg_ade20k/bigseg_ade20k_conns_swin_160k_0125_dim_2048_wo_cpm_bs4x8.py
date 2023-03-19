_base_=['bigseg_ade20k_conns_swin_160k_025_dim_2048_wo_cpm_bs4x8.py']
model=dict(decode_head=dict(indice_channel_index=0)) # 1/8 scale
evaluation = dict(interval=8000, metric='mIoU', pre_eval=True)
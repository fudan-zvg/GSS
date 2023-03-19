_base_=['bigseg_ade20k_conns_swin_80k_025_dim_2048_wo_cpm_bs4x8.py']
runner = dict(max_iters=160000)
#evaluation = dict(interval=100, metric='mIoU', pre_eval=True)
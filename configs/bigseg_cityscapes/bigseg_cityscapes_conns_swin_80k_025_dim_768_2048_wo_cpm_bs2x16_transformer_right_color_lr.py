_base_=['bigseg_cityscapes_conns_swin_160k_025_dim_768_2048_wo_cpm_bs2x8_transformer_right_color_lr.py']

evaluation = dict(interval=170000, metric='mIoU', pre_eval=True)

runner = dict(max_iters=80000)
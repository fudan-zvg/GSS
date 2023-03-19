_base_=['bigseg_mseg_conns_swin_160k.py']

data = dict(samples_per_gpu=4)
checkpoint_config = dict(by_epoch=False, interval=2000)
evaluation = dict(interval=16000, metric='mIoU', pre_eval=True)

_base_=['bigseg_mseg_conns_swin_160k_universe.py']
model=dict(type='MultiDomainEncoderDecoder')
data = dict(samples_per_gpu=4)
checkpoint_config = dict(by_epoch=False, interval=1000)
evaluation = dict(interval=1000, metric='mIoU', pre_eval=True)

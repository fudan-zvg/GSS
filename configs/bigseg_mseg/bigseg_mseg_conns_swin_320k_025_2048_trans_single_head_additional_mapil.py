_base_=['bigseg_mseg_conns_swin_320k_025_2048_trans_additioanl_mapil.py']

model = dict(
    decode_head=dict(
        type='BigSegAggHeadWoCPMTransformerSingleFusion',
))
checkpoint_config = dict(by_epoch=False, interval=8000)
evaluation = dict(interval=40000, metric='mIoU', pre_eval=True)
data = dict(samples_per_gpu=4)
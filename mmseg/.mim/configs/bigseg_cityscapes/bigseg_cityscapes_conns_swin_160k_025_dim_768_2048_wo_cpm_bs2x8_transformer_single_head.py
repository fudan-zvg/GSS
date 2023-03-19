_base_=['bigseg_cityscapes_conns_swin_160k_025_dim_768_2048_wo_cpm_bs2x8_transformer.py']

model = dict(decode_head=dict(type='BigSegAggHeadWoCPMTransformerSingleFusion'))

# checkpoint_config = dict(by_epoch=False, interval=16000)
# evaluation = dict(interval=16001, metric='mIoU', pre_eval=True)
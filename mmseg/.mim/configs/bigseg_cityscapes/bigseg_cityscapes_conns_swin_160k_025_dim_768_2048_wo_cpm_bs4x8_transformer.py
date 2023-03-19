_base_=['bigseg_cityscapes_conns_swin_160k_025_dim_2048_wo_cpm_bs4x8.py']

model = dict(decode_head=dict(
    type='BigSegAggHeadWoCPMTransformer',
    indice_seg_channel=512,
    indice_cls_channel=2048,
))

# checkpoint_config = dict(by_epoch=False, interval=16000)
evaluation = dict(interval=16001, metric='mIoU', pre_eval=True)
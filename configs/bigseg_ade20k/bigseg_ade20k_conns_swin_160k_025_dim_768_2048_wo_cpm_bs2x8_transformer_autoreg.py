_base_=['bigseg_ade20k_conns_swin_160k_025_dim_2048_wo_cpm_bs4x8.py']
data = dict(samples_per_gpu=2)
model = dict(decode_head=dict(
    type='BigSegAggHeadWoCPMTransformerAutoreg',
    indice_channel_index=0,
    #pixel_channel_index=0,
    indice_seg_channel=512,
    indice_cls_channel=2048,
))
# checkpoint_config = dict(by_epoch=False, interval=16000)
evaluation = dict(interval=16000, metric='mIoU', pre_eval=True)
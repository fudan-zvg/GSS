_base_=['bigseg_mseg_conns_swin_320k_additional_mapil.py']

model = dict(
    type='MultiDomainEncoderDecoder',
    backbone=dict(with_cp=False),
    decode_head=dict(
        type='BigSegAggHeadWoCPMTransformer',
        indice_channel_index=1,
        pixel_channel_index=1,
        indice_seg_channel=512,
        indice_cls_channel=2048,
        # indice_seg_channel=2048,
))
checkpoint_config = dict(by_epoch=False, interval=8000)
evaluation = dict(interval=40000, metric='mIoU', pre_eval=True)

runner = dict(max_iters=320000)

optimizer = dict(lr=0.00006 * 2)

data = dict(samples_per_gpu=4,
            workers_per_gpu=4,)
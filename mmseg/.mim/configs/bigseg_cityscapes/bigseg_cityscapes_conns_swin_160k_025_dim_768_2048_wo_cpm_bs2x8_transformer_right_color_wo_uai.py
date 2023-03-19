_base_=['bigseg_cityscapes_conns_swin_160k_025_dim_768_2048_wo_cpm_bs2x8_transformer_right_color.py']

model=dict(
    decode_head=dict(
        type='BigSegAggHeadWoCPMTransformerUAI',
        loss_decode=[
                    dict(
                        type='CrossEntropyLoss',
                        loss_name='loss_ce',
                        use_sigmoid=False,
                        loss_weight=1)
                ],
))

checkpoint_config = dict(by_epoch=False, interval=16000)
evaluation = dict(interval=16001, metric='mIoU', pre_eval=True)
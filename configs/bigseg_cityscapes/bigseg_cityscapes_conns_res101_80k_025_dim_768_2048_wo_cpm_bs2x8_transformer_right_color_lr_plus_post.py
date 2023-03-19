_base_=['bigseg_cityscapes_conns_res101_80k_025_dim_768_2048_wo_cpm_bs2x8_transformer_right_color_lr_plus.py']

model=dict(
    backbone=dict(frozen_stages=4),
    decode_head=dict(
    type='PostSwinBigSegAggHeadWoCPMTransformer',
    # indice_channel_index=1, # dual with oom
    post_seg_channel=128, # keep same with the Transformer in original decode_head
    post_swin_num_head=4,
    post_swin_depth=1,
    post_swin_window_size=7,
    loss_decode = dict(
        _delete_=True,
        type='CrossEntropyLoss',
        use_sigmoid=False,
        loss_weight=1.0),
    ),
    test_cfg=dict(mode='slide', crop_size=(768, 768), stride=(512, 512))
)
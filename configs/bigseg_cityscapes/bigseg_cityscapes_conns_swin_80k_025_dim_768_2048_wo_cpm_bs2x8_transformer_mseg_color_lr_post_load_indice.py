_base_=['bigseg_cityscapes_conns_swin_80k_025_dim_768_2048_wo_cpm_bs2x8_transformer_mseg_color_lr_post.py']

PALETTE = [[125, 167, 9], [119, 169, 72], [55, 15, 50], [223, 217, 173], [165, 154, 206], [176, 154, 172], [167, 109, 212], [166, 115, 169], [234, 107, 138], [115, 161, 137], [167, 154, 147], [171, 62, 107], [169, 57, 178], [229, 113, 209], [222, 156, 146], [227, 162, 83], [227, 164, 112], [231, 168, 17], [230, 109, 177], [0, 0, 0]]

model=dict(
    backbone=dict(frozen_stages=4),
    type='DalleDecoderLoadOnly',
    num_classes=19,
    palette=PALETTE,
    load_dir = 'work_dirs/bigseg_cityscapes_conns_swin_80k_025_dim_768_2048_wo_cpm_bs2x8_transformer_mseg_color_lr/indice_iter_80000/val/',
decode_head = dict(
    type='PostSwinBigSegAggHeadWoCPMTransformer',
    # indice_channel_index=1, # dual with oom
    post_seg_channel=128,  # keep same with the Transformer in original decode_head
    post_swin_num_head=4,
    post_swin_depth=1,
    post_swin_window_size=7,
    loss_decode=dict(
        _delete_=True,
        type='CrossEntropyLoss',
        use_sigmoid=False,
        loss_weight=1.0),
))
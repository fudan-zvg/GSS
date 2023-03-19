_base_=['bigseg_cityscapes_conns_swin_160k_025_dim_768_2048_wo_cpm_bs2x8_transformer_right_color_lr.py']
PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
           [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
           [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
           [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
           [0, 80, 100], [0, 0, 230], [119, 11, 32], [0, 0, 0]]



evaluation = dict(interval=40000000, metric='mIoU', pre_eval=True)
data = dict(samples_per_gpu=1, workers_per_gpu=2)
optimizer = dict(lr=1.5e-3 * 0.5)
# lr_config = dict(min_lr_ratio=1e-3,)
runner = dict(max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=8000)
log_config = dict(interval=20)

load_from = 'work_dirs/bigseg_cityscapes_conns_swin_160k_025_dim_768_2048_wo_cpm_bs2x16_transformer_right_color_lr/iter_160000.pth'

model=dict(
    backbone=dict(frozen_stages=4),
    type='DalleDecoderLoadOnly',
    num_classes=19,
    palette=PALETTE,
    load_dir = 'work_dirs/bigseg_cityscapes_conns_swin_80k_025_dim_768_2048_wo_cpm_bs2x8_transformer_right_color_lr/indice_iter_80000/val/',
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


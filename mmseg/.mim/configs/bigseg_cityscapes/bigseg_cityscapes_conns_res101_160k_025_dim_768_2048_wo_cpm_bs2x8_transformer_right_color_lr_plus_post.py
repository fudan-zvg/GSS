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

evaluation = dict(interval=40000000, metric='mIoU', pre_eval=True)
data = dict(samples_per_gpu=1, workers_per_gpu=2)
optimizer = dict(lr=1.5e-3 * 0.5)
# lr_config = dict(min_lr_ratio=1e-3,)
runner = dict(max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=8000)
log_config = dict(interval=20)

load_from = 'work_dirs/bigseg_cityscapes_conns_res101_80k_025_dim_768_2048_wo_cpm_bs2x8_transformer_right_color_lr_plus/iter_32000.pth'
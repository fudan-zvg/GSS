_base_=['bigseg_ade20k_conns_swin_160k_025_dim_768_2048_wo_cpm_bs4x8_transformer.py']

model=dict(
    backbone=dict(frozen_stages=4),
    decode_head=dict(
    type='PostSwinBigSegAggHeadWoCPMTransformer',
    post_seg_channel=128, # keep same with the Transformer in original decode_head
    post_swin_num_head=4,
    post_swin_depth=2,
    post_swin_window_size=7,
    loss_decode = dict(
        _delete_=True,
        type='CrossEntropyLoss',
        use_sigmoid=False,
        loss_weight=1.0),
))

evaluation = dict(interval=2000, metric='mIoU', pre_eval=True)
data = dict(samples_per_gpu=2)
optimizer = dict(lr=1.5e-3)
# lr_config = dict(min_lr_ratio=1e-3,)
runner = dict(max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=8000)
log_config = dict(interval=20)

load_from = 'work_dirs/bigseg_ade20k_conns_swin_160k_025_dim_768_2048_wo_cpm_bs4x8_transformer/iter_32000.pth'

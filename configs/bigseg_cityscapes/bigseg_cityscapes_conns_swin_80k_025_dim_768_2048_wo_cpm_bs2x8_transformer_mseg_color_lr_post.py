_base_=['bigseg_cityscapes_conns_swin_80k_025_dim_768_2048_wo_cpm_bs2x8_transformer_mseg_color_lr.py']

PALETTE = [[125, 167, 9], [119, 169, 72], [55, 15, 50], [223, 217, 173], [165, 154, 206], [176, 154, 172], [167, 109, 212], [166, 115, 169], [234, 107, 138], [115, 161, 137], [167, 154, 147], [171, 62, 107], [169, 57, 178], [229, 113, 209], [222, 156, 146], [227, 162, 83], [227, 164, 112], [231, 168, 17], [230, 109, 177], [0, 0, 0]]

model=dict(
    backbone=dict(
        frozen_stages=4,
        init_cfg=dict(
                    type='Pretrained',
                    checkpoint=
                    '/home/chenyurui/pj/mmsegmentation/ckp/swin_large_patch4_window12_384_22k_20220412-6580f57d.pth'
                )),
    decode_head=dict(
    type='PostSwinBigSegAggHeadWoCPMTransformer',
    palette=PALETTE,
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

load_from = 'work_dirs/bigseg_cityscapes_conns_swin_80k_025_dim_768_2048_wo_cpm_bs2x8_transformer_mseg_color_lr/iter_32000.pth'

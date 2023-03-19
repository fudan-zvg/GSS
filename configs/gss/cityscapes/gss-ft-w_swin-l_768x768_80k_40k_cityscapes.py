_base_ = [
    '../_base_/datasets/cityscapes_768x768.py',
    '../_base_/models/gss-ft-w_swin-l.py',
    '../_base_/default_runtime.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
img_size = (768, 768)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=192,
        patch_size=4,
        window_size=12,
        mlp_ratio=4,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='LN', requires_grad=True),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            '/home/chenjiaqi/pj/mmsegmentation/ckp/swin_large_patch4_window12_384_22k_20220412-6580f57d.pth'
        ),
        with_cp=False,
        frozen_stages=4),
    decode_head=dict(
        num_classes=19,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        palette=[[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                 [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
                 [107, 142, 35], [152, 251, 152], [70, 130, 180],
                 [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
                 [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32],
                 [0, 0, 0]],
        post_swin_num_head=4,
        post_swin_depth=1,
        post_swin_window_size=7),
    test_cfg=dict(mode='slide', crop_size=(768, 768), stride=(512, 512)))

load_from = 'work_dirs/bigseg_cityscapes_conns_swin_160k_025_dim_768_2048_wo_cpm_bs2x16_transformer_right_color_lr/iter_160000.pth'

optimizer = dict(
    type='AdamW',
    lr=0.00075,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
optimizer_config = dict()
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=8000)
evaluation = dict(interval=8000, metric='mIoU', pre_eval=True)
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
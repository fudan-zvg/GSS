# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
img_size = (768, 768)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=224,
        embed_dims=96,
        patch_size=4,
        window_size=7,
        mlp_ratio=4,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=backbone_norm_cfg,
        frozen_stages=4,
    ),
    decode_head=dict(
        type='PostBigSegAggHeadRelaxE08Hunger',
        in_channels=[384, 192, 768, 1536],
        in_index=[1, 0, 2, 3],
        channels=384,
        dropout_ratio=0.1,
        img_size=img_size,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_name='loss_ce', use_sigmoid=False, loss_weight=1.0),
            dict(type='CrossEntropyLoss', loss_name='loss_ce_pixel', use_sigmoid=False, loss_weight=1.0),
            dict(type='CrossEntropyLoss', loss_name='loss_ce_dense', use_sigmoid=False, loss_weight=0.4),
        ]),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

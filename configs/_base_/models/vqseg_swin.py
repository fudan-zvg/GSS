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
        norm_cfg=backbone_norm_cfg),
    neck=dict(
        type='FPN',
        in_channels=[192, 384, 768, 1536],
        out_channels=384,
        num_outs=4),
    decode_head=dict(
        type='VQSegHead',
        in_channels=384,
        channels=384,
        # channels=sum([192, 384, 768, 1536]),
        # input_transform='resize_concat',
        d_vae_type='dalle',
        task_type='seg',
        img_size=img_size,
        in_index=1,
        # num_classes=19,
        # dropout_ratio=0,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

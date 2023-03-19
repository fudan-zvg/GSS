# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
img_size = (768, 768)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='BEiT',
        img_size=(512, 512),
        patch_size=16,
        in_channels=3,
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4,
        out_indices=(3, 5, 7, 11),
        qv_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_cfg=dict(type='LN', eps=1e-6),
        act_cfg=dict(type='GELU'),
        norm_eval=False,
        init_values=0.1),
    decode_head=dict(
        type='BigSegAggHeadRelaxE08Hunger',
        in_channels=[768, 768, 768, 768],
        in_index=[1, 0, 2, 3],
        channels=384,
        dropout_ratio=0.1,
        img_size=img_size,
        norm_cfg=norm_cfg,
        align_corners=False,
        num_classes=150,
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_name='loss_ce', use_sigmoid=False, loss_weight=1.0),
            dict(type='CrossEntropyLoss', loss_name='loss_ce_pixel', use_sigmoid=False, loss_weight=1.0),
            dict(type='CrossEntropyLoss', loss_name='loss_ce_dense', use_sigmoid=False, loss_weight=0.4),
        ]),
    # model training and testing settings
    # train_cfg=dict(),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

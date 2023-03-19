norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
stage=1
model=dict(
    type='UViM',
    stage=stage,
    pretrained=None,
    backbone=dict(
        type='VisionTransformer',
        img_size=768,
        patch_size=16,
        in_channels=3,
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        pretrained=None,
        with_cls_token=False,
        init_cfg=dict(type='Pretrained', checkpoint='ckp/jx_vit_base_p16_224-80ecf9dd.pth'),
    ),
    decode_head=dict(
        type='VQVAEHeadTransformerSwinStage2',
        stage=stage,
        num_classes=150,
        channels=256,
        patch_size=16,
        ignore_index=255,
        oracle_depth=6,
        backbone_embed_dim=768,
        base_model_depth=12,
        mlp_dim=3072,
        decoder_dim=192,
        num_heads=12,
        dict_size=4096,
        codeword_dim=768,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, loss_name='loss_ce'),
        ]),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole')

)

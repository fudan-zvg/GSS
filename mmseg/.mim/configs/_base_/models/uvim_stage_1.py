norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
stage=1
model=dict(
    type='UViM',
    stage=stage,
    pretrained=None,
    backbone=dict(
        type='NoneBackbone'
    ),
    decode_head=dict(
        type='VQVAEHead',
        stage=stage,
        num_classes=150,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, loss_name='loss_ce'),
            dict(type='MSELoss', loss_weight=1.0, loss_name='loss_vq'),
            dict(type='MSELoss', loss_weight=0.25, loss_name='loss_commit')
        ]),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole')

)
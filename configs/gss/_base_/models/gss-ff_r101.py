norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
img_size = (768, 768)

model = dict(
    type='EncoderDecoder',
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='GenerativeSegHeadFF',
        in_channels=[512, 256, 1024, 2048],
        in_index=[1, 0, 2, 3],
        channels=384,
        dropout_ratio=0.1,
        img_size=(768, 768),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        num_classes=19,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                loss_name='loss_ce',
                use_sigmoid=False,
                loss_weight=0.8),
            dict(
                type='CrossEntropyLoss',
                loss_name='loss_ce_pixel',
                use_sigmoid=False,
                loss_weight=0.2)
        ],
        palette=[[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                 [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
                 [107, 142, 35], [152, 251, 152], [70, 130, 180],
                 [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
                 [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32],
                 [0, 0, 0]],
        indice_channel_index=1,
        pixel_channel_index=1,
        indice_seg_channel=512,
        indice_cls_channel=2048,
        swin_depth=6),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(768, 768), stride=(512, 512)))

_base_ = ['vqseg_agg_swin_large_patch4_window12_768x768_pretrain_384x384_22K_300e_cityscapes.py']
runner = dict(type='EpochBasedRunner', max_epochs=300)
norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    auxiliary_head=[
        dict(
            type='FCNHead',
            in_channels=384,
            in_index=1,
            channels=256,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=19,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4,
                # class_weight=[
                #     2.5959933, 6.7415504, 3.5354059, 9.8663225, 9.690899, 9.369352,
                #     10.289121, 9.953208, 4.3097677, 9.490387, 7.674431, 9.396905,
                #     10.347791, 6.3927646, 10.226669, 10.241062, 10.280587,
                #     10.396974, 10.055647
                # ]
            )),
        dict(
            type='FCNHead',
            in_channels=768,
            in_index=2,
            channels=256,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=19,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4,
            #     class_weight=[
            #     2.5959933, 6.7415504, 3.5354059, 9.8663225, 9.690899, 9.369352,
            #     10.289121, 9.953208, 4.3097677, 9.490387, 7.674431, 9.396905,
            #     10.347791, 6.3927646, 10.226669, 10.241062, 10.280587,
            #     10.396974, 10.055647
            # ]
            )),
        dict(
            type='FCNHead',
            in_channels=1536,
            in_index=3,
            channels=256,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=19,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4,
                # class_weight=[
                #     2.5959933, 6.7415504, 3.5354059, 9.8663225, 9.690899, 9.369352,
                #     10.289121, 9.953208, 4.3097677, 9.490387, 7.674431, 9.396905,
                #     10.347791, 6.3927646, 10.226669, 10.241062, 10.280587,
                #     10.396974, 10.055647
                # ]
            )),
    ],
)
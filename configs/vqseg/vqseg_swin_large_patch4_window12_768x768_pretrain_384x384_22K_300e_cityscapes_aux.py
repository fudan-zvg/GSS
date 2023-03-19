_base_ = ['vqseg_swin_large_patch4_window12_768x768_pretrain_384x384_22K_300e_cityscapes.py']
norm_cfg = dict(type='SyncBN', requires_grad=True)
img_size = (768, 768)
model = dict(
        decoder_head=dict(num_classes=19),
        auxiliary_head=dict(
                type='VQSegHead',
                in_channels=384,
                channels=384,
                # channels=sum([192, 384, 768, 1536]),
                # input_transform='resize_concat',
                d_vae_type='dalle',
                task_type='seg',
                img_size=img_size,
                in_index=2,
                num_classes=19,
                # dropout_ratio=0,
                norm_cfg=norm_cfg,
                align_corners=False,
                loss_decode=dict(
                        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
)
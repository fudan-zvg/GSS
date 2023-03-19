_base_ = [
    '../_base_/models/bigseg_relax_e08_hunger.py', '../_base_/datasets/cityscapes_for_vqseg_768x768.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_300e_swin.py'
]
PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
           [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
           [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
           [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
           [0, 80, 100], [0, 0, 230], [119, 11, 32], [0, 0, 0]]

checkpoint_file = 'ckp/swin_large_patch4_window12_384_22k_20220412-6580f57d.pth'
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        pretrain_img_size=384,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        patch_norm=True),
    decode_head=dict(num_classes=19,
                     palette=PALETTE,
                     loss_decode=[
                         dict(type='CrossEntropyLoss',
                              loss_name='loss_ce',
                              use_sigmoid=False,
                              loss_weight=0.8),
                         dict(type='CrossEntropyLoss',
                              loss_name='loss_ce_pixel',
                              use_sigmoid=False,
                              loss_weight=0.2,
                              class_weight=[0.8373, 0.9180, 0.8660, 1.0345, 1.0166, 0.9969, 0.9754,
                        1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037,
                        1.0865, 1.0955, 1.0865, 1.1529, 1.0507]
                              ),
                         # dict(type='CrossEntropyLoss', loss_name='loss_ce_dense', use_sigmoid=False, loss_weight=0.0),
                     ]
                     ))

data = dict(samples_per_gpu=4)
optimizer = dict(lr=1.5e-3)
lr_config = dict(min_lr_ratio=1e-3,)


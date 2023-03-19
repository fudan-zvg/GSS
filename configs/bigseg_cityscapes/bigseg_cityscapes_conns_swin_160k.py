_base_ = [
    '../_base_/models/bigseg_relax_e08_hunger.py', '../_base_/datasets/cityscapes_for_vqseg_768x768.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
checkpoint_file = '/home/chenjiaqi/pj/mmsegmentation/ckp/swin_large_patch4_window12_384_22k_20220412-6580f57d.pth'

PALETTE = [[222, 222, 145], [18, 30, 7], [8, 23, 47], [30, 6, 96], [1, 13, 164], [12, 28, 191], [25, 52, 32], [29, 48, 52], [15, 51, 95], [25, 56, 167], [25, 42, 210], [27, 81, 31], [9, 88, 54], [27, 92, 113], [11, 99, 151], [26, 110, 183], [24, 130, 26], [4, 122, 75], [3, 132, 98], [26, 147, 167], [17, 132, 197], [5, 169, 28], [19, 184, 67], [0, 190, 122], [12, 167, 147], [6, 161, 196], [2, 205, 3], [5, 220, 61], [23, 225, 107], [7, 217, 157], [25, 208, 191], [74, 10, 8], [69, 30, 69], [56, 4, 98], [61, 29, 164], [60, 10, 194], [60, 52, 19], [74, 69, 52], [65, 68, 116], [81, 41, 161], [70, 60, 197], [66, 81, 14], [55, 107, 61], [76, 110, 108], [74, 104, 162], [72, 94, 197], [60, 133, 16], [69, 128, 67], [59, 148, 104], [65, 133, 154], [68, 128, 183], [79, 181, 11], [76, 170, 56], [71, 175, 103], [53, 162, 137], [53, 182, 183], [51, 229, 26], [51, 202, 51], [69, 213, 122], [63, 213, 161], [71, 203, 197], [120, 11, 31], [124, 3, 68], [131, 2, 98], [113, 1, 162], [102, 13, 209], [109, 50, 30], [126, 41, 47], [107, 46, 118], [112, 49, 147], [109, 41, 189], [103, 83, 15], [126, 99, 70], [124, 101, 104], [131, 103, 159], [128, 110, 183], [119, 148, 9], [112, 137, 50], [123, 127, 116], [107, 124, 167], [102, 148, 203], [124, 180, 15], [116, 168, 65], [104, 182, 102], [111, 164, 163], [105, 174, 191], [102, 218, 20], [126, 203, 64], [108, 215, 109], [110, 221, 157], [107, 230, 192], [160, 25, 11], [165, 12, 65], [153, 2, 117], [182, 21, 141], [160, 19, 188], [176, 58, 19], [175, 58, 56], [170, 69, 93], [176, 42, 146], [157, 44, 211], [157, 105, 2], [180, 98, 73], [182, 85, 92], [169, 93, 152], [156, 89, 202], [157, 144, 22], [180, 151, 77], [154, 146, 118], [162, 136, 143], [171, 134, 184], [170, 174, 15], [178, 180, 65], [176, 183, 120], [175, 169, 147], [181, 165, 197], [156, 227, 3], [167, 218, 61], [160, 216, 119], [164, 251, 141], [177, 201, 251], [231, 30, 13], [219, 6, 59], [211, 26, 122], [216, 16, 153], [209, 12, 192], [216, 70, 15], [215, 46, 60], [234, 61, 112], [224, 53, 157], [227, 49, 207], [221, 108, 8], [220, 93, 73], [230, 111, 113], [218, 89, 143], [231, 90, 195], [227, 144, 22], [208, 137, 49], [210, 128, 116], [225, 135, 157], [221, 135, 193], [211, 174, 18], [222, 185, 50], [229, 183, 93], [233, 162, 155], [255, 167, 205], [211, 215, 15], [232, 225, 71], [0, 0, 0], [255, 255, 255], [215, 216, 196]]

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        embed_dims=192,
        pretrain_img_size=384,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        patch_norm=True),
    decode_head=dict(
        in_channels=[384, 192, 768, 1536],
        in_index=[1, 0, 2, 3],
        num_classes=150,
        palette=PALETTE,
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_name='loss_ce', use_sigmoid=False, loss_weight=0.8),
            dict(type='CrossEntropyLoss', loss_name='loss_ce_pixel', use_sigmoid=False, loss_weight=0.2),
            # dict(type='CrossEntropyLoss', loss_name='loss_ce_dense', use_sigmoid=False, loss_weight=0.0),
        ]
    ),
    # auxiliary_head=dict(
    #     type='FCNHead',
    #     in_channels=768,
    #     in_index=2,
    #     channels=256,
    #     num_convs=1,
    #     concat_input=False,
    #     dropout_ratio=0.1,
    #     num_classes=19,
    #     norm_cfg=norm_cfg,
    #     align_corners=False,
    #     loss_decode=dict(
    #         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4))
)

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=1.5e-3,  # lr=0.00006, for ade20k
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

lr_config = dict(
    # _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=2)



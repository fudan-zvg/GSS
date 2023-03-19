_base_ = [
    '../_base_/models/bigseg_relax_e08_hunger.py', '../_base_/datasets/cityscapes_for_vqseg_768x768.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_300e_swin.py'
]
# PALETTE = [[143, 176, 203], [8, 88, 73], [10, 95, 140], [6, 89, 202], [0, 167, 12], [12, 174, 75], [1, 169, 136], [10, 167, 198], [132, 14, 5], [143, 6, 73], [135, 9, 132], [136, 1, 202], [140, 84, 7], [138, 200, 0], [134, 0, 139], [137, 93, 255], [255, 164, 0], [255, 255, 255], [138, 161, 142], [132, 167, 76],]
PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
           [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
           [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
           [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 0, 0],
           [0, 80, 100], [0, 0, 230], [119, 11, 32], [255, 255, 255]]


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
                     palette=PALETTE))

data = dict(samples_per_gpu=4)
optimizer = dict(lr=1.5e-3)
lr_config = dict(min_lr_ratio=1e-3,)


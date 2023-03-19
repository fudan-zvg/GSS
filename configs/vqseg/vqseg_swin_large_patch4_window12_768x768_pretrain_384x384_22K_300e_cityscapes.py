_base_ = [
    '../_base_/models/vqseg_swin.py', '../_base_/datasets/cityscapes_for_vqseg_768x768.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_300e_swin.py'
]

# from upernet_swin_large_pretrain_384_window_12
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
    decode_head=dict(num_classes=19))

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=3)

# where is the reference of this configuration file?
# from swin large win12
# model = dict(
#     backbone=dict(
#         init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
#         pretrain_img_size=384,
#         window_size=12))

# from swin large win7
# model = dict(
#     backbone=dict(
#         # init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
#         embed_dims=192,
#         depths=[2, 2, 18, 2],
#         num_heads=[6, 12, 24, 48]),

# from swin tiny win7 1k
# model = dict(
#     backbone=dict(
#         use_abs_pos_embed=False,
#         drop_path_rate=0.3,
#         patch_norm=True),


_base_ = 'uvim_cityscapes_stage1_80k_vit_swin_clip.py'
model=dict(
    test_cfg=dict(mode='slide', crop_size=256, stride=170)
)
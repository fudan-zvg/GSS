_base_ = [
    '../_base_/models/vqseg_vit_large.py', '../_base_/datasets/cityscapes_for_vqseg_768x768.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_300e.py'
]

norm_cfg = dict(type='SyncBN', requires_grad=True)
crop_size = (768, 768)
model = dict(
    pretrained=None,
    backbone=dict(
        drop_rate=0.,
        init_cfg=dict(
            type='Pretrained', checkpoint='pretrain/vit_large_p16.pth')),
    decode_head=dict(
        d_vae_type='taming',
        task_type='seg'
    ),
    test_cfg=dict(mode='whole'))

optimizer = dict(
    weight_decay=0.0,
    paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10.)}))

data = dict(samples_per_gpu=1)

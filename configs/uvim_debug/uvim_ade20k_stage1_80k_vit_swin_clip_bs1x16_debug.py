_base_ = ['../uvim/uvim_ade20k_stage1_80k_vit_swin.py']


model=dict(
    decode_head=dict(
        type='VQVAEHeadTransformerSwinNonUnlable',        
        num_classes=151,
        ),
)

optimizer_config = dict(grad_clip=dict(max_norm=1, norm_type=2))

checkpoint_config = dict(by_epoch=False, interval=1000)
evaluation = dict(interval=81000, metric='mIoU', pre_eval=True)
_base_ = [
    '../_base_/models/uvim_stage_2.py', '../_base_/datasets/mseg_for_vqseg.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

backbone_norm_cfg = dict(type='LN', requires_grad=True)

backbone_checkpoint_file = '/home/chenjiaqi/pj/mmsegmentation/ckp/swin_large_patch4_window12_384_22k_20220412-6580f57d.pth'

stage_1_checkpoint_file = 'work_dirs/uvim_mseg_stage1_160k_vit_swin_clip/iter_160000.pth'

model=dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=192,
        patch_size=4,
        window_size=12,
        mlp_ratio=4,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='LN', requires_grad=True),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            '/home/chenjiaqi/pj/mmsegmentation/ckp/swin_large_patch4_window12_384_22k_20220412-6580f57d.pth'
        )),
    decode_head=dict(
        _delete_=True,
        type='VQVAEHeadTransformerStage2',
        oracle_depth=6,
        base_model_depth=12,
        num_heads=12,
        num_classes=194,
        patch_size=16,
        embed_dims=768,
        mlp_ratio=4,
        dict_size=4096,
        codeword_dim=768,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0,
                loss_name='loss_ce')
        ],
        # init_cfg=dict(type='Pretrained', checkpoint=decode_head_checkpoint_file),
    ),
    # test_cfg=dict(mode='slide', crop_size=256, stride=170)
)

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=1e-3,
    betas=(0.9, 0.999),
    weight_decay=1e-6,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

lr_config = dict(
    # _delete_=True,
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=4000,
    warmup_ratio=1e-6,
    min_lr_ratio=1e-3,
    by_epoch=False,
    warmup_by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=1)
optimizer_config = dict(grad_clip=dict(max_norm=1, norm_type=2))
runner = dict(type='IterBasedRunner', max_iters=160000)
checkpoint_config = dict(by_epoch=False, interval=16000)
evaluation = dict(interval=170000, metric='mIoU', pre_eval=True)
load_from = stage_1_checkpoint_file




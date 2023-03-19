_base_ = ['uvim_ade20k_stage1_40k_vit.py']

model=dict(
    decode_head=dict(
        type='VQVAEHeadTransformerSwin',
        oracle_depth=6,
        base_model_depth=12,
        num_heads=12,
        mlp_dim=3072,
        decoder_dim=192,
    ))

data = dict(samples_per_gpu=1)
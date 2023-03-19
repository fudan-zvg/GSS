_base_ = ['uvim_ade20k_stage1_40k.py']

model=dict(
    decode_head=dict(
        type='VQVAEHeadTransformer',
        oracle_depth=6,
        base_model_depth=12,
        mlp_dim=192,
        decoder_dim=192,
))

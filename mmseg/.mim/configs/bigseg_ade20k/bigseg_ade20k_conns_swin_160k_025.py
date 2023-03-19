_base_ = ['bigseg_ade20k_conns_swin_160k.py']

model= dict(
    backbone=dict(with_cp=False),
    decode_head=dict(
        indice_channel_index=1,
        pixel_channel_index=1,
    )
)
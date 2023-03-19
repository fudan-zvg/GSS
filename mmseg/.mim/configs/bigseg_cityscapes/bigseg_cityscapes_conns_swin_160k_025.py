_base_ = ['bigseg_cityscapes_conns_swin_160k.py']

model= dict(
    backbone=dict(with_cp=False),
    decode_head=dict(
        indice_channel_index=1,
        pixel_channel_index=1,
    )
)
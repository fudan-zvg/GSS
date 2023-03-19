_base_ = ['bigseg_ade20k_conns_swin_160k.py']
model = dict(decode_head=dict(
    loss_decode=[
        dict(type='CrossEntropyLoss', loss_name='loss_ce', use_sigmoid=False, loss_weight=1),
        dict(type='CrossEntropyLoss', loss_name='loss_ce_pixel', use_sigmoid=False, loss_weight=1),
        # dict(type='CrossEntropyLoss', loss_name='loss_ce_dense', use_sigmoid=False, loss_weight=0.0),
    ]
))
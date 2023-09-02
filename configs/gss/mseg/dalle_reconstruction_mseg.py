_base_ = [
    './gss-ff_swin-l_512x512_160k_mseg.py'
]

model=dict(
    pretrained=None,
    backbone=dict(
        _delete_=True,
        type='NoneBackbone'
    ),
    decode_head=dict(
        reconstruction_eval=True
        # add your own color list here
        # ------- begin -------
        # palette=[[10, 22, 26], [26, 12, 47], ..., [222, 220, 182]]
        # ------- end ---------
    ),
    test_cfg=dict(_delete_=True, mode='whole')
)
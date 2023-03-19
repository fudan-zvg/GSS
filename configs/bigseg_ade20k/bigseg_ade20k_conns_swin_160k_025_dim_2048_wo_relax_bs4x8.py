_base_=['bigseg_ade20k_conns_swin_160k_025_dim_2048.py']
runner = dict(max_iters=80000)
optimizer = dict(lr=0.00006 * 1.5)
data = dict(samples_per_gpu=3)
model=dict(
    backbone=dict(
        with_cp=False,
        window_size=7
    ),
    decode_head=dict(type='BigSegAggHeadHungerCPM'))
_base_=['bigseg_ade20k_conns_swin_160k_025_dim_2048.py']
model=dict(
    decode_head=dict(
        type='BigSegAggHeadWoCPM',
        num_classes=150))
runner = dict(max_iters=80000)
optimizer = dict(lr=0.00006 * 2)
data = dict(samples_per_gpu=4)
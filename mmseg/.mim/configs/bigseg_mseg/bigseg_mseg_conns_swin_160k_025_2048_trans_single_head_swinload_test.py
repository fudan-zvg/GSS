_base_=['bigseg_mseg_conns_swin_160k_025_2048_trans.py']

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='/home/chenjiaqi/pj/unilm/beit/swin_test.pth')),
    decode_head=dict(
        type='BigSegAggHeadWoCPMTransformerSingleFusion',
))
checkpoint_config = dict(by_epoch=False, interval=8000)
evaluation = dict(interval=40000, metric='mIoU', pre_eval=True)
# runner = dict(max_iters=160000)
data = dict(samples_per_gpu=2)
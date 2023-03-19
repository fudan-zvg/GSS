_base_=['../bigseg_ade20k/bigseg_ade20k_conns_swin_160k_025_dim_2048_wo_cpm_bs4x8.py']

model = dict(
    backbone=dict(
        _delete_=True,
        type='NoneBackbone'),
    decode_head=dict(
    type='BigSegAggHeadWoCPMTransformerTaming',
    indice_seg_channel=512,
    indice_cls_channel=2048,
))
# checkpoint_config = dict(by_epoch=False, interval=16000)
evaluation = dict(interval=16001, metric='mIoU', pre_eval=True)
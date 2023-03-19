_base_=['bigseg_cityscapes_conns_swin_160k_025_dim_2048_wo_cpm_bs4x8.py']

PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
           [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
           [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
           [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
           [0, 80, 100], [0, 0, 230], [119, 11, 32], [0, 0, 0]]

model = dict(decode_head=dict(
    type='BigSegAggHeadWoCPMTransformer',
    indice_seg_channel=512,
    indice_cls_channel=2048,
    palette=PALETTE,
    num_classes=19
))

# checkpoint_config = dict(by_epoch=False, interval=16000)
evaluation = dict(interval=16001, metric='mIoU', pre_eval=True)
_base_=['bigseg_cityscapes_conns_swin_160k_025_dim_768_2048_wo_cpm_bs4x8_transformer.py']

PALETTE = [[222, 222, 145], [18, 30, 7], [8, 23, 47], [30, 6, 96], [1, 13, 164], [12, 28, 191], [25, 52, 32], [29, 48, 52], [15, 51, 95], [25, 56, 167], [25, 42, 210], [27, 81, 31], [9, 88, 54], [27, 92, 113], [11, 99, 151], [26, 110, 183], [24, 130, 26], [4, 122, 75], [3, 132, 98], [26, 147, 167]]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2
)

model=dict(
    type='DalleDecoderLoadOnly',
    num_classes=19,
    palette=PALETTE,
    load_dir = 'work_dirs/bigseg_cityscapes_conns_swin_160k_025_dim_768_2048_wo_cpm_bs2x8_transformer/indice_iter_112000/val/'
    # backbone=dict(
    #     _delete_=True,
    #     type='ExampleBackbone'
    # ),
    # decode_head=dict(
        # type='ExampleDecodeHead')
)
# checkpoint_config = dict(by_epoch=False, interval=16000)
# evaluation = dict(interval=16001, metric='mIoU', pre_eval=True)
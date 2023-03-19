_base_=['bigseg_cityscapes_conns_swin_160k_025_dim_768_2048_wo_cpm_bs4x8_transformer.py']

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2
)
model=dict(decode_head=dict(type='BigSegAggHeadWoCPMTransformerSave',save_path = 'work_dirs/bigseg_cityscapes_conns_swin_160k_025_dim_768_2048_wo_cpm_bs2x16_transformer_right_color_lr/indice_iter_160000/'))
# checkpoint_config = dict(by_epoch=False, interval=16000)
# evaluation = dict(interval=16001, metric='mIoU', pre_eval=True)
_base_=['bigseg_cityscapes_conns_swin_80k_025_dim_768_2048_wo_cpm_bs2x8_transformer_right_color_lr.py']

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2
)
model=dict(
    decode_head=dict(
        type='BigSegAggHeadWoCPMTransformerSave',
        save_path = 'work_dirs/bigseg_cityscapes_conns_swin_80k_025_dim_768_2048_wo_cpm_bs2x8_transformer_right_color_lr/indice_iter_80000/'))
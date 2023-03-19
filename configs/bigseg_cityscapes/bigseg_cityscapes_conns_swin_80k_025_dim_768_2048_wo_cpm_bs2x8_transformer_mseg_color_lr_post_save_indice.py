_base_=['bigseg_cityscapes_conns_swin_80k_025_dim_768_2048_wo_cpm_bs2x8_transformer_mseg_color_lr_post.py']

PALETTE = [[125, 167, 9], [119, 169, 72], [55, 15, 50], [223, 217, 173], [165, 154, 206], [176, 154, 172], [167, 109, 212], [166, 115, 169], [234, 107, 138], [115, 161, 137], [167, 154, 147], [171, 62, 107], [169, 57, 178], [229, 113, 209], [222, 156, 146], [227, 162, 83], [227, 164, 112], [231, 168, 17], [230, 109, 177], [0, 0, 0]]

model=dict(
    decode_head=dict(
        type='BigSegAggHeadWoCPMTransformerSave',
        save_path = 'work_dirs/bigseg_cityscapes_conns_swin_80k_025_dim_768_2048_wo_cpm_bs2x8_transformer_mseg_color_lr/indice_iter_80000/'))

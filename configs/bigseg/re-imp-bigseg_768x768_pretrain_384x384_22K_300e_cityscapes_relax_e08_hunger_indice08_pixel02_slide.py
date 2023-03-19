_base_ = 're-imp-bigseg_768x768_pretrain_384x384_22K_300e_cityscapes_relax_e08_hunger_indice08_pixel02.py'
model=dict(test_cfg=dict(mode='slide', crop_size=(769, 769), stride=(513, 513)))


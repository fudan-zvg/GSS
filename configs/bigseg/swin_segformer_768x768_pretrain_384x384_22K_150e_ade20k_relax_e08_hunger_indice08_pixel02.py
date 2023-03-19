_base_ = ['swin_segformer_768x768_pretrain_384x384_22K_300e_ade20k_relax_e08_hunger_indice08_pixel02.py']
runner = dict(type='EpochBasedRunner', max_epochs=150)
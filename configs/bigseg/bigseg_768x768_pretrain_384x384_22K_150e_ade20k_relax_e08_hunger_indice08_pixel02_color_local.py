_base_ = ['bigseg_768x768_pretrain_384x384_22K_150e_ade20k_relax_e08_hunger_indice08_pixel02_color.py']

model = dict(
    decode_head = dict(
        type='BigSegAggHeadRelaxE08HungerLocal'
    )
)
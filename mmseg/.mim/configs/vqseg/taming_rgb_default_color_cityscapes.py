_base_=['mask_vqseg_agg_swin_large_patch4_window12_768x768_pretrain_384x384_22K_300e_cityscapes.py']
model=dict(decode_head=dict(d_vae_type='taming', task_type='recon'))
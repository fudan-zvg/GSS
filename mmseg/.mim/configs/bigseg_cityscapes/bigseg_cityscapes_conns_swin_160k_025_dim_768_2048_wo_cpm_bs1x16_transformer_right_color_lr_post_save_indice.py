_base_=['bigseg_cityscapes_conns_swin_160k_025_dim_768_2048_wo_cpm_bs1x16_transformer_right_color_lr_post.py']

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2
)
model=dict(decode_head=dict(type='PostSwinBigSegAggHeadWoCPMTransformerSave',save_path = 'work_dirs/bigseg_cityscapes_conns_swin_160k_025_dim_768_2048_wo_cpm_bs1x16_transformer_right_color_lr_post/indice_iter_40000/'))
# checkpoint_config = dict(by_epoch=False, interval=16000)
# evaluation = dict(interval=16001, metric='mIoU', pre_eval=True)
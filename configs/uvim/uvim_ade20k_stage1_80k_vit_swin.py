_base_ = ['uvim_ade20k_stage1_40k_vit_swin.py']


runner = dict(type='IterBasedRunner', max_iters=80000)
checkpoint_config = dict(by_epoch=False, interval=4000)
evaluation = dict(interval=80000, metric='mIoU', pre_eval=True)
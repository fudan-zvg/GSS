import torch
root_dir = '/home/chenjiaqi/pj/mmsegmentation/'
# ade20k dir (Swin, Swin)
# model_dir = 'work_dirs/bigseg_ade20k_conns_swin_160k_025_dim_768_2048_wo_cpm_bs4x8_transformer/iter_160000.pth'
# color2cls_dir = 'work_dirs/bigseg_ade20k_conns_swin_160k_025_dim_768_2048_wo_cpm_bs4x8_transformer_post/iter_40000.pth'
# target_dir = 'work_dirs/bigseg_ade20k_conns_swin_160k_025_dim_768_2048_wo_cpm_bs4x8_transformer_post_composite/swin_160k_post_swin_40k_from_32k.pth'

# ade20k dir (ResNet, Swin)
# model_dir = 'work_dirs/bigseg_ade20k_conns_res101_160k_025_dim_768_2048_wo_cpm_bs4x8_transformer_plus/iter_160000.pth'
# color2cls_dir = 'work_dirs/bigseg_ade20k_conns_swin_160k_025_dim_768_2048_wo_cpm_bs4x8_transformer_post/iter_40000.pth'
# target_dir = 'work_dirs/bigseg_ade20k_conns_res101_160k_025_dim_768_2048_wo_cpm_bs4x8_transformer_plus_post/resnet_160k_post_swin_40k_from_32k.pth'

# cityscapes (Swin(160k), Swin)
# model_dir = 'work_dirs/bigseg_cityscapes_conns_swin_160k_025_dim_768_2048_wo_cpm_bs2x8_transformer_right_color_lr/iter_160000.pth'
# color2cls_dir = 'work_dirs/bigseg_cityscapes_conns_swin_160k_025_dim_768_2048_wo_cpm_bs2x8_transformer_right_color_lr_post/iter_40000.pth'
# target_dir = 'work_dirs/bigseg_cityscapes_conns_swin_160k_025_dim_768_2048_wo_cpm_bs2x8_transformer_right_color_lr_post/ckp.pth'

# cityscapes (Swin(80k), Swin)
# model_dir = 'work_dirs/bigseg_cityscapes_conns_swin_80k_025_dim_768_2048_wo_cpm_bs2x8_transformer_right_color_lr/iter_80000.pth'
# color2cls_dir = 'work_dirs/bigseg_cityscapes_conns_swin_160k_025_dim_768_2048_wo_cpm_bs2x8_transformer_right_color_lr_post/iter_40000.pth'
# target_dir = 'work_dirs/bigseg_cityscapes_conns_swin_80k_025_dim_768_2048_wo_cpm_bs2x8_transformer_right_color_lr_post/swin_80k_post_swin_40k_from_32k.pth'

# cityscapes (Swin(90k), Swin)
# model_dir = 'work_dirs/bigseg_cityscapes_conns_swin_90k_025_dim_768_2048_wo_cpm_bs2x8_transformer_right_color_lr/iter_90000.pth'
# color2cls_dir = 'work_dirs/bigseg_cityscapes_conns_swin_160k_025_dim_768_2048_wo_cpm_bs2x8_transformer_right_color_lr_post/iter_40000.pth'
# target_dir = 'work_dirs/bigseg_cityscapes_conns_swin_90k_025_dim_768_2048_wo_cpm_bs2x8_transformer_right_color_lr_post/swin_90k_post_swin_40k_from_32k.pth'

# cityscapes (Swin(80k), Swin Mseg-Maskige)
# model_dir = 'work_dirs/bigseg_cityscapes_conns_swin_80k_025_dim_768_2048_wo_cpm_bs2x8_transformer_mseg_color_lr/iter_80000.pth'
# color2cls_dir = 'work_dirs/bigseg_cityscapes_conns_swin_80k_025_dim_768_2048_wo_cpm_bs2x8_transformer_mseg_color_lr_post/iter_40000.pth'
# target_dir = 'work_dirs/bigseg_cityscapes_conns_swin_80k_025_dim_768_2048_wo_cpm_bs2x8_transformer_mseg_color_lr_post/swin_80k_post_swin_40k_from_32k.pth'

# cityscapes (ResNet, Swin)
# model_dir = 'work_dirs/bigseg_cityscapes_conns_res101_80k_025_dim_768_2048_wo_cpm_bs2x8_transformer_right_color_lr_plus/iter_80000.pth'
# color2cls_dir = 'work_dirs/bigseg_cityscapes_conns_swin_160k_025_dim_768_2048_wo_cpm_bs2x8_transformer_right_color_lr_post/iter_40000.pth'
# target_dir = 'work_dirs/bigseg_cityscapes_conns_res101_80k_025_dim_768_2048_wo_cpm_bs2x8_transformer_right_color_lr_plus_post/resnet_80k_post_swin_40k_from_32k.pth'

# cityscapes (ResNet, Swin)
model_dir = 'work_dirs/bigseg_cityscapes_conns_res101_90k_025_dim_768_2048_wo_cpm_bs2x8_transformer_right_color_lr_plus/iter_90000.pth'
color2cls_dir = 'work_dirs/bigseg_cityscapes_conns_swin_160k_025_dim_768_2048_wo_cpm_bs2x8_transformer_right_color_lr_post/iter_40000.pth'
target_dir = 'work_dirs/bigseg_cityscapes_conns_res101_90k_025_dim_768_2048_wo_cpm_bs2x8_transformer_right_color_lr_plus_post/resnet_90k_post_swin_40k_from_32k.pth'

# mseg (HRNet, Swin)
# model_dir = 'work_dirs/bigseg_mseg_conns_hrnet48_160k_025_2048_trans_single_head_transformer_plus/iter_160000.pth'
# color2cls_dir = 'work_dirs/bigseg_mseg_conns_swin_160k_025_2048_trans_single_head_post/iter_40000.pth'
# target_dir = 'work_dirs/bigseg_mseg_conns_hrnet48_160k_025_2048_trans_single_head_transformer_plus_post/hrnet_plus_160k_post_swin_160k.pth'

# mseg (Swin, Swin)
# model_dir = 'work_dirs/bigseg_mseg_conns_swin_160k_025_2048_trans_single_head/iter_160000.pth'
# color2cls_dir = 'work_dirs/bigseg_mseg_conns_swin_160k_025_2048_trans_single_head_post/iter_40000.pth'
# target_dir = 'work_dirs/bigseg_mseg_conns_swin_160k_025_2048_trans_single_head_post/swin_160k_post_swin_160k.pth'


model_ckp = torch.load(root_dir + model_dir)
color2cls_ckp = torch.load(root_dir + color2cls_dir)

composite_big_ckp = dict()
composite_big_ckp['meta'] = model_ckp['meta']
composite_big_ckp['optimizer'] = model_ckp['optimizer']
composite_big_ckp['state_dict'] = model_ckp

# for model=swin+GSS, post=Swin+GSS
# for key, value in color2cls_ckp['state_dict'].items():
#     if key in model_ckp['state_dict']:
#         composite_big_ckp['state_dict'][key] = model_ckp['state_dict'][key]
#         print(key, 'is from seg model checkpoint')
#     else:
#         composite_big_ckp['state_dict'][key] = color2cls_ckp['state_dict'][key]
#         print(key, 'is from color2cls model checkpoint')

# for model=hrnet(or ResNet)+GSS, post=Swin+GSS
for key, value in model_ckp['state_dict'].items():
    composite_big_ckp['state_dict'][key] = model_ckp['state_dict'][key]
    print('[Seg model] ', key)

for key , value in color2cls_ckp['state_dict'].items():
    if 'decode_head' in key and key not in model_ckp['state_dict'] and 'pixel' not in key:
        composite_big_ckp['state_dict'][key] = color2cls_ckp['state_dict'][key]
        print('[Post model] ', key)

torch.save(composite_big_ckp, root_dir + target_dir)

if __name__ == '__main__':
    pass

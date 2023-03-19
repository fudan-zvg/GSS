import torch
root_dir = '/home/chenjiaqi/pj/mmsegmentation/'
# ade20k
# stage_1_dir = 'work_dirs/uvim_ade20k_stage1_80k_vit_swin_clip/iter_80000.pth'
# stage_2_dir = 'work_dirs/uvim_ade20k_stage2_160k_vit_swin_clip/iter_160000.pth'
# target_dir = 'work_dirs/uvim_ade20k_stage2_160k_vit_swin_clip/ckp.pth'

# cityscapes
# stage_1_dir = 'work_dirs/uvim_cityscapes_stage1_80k_vit_swin_clip/iter_80000.pth'
# stage_2_dir = 'work_dirs/uvim_cityscapes_stage2_160k_vit_swin_clip/iter_144000.pth'
# target_dir = 'work_dirs/uvim_cityscapes_stage2_160k_vit_swin_clip/ckp_iter144k.pth'

# mseg uvim
# stage_1_dir = 'work_dirs/uvim_mseg_stage1_160k_vit_swin_clip/iter_160000.pth'
# stage_2_dir = 'work_dirs/uvim_mseg_stage2_160k_vit_swin_clip/iter_160000.pth'
# target_dir = 'work_dirs/uvim_mseg_stage2_160k_vit_swin_clip/ckp_iter160k_full.pth'

# mseg uvim corr
stage_1_dir = 'work_dirs/uvim_mseg_stage1_160k_vit_swin_clip/iter_160000.pth'
stage_2_dir = 'work_dirs/uvim_mseg_stage2_160k_vit_swin_clip_corr/iter_160000.pth'
target_dir = 'work_dirs/uvim_mseg_stage2_160k_vit_swin_clip_corr/ckp_iter160k_full.pth'

stage_1_ckp = torch.load(root_dir + stage_1_dir)
stage_2_ckp = torch.load(root_dir + stage_2_dir)

composite_big_ckp = dict()
composite_big_ckp['meta'] = stage_2_ckp['meta']
composite_big_ckp['optimizer'] = stage_2_ckp['optimizer']
composite_big_ckp['state_dict'] = stage_2_ckp['state_dict']

for key, value in stage_1_ckp['state_dict'].items():
    if 'decode_head.gt_embedding_steam_dec' in key \
            or 'decode_head.decoder' in key \
            or 'decode_head.upsample_layers' in key \
            or 'decode_head.img_embedding_steam_dec' in key \
            or 'decode_head.base_model' in key \
            or 'decode_head.conv_seg' in key \
            or 'decode_head.emb' in key:
                composite_big_ckp['state_dict'][key] = stage_1_ckp['state_dict'][key]
                print(key, 'is from stage 1 model checkpoint')
torch.save(composite_big_ckp, root_dir + target_dir)

if __name__ == '__main__':
    pass

import torch 
from mmseg.models.backbones import SwinTransformer
checkpoint_path = '/home/chenjiaqi/pj/unilm/beit/work_dirs/checkpoint-0.pth'

checkpoint = torch.load(checkpoint_path)
reconstructed_ckp = checkpoint['model']
reconstructed_ckp = {para_name.lstrip('swin').lstrip('.'):para for para_name,para in reconstructed_ckp.items() if 'swin' in para_name }
reconstructed_ckp.pop('mask_token')
# import pdb;pdb.set_trace()

torch.save(reconstructed_ckp,'/home/chenjiaqi/pj/mmsegmentation/ckp/swin_pretrain_by_beit.pth')
swin = SwinTransformer(
        embed_dims=192,
        pretrain_img_size=384,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        patch_norm=True)

swin.load_state_dict(reconstructed_ckp)

if __name__ == '__main__':
    pass

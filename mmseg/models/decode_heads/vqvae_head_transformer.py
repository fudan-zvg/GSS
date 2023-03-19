import torch
from torch import nn
import logging
from .nearest_embed_utils import NearestEmbed
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.ops import resize
from ..builder import build_loss
from ..losses import accuracy
import numpy as np
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from .modules import VisionTransformer
from .modules import SwinTransformer


@HEADS.register_module()
class VQVAEHeadTransformer(BaseDecodeHead):
    def __init__(self,
                 stage,
                 num_classes,
                 channels=256,
                 patch_size=16,
                 ignore_index=255,
                 oracle_depth=6,
                 embed_dims=768,
                 base_model_depth=12,
                 mlp_ratio=4,
                 num_heads=12,
                 dict_size=4096,
                 codeword_dim=768,
                 # code_len=256,
                 **kwargs):
        super(VQVAEHeadTransformer, self).__init__(channels=channels, in_channels=3, num_classes=num_classes,**kwargs)
        self.patch_size = patch_size
        self.stage = stage
        self.dict_size = dict_size
        self.ignore_index = ignore_index
        self.emb = NearestEmbed(dict_size, codeword_dim)
        # A ViT Encoder structure for to embed GT into guide-code
        self.guide_code_encoder = None # ADD AN VIT MODEL
        self.gt_embedding_steam_dec = nn.Conv2d(
            codeword_dim, embed_dims, kernel_size=1, stride=1)
        self.img_embedding_steam_dec = nn.Conv2d(
            3, embed_dims, kernel_size=patch_size, stride=patch_size)
        # self.language_model = None
        self.guide_code_head = nn.Conv2d(embed_dims, codeword_dim, kernel_size=1)
        self.oracle = VisionTransformer(
            img_size=768,
            patch_size=patch_size,
            in_channels=3 + self.num_classes + 1,
            embed_dims=embed_dims,
            num_layers=oracle_depth,
            num_heads=num_heads,
            pretrained=None,
            with_cls_token=False,
            init_cfg=None)
        self.base_model = VisionTransformer(
            img_size=768,
            patch_size=patch_size,
            in_channels=3,
            embed_dims=768,
            num_layers=base_model_depth,
            num_heads=num_heads,
            pretrained=None,
            with_cls_token=False,
            with_patch_embed=False,
            init_cfg=dict(type='Pretrained', checkpoint='ckp/jx_vit_base_p16_224-80ecf9dd.pth'),
        )
        self.upsample_layers = nn.Sequential(
            nn.ConvTranspose2d(embed_dims, channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(channels),
            nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(channels),
            nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(channels),
            nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1),
        )
        self.conv_seg = nn.Conv2d(channels, self.num_classes + 1, kernel_size=1)
        self.init_weights()

    def init_weights(self):
        for l in self.modules():
            if isinstance(l, nn.Linear) or isinstance(l, nn.Conv2d):
                l.weight.detach().normal_(0, 0.02)
                torch.fmod(l.weight, 0.04)
                nn.init.constant_(l.bias, 0)
        # self.encoder[-1].weight.detach().fill_(1 / 40)
        self.emb.weight.detach().normal_(0, 0.02)
        torch.fmod(self.emb.weight, 0.04)
        self.oracle.init_weights()
        self.base_model.init_weights()

    def encode(self, x):
        outs =  self.oracle(x)[-1]
        return self.guide_code_head(outs)

    def decode(self, x):
        outs = self.base_model(x)[-1]
        return self.upsample_layers(torch.tanh(outs))

    def sample(self, size):
        sample = torch.randn(size, self.codeword_dim, self.f,
                             self.f, requires_grad=False),
        if self.cuda():
            sample = sample.cuda()
        emb, _ = self.emb(sample)
        return self.decode(emb.view(size, self.codeword_dim, self.f, self.f)).cpu()

    def forward(self, img, gt_semantic_seg):
        gt_semantic_seg_logit = F.one_hot(
            gt_semantic_seg.to(torch.long), self.num_classes + 1).squeeze(1).permute(0, 3, 1, 2).to(torch.float)
        x = torch.cat([img, gt_semantic_seg_logit], dim=1)
        z_e = self.encode(x)
        self.f = z_e.shape[-1]
        z_q, argmin = self.emb(z_e, weight_sg=True)
        emb, _ = self.emb(z_e.detach())
        ctx_dec = self.img_embedding_steam_dec(img)
        z_q = self.gt_embedding_steam_dec(z_q)
        z_q = resize(z_q, size=ctx_dec.shape[-2:], mode='bilinear')
        z_q = z_q + ctx_dec
        z = self.decode(z_q)
        seg_logit = self.conv_seg(z)

        return seg_logit, z_e, emb, argmin

    def forward_train(self, inputs, img, img_metas, gt_semantic_seg, train_cfg):
        gt_semantic_seg[gt_semantic_seg == self.ignore_index] = self.num_classes
        seg_logit, z_e, emb, argmin = self.forward(img, gt_semantic_seg)
        return self.losses(seg_logit, gt_semantic_seg, z_e, emb, argmin)

    def forward_test(self, inputs, img, img_metas, gt_semantic_seg, test_cfg):
        gt_semantic_seg = gt_semantic_seg[0]
        gt_semantic_seg[gt_semantic_seg == self.ignore_index] = self.num_classes
        seg_logit, z_e, emb, argmin = self.forward(img, gt_semantic_seg)
        import ipdb
        ipdb.set_trace()
        from torchvision.utils import save_image, make_grid
        save_image(seg_logit.argmax(1) / 195.0, 'work_dirs/uvim_stage_1_mseg/' + img_metas[0]['ori_filename'].split('/')[-1].split('.')[0] + '_orcle.png')
        save_image(gt_semantic_seg / 195.0,
                   'work_dirs/uvim_stage_1_mseg/' + img_metas[0]['ori_filename'].split('/')[-1].split('.')[0] + '_gt_orcle.png')
        print('cjq debug save')
        return seg_logit

    def losses(self, seg_logit, seg_label, z_e, emb, argmin):
        """Compute segmentation loss."""
        loss = dict()
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)
        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                if loss_decode.loss_name == 'loss_ce':
                    loss[loss_decode.loss_name] = loss_decode(
                        seg_logit,
                        seg_label,
                        weight=seg_weight,
                        ignore_index=-1)
                elif loss_decode.loss_name == 'loss_vq':
                    loss[loss_decode.loss_name] = loss_decode(pred=emb, label=z_e)
                elif loss_decode.loss_name == 'loss_commit':
                    loss[loss_decode.loss_name] = loss_decode(pred=z_e, label=emb)
            else:
                if loss_decode.loss_name == 'loss_ce':
                    loss[loss_decode.loss_name] += loss_decode(
                        seg_logit,
                        seg_label,
                        weight=seg_weight,
                        ignore_index=-1)
                elif loss_decode.loss_name == 'loss_vq':
                    loss[loss_decode.loss_name] += loss_decode(pred=emb, label=z_e)
                elif loss_decode.loss_name == 'loss_commit':
                    loss[loss_decode.loss_name] += loss_decode(pred=z_e, label=emb)
        loss['acc_seg'] = accuracy(seg_logit, seg_label, ignore_index=self.ignore_index)
        # loss['unique'], loss['embedding_counts'] = np.unique(argmin, return_counts=True)

        return loss


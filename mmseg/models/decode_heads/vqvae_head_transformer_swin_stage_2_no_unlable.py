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
from .modules import NoneBackbone

@HEADS.register_module()
class VQVAEHeadTransformerSwinStage2NonUnlable(BaseDecodeHead):
    def __init__(self,
                 stage,
                 num_classes,
                 channels=256,
                 patch_size=16,
                 ignore_index=255,
                 oracle_depth=6,
                 base_model_depth=12,
                 mlp_dim=3072,
                 decoder_dim=192,
                 backbone_embed_dim=768,
                 num_heads=12,
                 dict_size=4096,
                 codeword_dim=768,
                 # code_len=256,
                 **kwargs):
        super(VQVAEHeadTransformerSwinStage2NonUnlable, self).__init__(channels=channels, in_channels=3, num_classes=num_classes,**kwargs)
        self.patch_size = patch_size
        self.stage = stage
        self.dict_size = dict_size
        self.ignore_index = ignore_index
        self.emb = NearestEmbed(dict_size, codeword_dim)
        # A ViT Encoder structure for to embed GT into guide-code
        self.guide_code_head = nn.Conv2d(mlp_dim, codeword_dim, kernel_size=1)
        self.gt_embedding_steam_dec = nn.Conv2d(
            codeword_dim, decoder_dim, kernel_size=1, stride=1)
        self.guide_code_head_for_lm = nn.Conv2d(backbone_embed_dim, self.dict_size, kernel_size=1)
        self.encoder = VisionTransformer(
            img_size=768,
            patch_size=patch_size,
            in_channels=3 + self.num_classes,
            embed_dims=mlp_dim,
            num_layers=oracle_depth,
            num_heads=num_heads,
            pretrained=None,
            with_cls_token=False,
            init_cfg=None)
        # self.decoder = NoneBackbone()
        self.decoder = SwinTransformer(
            pretrain_img_size=384,
            in_channels=3,
            embed_dims=decoder_dim,
            patch_size=4,
            window_size=12,
            mlp_ratio=4,
            depths=[2, 2, 18, 2],
            num_heads=[6, 12, 24, 48],
            strides=(4, 2, 2, 2),
            out_indices=(0, 1, 2, 3),
            qkv_bias=True,
            qk_scale=None,
            patch_norm=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.3,
            use_abs_pos_embed=False,
            act_cfg=dict(type='GELU'),
            norm_cfg=dict(type='LN', requires_grad=True),
            init_cfg=dict(
                type='Pretrained',
                checkpoint='/home/chenjiaqi/pj/mmsegmentation/ckp/swin_large_patch4_window12_384_22k_20220412-6580f57d.pth'),
            with_cp=False
        )
        self.upsample_layers = nn.Sequential(
            nn.ConvTranspose2d(1536, channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(channels),
            nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(channels),
            nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(channels),
            nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1),
        )
        self.conv_seg = nn.Conv2d(channels, self.num_classes, kernel_size=1)
        self.init_weights()

    def init_weights(self):
        for l in self.modules():
            if isinstance(l, nn.Linear) or isinstance(l, nn.Conv2d):
                l.weight.detach().normal_(0, 0.02)
                torch.fmod(l.weight, 0.04)
                # nn.init.constant_(l.bias, 0)
        self.emb.weight.detach().normal_(0, 0.02)
        torch.fmod(self.emb.weight, 0.04)
        self.encoder.init_weights()
        self.decoder.init_weights()

    def encode(self, x):
        outs =  self.encoder(x)[-1]
        return self.guide_code_head(outs)

    def decode(self, x, guide_code=None):
        if guide_code is not None:
            guide_code = self.gt_embedding_steam_dec(guide_code)
        outs = self.decoder(x, guide_code)[-1]
        return self.upsample_layers(torch.tanh(outs))

    def sample(self, size):
        sample = torch.randn(size, self.codeword_dim, self.f,
                             self.f, requires_grad=False),
        if self.cuda():
            sample = sample.cuda()
        emb, _ = self.emb(sample)
        return self.decode(emb.view(size, self.codeword_dim, self.f, self.f)).cpu()

    def forward(self, inputs):
        #import pdb;pdb.set_trace()
        z_logit = self.guide_code_head_for_lm(inputs[-1])
        return z_logit

    def forward_train(self, inputs, img, img_metas, gt_semantic_seg, train_cfg):
        z_logit = self.forward(inputs)
        with torch.no_grad():
            gt_semantic_seg[gt_semantic_seg == self.ignore_index] = self.num_classes-1
            gt_semantic_seg_logit = F.one_hot(
                gt_semantic_seg.to(torch.long), self.num_classes).squeeze(1).permute(0, 3, 1, 2).to(torch.float)
            x = torch.cat([img, gt_semantic_seg_logit], dim=1)
            z_e = self.encode(x)
            self.f = z_e.shape[-1]
            z_q, argmin = self.emb(z_e, weight_sg=True)
        return self.losses(z_logit, argmin.unsqueeze(1))

    def forward_test(self, inputs, img, img_metas, gt_semantic_seg, test_cfg):
        # return self.forward_test_recon(inputs, img, img_metas, gt_semantic_seg, test_cfg)
        z_logit = self.forward(inputs)
        z_indice = z_logit.argmax(1)
        z_q = self.emb.weight[:, z_indice].permute(1, 0, 2, 3)
        # emb, _ = self.emb(z_e.detach())
        z = self.decode(img, guide_code=z_q)
        seg_logit = self.conv_seg(z)
        return seg_logit

    def forward_test_recon(self, inputs, img, img_metas, gt_semantic_seg, test_cfg):
        # scale = 0.8
        gt_semantic_seg = gt_semantic_seg[0]
        # h, w = gt_semantic_seg.shape[-2:]
        # gt_semantic_seg[gt_semantic_seg == self.ignore_index] = self.num_classes
        # gt_semantic_seg = F.interpolate(
        #     F.one_hot(gt_semantic_seg.to(torch.long), self.num_classes + 1).squeeze(1).permute(0, 3, 1, 2).to(
        #         torch.float),
        #     size=(int(h * scale), int(w * scale)), mode='bilinear').argmax(1).unsqueeze(1)
        # gt_semantic_seg[gt_semantic_seg == self.num_classes] = self.ignore_index
        # img = resize(img, size=[int(h * scale), int(w * scale)], mode='bilinear')
        gt_semantic_seg[gt_semantic_seg == self.ignore_index] = self.num_classes-1
        gt_semantic_seg_logit = F.one_hot(gt_semantic_seg.to(torch.long), self.num_classes).squeeze(1).permute(0, 3, 1, 2).to(torch.float)
        x = torch.cat([img, gt_semantic_seg_logit], dim=1)
        z_e = self.encode(x)
        self.f = z_e.shape[-1]
        z_q, argmin = self.emb(z_e, weight_sg=True)
        # z_q = self.emb.weight[:, argmin].permute(1, 0, 2, 3)
        # z_q = torch.gather(self.emb.weight, 1, argmin)
        z = self.decode(img, guide_code=z_q)
        seg_logit = self.conv_seg(z)
        return seg_logit
    def losses(self, z_logit, z_label):
        """Compute segmentation loss."""
        loss = dict()
        z_logit = resize(
            input=z_logit,
            size=z_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(z_logit, z_label)
        else:
            seg_weight = None
        z_label = z_label.squeeze(1)
        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                if loss_decode.loss_name == 'loss_ce':
                    loss[loss_decode.loss_name] = loss_decode(
                        z_logit,
                        z_label,
                        weight=seg_weight,
                        ignore_index=-100)
            else:
                if loss_decode.loss_name == 'loss_ce':
                    loss[loss_decode.loss_name] += loss_decode(
                        z_logit,
                        z_label,
                        weight=seg_weight,
                        ignore_index=-100)
        loss['acc_seg'] = accuracy(z_logit, z_label, ignore_index=-100)
        # loss['unique'], loss['embedding_counts'] = np.unique(argmin, return_counts=True)

        return loss


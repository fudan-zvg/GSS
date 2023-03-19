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
from torchvision.utils import save_image, make_grid


@HEADS.register_module()
class VQVAEHeadTransformerSwin(BaseDecodeHead):
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
                 num_heads=12,
                 dict_size=4096,
                 codeword_dim=768,
                 # code_len=256,
                 **kwargs):
        super(VQVAEHeadTransformerSwin, self).__init__(channels=channels, in_channels=3, num_classes=num_classes,**kwargs)
        self.patch_size = patch_size
        self.stage = stage
        self.dict_size = dict_size
        self.ignore_index = ignore_index
        self.emb = NearestEmbed(dict_size, codeword_dim)
        # A ViT Encoder structure for to embed GT into guiding-code
        # self.language_model = None
        self.guide_code_head = nn.Conv2d(mlp_dim, codeword_dim, kernel_size=1)
        self.gt_embedding_steam_dec = nn.Conv2d(
            codeword_dim, decoder_dim, kernel_size=1, stride=1)
        self.encoder = VisionTransformer(
            img_size=768,
            patch_size=patch_size,
            in_channels=3 + self.num_classes + 1,
            embed_dims=mlp_dim,
            num_layers=oracle_depth,
            num_heads=num_heads,
            pretrained=None,
            with_cls_token=False,
            init_cfg=None)
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
        self.conv_seg = nn.Conv2d(channels, self.num_classes + 1, kernel_size=1)
        self.init_weights()
        # self.visual_palette = torch.tensor([[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
        #                                     [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
        #                                     [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
        #                                     [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
        #                                     [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
        #                                     [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
        #                                     [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
        #                                     [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
        #                                     [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
        #                                     [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
        #                                     [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
        #                                     [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
        #                                     [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
        #                                     [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
        #                                     [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
        #                                     [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
        #                                     [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
        #                                     [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
        #                                     [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
        #                                     [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
        #                                     [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
        #                                     [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
        #                                     [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
        #                                     [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
        #                                     [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
        #                                     [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
        #                                     [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
        #                                     [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
        #                                     [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
        #                                     [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
        #                                     [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
        #                                     [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
        #                                     [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
        #                                     [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
        #                                     [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
        #                                     [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
        #                                     [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
        #                                     [102, 255, 0], [92, 0, 255], [0, 0, 0]])

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

    def forward(self, img, gt_semantic_seg):
        gt_semantic_seg_logit = F.one_hot(gt_semantic_seg.to(torch.long), self.num_classes + 1).squeeze(1).permute(0, 3, 1, 2).to(torch.float)
        x = torch.cat([img, gt_semantic_seg_logit], dim=1)
        z_e = self.encode(x)
        self.f = z_e.shape[-1]

        z_q, argmin = self.emb(z_e, weight_sg=True)
        emb, _ = self.emb(z_e.detach())
        z = self.decode(img, guide_code=z_q)
        seg_logit = self.conv_seg(z)
        return seg_logit, z_e, emb, argmin

    def forward_train(self, inputs, img, img_metas, gt_semantic_seg, train_cfg):
        gt_semantic_seg[gt_semantic_seg == self.ignore_index] = self.num_classes
        seg_logit, z_e, emb, argmin = self.forward(img, gt_semantic_seg)
        return self.losses(seg_logit, gt_semantic_seg, z_e, emb, argmin)

    def forward_test(self, inputs, img, img_metas, gt_semantic_seg, test_cfg):
        gt_semantic_seg = gt_semantic_seg[0]
        gt_semantic_seg[gt_semantic_seg == self.ignore_index] = self.num_classes
        # scale = 0.5
        # h, w = gt_semantic_seg.shape[-2:]
        # gt_semantic_seg[gt_semantic_seg == self.ignore_index] = self.num_classes
        # gt_semantic_seg = F.interpolate(
        #     F.one_hot(gt_semantic_seg.to(torch.long), self.num_classes + 1).squeeze(1).permute(0, 3, 1, 2).to(
        #         torch.float),
        #     size=(int(h * scale), int(w * scale)), mode='bilinear').argmax(1).unsqueeze(1)
        # gt_semantic_seg[gt_semantic_seg == self.num_classes] = self.ignore_index
        # img = resize(img, size=[int(h * scale), int(w * scale)], mode='bilinear')
        seg_logit, z_e, emb, argmin = self.forward(img, gt_semantic_seg)
        # save_image(self.encode_to_segmap_visual(seg_logit.argmax(1).unsqueeze(1).long()) / 255.0,
        #            'work_dirs/visualization_stage_1/uvim/' +
        #            img_metas[0]['ori_filename'].split('/')[-1].split('.')[0] + '_pred.png')
        # print('cjq save uvim')
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
                        ignore_index=-100)
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
                        ignore_index=-100)
                elif loss_decode.loss_name == 'loss_vq':
                    loss[loss_decode.loss_name] += loss_decode(pred=emb, label=z_e)
                elif loss_decode.loss_name == 'loss_commit':
                    loss[loss_decode.loss_name] += loss_decode(pred=z_e, label=emb)
        loss['acc_seg'] = accuracy(seg_logit, seg_label, ignore_index=-100)
        # loss['unique'], loss['embedding_counts'] = np.unique(argmin, return_counts=True)

        return loss

    def encode_to_segmap_visual(self, indice):
        PALETTE_ = self.visual_palette.clone().to(indice.device)
        _indice = indice.clone().detach()
        _indice[_indice > self.num_classes] = self.num_classes
        return PALETTE_[_indice.long()].squeeze(1).permute(0, 3, 1, 2)


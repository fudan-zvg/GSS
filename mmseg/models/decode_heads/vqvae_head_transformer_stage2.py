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
class VQVAEHeadTransformerStage2(BaseDecodeHead):
    def __init__(self,
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
                 backbone_embed_dim=1536,
                 # code_len=256,
                 **kwargs):
        super(VQVAEHeadTransformerStage2, self).__init__(channels=channels, in_channels=3, num_classes=num_classes,**kwargs)
        self.patch_size = patch_size
        self.dict_size = dict_size
        self.ignore_index = ignore_index
        self.emb = NearestEmbed(dict_size, codeword_dim)
        # A ViT Encoder structure for to embed GT into guide-code
        self.guide_code_encoder = None # ADD AN VIT MODEL
        self.gt_embedding_steam_dec = nn.Conv2d(
            codeword_dim, embed_dims, kernel_size=1, stride=1)
        self.img_embedding_steam_dec = nn.Conv2d(
            3, embed_dims, kernel_size=patch_size, stride=patch_size)
        self.guide_code_head_for_lm = nn.Conv2d(backbone_embed_dim, self.dict_size, kernel_size=1)
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
            # init_cfg=dict(type='Pretrained', checkpoint='ckp/jx_vit_base_p16_224-80ecf9dd.pth'),
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
        self.conv_seg = nn.Conv2d(channels, self.num_classes + 1, kernel_size=1) # I have added 1 for num_classes in config
        self.init_weights()
        self.palette = torch.tensor([[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
               [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
               [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
               [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
               [0, 80, 100], [0, 0, 230], [119, 11, 32]])

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

    def forward(self, inputs):
        z_logit = self.guide_code_head_for_lm(inputs[-1])
        return z_logit

    # def forward(self, img, gt_semantic_seg):
    #     gt_semantic_seg_logit = F.one_hot(
    #         gt_semantic_seg.to(torch.long), self.num_classes + 1).squeeze(1).permute(0, 3, 1, 2).to(torch.float)
    #     x = torch.cat([img, gt_semantic_seg_logit], dim=1)
    #     z_e = self.encode(x)
    #     self.f = z_e.shape[-1]
    #     z_q, argmin = self.emb(z_e, weight_sg=True)
    #     emb, _ = self.emb(z_e.detach())
    #     ctx_dec = self.img_embedding_steam_dec(img)
    #     z_q = self.gt_embedding_steam_dec(z_q)
    #     z_q = resize(z_q, size=ctx_dec.shape[-2:], mode='bilinear')
    #     z_q = z_q + ctx_dec
    #     z = self.decode(z_q)
    #     seg_logit = self.conv_seg(z)
    #     return seg_logit, z_e, emb, argmin

    def forward_train(self, inputs, img, img_metas, gt_semantic_seg, train_cfg):
        z_logit = self.forward(inputs)
        with torch.no_grad():
            gt_semantic_seg[gt_semantic_seg == self.ignore_index] = self.num_classes
            gt_semantic_seg_logit = F.one_hot(
                gt_semantic_seg.to(torch.long), self.num_classes + 1).squeeze(1).permute(0, 3, 1, 2).to(torch.float)
            x = torch.cat([img, gt_semantic_seg_logit], dim=1)
            z_e = self.encode(x)
            self.f = z_e.shape[-1]
            z_q, argmin = self.emb(z_e, weight_sg=True)
        return self.losses(z_logit, argmin.unsqueeze(1))

    # def forward_train(self, inputs, img, img_metas, gt_semantic_seg, train_cfg):
    #     gt_semantic_seg[gt_semantic_seg == self.ignore_index] = self.num_classes
    #     seg_logit, z_e, emb, argmin = self.forward(img, gt_semantic_seg)
    #     return self.losses(seg_logit, gt_semantic_seg, z_e, emb, argmin)

    def forward_test(self, inputs, img, img_metas, gt_semantic_seg, test_cfg):
        # return self.forward_test_recon(inputs, img, img_metas, gt_semantic_seg, test_cfg)
        z_logit = self.forward(inputs)
        z_indice = z_logit.argmax(1)
        z_q = self.emb.weight[:, z_indice].permute(1, 0, 2, 3)
        # emb, _ = self.emb(z_e.detach())
        ctx_dec = self.img_embedding_steam_dec(img)
        z_q = self.gt_embedding_steam_dec(z_q)
        z_q = resize(z_q, size=ctx_dec.shape[-2:], mode='bilinear')
        z_q = z_q + ctx_dec
        z = self.decode(z_q)
        seg_logit = self.conv_seg(z)[:,:self.num_classes,:,:]
        # we have to output an semantic prediction to run the framework
        # save_image(seg_logit.argmax(1) / 195.0, 'work_dirs/uvim_stage_2_mseg_corr/' + img_metas[0]['ori_filename'].split('/')[-1].split('.')[0] + '_orcle.png')
        # save_image(gt_semantic_seg[0].unsqueeze(0) / 195.0,
        #            'work_dirs/uvim_stage_2_mseg_corr/' + img_metas[0]['ori_filename'].split('/')[-1].split('.')[0] + '_gt_orcle.png')
        # print('cjq debug save')
        # vq_indices_show = F.interpolate(vq_indices.float(), size=seg_logist.shape[-2:], mode='bilinear')
        # gt_semantic_seg[0] = gt_semantic_seg[0].unsqueeze(0)
        # gt_semantic_seg[0] = F.interpolate(gt_semantic_seg[0].float(), size=seg_logist.shape[-2:], mode='bilinear')
        #
        # error = torch.abs(gt_semantic_seg[0] - seg_logist.argmax(1).unsqueeze(1))
        # error[gt_semantic_seg[0] >= self.num_classes] = 0
        # import ipdb
        # ipdb.set_trace()

        return seg_logit

    # def forward_test(self, inputs, img, img_metas, gt_semantic_seg, test_cfg):
    #     gt_semantic_seg = gt_semantic_seg[0]
    #     gt_semantic_seg[gt_semantic_seg == self.ignore_index] = self.num_classes
    #     seg_logit, z_e, emb, argmin = self.forward(img, gt_semantic_seg)
    #     return seg_logit

    def losses(self, seg_logit, seg_label):
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
            else:
                if loss_decode.loss_name == 'loss_ce':
                    loss[loss_decode.loss_name] += loss_decode(
                        seg_logit,
                        seg_label,
                        weight=seg_weight,
                        ignore_index=-1)
        loss['acc_seg'] = accuracy(seg_logit, seg_label, ignore_index=self.ignore_index)
        # loss['unique'], loss['embedding_counts'] = np.unique(argmin, return_counts=True)

        return loss

    def encode_to_segmap(self, indice):
        PALETTE_ = self.palette.clone().to(indice.device)
        _indice = indice.clone().detach()
        _indice[_indice > self.num_classes] = self.num_classes
        return PALETTE_[_indice.long()].squeeze(1).permute(0, 3, 1, 2)


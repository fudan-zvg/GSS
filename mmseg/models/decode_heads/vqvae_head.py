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

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, bn=False):
        super(ResBlock, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels

        layers = [
            nn.ReLU(),
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels,
                      kernel_size=1, stride=1, padding=0)
        ]
        if bn:
            layers.insert(2, nn.BatchNorm2d(out_channels))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.convs(x)

@HEADS.register_module()
class VQVAEHead(BaseDecodeHead):
    def __init__(self,
                 stage,
                 channels,
                 num_classes,
                 num_vocab=10,
                 patch_size=16,
                 ignore_index=255,
                 loss_config=dict(
                     vq_coef=1,
                     commit_coef=0.5),
                 **kwargs):
        super(VQVAEHead, self).__init__(channels=channels, in_channels=3, num_classes=num_classes,**kwargs)
        self.patch_size = patch_size
        self.stage = stage
        self.channels = channels
        self.num_vocab = num_vocab
        self.ignore_index = ignore_index
        # self.mse + self.vq_coef*self.vq_loss + self.commit_coef*self.commit_loss
        self.vq_coef = loss_config['vq_coef']
        self.commit_coef = loss_config['commit_coef']
        self.emb = NearestEmbed(num_vocab, channels)
        # A ViT Encoder structure for to embed GT into guide-code
        self.guide_code_encoder = None # ADD AN VIT MODEL
        self.gt_embedding_steam_enc = nn.Conv2d(
            num_classes + 1, self.channels, kernel_size=self.patch_size, stride=self.patch_size)
        self.img_embedding_steam_enc = nn.Conv2d(
            3, self.channels, kernel_size=self.patch_size, stride=self.patch_size)
        self.img_embedding_steam_dec = nn.Conv2d(
            3, self.channels, kernel_size=self.patch_size, stride=self.patch_size)
        self.language_model = None
        if self.stage == 2:
            self.guide_code_head_for_lm = nn.Conv2d(self.channels, self.num_vocab, kernel_size=1)
        self.encoder = nn.Sequential(
            ResBlock(channels, channels, bn=True),
            nn.BatchNorm2d(channels),
            ResBlock(channels, channels, bn=True),
            nn.BatchNorm2d(channels),
            ResBlock(channels, channels, bn=True),
            nn.BatchNorm2d(channels),
            ResBlock(channels, channels, bn=True),
            nn.BatchNorm2d(channels),
            ResBlock(channels, channels, bn=True),
            nn.BatchNorm2d(channels),
            ResBlock(channels, channels, bn=True),
            nn.BatchNorm2d(channels),
            ResBlock(channels, channels, bn=True),
            nn.BatchNorm2d(channels),
            ResBlock(channels, channels, bn=True),
            nn.BatchNorm2d(channels),

        )
        self.decoder = nn.Sequential(
            # 1
            ResBlock(channels, channels),
            nn.BatchNorm2d(channels),
            # 2
            ResBlock(channels, channels),
            nn.BatchNorm2d(channels),
            # 3
            nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(channels),
            ResBlock(channels, channels, bn=True),
            nn.BatchNorm2d(channels),
            # 4
            nn.ConvTranspose2d(
                channels, channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(channels),
            ResBlock(channels, channels, bn=True),
            nn.BatchNorm2d(channels),
            nn.ConvTranspose2d(
                channels, channels, kernel_size=4, stride=2, padding=1),
        )
        self.conv_seg = nn.Conv2d(channels, self.num_classes, kernel_size=1)
        self.init_weights()

    def init_weights(self):
        for l in self.modules():
            if isinstance(l, nn.Linear) or isinstance(l, nn.Conv2d):
                l.weight.detach().normal_(0, 0.02)
                torch.fmod(l.weight, 0.04)
                nn.init.constant_(l.bias, 0)
        self.encoder[-1].weight.detach().fill_(1 / 40)
        self.emb.weight.detach().normal_(0, 0.02)
        torch.fmod(self.emb.weight, 0.04)
    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return torch.tanh(self.decoder(x))

    def sample(self, size):
        sample = torch.randn(size, self.channels, self.f,
                             self.f, requires_grad=False),
        if self.cuda():
            sample = sample.cuda()
        emb, _ = self.emb(sample)
        return self.decode(emb.view(size, self.channels, self.f, self.f)).cpu()

    def forward(self, img, gt_semantic_seg):
        gt_semantic_seg[gt_semantic_seg == self.ignore_index] = self.num_classes
        gt_semantic_seg_logit = F.one_hot(gt_semantic_seg.to(torch.long), self.num_classes + 1).squeeze(1).permute(0, 3, 1, 2).to(torch.float)
        ctx_enc = self.img_embedding_steam_enc(img)
        x = self.gt_embedding_steam_enc(gt_semantic_seg_logit)
        x = x + ctx_enc
        z_e = self.encode(x)
        self.f = z_e.shape[-1]
        z_q, argmin = self.emb(z_e, weight_sg=True)
        emb, _ = self.emb(z_e.detach())
        ctx_dec = self.img_embedding_steam_dec(img)
        z_q = z_q + ctx_dec
        z = self.decode(z_q)
        seg_logit = self.conv_seg(z)
        return seg_logit, z_e, emb, argmin

    def forward_train(self, inputs, img, img_metas, gt_semantic_seg, train_cfg):
        seg_logit, z_e, emb, argmin = self.forward(img, gt_semantic_seg)
        return self.losses(seg_logit, gt_semantic_seg, z_e, emb, argmin)

    def forward_test(self, inputs, img, img_metas, gt_semantic_seg, test_cfg):
        gt_semantic_seg = gt_semantic_seg[0]
        seg_logit, z_e, emb, argmin = self.forward(img, gt_semantic_seg)
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
                        ignore_index=self.ignore_index)
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
                        ignore_index=self.ignore_index)
                elif loss_decode.loss_name == 'loss_vq':
                    loss[loss_decode.loss_name] += loss_decode(pred=emb, label=z_e)
                elif loss_decode.loss_name == 'loss_commit':
                    loss[loss_decode.loss_name] += loss_decode(pred=z_e, label=emb)
        loss['acc_seg'] = accuracy(seg_logit, seg_label, ignore_index=self.ignore_index)
        # loss['unique'], loss['embedding_counts'] = np.unique(argmin, return_counts=True)

        return loss


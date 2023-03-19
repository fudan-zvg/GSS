import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
import scipy.io as sio
import mmcv
import os
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from torchvision.utils import save_image, make_grid
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from ..utils.dalle_d_vae import get_dalle_vae, map_pixels, unmap_pixels
from ..losses import accuracy
import torch
from mmcv.cnn import ConvModule

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.ops import resize

@HEADS.register_module()
class PostBigSegAggHeadRelaxE08HungerConv3_5x5(BaseDecodeHead):
    """
    Args:
        norm_layer (dict): Config dict for input normalization.
            Default: norm_layer=dict(type='LN', eps=1e-6, requires_grad=True).
        num_convs (int): Number of decoder convolutions. Default: 1.
        up_scale (int): The scale factor of interpolate. Default:4.
        kernel_size (int): The kernel size of convolution when decoding
            feature information from backbone. Default: 3.
        init_cfg (dict | list[dict] | None): Initialization config dict.
            Default: dict(
                     type='Constant', val=1.0, bias=0, layer='LayerNorm').
    """

    def __init__(self,
                 channels,
                 img_size,
                 init_cfg=[
                     dict(type='Constant', val=1.0, bias=0, layer='LayerNorm'),
                     dict(
                         type='Normal',
                         std=0.01,
                         override=dict(name='conv_seg'))],
                 norm_layer=dict(type='LN', eps=1e-6, requires_grad=True),
                 interpolate_mode='bilinear',
                 palette=None,
                 **kwargs):
        super(PostBigSegAggHeadRelaxE08HungerConv3_5x5, self).__init__(init_cfg=init_cfg, input_transform='multiple_select', channels=channels, **kwargs)
        num_inputs = len(self.in_channels)
        assert num_inputs == len(self.in_index)
        self.channels = channels
        self.palette = torch.tensor(palette)
        self.interpolate_mode = interpolate_mode
        self.img_size = img_size
        self.vocab_size = 8192
        self.indice_ignore_index = self.vocab_size
        self.pixel_ignore_index = 255
        self.ignore_index = self.pixel_ignore_index
        _, self.norm = build_norm_layer(norm_layer, self.channels)

        # dense classificator
        self.conv_seg = nn.Conv2d(channels, self.vocab_size, kernel_size=1)
        self.conv_seg_pixel = nn.Conv2d(channels, self.num_classes, kernel_size=1)

        # input translation
        self.convs = nn.ModuleList()
        self.convs_pixel = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
            self.convs_pixel.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        self.cls_segmap = nn.Sequential(
            ConvModule(
                in_channels=3,
                out_channels=128,
                kernel_size=5,
                stride=1,
                padding=2,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            ConvModule(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            ConvModule(
                in_channels=128,
                out_channels=self.num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
                norm_cfg=None,
                act_cfg=None)
        )

        # fusion blocks
        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)
        self.fusion_conv_pixel = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)
        self.d_vae = get_dalle_vae(
            weight_path="ckp",
            device="cuda")

    def forward(self, x):
        out = self.cls_seg(x)
        return out

    def get_gt_vq_indices(self, gt_semantic_seg):
        gt_segmap = map_pixels(self.encode_to_segmap(gt_semantic_seg) / 255.0)
        return self.d_vae.get_codebook_indices(gt_segmap)

    def feature_aggregation(self, inputs):
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        out = self.fusion_conv(torch.cat(outs, dim=1))
        return out

    def feature_aggregation_for_pixel(self, inputs):
        outs_for_pixel = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs_pixel[idx]
            outs_for_pixel.append(
                resize(
                    input=conv(x),
                    size=inputs[1].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))
        out_for_pixel = self.fusion_conv_pixel(torch.cat(outs_for_pixel, dim=1))
        return out_for_pixel

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        # get vq indices from gt by dalle
        with torch.no_grad():
            inputs = self._transform_inputs(inputs)
            x = self.feature_aggregation(inputs)
            h, w = x.shape[-2:]
            vq_logits = self.forward(x).view(-1, self.vocab_size, h, w)
            # get the pixel-wise prediction from indice prediction
            pixel_segmap_from_indice_pred = self.d_vae.decode(vq_logits.argmax(1).unsqueeze(1), img_size=[h, w])
            pixel_segmap_from_indice_pred = unmap_pixels(torch.sigmoid(pixel_segmap_from_indice_pred[:, :3]))
            # pixel_prob_from_indice_pred = self.get_cls_prob_from_segmap(torch.tensor(pixel_segmap_from_indice_pred), keep_ignore_index=False)
        logits = self.cls_segmap(pixel_segmap_from_indice_pred)
        losses = self.losses(logits, gt_semantic_seg)
        return losses
        # with torch.no_grad():
        #     input_segmap = map_pixels(self.encode_to_segmap(gt_semantic_seg) / 255.0)
        #     input_ids = self.d_vae.get_codebook_indices(input_segmap)
        #     h, w = input_ids.shape[-2:]
        #     rec_segmap = self.d_vae.decode(input_ids, img_size=[h, w])
        #     rec_segmap = unmap_pixels(torch.sigmoid(rec_segmap[:, :3])) * 255
        # logits = self.cls_segmap(rec_segmap)
        # losses = self.losses(logits, gt_semantic_seg)
        # return losses

    def forward_test(self, inputs, img_metas, gt_semantic_seg, test_cfg):
        inputs = self._transform_inputs(inputs)
        # return self._forward_test_recon_with_dalle(gt_semantic_seg, img_metas)
        x = self.feature_aggregation(inputs)
        h, w = x.shape[-2:]
        vq_logist = self.forward(x).view(-1, self.vocab_size, h, w)
        h, w = vq_logist.shape[-2:]
        vq_indices = vq_logist.argmax(1).unsqueeze(1)
        rec_segmap = self.d_vae.decode(vq_indices, img_size=[h, w])
        rec_segmap = unmap_pixels(torch.sigmoid(rec_segmap[:, :3]))
        seg_logits = self.cls_segmap(rec_segmap)

        # seg_pred = self.decode_from_segmap(torch.tensor(rec_segmap), keep_ignore_index=False)
        # seg_pred[seg_pred == self.num_classes] = 0  # b, h, w, c
        # seg_logist = F.one_hot(seg_pred.to(torch.int64), self.num_classes).squeeze(1).permute(0, 3, 1, 2).to(
        #     torch.float)

        # save images
        gt_semantic_seg[0] = gt_semantic_seg[0].unsqueeze(0)
        gt_semantic_seg[0] = F.interpolate(gt_semantic_seg[0].float(), size=seg_logits.shape[-2:], mode='bilinear')
        error = gt_semantic_seg[0] - seg_logits.argmax(1).unsqueeze(1)
        error[gt_semantic_seg[0] >= self.num_classes] = 0
        save_image(torch.cat([map_pixels(self.encode_to_segmap(gt_semantic_seg[0].long()) / 255.0),
                              rec_segmap,
                              map_pixels(rec_segmap),
                              map_pixels(self.encode_to_segmap(seg_logits.argmax(1).unsqueeze(1).long()) / 255.0),
                              torch.Tensor.repeat(error, 3, 1, 1, 1, 1).squeeze(2).permute(1, 0, 2, 3)],
                             dim=0), 'work_dirs/post_bigseg_768x768_pretrain_384x384_22K_300e_ade20k_relax_e08_hunger_indice08_pixel02_no_act_no_norm_conv3/show_val_30epoch/val_' + img_metas[0]['ori_filename'].split('/')[-1])
        print('cjq save images')

        # save indice data, includes semantic_seg_pred (19 classes), gt_semantic_seg (19 classes), vq_ind_pred (8192 classes), vq_indice_gt (8192 classes)
        # gt_semantic_seg = gt_semantic_seg[0]
        # gt_semantic_seg[gt_semantic_seg == 255] = self.num_classes
        # gt_semantic_seg = F.interpolate(
        #     F.one_hot(gt_semantic_seg.to(torch.long), self.num_classes + 1).squeeze(1).permute(0, 3, 1, 2).to(
        #         torch.float),
        #     size=(h * 8, w * 8), mode='bilinear').argmax(1).unsqueeze(1)
        # gt_semantic_seg[gt_semantic_seg == self.num_classes] = 255
        # gt_semantic_seg_indices = self.get_gt_vq_indices(gt_semantic_seg).unsqueeze(1)  # % 100
        #
        # sio.savemat('work_dirs/mask_vqseg_agg_swin_large_patch4_window12_768x768_pretrain_384x384_22K_300e_cityscapes/annal/' + img_metas[0]['ori_filename'].split('/')[-1].split('.')[0] + '.mat',
        #             mdict={'semantic_seg_pred': seg_pred.cpu().numpy(),
        #                    'gt_semantic_seg': gt_semantic_seg.cpu().numpy(),
        #                    'vq_indices': vq_indices.cpu().numpy(),
        #                    'vq_indice_gt': gt_semantic_seg_indices.cpu().numpy()})
        return seg_logits

    def _forward_test_recon_with_dalle(self, gt_semantic_seg, img_metas):
        assert isinstance(gt_semantic_seg, list)
        results = []
        for gt_semantic_seg_item in gt_semantic_seg:
            input_segmap = map_pixels(self.encode_to_segmap(gt_semantic_seg_item) / 255.0)
            input_ids = self.d_vae.get_codebook_indices(input_segmap)
            h, w = input_ids.shape[-2:]
            rec_segmap = self.d_vae.decode(input_ids, img_size=[h, w])
            rec_segmap = unmap_pixels(torch.sigmoid(rec_segmap[:, :3])) * 255
            # seg_indices = self.decode_from_segmap(rec_segmap, keep_ignore_index=False)
            # seg_logist = F.one_hot(seg_indices.to(torch.int64), self.num_classes + 1).squeeze(1).permute(0, 3, 1, 2).to(
            #     torch.float)[:,:self.num_classes,:,:]
            results.append(rec_segmap)
            # error = gt_semantic_seg_item - seg_logist.argmax(1).unsqueeze(1)
            # error[(gt_semantic_seg_item == self.num_classes).unsqueeze(1)] = 0
            # save_image(torch.cat([map_pixels(encode_to_segmap(gt_semantic_seg_item.long()) / 255.0),
            #                       map_pixels(encode_to_segmap(seg_logist.argmax(1).unsqueeze(1).long()) / 255.0),
            #                       torch.Tensor.repeat(error, 3, 1, 1, 1, 1).squeeze(2).permute(1, 0, 2, 3)],
            #                      dim=0), 'work_dirs/dalle768x768/gt_' + img_metas[0]['ori_filename'].split('/')[-1])
            # print('cjq debug dalle ok')
        rec_segmap = torch.cat(results, dim=0)
        seg_logits = self.cls_segmap(rec_segmap)
        return seg_logits

    def encode_to_segmap(self, indice):
        PALETTE_ = self.palette.clone().to(indice.device)
        _indice = indice.clone().detach()
        _indice[_indice > self.num_classes] = self.num_classes
        return PALETTE_[_indice.long()].squeeze(1).permute(0, 3, 1, 2)

    def decode_from_segmap(self, segmap, keep_ignore_index):
        PALETTE_ = self.palette.clone().to(segmap.device) \
            if keep_ignore_index \
            else self.palette[:-1].clone().to(segmap.device)
        B, C, H, W = segmap.shape
        p = torch.Tensor.repeat(PALETTE_, B, H, W, 1, 1).permute(0, 3, 4, 1, 2)
        if keep_ignore_index:
            segmap = torch.Tensor.repeat(segmap, self.num_classes + 1, 1, 1, 1, 1).permute(1, 0, 2, 3, 4)
        else:
            segmap = torch.Tensor.repeat(segmap, self.num_classes, 1, 1, 1, 1).permute(1, 0, 2, 3, 4)
        # return torch.abs(segmap - p).sum(2).argmin(1).unsqueeze(1)
        return ((segmap - p) ** 2).sum(2).argmin(1).unsqueeze(1)

    def get_cls_prob_from_segmap(self, segmap, keep_ignore_index):
        PALETTE_ = self.palette.clone().to(segmap.device) \
            if keep_ignore_index \
            else self.palette[:-1].clone().to(segmap.device)
        B, C, H, W = segmap.shape
        p = torch.Tensor.repeat(PALETTE_, B, H, W, 1, 1).permute(0, 3, 4, 1, 2)
        if keep_ignore_index:
            segmap = torch.Tensor.repeat(segmap, self.num_classes + 1, 1, 1, 1, 1).permute(1, 0, 2, 3, 4)
        else:
            segmap = torch.Tensor.repeat(segmap, self.num_classes, 1, 1, 1, 1).permute(1, 0, 2, 3, 4)
        return ((segmap - p) ** 2).sum(2)
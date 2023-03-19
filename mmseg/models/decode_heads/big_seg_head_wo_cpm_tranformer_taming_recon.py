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
from ..losses import accuracy
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.ops import resize
from omegaconf import OmegaConf
from taming.models.cond_transformer import Net2NetTransformer

from mmseg.models.backbones.swin import SwinBlockSequence
@HEADS.register_module()
class BigSegAggHeadWoCPMTransformerTaming(BaseDecodeHead):
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
                 indice_channel_index=0,
                 pixel_channel_index=1,
                 interpolate_mode='bilinear',
                 palette=None,
                 indice_seg_channel=None,
                 indice_cls_channel=2048,
                 swin_num_head=16,
                 swin_depth=2,
                 swin_window_size=7,
                 taming_conifg='ckp/2020-11-20T21-45-44_ade20k_transformer/configs/2020-11-20T21-45-44-project.yaml',
                 taming_ckp='ckp/2020-11-20T21-45-44_ade20k_transformer/checkpoints/last.ckpt',
                 **kwargs):
        super(BigSegAggHeadWoCPMTransformerTaming, self).__init__(init_cfg=init_cfg, input_transform='multiple_select', channels=channels, **kwargs)
        num_inputs = len(self.in_channels)
        assert num_inputs == len(self.in_index)
        self.channels = channels
        self.palette = torch.tensor(palette)
        self.interpolate_mode = interpolate_mode
        self.img_size = img_size
        self.vocab_size = 8192
        self.indice_ignore_index = self.vocab_size
        self.pixel_ignore_index = 255
        _, self.norm = build_norm_layer(norm_layer, self.channels)
        self.pixel_channel_index = pixel_channel_index
        self.indice_channel_index = indice_channel_index
        self.indice_seg_channel = indice_seg_channel if indice_seg_channel is not None else channels
        self.indice_cls_channel = indice_cls_channel
        config = OmegaConf.load(taming_conifg)
        # import ipdb
        # ipdb.set_trace()
        self.taming = Net2NetTransformer(**config.model.params)
        self.taming.load_state_dict(torch.load(taming_ckp)['state_dict'], strict=False)
        # self.d_vae = get_dalle_vae(
        #     weight_path="ckp",
        #     device="cuda")
        self.swin_num_head = swin_num_head
        self.swin_depth = swin_depth
        self.swin_window_size = swin_window_size
        self.visual_palette = torch.tensor([[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
               [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
               [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
               [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
               [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
               [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
               [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
               [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
               [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
               [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
               [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
               [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
               [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
               [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
               [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
               [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
               [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
               [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
               [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
               [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
               [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
               [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
               [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
               [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
               [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
               [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
               [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
               [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
               [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
               [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
               [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
               [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
               [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
               [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
               [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
               [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
               [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
               [102, 255, 0], [92, 0, 255], [0, 0, 0]])
        # dense classificator
        # self.conv_before_seg = ConvModule(
        #             in_channels=self.indice_seg_channel,
        #             out_channels=self.indice_cls_channel,
        #             kernel_size=1,
        #             stride=1,
        #             norm_cfg=self.norm_cfg,
        #             act_cfg=self.act_cfg)
        # self.conv_seg = nn.Conv2d(self.indice_cls_channel, self.vocab_size, kernel_size=1)
        # self.conv_seg_pixel = nn.Conv2d(channels, self.num_classes, kernel_size=1)
        # _, self.swin_ln = build_norm_layer(dict(type='LN'), self.indice_seg_channel)

        # input translation
        # self.convs = nn.ModuleList()
        # self.convs_pixel = nn.ModuleList()
        # for i in range(num_inputs):
        #     self.convs.append(
        #         ConvModule(
        #             in_channels=self.in_channels[i],
        #             out_channels=self.channels,
        #             kernel_size=1,
        #             stride=1,
        #             norm_cfg=self.norm_cfg,
        #             act_cfg=self.act_cfg)
        #         )
        #     self.convs_pixel.append(
        #         ConvModule(
        #             in_channels=self.in_channels[i],
        #             out_channels=self.channels,
        #             kernel_size=1,
        #             stride=1,
        #             norm_cfg=self.norm_cfg,
        #             act_cfg=self.act_cfg))

        # fusion blocks
        # self.fusion_conv = ConvModule(
        #     in_channels=self.channels * num_inputs,
        #     out_channels=self.indice_seg_channel,
        #     kernel_size=1,
        #     norm_cfg=self.norm_cfg)
        # self.fusion_conv_pixel = ConvModule(
        #     in_channels=self.channels * num_inputs,
        #     out_channels=self.channels,
        #     kernel_size=1,
        #     norm_cfg=self.norm_cfg)
        #
        # # swin transformer block
        # self.transformer_block = SwinBlockSequence(
        #     embed_dims=self.indice_seg_channel,
        #     num_heads=self.swin_num_head,
        #     feedforward_channels=self.indice_seg_channel * 2,
        #     depth=self.swin_depth,
        #     window_size=self.swin_window_size,
        #     qkv_bias=True,
        #     qk_scale=None,
        #     drop_rate=0,
        #     attn_drop_rate=0,
        #     drop_path_rate=0,
        #     downsample=None,
        #     norm_cfg=dict(type='LN'),
        #     with_cp=False,
        #     init_cfg=None)

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
            x = self.convs[idx](x)
            # torch.cuda.empty_cache()
            outs.append(
                resize(
                    input=x,
                    size=inputs[self.indice_channel_index].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        out = self.fusion_conv(torch.cat(outs, dim=1))
        return out

    def feature_aggregation_for_pixel(self, inputs):
        outs_for_pixel = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs_pixel[idx]
            # torch.cuda.empty_cache()
            outs_for_pixel.append(
                resize(
                    input=conv(x),
                    size=inputs[self.pixel_channel_index].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))
        out_for_pixel = self.fusion_conv_pixel(torch.cat(outs_for_pixel, dim=1))
        return out_for_pixel

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        inputs = self._transform_inputs(inputs)
        x = self.feature_aggregation(inputs)
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x, hw, x_down, hw_down = self.transformer_block(x, (h, w))
        x = self.swin_ln(x)
        x = x.transpose(1, 2).view(b, c, h, w)
        x_p = self.feature_aggregation_for_pixel(inputs)
        h_p, w_p = x_p.shape[-2:]
        x = self.conv_before_seg(x)
        vq_logits = self.forward(x).view(-1, self.vocab_size, h, w)
        pixel_logits = self.conv_seg_pixel(x_p).view(-1, self.num_classes, h_p, w_p)
        pixel_logits = F.interpolate(pixel_logits, size=(h * 8, w * 8), mode='bilinear')

        # get vq indices from gt by dalle
        with torch.no_grad():
            # interpolate only preform when 1/4 scale was used
            gt_semantic_seg[gt_semantic_seg == self.pixel_ignore_index] = self.num_classes
            gt_semantic_seg = F.interpolate(
                F.one_hot(gt_semantic_seg.to(torch.long), self.num_classes + 1).squeeze(1).permute(0, 3, 1, 2).to(
                    torch.float),
                size=(h * 8, w * 8), mode='bilinear').argmax(1).unsqueeze(1)
            gt_semantic_seg[gt_semantic_seg == self.num_classes] = self.pixel_ignore_index

            # get non-ignore gt_indice
            pixel_pred = pixel_logits.argmax(1).unsqueeze(1)
            gt_semantic_seg_for_recon = torch.zeros_like(gt_semantic_seg)
            gt_semantic_seg_for_recon[gt_semantic_seg != self.pixel_ignore_index] = gt_semantic_seg[gt_semantic_seg != self.pixel_ignore_index].clone()
            gt_semantic_seg_for_recon[gt_semantic_seg == self.pixel_ignore_index] = pixel_pred[gt_semantic_seg == self.pixel_ignore_index].clone()
            gt_semantic_seg_indices = self.get_gt_vq_indices(gt_semantic_seg_for_recon).unsqueeze(1) # % 100

            # get ignore mask
            ignore_map = torch.ones_like(gt_semantic_seg, device=gt_semantic_seg.device)
            ignore_map[gt_semantic_seg >= self.num_classes] = 0
            ignore_mask = F.max_pool2d(ignore_map.float(), kernel_size=(8, 8), stride=(8, 8))

            # indice_map_mask = relaxation_map * ignore_mask
            indice_map_mask = ignore_mask

            # get final gt indices
            masked_gt_semantic_seg_indices = gt_semantic_seg_indices.clone()
            masked_gt_semantic_seg_indices[indice_map_mask == 0] = self.indice_ignore_index

            # 10.16 32号上的第一个bigseg实验没有ignore，而是按照teacher student的范式监督了student
            gt_semantic_seg_indices[ignore_mask == 0] = self.indice_ignore_index

            # error map
            # error_map = torch.zeros_like(gt_semantic_seg_indices, device=gt_semantic_seg.device)
            # # the correct predicted pixel will be ignored
            # error_map[vq_logits.argmax(1).unsqueeze(1) != gt_semantic_seg_indices] = 1
            # save_image(torch.cat([
            #     map_pixels(self.encode_to_segmap(gt_semantic_seg) / 255.0),  # gt
            #     map_pixels(self.encode_to_segmap(gt_semantic_seg_for_recon) / 255.0),
            #     # pixel_segmap_from_indice_pred / 255.0,
            #     # map_pixels(self.encode_to_segmap(pixel_pred_from_indice_pred) / 255.0), # prediction
            #     torch.Tensor.repeat(F.interpolate(error_map.float(), size=gt_semantic_seg.shape[-2:]),
            #                       3, 1, 1, 1, 1).squeeze(2).permute(1, 0, 2, 3),  # indice error map
            #     torch.Tensor.repeat(ignore_map.float(), 3, 1, 1, 1, 1).squeeze(2).permute(1, 0, 2, 3), # ignore map
            #     torch.Tensor.repeat(F.interpolate(indice_map_mask.float(), size=gt_semantic_seg.shape[-2:]), 3, 1, 1, 1, 1).squeeze(2).permute(1, 0, 2, 3),
            #     torch.Tensor.repeat(F.interpolate(gt_semantic_seg_indices.float(), size=gt_semantic_seg.shape[-2:]),
            #                         3, 1, 1, 1, 1).squeeze(2).permute(1, 0, 2, 3) / 8192.0,
            #     torch.Tensor.repeat(F.interpolate(vq_logits.argmax(1).unsqueeze(1).float(), size=gt_semantic_seg.shape[-2:]),
            #                         3, 1, 1, 1, 1).squeeze(2).permute(1, 0, 2, 3)/ 8192.0,
            # ],
            #     dim=0), 'work_dirs/bigseg_ade20k_conns_swin_160k_wo_cpm/show_train_16k/' + img_metas[0]['ori_filename'].split('/')[-1])
            # print('cjq save images')
        losses = self.losses(indice_seg_logit=vq_logits,
                             pixel_seg_logit=pixel_logits,
                             masked_indice_seg_label=masked_gt_semantic_seg_indices,
                             full_indice_seg_label=gt_semantic_seg_indices,
                             pixel_seg_label=gt_semantic_seg,
                             pred_pixel_from_pred_indice=None)
        return losses

    def forward_test(self, inputs, img_metas, gt_semantic_seg, test_cfg):
        # print('cjq debug:', gt_semantic_seg[0].shape)
        # inputs = self._transform_inputs(inputs)
        # return self._forward_test_recon_with_dalle(gt_semantic_seg, img_metas)
        return self._forward_test_recon_with_taming(gt_semantic_seg, img_metas)
        # x = self.feature_aggregation(inputs)
        # b, c, h, w = x.shape
        # x = x.flatten(2).transpose(1, 2)
        # x, hw, x_down, hw_down = self.transformer_block(x, (h, w))
        # x = self.swin_ln(x)
        # x = x.transpose(1, 2).view(b, c, h, w)
        # x = self.conv_before_seg(x)
        # # x_p = self.feature_aggregation_for_pixel(inputs)
        # vq_logist = self.forward(x).view(-1, self.vocab_size, h, w)
        # vq_indices = vq_logist.argmax(1).unsqueeze(1)

        # x_p = self.feature_aggregation_for_pixel(inputs)
        # h_p, w_p = x_p.shape[-2:]
        # pixel_logits = self.conv_seg_pixel(x_p).view(-1, self.num_classes, h_p, w_p)
        # pixel_logits = F.interpolate(pixel_logits, size=(h * 8, w * 8), mode='bilinear')

        # rec_segmap = self.d_vae.decode(vq_indices, img_size=[h, w])
        # rec_segmap = unmap_pixels(torch.sigmoid(rec_segmap[:, :3])) * 255
        # seg_pred = self.decode_from_segmap(torch.tensor(rec_segmap), keep_ignore_index=False, prob=False)
        # # seg_pred[seg_pred == self.num_classes] = 0  # b, h, w, c
        # seg_logist = F.one_hot(seg_pred.to(torch.int64), self.num_classes).squeeze(1).permute(0, 3, 1, 2).to(
        #     torch.float)

        # save images
        # vq_indices_show = F.interpolate(vq_indices.float(), size=seg_logist.shape[-2:], mode='bilinear')
        # gt_semantic_seg[0] = gt_semantic_seg[0].unsqueeze(0)
        # gt_semantic_seg[0] = F.interpolate(gt_semantic_seg[0].float(), size=seg_logist.shape[-2:], mode='bilinear')

        # error = torch.abs(gt_semantic_seg[0] - seg_logist.argmax(1).unsqueeze(1))
        # error[gt_semantic_seg[0] >= self.num_classes] = 0
        # import ipdb
        # ipdb.set_trace()

        # save_image(torch.cat([map_pixels(self.encode_to_segmap(gt_semantic_seg[0].long()) / 255.0),
        #                       rec_segmap / 255.0,
        #                       map_pixels(self.encode_to_segmap(seg_logist.argmax(1).unsqueeze(1).long()) / 255.0),
        #                       torch.Tensor.repeat(error, 3, 1, 1, 1, 1).squeeze(2).permute(1, 0, 2, 3)],
        #                      dim=0), 'work_dirs/bigseg_mseg_conns_swin_160k_025_2048_trans/kitti_show_24k/val_24k_' + img_metas[0]['ori_filename'].split('/')[-1])
        # print('cjq save images')

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

        # gt_semantic_seg = gt_semantic_seg[0].unsqueeze(0)
        # gt_semantic_seg[gt_semantic_seg == self.pixel_ignore_index] = self.num_classes
        # seg_logist = F.one_hot(gt_semantic_seg.to(torch.long), self.num_classes + 1).squeeze(1).permute(0, 3, 1, 2).to(torch.float)[:,:-1,:,:]
        # return seg_logist

    def _forward_test_recon_with_taming(self, gt_semantic_seg, img_metas):
        mask = gt_semantic_seg[0].unsqueeze(0)
        mask[mask == 255] = 150  # 1, 1, 512, 512
        mask += 1
        mask[mask == 151] = 0 # 1, 1, 512, 512
        cxxx = F.one_hot(mask.to(torch.int64), self.num_classes + 1)[0].to(torch.float).permute(0, 3, 1, 2) # 1 151, 512, 512
        quant_c, c_indices = self.taming.encode_to_c(cxxx)
        cond_rec = self.taming.cond_stage_model.decode(quant_c)
        pred = cond_rec.argmax(1)
        pred[pred == 0] = 151
        pred -= 1
        seg_logit = F.one_hot(pred.to(torch.int64), self.num_classes + 1).squeeze(1).permute(0, 3, 1, 2).to(
            torch.float)
        # gt_semantic_seg[0] = gt_semantic_seg[0].unsqueeze(0)
        # save prediction
        # save_image(self.encode_to_segmap_visual(seg_logit.argmax(1).unsqueeze(1).long()) / 255.0,
        #            'work_dirs/visualization_stage_1/taming_150_zero_reduce_non_correct/' + img_metas[0]['ori_filename'].split('/')[-1].split('.')[0] + '_pred.png')
        # # save ground truth
        # save_image(self.encode_to_segmap_visual(mask.long()) / 255.0,
        #            'work_dirs/visualization_stage_1/gt/' + img_metas[0]['ori_filename'].split('/')[-1].split('.')[0] + '_gt.png')
        # # save the self-designed color
        # save_image(self.encode_to_segmap(cond_rec.argmax(1).unsqueeze(1).long()) / 255.0,
        #            'work_dirs/visualization_stage_1/taming150/' + img_metas[0]['ori_filename'].split('/')[-1].split('.')[0] + '_pred_our_color.png')
        # save_image(self.encode_to_segmap(mask.long()) / 255.0,
        #            'work_dirs/visualization_stage_1/gt/' + img_metas[0]['ori_filename'].split('/')[-1].split('.')[0] + '_gt_our_color.png')
        #
        # mask = F.interpolate(mask.float(), size=cond_rec.shape[-2:], mode='nearest')
        # error = torch.abs(mask - cond_rec.argmax(1).unsqueeze(1))
        # error[mask >= self.num_classes] = 0
        # error[error > 0] = 1
        # save_image(torch.cat([self.encode_to_segmap(mask.long()) / 255.0,
        #                       self.encode_to_segmap(cond_rec.argmax(1).unsqueeze(1).long()) / 255.0,
        #                       torch.Tensor.repeat(error, 3, 1, 1, 1, 1).squeeze(2).permute(1, 0, 2, 3)],
        #                      dim=0), 'work_dirs/visualization_stage_1/taming150/' + img_metas[0]['ori_filename'].split('/')[-1].split('.')[0] + '_gt_pred_error.png')
        # print('cjq save images')
        return seg_logit

    def _forward_test_recon_with_dalle(self, gt_semantic_seg, img_metas):
        assert isinstance(gt_semantic_seg, list)
        results = []
        for gt_semantic_seg_item in gt_semantic_seg:
            input_segmap = map_pixels(self.encode_to_segmap(gt_semantic_seg_item) / 255.0)
            input_ids = self.d_vae.get_codebook_indices(input_segmap)
            h, w = input_ids.shape[-2:]
            rec_segmap = self.d_vae.decode(input_ids, img_size=[h, w])
            rec_segmap = unmap_pixels(torch.sigmoid(rec_segmap[:, :3])) * 255
            seg_indices = self.decode_from_segmap(rec_segmap, keep_ignore_index=True)
            seg_logist = F.one_hot(seg_indices.to(torch.int64), self.num_classes + 1).squeeze(1).permute(0, 3, 1, 2).to(
                torch.float)
            results.append(seg_logist) # [:,:self.num_classes,:,:]
            seg_indices[seg_indices == 150] = 255
            save_image(rec_segmap / 255.0,
                       'work_dirs/visualization_stage_1/dalle/' +
                       img_metas[0]['ori_filename'].split('/')[-1].split('.')[0] + '_rec.png')
            # save images
            save_image(self.encode_to_segmap_visual(seg_indices.long()) / 255.0,
                       'work_dirs/visualization_stage_1/dalle/' +
                       img_metas[0]['ori_filename'].split('/')[-1].split('.')[0] + '_pred.png')
            # save ground truth
            save_image(self.encode_to_segmap_visual(gt_semantic_seg_item.long()) / 255.0,
                       'work_dirs/visualization_stage_1/gt/' + img_metas[0]['ori_filename'].split('/')[-1].split('.')[
                           0] + '_gt.png')
            # save the self-designed color
            save_image(self.encode_to_segmap(seg_indices.long()) / 255.0,
                       'work_dirs/visualization_stage_1/dalle/' +
                       img_metas[0]['ori_filename'].split('/')[-1].split('.')[0] + '_pred_our_color.png')
            save_image(self.encode_to_segmap(gt_semantic_seg_item.long()) / 255.0,
                       'work_dirs/visualization_stage_1/gt/' + img_metas[0]['ori_filename'].split('/')[-1].split('.')[
                           0] + '_gt_our_color.png')
            seg_indices = F.interpolate(seg_indices.float(), size=gt_semantic_seg_item.shape[-2:], mode='bilinear').long()
            # mask = F.interpolate(gt_semantic_seg_item.float(), size=seg_logist.shape[-2:], mode='nearest')
            error = torch.abs(gt_semantic_seg_item.unsqueeze(0) - seg_indices)
            error[gt_semantic_seg_item.unsqueeze(0) >= self.num_classes] = 0
            error[error > 0] = 1
            save_image(torch.cat([self.encode_to_segmap(gt_semantic_seg_item.long()) / 255.0,
                                  self.encode_to_segmap(seg_indices.long()) / 255.0,
                                  torch.Tensor.repeat(error, 3, 1, 1, 1, 1).squeeze(2).permute(1, 0, 2, 3)],
                                 dim=0), 'work_dirs/visualization_stage_1/dalle/' +
                       img_metas[0]['ori_filename'].split('/')[-1].split('.')[0] + '_gt_pred_error.png')
            print('cjq save images')

            # save images
            # gt_semantic_seg[0] = gt_semantic_seg[0].unsqueeze(0)
            # gt_semantic_seg[0] = F.interpolate(gt_semantic_seg[0].float(), size=seg_logist.shape[-2:], mode='bilinear')
            # error = gt_semantic_seg[0] - seg_logist.argmax(1).unsqueeze(1)
            # error[gt_semantic_seg[0] >= self.num_classes] = 0
            # error[error > 0] = 1
            # save_image(torch.cat([map_pixels(self.encode_to_segmap(gt_semantic_seg[0].long()) / 255.0),
            #                       map_pixels(self.encode_to_segmap(seg_logist.argmax(1).unsqueeze(1).long()) / 255.0),
            #                       torch.Tensor.repeat(error, 3, 1, 1, 1, 1).squeeze(2).permute(1, 0, 2, 3)],
            #                      dim=0), 'work_dirs/ade20k_change_color/show/val_' + img_metas[0]['ori_filename'].split('/')[-1])
            # print('cjq save images')

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
        return torch.cat(results, dim=0)

    def encode_to_segmap(self, indice):

        PALETTE_ = self.palette.clone().to(indice.device)
        _indice = indice.clone().detach()
        _indice[_indice > self.num_classes] = self.num_classes
        return PALETTE_[_indice.long()].squeeze(1).permute(0, 3, 1, 2)

    def decode_from_segmap(self, segmap, keep_ignore_index, prob=False):
        PALETTE_ = self.palette.clone().to(segmap.device) \
            if keep_ignore_index \
            else self.palette[:-1].clone().to(segmap.device) # N, C
        B, C, H, W = segmap.shape # B, N, C, H, W
        N, _ = PALETTE_.shape
        p = PALETTE_.reshape(1, N, C, 1, 1)
        # p = torch.Tensor.repeat(PALETTE_, B, H, W, 1, 1).permute(0, 3, 4, 1, 2) # B, N, C, H, W
        if keep_ignore_index:
            segmap = torch.Tensor.repeat(segmap, self.num_classes + 1, 1, 1, 1, 1).permute(1, 0, 2, 3, 4)
        else:
            segmap = segmap.reshape(B, 1, C, H, W)
            # segmap = torch.Tensor.repeat(segmap, self.num_classes, 1, 1, 1, 1).permute(1, 0, 2, 3, 4)
        # return torch.abs(segmap - p).sum(2).argmin(1).unsqueeze(1)
        if prob:
            return ((segmap - p) ** 2).sum(2)
        else:
            return ((segmap - p) ** 2).sum(2).argmin(1).unsqueeze(1)

    def encode_to_segmap_visual(self, indice):
        PALETTE_ = self.visual_palette.clone().to(indice.device)
        _indice = indice.clone().detach()
        _indice[_indice > self.num_classes] = self.num_classes
        return PALETTE_[_indice.long()].squeeze(1).permute(0, 3, 1, 2)

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self,
               indice_seg_logit,
               pixel_seg_logit,
               masked_indice_seg_label,
               full_indice_seg_label,
               pixel_seg_label,
               pred_pixel_from_pred_indice):
        """Compute segmentation loss."""
        loss = dict()
        indice_seg_logit = resize(
            input=indice_seg_logit,
            size=masked_indice_seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        pixel_seg_logit = resize(
            input=pixel_seg_logit,
            size=pixel_seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            indice_seg_weight = self.sampler.sample(indice_seg_logit, masked_indice_seg_label)
            dense_indice_seg_weight = self.sampler.sample(indice_seg_logit, full_indice_seg_label)
            pixel_seg_weight = self.sampler.sample(pixel_seg_logit, pixel_seg_label)
        else:
            indice_seg_weight = None
            dense_indice_seg_weight = None
            pixel_seg_weight = None
        masked_indice_seg_label = masked_indice_seg_label.squeeze(1)
        pixel_seg_label = pixel_seg_label.squeeze(1)
        full_indice_seg_label = full_indice_seg_label.squeeze(1)
        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                if loss_decode.loss_name == 'loss_ce':
                    loss[loss_decode.loss_name] = loss_decode(
                        indice_seg_logit,
                        masked_indice_seg_label,
                        weight=indice_seg_weight,
                        ignore_index=self.indice_ignore_index)
                elif loss_decode.loss_name == 'loss_ce_pixel':
                    loss[loss_decode.loss_name] = loss_decode(
                        pixel_seg_logit,
                        pixel_seg_label,
                        weight=pixel_seg_weight,
                        ignore_index=self.pixel_ignore_index)
                elif loss_decode.loss_name == 'loss_ce_dense':
                    loss[loss_decode.loss_name] = loss_decode(
                        indice_seg_logit,
                        full_indice_seg_label,
                        weight=dense_indice_seg_weight,
                        ignore_index=self.indice_ignore_index)
            else:
                if loss_decode.loss_name == 'loss_ce':
                    loss[loss_decode.loss_name] += loss_decode(
                        indice_seg_logit,
                        masked_indice_seg_label,
                        weight=indice_seg_weight,
                        ignore_index=self.indice_ignore_index)
                elif loss_decode.loss_name == 'loss_ce_pixel':
                    loss[loss_decode.loss_name] += loss_decode(
                        pixel_seg_logit,
                        pixel_seg_label,
                        weight=pixel_seg_weight,
                        ignore_index=self.pixel_ignore_index)
                elif loss_decode.loss_name == 'loss_ce_dense':
                    loss[loss_decode.loss_name] += loss_decode(
                        indice_seg_logit,
                        full_indice_seg_label,
                        weight=dense_indice_seg_weight,
                        ignore_index=self.indice_ignore_index)

        if pred_pixel_from_pred_indice is not None:
            pixel_seg_logist_from_pred_indice = F.one_hot(pred_pixel_from_pred_indice.to(torch.int64), self.num_classes).squeeze(1).permute(0, 3, 1, 2).to(
                torch.float)
            loss['acc_seg'] = accuracy(
                pixel_seg_logist_from_pred_indice, pixel_seg_label, ignore_index=self.pixel_ignore_index)
        loss['acc_seg_aux'] = accuracy(
            pixel_seg_logit, pixel_seg_label, ignore_index=self.pixel_ignore_index)
        loss['acc_seg_indice'] = accuracy(
            indice_seg_logit, full_indice_seg_label, ignore_index=self.indice_ignore_index)
        return loss

    # visualization
    # save_image(torch.cat([map_pixels(encode_to_segmap(gt_semantic_seg_item.long()) / 255.0),
    #                       map_pixels(encode_to_segmap(seg_logist.argmax(1).unsqueeze(1).long()) / 255.0),
    #                       torch.Tensor.repeat(gt_semantic_seg_item - seg_logist.argmax(1).unsqueeze(1), 3, 1, 1, 1, 1).squeeze(2).permute(1, 0, 2, 3)],
    #                      dim=0), 'work_dirs/vqseg_vit-large_8x1_768x768_300e_cityscapes_gt_test_2049x1025/show/gt_' + img_metas[0]['ori_filename'].split('/')[-1] + '_debug.png')
    # print('cjq debug ok')
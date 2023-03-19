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

from mmseg.models.backbones.swin import SwinBlockSequence
from .big_seg_head_wo_cpm_tranformer import BigSegAggHeadWoCPMTransformer

@HEADS.register_module()
class PostSwinBigSegAggHeadWoCPMTransformer(BigSegAggHeadWoCPMTransformer):
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
                 post_seg_channel,
                 post_swin_num_head,
                 post_swin_depth,
                 post_swin_window_size,
                 **kwargs):
        super(PostSwinBigSegAggHeadWoCPMTransformer, self).__init__(**kwargs)
        self.post_seg_channel = post_seg_channel
        self.post_swin_num_head = post_swin_num_head
        self.post_swin_depth = post_swin_depth
        self.post_swin_window_size = post_swin_window_size
        self.projection = ConvModule(
                    in_channels=3,
                    out_channels=self.post_seg_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg)
        self.post_transformer_block = SwinBlockSequence(
                    embed_dims=self.post_seg_channel,
                    num_heads=self.post_swin_num_head,
                    feedforward_channels=self.post_seg_channel * 2,
                    depth=self.post_swin_depth,
                    window_size=self.post_swin_window_size,
                    qkv_bias=True,
                    qk_scale=None,
                    drop_rate=0,
                    attn_drop_rate=0,
                    drop_path_rate=0,
                    downsample=None,
                    norm_cfg=dict(type='LN'),
                    with_cp=False,
                    init_cfg=None)
        self.cls_segmap = ConvModule(
                in_channels=self.post_seg_channel,
                out_channels=self.num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
                norm_cfg=None,
                act_cfg=None)
        _, self.post_swin_ln = build_norm_layer(dict(type='LN'), self.post_seg_channel)
        self.conv_seg_pixel = None
        self.convs_pixel = None
        self.fusion_conv_pixel = None
        # self.visual_palette = torch.tensor([[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
        #        [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
        #        [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
        #        [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
        #        [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
        #        [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
        #        [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
        #        [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
        #        [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
        #        [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
        #        [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
        #        [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
        #        [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
        #        [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
        #        [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
        #        [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
        #        [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
        #        [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
        #        [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
        #        [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
        #        [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
        #        [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
        #        [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
        #        [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
        #        [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
        #        [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
        #        [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
        #        [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
        #        [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
        #        [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
        #        [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
        #        [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
        #        [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
        #        [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
        #        [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
        #        [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
        #        [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
        #        [102, 255, 0], [92, 0, 255], [0, 0, 0]])


    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        with torch.no_grad():
            inputs = self._transform_inputs(inputs)
            x = self.feature_aggregation(inputs)
            b, c, h, w = x.shape
            x = x.flatten(2).transpose(1, 2)
            x, hw, x_down, hw_down = self.transformer_block(x, (h, w))
            x = self.swin_ln(x)
            x = x.transpose(1, 2).view(b, c, h, w)
            x = self.conv_before_seg(x)
            vq_logits = self.forward(x).view(-1, self.vocab_size, h, w)
            # get the pixel-wise prediction from indice prediction
            pixel_segmap_from_indice_pred = self.d_vae.decode(vq_logits.argmax(1).unsqueeze(1), img_size=[h, w])
            pixel_segmap_from_indice_pred = unmap_pixels(torch.sigmoid(pixel_segmap_from_indice_pred[:, :3]))
            # pixel_prob_from_indice_pred = self.get_cls_prob_from_segmap(torch.tensor(pixel_segmap_from_indice_pred), keep_ignore_index=False)
        b, c, h, w = pixel_segmap_from_indice_pred.shape
        x = self.projection(pixel_segmap_from_indice_pred)
        x = x.flatten(2).transpose(1, 2)
        x, hw, x_down, hw_down = self.post_transformer_block(x, (h, w))
        x = self.post_swin_ln(x)
        x = x.transpose(1, 2).view(b, self.post_seg_channel, h, w)
        logits = self.cls_segmap(x)
        losses = self.losses(logits, gt_semantic_seg)
        return losses

    def forward_test(self, inputs, img_metas, gt_semantic_seg, test_cfg):
        inputs = self._transform_inputs(inputs)
        x = self.feature_aggregation(inputs)
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x, hw, x_down, hw_down = self.transformer_block(x, (h, w))
        x = self.swin_ln(x)
        x = x.transpose(1, 2).view(b, c, h, w)
        x = self.conv_before_seg(x)
        # x_p = self.feature_aggregation_for_pixel(inputs)
        vq_logist = self.forward(x).view(-1, self.vocab_size, h, w)
        vq_indices = vq_logist.argmax(1).unsqueeze(1)
        rec_segmap = self.d_vae.decode(vq_indices, img_size=[h, w])
        rec_segmap = unmap_pixels(torch.sigmoid(rec_segmap[:, :3]))
        rec_segmap = self.get_recon_from_dalle(gt_semantic_seg, img_metas) / 255.0
        b, c, h, w = rec_segmap.shape
        x = self.projection(rec_segmap)
        x = x.flatten(2).transpose(1, 2)
        x, hw, x_down, hw_down = self.post_transformer_block(x, (h, w))
        x = self.post_swin_ln(x)
        x = x.transpose(1, 2).view(b, self.post_seg_channel, h, w)
        seg_logits = self.cls_segmap(x)
        # seg_pred = self.decode_from_segmap(torch.tensor(rec_segmap) * 255, keep_ignore_index=False, prob=False)
        # # seg_pred[seg_pred == self.num_classes] = 0  # b, h, w, c
        # seg_logits = F.one_hot(seg_pred.to(torch.int64), self.num_classes).squeeze(1).permute(0, 3, 1, 2).to(
        #     torch.float)
        # for dalle recon
        # save_image(rec_segmap,
        #            'work_dirs/visualization_stage_2/cityscapes/' +
        #            img_metas[0]['ori_filename'].split('/')[-1].split('.')[0] + '_rec.png')
        # # print('cjq debug save')
        # save_image(self.encode_to_segmap_visual(seg_logits.argmax(1).unsqueeze(1).long()) / 255.0,
        #            'work_dirs/visualization_stage_2/cityscapes/' +
        #            img_metas[0]['ori_filename'].split('/')[-1].split('.')[0] + '_pred.png')
        # save_image(self.encode_to_segmap(seg_logits.argmax(1).unsqueeze(1).long()) / 255.0,
        #            'work_dirs/visualization_stage_2/cityscapes/' +
        #            img_metas[0]['ori_filename'].split('/')[-1].split('.')[0] + '_pred_our_color.png')
        # print('cjq debug save')
        # for learnbale model
        # save_image(rec_segmap,
        #            'work_dirs/visualization_stage_2/ade20k/' +
        #            img_metas[0]['ori_filename'].split('/')[-1].split('.')[0] + '_model_pred_rec.png')
        # print('cjq debug save')
        # save_image(self.encode_to_segmap(seg_logits.argmax(1).unsqueeze(1).long()) / 255.0,
        #            'work_dirs/visualization_stage_2/ade20k/' +
        #            img_metas[0]['ori_filename'].split('/')[-1].split('.')[0] + '_learnable_pred.png')
        # print('cjq debug save')
        return seg_logits

    def get_recon_from_dalle(self, gt_semantic_seg, img_metas):
        assert isinstance(gt_semantic_seg, list)
        results = []
        for gt_semantic_seg_item in gt_semantic_seg:
            input_segmap = map_pixels(self.encode_to_segmap(gt_semantic_seg_item) / 255.0)
            input_ids = self.d_vae.get_codebook_indices(input_segmap)
            h, w = input_ids.shape[-2:]
            rec_segmap = self.d_vae.decode(input_ids, img_size=[h, w])
            rec_segmap = unmap_pixels(torch.sigmoid(rec_segmap[:, :3])) * 255
            # seg_indices = self.decode_from_segmap(rec_segmap, keep_ignore_index=True)
            # save_image(rec_segmap / 255.0,
            #            'work_dirs/visualization_stage_1/dalle/' +
            #            img_metas[0]['ori_filename'].split('/')[-1].split('.')[0] + '_rec.png')
            # # save images
            # save_image(self.encode_to_segmap_visual(seg_indices.long()) / 255.0,
            #            'work_dirs/visualization_stage_1/dalle/' +
            #            img_metas[0]['ori_filename'].split('/')[-1].split('.')[0] + '_pred.png')
            # # # save ground truth
            # save_image(self.encode_to_segmap_visual(gt_semantic_seg_item.long()) / 255.0,
            #            'work_dirs/visualization_stage_1/gt/' + img_metas[0]['ori_filename'].split('/')[-1].split('.')[
            #                0] + '_gt.png')
            # # # save the self-designed color
            # save_image(self.encode_to_segmap(seg_indices.long()) / 255.0,
            #            'work_dirs/visualization_stage_1/dalle/' +
            #            img_metas[0]['ori_filename'].split('/')[-1].split('.')[0] + '_pred_our_color.png')
            # save_image(self.encode_to_segmap(gt_semantic_seg_item.long()) / 255.0,
            #            'work_dirs/visualization_stage_1/gt/' + img_metas[0]['ori_filename'].split('/')[-1].split('.')[
            #                0] + '_gt_our_color.png')
            # # seg_indices = F.interpolate(seg_indices.float(), size=gt_semantic_seg_item.shape[-2:],
            #                             mode='bilinear').long()
            # # mask = F.interpolate(gt_semantic_seg_item.float(), size=seg_logist.shape[-2:], mode='nearest')
            # error = torch.abs(gt_semantic_seg_item.unsqueeze(0) - seg_indices)
            # error[gt_semantic_seg_item.unsqueeze(0) >= self.num_classes] = 0
            # error[error > 0] = 1
            # save_image(torch.cat([self.encode_to_segmap(gt_semantic_seg_item.long()) / 255.0,
            #                       self.encode_to_segmap(seg_indices.long()) / 255.0,
            #                       torch.Tensor.repeat(error, 3, 1, 1, 1, 1).squeeze(2).permute(1, 0, 2, 3)],
            #                      dim=0), 'work_dirs/visualization_stage_1/dalle/' +
            #            img_metas[0]['ori_filename'].split('/')[-1].split('.')[0] + '_gt_pred_error.png')
        return rec_segmap

    @force_fp32(apply_to=('seg_logit', ))
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
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logit,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logit,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)

        loss['acc_seg'] = accuracy(
            seg_logit, seg_label, ignore_index=self.ignore_index)
        return loss

    def encode_to_segmap_visual(self, indice):
        PALETTE_ = self.visual_palette.clone().to(indice.device)
        _indice = indice.clone().detach()
        _indice[_indice > self.num_classes] = self.num_classes
        return PALETTE_[_indice.long()].squeeze(1).permute(0, 3, 1, 2)
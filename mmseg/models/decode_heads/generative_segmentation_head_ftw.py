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
from .generative_segmentation_head_ff import GenerativeSegHeadFF


@HEADS.register_module()
class GenerativeSegHeadFTW(GenerativeSegHeadFF):
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
        super(GenerativeSegHeadFTW, self).__init__(**kwargs)
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


    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        with torch.no_grad():
            inputs = self._transform_inputs(inputs)
            x = self.feature_aggregation(inputs)
            b, c, h, w = x.shape
            x = x.flatten(2).transpose(1, 2).contiguous()
            x, hw, x_down, hw_down = self.transformer_block(x, (h, w))
            x = self.swin_ln(x)
            x = x.transpose(1, 2).view(b, c, h, w).contiguous()
            x = self.conv_before_seg(x)
            vq_logits = self.forward(x).view(-1, self.vocab_size, h, w).contiguous()
            # get the pixel-wise prediction from indice prediction
            pixel_segmap_from_indice_pred = self.d_vae.decode(vq_logits.argmax(1).unsqueeze(1), img_size=[h, w]).contiguous()
            pixel_segmap_from_indice_pred = unmap_pixels(torch.sigmoid(pixel_segmap_from_indice_pred[:, :3]))
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
        vq_logist = self.forward(x).view(-1, self.vocab_size, h, w)
        vq_indices = vq_logist.argmax(1).unsqueeze(1)
        rec_segmap = self.d_vae.decode(vq_indices, img_size=[h, w])
        rec_segmap = unmap_pixels(torch.sigmoid(rec_segmap[:, :3]))
        b, c, h, w = rec_segmap.shape
        x = self.projection(rec_segmap)
        x = x.flatten(2).transpose(1, 2)
        x, hw, x_down, hw_down = self.post_transformer_block(x, (h, w))
        x = self.post_swin_ln(x)
        x = x.transpose(1, 2).view(b, self.post_seg_channel, h, w)
        seg_logits = self.cls_segmap(x)
        return seg_logits

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
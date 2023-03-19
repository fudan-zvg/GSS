# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
import torch.nn.functional as F

from ..builder import HEADS
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class MSegHead(BaseDecodeHead):
    """Fully Convolution Networks for Semantic Segmentation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
        dilation (int): The dilation rate for convs in the head. Default: 1.

        # Upsampling
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.upsample(x[1], size=(x0_h, x0_w), mode='bilinear')
        x2 = F.upsample(x[2], size=(x0_h, x0_w), mode='bilinear')
        x3 = F.upsample(x[3], size=(x0_h, x0_w), mode='bilinear')

        x = torch.cat([x[0], x1, x2, x3], 1)
        # Perform two 1x1 convolutions on concat representation
        x = self.last_layer(x)

        # Bilinear upsampling of output
        x = F.interpolate(x, size=(h,w), mode='bilinear', align_corners=True)
        if self.training:
            main_loss = self.criterion(x, y)
            return x.max(1)[1], main_loss, main_loss * 0
        else:
            return x
    """

    def __init__(self,
                 **kwargs):
        super(MSegHead, self).__init__(**kwargs)

    def forward(self, inputs):
        """Forward function."""
        # Upsampling
        x0_h, x0_w = inputs[1].size(2), inputs[1].size(3)
        # import ipdb;ipdb.set_trace()
        x0 = F.upsample(inputs[0], size=(x0_h, x0_w), mode='bilinear')
        x1 = F.upsample(inputs[1], size=(x0_h, x0_w), mode='bilinear')
        x2 = F.upsample(inputs[2], size=(x0_h, x0_w), mode='bilinear')
        x3 = F.upsample(inputs[3], size=(x0_h, x0_w), mode='bilinear')
        x = torch.cat([x0, x1, x2, x3], 1)
        # Perform two 1x1 convolutions on concat representation
        output = self.cls_seg(x)

        return output

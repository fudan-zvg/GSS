# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .encoder_decoder import EncoderDecoder
from mseg.taxonomy.taxonomy_converter import TaxonomyConverter


@SEGMENTORS.register_module()
class MultiDomainEncoderDecoder(EncoderDecoder):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 **kwargs):
        super(MultiDomainEncoderDecoder, self).__init__(**kwargs)
        self.universal2test_mappings = TaxonomyConverter().convs
        for (dataset_name, conv) in self.universal2test_mappings.items():
            self.universal2test_mappings[dataset_name] = conv.cuda()

    def inference(self, img, img_meta, rescale, **kwargs):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """
        if len(img_meta) == 0:
            seg_logit = self.whole_inference(img, img_meta, rescale, **kwargs)
            output = F.softmax(seg_logit, dim=1)
            return output
        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale, **kwargs)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale, **kwargs)
        output = F.softmax(seg_logit, dim=1)
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))

        return output

    def simple_test(self, img, img_meta, rescale=True, **kwargs):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale, **kwargs)
        num_cls = seg_logit.shape[1]
        seg_logit = F.one_hot(seg_logit.argmax(1).long(), num_cls).float().squeeze(1).permute(0, 3, 1, 2)
        # print('\ncjq debug seg_pred before u2s:', seg_logit.argmax(1))
        # seg_logit = F.softmax(seg_logit, dim=1)

        seg_logit = self.universal2test_mappings[img_meta[0]['dataset_name']](seg_logit)
        seg_pred = seg_logit.argmax(dim=1)
        # print('\ncjq debug seg_pred after u2s:', seg_pred)
        # import os
        # file_dir = 'work_dirs/visualization_stage_2/mseg_learnable/' + img_meta[0]['dataset_name'] + '/'
        # if not os.path.exists(file_dir):
        #     os.makedirs(file_dir)
        # from torchvision.utils import save_image, make_grid
        # save_image(self.decode_head.v_encode_to_segmap(seg_pred.long()) / 255.0,
        #            file_dir +
        #            img_meta[0]['ori_filename'].split('/')[-1].split('.')[0] + '_learnable_pred.png')
        # print('cjq debug save')
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

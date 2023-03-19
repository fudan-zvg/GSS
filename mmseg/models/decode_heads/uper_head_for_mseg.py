# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from .psp_head import PPM
from .uper_head import UPerHead
from mseg.taxonomy.taxonomy_converter import TaxonomyConverter
from mseg.utils.names_utils import get_universal_class_names, load_class_names
@HEADS.register_module()
class UniUPerHead(UPerHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, dataset_name, train_taxonomy, eval_taxonomy,**kwargs):
        super(UniUPerHead, self).__init__(**kwargs)
        tc = TaxonomyConverter()
        self.dataset_name = dataset_name
        if dataset_name in tc.test_datasets:
            self.dataset_type = 'mseg-val'
        elif dataset_name in tc.train_datasets:
            self.dataset_type = 'mseg-train'
        else:
            self.dataset_type = 'other'
        self.need_convert = not (train_taxonomy == eval_taxonomy)
        if self.need_convert:
            if self.dataset_type == 'val':
                self.taxonomy_transform = nn.Sequential(
                    nn.Softmax(dim=1),
                    tc.convs[dataset_name])
            elif self.dataset_type == 'train':
                class TaxonomyTransform(nn.Module):
                    def __init__(self):
                        id_to_uid_maps = tc.id_to_uid_maps[dataset_name]
                        uid_to_id_maps = dict([val, key] for key, val in id_to_uid_maps.items())
                        sorted(uid_to_id_maps.items(), key=lambda item: item[0])
                        self.prediction_mapping

        else:
            raise NotImplementedError
        self.num_eval_classes = len(load_class_names(dataset_name))

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        seg_logits = self.forward(inputs)
        losses = self.losses(seg_logits, gt_semantic_seg)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        logit = self.forward(inputs)
        logit = self.softmax(logit)
        if self.need_convert:
            logit = self.taxonomy_transform(logit)
        return logit


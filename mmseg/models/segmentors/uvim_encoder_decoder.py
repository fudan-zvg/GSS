# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor
from .encoder_decoder import EncoderDecoder

from mseg.taxonomy.taxonomy_converter import TaxonomyConverter


@SEGMENTORS.register_module()
class UViM(EncoderDecoder):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 stage=1,
                 **kwargs):
        super(UViM, self).__init__(**kwargs)
        self.stage = stage
        self.universal2test_mappings = TaxonomyConverter().convs
        for (dataset_name, conv) in self.universal2test_mappings.items():
            self.universal2test_mappings[dataset_name] = conv.cuda()

    def encode_decode(self, img, img_metas, **kwargs):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)
        out = self._decode_head_forward_test_uvim(x, img, img_metas, **kwargs)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def _decode_head_forward_train_uvim(self, x, img, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(x, img, img_metas,
                                                     gt_semantic_seg,
                                                     self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _decode_head_forward_test_uvim(self, x, img, img_metas, **kwargs):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, img, img_metas, kwargs['gt_semantic_seg'], self.test_cfg)
        # try:
        #     seg_logits = self.decode_head.forward_test(x, img, img_metas, kwargs['gt_semantic_seg'], self.test_cfg)
        # except (KeyError, TypeError):
        #     try:
        #         seg_logits = self.decode_head.forward_test(x, img, img_metas, None, self.test_cfg)
        #     except TypeError:
        #         seg_logits = self.decode_head.forward_test(x, img, img_metas, self.test_cfg)
        return seg_logits


    def forward_train(self, img, img_metas, gt_semantic_seg):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        #import pdb;pdb.set_trace() #vege_dog
        x = self.extract_feat(img)
        
        losses = dict()

        loss_decode = self._decode_head_forward_train_uvim(x, img, img_metas,
                                                      gt_semantic_seg)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)

        return losses

    def simple_test(self, img, img_meta, rescale=True, **kwargs):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale, **kwargs)
        #seg_logit = self.AR_inference(img, img_meta, rescale, **kwargs) #autoregressive inference
        num_cls = seg_logit.shape[1]
        seg_logit = F.one_hot(seg_logit.argmax(1).long(), num_cls).float().squeeze(1).permute(0, 3, 1, 2)
        # print('\ncjq debug seg_pred before u2s:', seg_logit.argmax(1))
        # seg_logit = F.softmax(seg_logit, dim=1)
        
        #seg_logit = self.universal2test_mappings[img_meta[0]['dataset_name']](seg_logit) #only on mseg dataset vege_dog
        seg_pred = seg_logit.argmax(dim=1)
        # print('\ncjq debug seg_pred after u2s:', seg_pred)
        # import os
        # file_dir = 'work_dirs/mseg_visualization_stage_2/' + img_meta[0]['dataset_name'] + '/'
        # if not os.path.exists(file_dir):
        #     os.makedirs(file_dir)
        # from torchvision.utils import save_image, make_grid
        # save_image(self.decode_head.encode_to_segmap(seg_pred.long()) / 255.0,
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
        #import pdb;pdb.set_trace()
        return seg_pred

    @torch.no_grad()
    def AR_inference(self, img, img_meta, rescale=True, **kwargs):
        """
        img:tensor[B,C,H,W]
        """
        #import pdb;pdb.set_trace()
        B = img.shape[0]
        tokens, hw_shape = self.backbone.patch_embed(img)   #tokens:tensor[B,N,C], hw_shape:tuple(hp,wp)
        B,N,C = tokens.shape
        tokens = tokens.view(B,hw_shape[0],hw_shape[1],-1)
        
        step = (1,1)
        for i in range(0,hw_shape[0],step[0]):
            for j in range(0,hw_shape[1],step[1]):
                x = tokens.detach().reshape(B,-1,C)
                x = self.backbone.pos_embeding(x, hw_shape)
                outs = self.backbone.token_forward(x,hw_shape)  #outs:list(tensor[B,C,hp,wp])
                z_logit = self.decode_head.forward(outs)    #z_logit:tensor[B,dict_num,hp,wp]
                z_indice = z_logit.argmax(1)    #z_indice:tensor[B,hp,wp]
                z_q = self.decode_head.emb.weight[:, z_indice].permute(1, 2, 3, 0)  #z_q:tensor[B,hp,wp]
                tokens[:,i:(i+step[0]),j:(j+step[1])] = z_q[:,i:i+step[0],j:j+step[1]]

        #import pdb;pdb.set_trace()
        tokens = tokens.permute(0,3,1,2)    #tokens:tensor[B,C,hp,wp]
        z = self.decode_head.decode(img, guide_code=tokens)
        seg_logit = self.decode_head.conv_seg(z)

        seg_logit= resize(
            input=seg_logit,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        
        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                # remove padding area
                resize_shape = img_meta[0]['img_shape'][:2]
                seg_logit = seg_logit[:, :, :resize_shape[0], :resize_shape[1]]
                size = img_meta[0]['ori_shape'][:2]
            seg_logit = resize(
                seg_logit,
                size=size,
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

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



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
from .big_seg_head_wo_cpm_tranformer import BigSegAggHeadWoCPMTransformer
from mmseg.models.backbones.swin import SwinBlockSequence

@HEADS.register_module()
class BigSegAggHeadWoCPMTransformerSave(BigSegAggHeadWoCPMTransformer):
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
                 save_path='work_dirs/bigseg_cityscapes_conns_swin_160k_025_dim_768_2048_wo_cpm_bs2x16_transformer_right_color_lr/indice_iter_16000/',
                 **kwargs):
        super(BigSegAggHeadWoCPMTransformerSave, self).__init__(**kwargs)
        self.root = '/home/chenyurui/pj/mmsegmentation'
        self.save_path = os.path.join(self.root, save_path)

    def forward_test(self, inputs, img_metas, gt_semantic_seg, test_cfg):
        #vege_dog
        #import pdb;pdb.set_trace()
        inputs = self._transform_inputs(inputs)
        x = self.feature_aggregation(inputs)
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x, hw, x_down, hw_down = self.transformer_block(x, (h, w))
        x = self.swin_ln(x)
        x = x.transpose(1, 2).view(b, c, h, w)
        x = self.conv_before_seg(x)
        vq_logit = self.forward(x).view(-1, self.vocab_size, h, w)
        vq_indices = vq_logit.argmax(1).unsqueeze(1)
        
        full_path = os.path.join(self.save_path, 'val', img_metas[0]['ori_filename'].split('.')[0] + '.pth')
        save_path, file_name = os.path.split(full_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(full_path, 'wb') as f:
            torch.save(dict(vq_indices=vq_indices, img_metas=img_metas), f)
        mask = gt_semantic_seg[0]
        mask[mask == self.pixel_ignore_index] = self.num_classes
        seg_logit = F.one_hot(mask.to(torch.int64), self.num_classes + 1).permute(0, 3, 1, 2)[:, :self.num_classes,:,:].to(torch.float)
        print('cjq save:', file_name)
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
            seg_indices = self.decode_from_segmap(rec_segmap, keep_ignore_index=False)
            seg_logist = F.one_hot(seg_indices.to(torch.int64), self.num_classes).squeeze(1).permute(0, 3, 1, 2).to(
                torch.float)
            seg_logist = F.interpolate(seg_logist, size=gt_semantic_seg_item.shape[-2:], mode='bilinear')
            results.append(seg_logist) # [:,:self.num_classes,:,:]
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
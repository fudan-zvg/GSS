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
from ..utils.dalle_d_vae import get_dalle_vae, map_pixels, unmap_pixels, encode_to_segmap, decode_from_segmap
# from .vqgan.vqgan import get_taming_vae
import torch
from mmcv.cnn import ConvModule

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.ops import resize

@HEADS.register_module()
class MaskVQSegAggHeadV3(BaseDecodeHead):
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
                 d_vae_type=None,
                 task_type=None,
                 init_cfg=[
                     dict(type='Constant', val=1.0, bias=0, layer='LayerNorm'),
                     dict(
                         type='Normal',
                         std=0.01,
                         override=dict(name='conv_seg'))],
                 norm_layer=dict(type='LN', eps=1e-6, requires_grad=True),
                 interpolate_mode='bilinear',
                 **kwargs):
        super(MaskVQSegAggHeadV3, self).__init__(init_cfg=init_cfg, input_transform='multiple_select', channels=channels, **kwargs)
        self.channels = channels
        _, self.norm = build_norm_layer(norm_layer, self.channels)

        # from segformer head
        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)
        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

        if d_vae_type == 'dalle':
            self.d_vae = get_dalle_vae(
                weight_path="ckp",
                device="cuda")
            self.vocab_size = 8192
        elif d_vae_type == 'taming':
            self.d_vae = get_taming_vae()
            self.vocab_size = 1024
        else:
            raise NotImplementedError

        self.conv_seg = nn.Conv2d(channels, self.vocab_size, kernel_size=1)
        self.img_size = img_size
        self.d_vae_type = d_vae_type
        self.task_type = task_type
        self.ignore_index = self.vocab_size
        self.PALETTE = torch.tensor([[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                                [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
                                [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
                                [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
                                [0, 80, 100], [0, 0, 230], [119, 11, 32]])

    def forward(self, x):
        out = self.cls_seg(x)
        return out

    def get_gt_vq_indices(self, gt_semantic_seg):
        gt_segmap = map_pixels(encode_to_segmap(gt_semantic_seg) / 255.0)
        return self.d_vae.get_codebook_indices(gt_segmap)

    def _feature_aggregation(self, inputs):
        inputs = self._transform_inputs(inputs)
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

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        x = self._feature_aggregation(inputs)
        if self.d_vae_type == 'dalle':
            result =  self._forward_train_seg_with_dalle(x, gt_semantic_seg, img_metas)
            return result
        elif self.d_vae_type == 'taming':
            return self._forward_train_seg_with_taming(x, gt_semantic_seg, img_metas)

    def forward_test(self, inputs, img_metas, gt_semantic_seg, test_cfg):
        with torch.no_grad():
            # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
            x = self._feature_aggregation(inputs)
            if self.d_vae_type == 'dalle' and self.task_type == 'recon':
                return self._forward_test_recon_with_dalle(gt_semantic_seg, img_metas)
            elif self.d_vae_type == 'dalle' and self.task_type == 'seg':
                return self._forward_test_seg_with_dalle(x, gt_semantic_seg, img_metas)
            elif self.d_vae_type == 'taming' and self.task_type == 'recon':
                return self._forward_test_recon_with_taming(gt_semantic_seg, img_metas)
            elif self.d_vae_type == 'taming' and self.task_type == 'seg':
                return self._forward_test_seg_with_taming(x, gt_semantic_seg, img_metas)
            else:
                return self._forward_test_output_gt(gt_semantic_seg)

    def _forward_test_recon_with_dalle(self, gt_semantic_seg, img_metas):
        assert isinstance(gt_semantic_seg, list)
        results = []
        for gt_semantic_seg_item in gt_semantic_seg:
            gt_semantic_seg_item[gt_semantic_seg_item == 255] = self.num_classes
            input_segmap = map_pixels(encode_to_segmap(gt_semantic_seg_item) / 255.0)
            input_ids = self.d_vae.get_codebook_indices(input_segmap)
            h, w = input_ids.shape[-2:]
            rec_segmap = self.d_vae.decode(input_ids, img_size=[h, w])
            rec_segmap = unmap_pixels(torch.sigmoid(rec_segmap[:, :3])) * 255
            seg_indices = decode_from_segmap(rec_segmap)
            seg_logist = F.one_hot(seg_indices.to(torch.int64), self.num_classes + 1).squeeze(1).permute(0, 3, 1, 2).to(
                torch.float)[:,:self.num_classes,:,:]
            results.append(seg_logist)
            # error = gt_semantic_seg_item - seg_logist.argmax(1).unsqueeze(1)
            # error[(gt_semantic_seg_item == self.num_classes).unsqueeze(1)] = 0
            # save_image(torch.cat([map_pixels(encode_to_segmap(gt_semantic_seg_item.long()) / 255.0),
            #                       map_pixels(encode_to_segmap(seg_logist.argmax(1).unsqueeze(1).long()) / 255.0),
            #                       torch.Tensor.repeat(error, 3, 1, 1, 1, 1).squeeze(2).permute(1, 0, 2, 3)],
            #                      dim=0), 'work_dirs/dalle768x768/gt_' + img_metas[0]['ori_filename'].split('/')[-1])
            # print('cjq debug dalle ok')
        return torch.cat(results, dim=0)

    def _forward_test_seg_with_dalle(self, inputs, gt_semantic_seg, img_metas):
        h, w = inputs.shape[-2:]
        vq_logist = self.forward(inputs).view(-1, self.vocab_size, h, w)
        h, w = vq_logist.shape[-2:]
        vq_indices = vq_logist.argmax(1).unsqueeze(1)
        rec_segmap = self.d_vae.decode(vq_indices, img_size=[h, w])
        rec_segmap = torch.tensor(unmap_pixels(torch.sigmoid(rec_segmap[:, :3])) * 255).to(vq_logist.device)
        seg_pred = self.decode_from_segmap(rec_segmap)
        seg_pred[seg_pred == self.num_classes] = 0  # b, h, w, c
        seg_logist = F.one_hot(seg_pred.to(torch.int64), self.num_classes).squeeze(1).permute(0, 3, 1, 2).to(torch.float)

        # save images
        gt_semantic_seg[0] = gt_semantic_seg[0].unsqueeze(0)
        gt_semantic_seg[0] = F.interpolate(gt_semantic_seg[0].float(), size=seg_logist.shape[-2:], mode='bilinear')
        error = gt_semantic_seg[0] - seg_logist.argmax(1).unsqueeze(1)
        error[gt_semantic_seg[0] >= self.num_classes] = 0
        save_image(torch.cat([map_pixels(encode_to_segmap(gt_semantic_seg[0].long()) / 255.0),
                              map_pixels(encode_to_segmap(seg_logist.argmax(1).unsqueeze(1).long()) / 255.0),
                              torch.Tensor.repeat(error, 3, 1, 1, 1, 1).squeeze(2).permute(1, 0, 2, 3)],
                             dim=0), 'work_dirs/mask_vqseg_agg_v3_swin_large_patch4_window12_768x768_pretrain_384x384_22K_300e_cityscapes/show/val_' + img_metas[0]['ori_filename'].split('/')[-1])
        # print('cjq save images')

        # save indice data, includes semantic_seg_pred (19 classes), gt_semantic_seg (19 classes), vq_ind_pred (8192 classes), vq_indice_gt (8192 classes)
        gt_semantic_seg = gt_semantic_seg[0]
        gt_semantic_seg[gt_semantic_seg == 255] = self.num_classes
        gt_semantic_seg = F.interpolate(
            F.one_hot(gt_semantic_seg.to(torch.long), self.num_classes + 1).squeeze(1).permute(0, 3, 1, 2).to(
                torch.float),
            size=(h * 8, w * 8), mode='bilinear').argmax(1).unsqueeze(1)
        gt_semantic_seg[gt_semantic_seg == self.num_classes] = 255
        gt_semantic_seg_indices = self.get_gt_vq_indices(gt_semantic_seg).unsqueeze(1)  # % 100

        sio.savemat('work_dirs/mask_vqseg_agg_v3_swin_large_patch4_window12_768x768_pretrain_384x384_22K_300e_cityscapes/annal/' + img_metas[0]['ori_filename'].split('/')[-1].split('.')[0] + '.mat',
                    mdict={'semantic_seg_pred': seg_pred.cpu().numpy(),
                           'gt_semantic_seg': gt_semantic_seg.cpu().numpy(),
                           'vq_indices': vq_indices.cpu().numpy(),
                           'vq_indice_gt': gt_semantic_seg_indices.cpu().numpy()})
        return seg_logist

    def _forward_train_seg_with_dalle(self, inputs, gt_semantic_seg, img_metas):
        h, w = inputs.shape[-2:]
        # get indices logist from ViT
        vq_logits = self.forward(inputs).view(-1, self.vocab_size, h, w)
        # get vq indices from gt by dalle
        with torch.no_grad():
            gt_semantic_seg[gt_semantic_seg == 255] = self.num_classes
            gt_semantic_seg = F.interpolate(F.one_hot(gt_semantic_seg.to(torch.long), self.num_classes + 1).squeeze(1).permute(0, 3, 1, 2).to(torch.float),
                                            size=(h * 8, w * 8), mode='bilinear').argmax(1).unsqueeze(1)
            gt_semantic_seg[gt_semantic_seg == self.num_classes] = 255
            gt_semantic_seg_indices = self.get_gt_vq_indices(gt_semantic_seg).unsqueeze(1) # % 100

            # 获取像素级的预测值，用于判断是否已经达到合格要求
            pred_pixel_segmap = self.d_vae.decode(vq_logits.argmax(1).unsqueeze(1), img_size=[h, w])
            pred_pixel_segmap = unmap_pixels(torch.sigmoid(pred_pixel_segmap[:, :3])) * 255
            pred_pixel = decode_from_segmap(pred_pixel_segmap)

            # 对pixel预测正确的局部进行松弛, 将正确像素标记为0，错误像素为1，便于进行腐蚀
            relaxation_map = torch.ones_like(gt_semantic_seg, device=gt_semantic_seg.device)
            relaxation_map[pred_pixel == gt_semantic_seg] = 0
            # relaxation
            relaxation_map = F.avg_pool2d(relaxation_map.float(), kernel_size=(64, 64), stride=(16, 16))
            relaxation_map = F.interpolate(relaxation_map.float(), size=(h, w), mode='bilinear')
            relaxation_map[relaxation_map >= 0.0001] = 1
            relaxation_map[relaxation_map < 1] = 0

            gt_semantic_seg_indices[relaxation_map == 0] = vq_logits.argmax(1).unsqueeze(1)[relaxation_map == 0]

            # error map
            # error_map = torch.zeros_like(gt_semantic_seg_indices, device=gt_semantic_seg.device)
            # # the correct predicted pixel will be ignored
            # error_map[vq_logits.argmax(1).unsqueeze(1) != gt_semantic_seg_indices] = 1
            # save_image(torch.cat([map_pixels(pred_pixel_segmap / 255.0), # prediction
            #                       map_pixels(gt_pixel_segmap / 255.0), # gt
            #                       torch.Tensor.repeat(F.interpolate(error_map.float(), size=gt_semantic_seg.shape[-2:]),
            #                                           3, 1, 1, 1, 1).squeeze(2).permute(1, 0, 2, 3),  # indice error map
            #                       torch.Tensor.repeat(ignore_map.float(), 3, 1, 1, 1, 1).squeeze(2).permute(1, 0, 2, 3), # ignore map
            #                       torch.Tensor.repeat(F.interpolate(indice_map_mask.float(), size=gt_semantic_seg.shape[-2:]), 3, 1, 1, 1, 1).squeeze(2).permute(1, 0, 2, 3)],
            #                      dim=0), 'work_dirs/mask_vqseg_agg_swin_large_patch4_window12_768x768_pretrain_384x384_22K_300e_cityscapes/show/mask_sup_indice_' + img_metas[0]['ori_filename'].split('/')[-1],
            #             nrow=len(pred_pixel_segmap))
        losses = self.losses(vq_logits, gt_semantic_seg_indices.clone().detach())
        # print('cjq debug loss', losses)

        # # visualization
        # save_image(torch.cat([map_pixels(encode_to_segmap(gt_semantic_seg.long()) / 255.0),
        #                       map_pixels(encode_to_segmap(rec_gt_by_dalle) / 255.0),
        #                       torch.Tensor.repeat(error_map, 3, 1, 1, 1, 1).squeeze(2).permute(1, 0, 2, 3),
        #                       torch.Tensor.repeat(F.interpolate(indice_map_mask, size=gt_semantic_seg.shape[-2:]), 3, 1, 1, 1, 1).squeeze(2).permute(1, 0, 2, 3),],
        #                      dim=0), 'work_dirs/vqseg_vit-large_dalle_8x1_768x768_300e_cityscapes/show_1/dalle_' + img_metas[0]['ori_filename'].split('/')[-1] + '_debug.png')
        # print('cjq debug ok')
        return losses

    def decode_from_segmap(self, segmap):
        PALETTE_ = self.PALETTE.clone().detach().to(segmap.device)
        B, C, H, W = segmap.shape
        p = torch.Tensor.repeat(PALETTE_, B, H, W, 1, 1).permute(0, 3, 4, 1, 2)
        segmap = torch.Tensor.repeat(segmap, 19, 1, 1, 1, 1).permute(1, 0, 2, 3, 4)
        return torch.abs(segmap - p).sum(2).argmin(1).unsqueeze(1)

    # visualization
    # save_image(torch.cat([map_pixels(encode_to_segmap(gt_semantic_seg_item.long()) / 255.0),
    #                       map_pixels(encode_to_segmap(seg_logist.argmax(1).unsqueeze(1).long()) / 255.0),
    #                       torch.Tensor.repeat(gt_semantic_seg_item - seg_logist.argmax(1).unsqueeze(1), 3, 1, 1, 1, 1).squeeze(2).permute(1, 0, 2, 3)],
    #                      dim=0), 'work_dirs/vqseg_vit-large_8x1_768x768_300e_cityscapes_gt_test_2049x1025/show/gt_' + img_metas[0]['ori_filename'].split('/')[-1] + '_debug.png')
    # print('cjq debug ok')
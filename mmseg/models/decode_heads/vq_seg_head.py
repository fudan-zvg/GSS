import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
import mmcv
import os
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from torchvision.utils import save_image, make_grid
from ..utils.dalle_d_vae import get_dalle_vae, map_pixels, unmap_pixels, encode_to_segmap, decode_from_segmap
# from .vqgan.vqgan import get_taming_vae
@HEADS.register_module()
class VQSegHead(BaseDecodeHead):
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
                 img_size,
                 d_vae_type='taming',
                 task_type='recon',
                 init_cfg=[
                     dict(type='Constant', val=1.0, bias=0, layer='LayerNorm'),
                     dict(
                         type='Normal',
                         std=0.01,
                         override=dict(name='conv_seg'))],
                 norm_layer=dict(type='LN', eps=1e-6, requires_grad=True),
                 **kwargs):
        super(VQSegHead, self).__init__(init_cfg=init_cfg, **kwargs)

        _, self.norm = build_norm_layer(norm_layer, self.channels)
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

        self.vq_cls = nn.Conv2d(self.channels,
                                self.vocab_size,
                                kernel_size=1)
        # self.vq_cls = nn.Conv2d(self.in_channels,
        #                         self.vocab_size,
        #                         kernel_size=1)
        self.img_size = img_size
        self.d_vae_type = d_vae_type
        self.task_type = task_type
        self.ignore_index = self.vocab_size

    def forward(self, x):
        out = self.vq_cls(x)
        return out

    def get_gt_vq_indices(self, gt_semantic_seg):
        gt_segmap = map_pixels(encode_to_segmap(gt_semantic_seg) / 255.0)
        return self.d_vae.get_codebook_indices(gt_segmap)

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        x = self._transform_inputs(inputs)
        n, c, h, w = x.shape
        x = x.reshape(n, c, h * w).transpose(2, 1).contiguous()
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(n, c, h, w).contiguous()
        if self.d_vae_type == 'dalle':
            return self._forward_train_seg_with_dalle(x, gt_semantic_seg, img_metas)
        elif self.d_vae_type == 'taming':
            return self._forward_train_seg_with_taming(x, gt_semantic_seg, img_metas)


    def forward_test(self, inputs, img_metas, gt_semantic_seg, test_cfg):
        x = self._transform_inputs(inputs)
        n, c, h, w = x.shape
        x = x.reshape(n, c, h * w).transpose(2, 1).contiguous()
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(n, c, h, w).contiguous()
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
        rec_segmap = unmap_pixels(torch.sigmoid(rec_segmap[:, :3])) * 255
        seg_pred = decode_from_segmap(rec_segmap)
        seg_pred[seg_pred == self.num_classes] = 0  # b, h, w, c
        seg_logist = F.one_hot(seg_pred.to(torch.int64), self.num_classes).squeeze(1).permute(0, 3, 1, 2).to(torch.float)

        # save images
        # gt_semantic_seg[0] = gt_semantic_seg[0].unsqueeze(0)
        # gt_semantic_seg[0] = F.interpolate(gt_semantic_seg[0].float(), size=seg_logist.shape[-2:], mode='bilinear')
        # error = gt_semantic_seg[0] - seg_logist.argmax(1).unsqueeze(1)
        # error[gt_semantic_seg[0] >= self.num_classes] = 0
        # save_image(torch.cat([map_pixels(encode_to_segmap(gt_semantic_seg[0].long()) / 255.0),
        #                       map_pixels(encode_to_segmap(seg_logist.argmax(1).unsqueeze(1).long()) / 255.0),
        #                       torch.Tensor.repeat(error, 3, 1, 1, 1, 1).squeeze(2).permute(1, 0, 2, 3)],
        #                      dim=0), 'work_dirs/vqseg_swin_large_patch4_window12_512x512_pretrain_384x384_22K_160k_ade20k/show/test_swin' + img_metas[0]['ori_filename'].split('/')[-1])
        return seg_logist

    def _forward_test_output_gt(self, gt_semantic_seg):
        results = []
        for gt_semantic_seg_item in gt_semantic_seg:
            gt_semantic_seg_item_ = gt_semantic_seg_item.clone().detach()
            gt_semantic_seg_item_[gt_semantic_seg_item_ == 255] = self.num_classes
            seg_logist = F.one_hot(gt_semantic_seg_item_.to(torch.int64), self.num_classes + 1).permute(0, 3, 1, 2).to(
                torch.float)[:, :self.num_classes, :, :]
            results.append(seg_logist)
        return torch.cat(results, dim=0)

    def _forward_test_seg_with_taming(self, inputs, gt_semantic_seg, img_metas):
        with torch.no_grad():
            h, w = inputs.shape[-2:]
            vq_logist = self.forward(inputs).view(-1, self.vocab_size, h, w)
            vq_indices = vq_logist.argmax(1).unsqueeze(1)
            rec_segmap = self.d_vae.decode_code(vq_indices)
            seg_pred = decode_from_segmap(rec_segmap * 255)
            seg_pred[seg_pred > self.num_classes] = self.num_classes
            seg_logist = F.one_hot(seg_pred.to(torch.int64), self.num_classes + 1
                                   ).squeeze(1).permute(0, 3, 1, 2).to(torch.float)[:, :self.num_classes,: ,:]
        #     error = gt_semantic_seg[0] - seg_logist.argmax(1).unsqueeze(1)
        #     error[(gt_semantic_seg[0] == self.num_classes).unsqueeze(1)] = 0
        #     save_image(torch.cat([map_pixels(encode_to_segmap(gt_semantic_seg[0].long()) / 255.0),
        #                           map_pixels(encode_to_segmap(seg_logist.argmax(1).unsqueeze(1).long()) / 255.0),
        #                           torch.Tensor.repeat(error, 3, 1, 1, 1, 1).squeeze(2).permute(1, 0, 2, 3)],
        #                          dim=0), 'work_dirs/vqseg_vit-large_taming_8x1_768x768_300e_cityscapes/show/taming_' + img_metas[0]['ori_filename'].split('/')[-1])
        # print('cjq save ok', len(gt_semantic_seg))
        return seg_logist

    def _forward_test_recon_with_taming(self, gt_semantic_seg, img_metas):
        assert isinstance(gt_semantic_seg, list)
        results = []
        for gt_semantic_seg_item in gt_semantic_seg:
            gt_semantic_seg_item[gt_semantic_seg_item == 255] = self.num_classes
            input_segmap = encode_to_segmap(gt_semantic_seg_item) / 255.0
            rec_segmap, _, _ = self.d_vae(input_segmap)
            seg_indices = decode_from_segmap(rec_segmap * 255.0)
            seg_logist = F.one_hot(seg_indices.to(torch.int64), self.num_classes + 1).squeeze(1).permute(0, 3, 1, 2).to(
                torch.float)[:, :self.num_classes, :, :]
            results.append(seg_logist)
            # gt_semantic_seg_item = F.interpolate(gt_semantic_seg_item.unsqueeze(0).float(), size=seg_logist.shape[-2:], mode='bilinear')[0]
            # error = torch.abs(gt_semantic_seg_item - seg_logist.argmax(1).unsqueeze(1))
            # error[(gt_semantic_seg_item == self.num_classes).unsqueeze(1)] = 0
            # save_image(torch.cat([map_pixels(encode_to_segmap(gt_semantic_seg_item.long()) / 255.0),
            #                       map_pixels(encode_to_segmap(seg_logist.argmax(1).unsqueeze(1).long()) / 255.0),
            #                       torch.Tensor.repeat(error, 3, 1, 1, 1, 1).squeeze(2).permute(1, 0, 2, 3)],
            #                      dim=0), 'work_dirs/taming_rgb_recon_ade20k/show/' + img_metas[0]['ori_filename'].split('/')[-1])
            # print('cjq save')
        return torch.cat(results, dim=0)

    def _forward_train_seg_with_dalle(self, inputs, gt_semantic_seg, img_metas):
        h, w = inputs.shape[-2:]
        # get indices logist from ViT
        vq_logits = self.forward(inputs).view(-1, self.vocab_size, h, w)
        # get vq indices from gt by dalle
        gt_semantic_seg[gt_semantic_seg == 255] = self.num_classes
        gt_semantic_seg = F.interpolate(F.one_hot(gt_semantic_seg.to(torch.long), self.num_classes + 1).squeeze(1).permute(0, 3, 1, 2).to(torch.float),
                                        size=(h * 8, w * 8), mode='bilinear').argmax(1).unsqueeze(1)
        gt_semantic_seg[gt_semantic_seg == self.num_classes] = 255
        gt_semantic_seg_indices = self.get_gt_vq_indices(gt_semantic_seg).unsqueeze(1) # % 100

        # rec_segmap = self.d_vae.decode(gt_semantic_seg_indices, img_size=[h, w])
        # rec_segmap = unmap_pixels(torch.sigmoid(rec_segmap[:, :3])) * 255
        # rec_gt_by_dalle = decode_from_segmap(rec_segmap)

        # calculate error map and ignore noice indice and ignore them
        error_map = torch.zeros_like(gt_semantic_seg, device=gt_semantic_seg.device)
        # error_map[rec_gt_by_dalle != gt_semantic_seg] = 1
        error_map[gt_semantic_seg >= self.num_classes] = 1
        indice_map_mask = F.max_pool2d(error_map.float(), kernel_size=(8, 8), stride=(8, 8))
        gt_semantic_seg_indices[indice_map_mask == 1] = self.ignore_index
        losses = self.losses(vq_logits, gt_semantic_seg_indices)

        # # visualization
        # save_image(torch.cat([map_pixels(encode_to_segmap(gt_semantic_seg.long()) / 255.0),
        #                       map_pixels(encode_to_segmap(rec_gt_by_dalle) / 255.0),
        #                       torch.Tensor.repeat(error_map, 3, 1, 1, 1, 1).squeeze(2).permute(1, 0, 2, 3),
        #                       torch.Tensor.repeat(F.interpolate(indice_map_mask, size=gt_semantic_seg.shape[-2:]), 3, 1, 1, 1, 1).squeeze(2).permute(1, 0, 2, 3),],
        #                      dim=0), 'work_dirs/vqseg_vit-large_dalle_8x1_768x768_300e_cityscapes/show_1/dalle_' + img_metas[0]['ori_filename'].split('/')[-1] + '_debug.png')
        # print('cjq debug ok')
        return losses

    def _forward_train_seg_with_taming(self, inputs, gt_semantic_seg, img_metas):
        h, w = inputs.shape[-2:]
        # get vq_logist prediction from ViT
        vq_logits = self.forward(inputs).view(-1, self.vocab_size, h, w)
        with torch.no_grad():
            # get gt_indices from dalle
            seg_map = encode_to_segmap(gt_semantic_seg)
            rec_segmap, _, (_, _, gt_semantic_seg_indices) = self.d_vae(seg_map / 255.0)  # % 100
            gt_semantic_seg_indices = gt_semantic_seg_indices.view(-1, 1, h, w)
            seg_indices = decode_from_segmap(rec_segmap * 255.0)

            # get prediction
            # pred_segmap = self.d_vae.decode_code(vq_logits.argmax(1).unsqueeze(1))
            # pred_seg_indices = decode_from_segmap(pred_segmap * 255)
            # pred_seg_indices[pred_seg_indices > self.num_classes] = self.num_classes
            # pred_segmap_from_pred_indices = encode_to_segmap(pred_seg_indices)
            # pred_seg_logist = F.one_hot(pred_seg_indices.to(torch.int64), self.num_classes + 1
            #                        ).squeeze(1).permute(0, 3, 1, 2).to(torch.float)[:, :self.num_classes, :, :]

            error_map = torch.zeros_like(seg_indices, device=seg_indices.device)
            error_map[seg_indices != gt_semantic_seg] = 1
            # error_map[pred_seg_indices != gt_semantic_seg] = 1
            error_map[gt_semantic_seg >= self.num_classes] = 1
            indice_map_mask = F.max_pool2d(error_map.float(), kernel_size=(16, 16), stride=(16, 16))

            # visualization
            # error_map_show = torch.Tensor.repeat(error_map, 3, 1, 1, 1, 1).squeeze(2).permute(1, 0, 2, 3)
            # indice_map_mask_show = F.interpolate(indice_map_mask, mode='nearest', scale_factor=16)
            # indice_map_mask_show = torch.Tensor.repeat(indice_map_mask_show, 3, 1, 1, 1, 1).squeeze(2).permute(1, 0, 2, 3)
            # save_image(torch.cat([(seg_map / 255.0).cpu(),
            #                       (pred_segmap_from_pred_indices / 255.0).cpu(),
            #                       error_map_show.cpu(), indice_map_mask_show.cpu()],
            #                      dim=0), 'work_dirs/vqseg_vit-large_taming_8x1_768x768_300e_cityscapes/train_show_' + img_metas[0]['ori_filename'].split('/')[-1] + '_debug.png')
            # print('ok cjq save train')
            gt_semantic_seg_indices = gt_semantic_seg_indices.view(-1, 1, h, w)
            gt_semantic_seg_indices[indice_map_mask == 1] = self.ignore_index
        losses = self.losses(vq_logits, gt_semantic_seg_indices.clone().detach())
        return losses


    # visualization
    # save_image(torch.cat([map_pixels(encode_to_segmap(gt_semantic_seg_item.long()) / 255.0),
    #                       map_pixels(encode_to_segmap(seg_logist.argmax(1).unsqueeze(1).long()) / 255.0),
    #                       torch.Tensor.repeat(gt_semantic_seg_item - seg_logist.argmax(1).unsqueeze(1), 3, 1, 1, 1, 1).squeeze(2).permute(1, 0, 2, 3)],
    #                      dim=0), 'work_dirs/vqseg_vit-large_8x1_768x768_300e_cityscapes_gt_test_2049x1025/show/gt_' + img_metas[0]['ori_filename'].split('/')[-1] + '_debug.png')
    # print('cjq debug ok')
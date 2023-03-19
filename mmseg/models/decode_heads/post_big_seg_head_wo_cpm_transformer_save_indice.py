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
from .post_big_seg_head_wo_cpm_tranformer import PostSwinBigSegAggHeadWoCPMTransformer

TEST_SPLIT_PALETTE = {
    'camvid-11': [[128, 0, 0], [128, 128, 0], [128, 128, 128], [64, 0, 128],
                  [192, 128, 128], [128, 64, 128], [64, 64, 0], [64, 64, 128],
                  [192, 192, 128], [0, 0, 192], [0, 128, 192], [0, 0, 0]],
    'kitti-19': [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                 [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
                 [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
                 [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100],
                 [0, 0, 230], [119, 11, 32], [0, 0, 0]],
    'pascal-context-60': [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
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
                          [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255], [0, 0, 0]],
    'scannet-20': [[8, 255, 51], [255, 6, 82], [255, 6, 51], [7, 255, 224],
                   [224, 255, 8], [6, 51, 255], [140, 140, 140], [20, 255, 0],
                   [255, 224, 0], [255, 82, 0], [255, 61, 6], [255, 184, 184],
                   [255, 112, 0], [140, 140, 140], [153, 0, 255], [255, 204, 0],
                   [0, 204, 255], [0, 41, 255], [184, 255, 0], [92, 0, 255], [0, 0, 0]],
    'voc2012': [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
                [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
                [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
                [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
                [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128], [0, 0, 0]],
    'wilddash-19': [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
                    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
                    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
                    [0, 80, 100], [0, 0, 230], [119, 11, 32], [0, 0, 0]],
}

@HEADS.register_module()
class PostSwinBigSegAggHeadWoCPMTransformerSave(PostSwinBigSegAggHeadWoCPMTransformer):
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
                 save_path='work_dirs/bigseg_cityscapes_conns_swin_160k_025_dim_768_2048_wo_cpm_bs2x16_transformer_right_color_lr_post/indice_iter_16000/',
                 **kwargs):
        super(PostSwinBigSegAggHeadWoCPMTransformerSave, self).__init__(**kwargs)
        self.root = '/home/chenjiaqi/pj/mmsegmentation'
        self.save_path = os.path.join(self.root, save_path)


    def forward_test(self, inputs, img_metas, gt_semantic_seg, test_cfg):
        # gt_semantic_seg = [torch.ones((1, 1, 512, 512)).cuda()]
        # return self._forward_test_recon_with_dalle(gt_semantic_seg, img_metas)
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

        full_path = os.path.join(self.save_path, 'val', img_metas[0]['ori_filename'].split('.')[0] + '.pth')
        save_path, file_name = os.path.split(full_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(full_path, 'wb') as f:
            torch.save(dict(vq_indices=vq_indices, img_metas=img_metas), f)


        mask = gt_semantic_seg[0]
        mask[mask == self.pixel_ignore_index] = self.num_classes
        seg_logits = F.one_hot(mask.to(torch.int64), self.num_classes + 1).permute(0, 3, 1, 2)[:, :self.num_classes,:,:].to(torch.float)
        # rec_segmap = self.d_vae.decode(vq_indices, img_size=[h, w])
        # rec_segmap = unmap_pixels(torch.sigmoid(rec_segmap[:, :3]))
        # # rec_segmap = self.get_recon_from_dalle(gt_semantic_seg, img_metas) / 255.0
        # b, c, h, w = rec_segmap.shape
        # x = self.projection(rec_segmap)
        # x = x.flatten(2).transpose(1, 2)
        # x, hw, x_down, hw_down = self.post_transformer_block(x, (h, w))
        # x = self.post_swin_ln(x)
        # x = x.transpose(1, 2).view(b, self.post_seg_channel, h, w)
        # seg_logits = self.cls_segmap(x)

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

        # mseg
        # filename: data/mseg_dataset/PASCAL_VOC_2012/JPEGImages/2007_000033.jpg


        # self.visual_palette = torch.tensor(TEST_SPLIT_PALETTE[img_metas[0]['dataset_name']])
        # import os
        # file_dir = 'work_dirs/visualization_stage_2/mseg_learnable/' + img_metas[0]['dataset_name'] + '/'
        # if not os.path.exists(file_dir):
        #     os.makedirs(file_dir)
        # import shutil
        # def mycopyfile(srcfile, dstpath):  # 复制函数
        #     if not os.path.isfile(srcfile):
        #         print("%s not exist!" % (srcfile))
        #     else:
        #         fpath, fname = os.path.split(srcfile)  # 分离文件名和路径
        #         if not os.path.exists(dstpath):
        #             os.makedirs(dstpath)  # 创建路径
        #         shutil.copy(srcfile, dstpath + fname)  # 复制文件
        #         print("copy %s -> %s" % (srcfile, dstpath + fname))
        # mycopyfile(srcfile=img_metas[0]['filename'], dstpath=file_dir)
        # save_image(self.v_encode_to_segmap(gt_semantic_seg[0].long()) / 255.0,
        #            file_dir +
        #            img_metas[0]['ori_filename'].split('/')[-1].split('.')[0] + '_gt.png')

        return seg_logits

    def v_encode_to_segmap(self, indice):
        PALETTE_ = self.visual_palette.clone().to(indice.device)
        _indice = indice.clone().detach()
        _indice[_indice == self.ignore_index] = len(PALETTE_) - 1
        return PALETTE_[_indice.long()].squeeze(1).permute(0, 3, 1, 2)

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
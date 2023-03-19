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
import os
import _pickle as cPickle
from ..utils.dalle_d_vae import get_dalle_vae, map_pixels, unmap_pixels
from torchvision.utils import save_image, make_grid
@SEGMENTORS.register_module()
class DalleDecoderLoadOnly(EncoderDecoder):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 palette,
                 num_classes,
                 load_dir='work_dirs/bigseg_cityscapes_conns_swin_160k_025_dim_768_2048_wo_cpm_bs2x16_transformer_right_color_lr/indice_iter_16000/val/',
                 **kwargs):
        super(DalleDecoderLoadOnly, self).__init__(**kwargs)
        self.universal2test_mappings = TaxonomyConverter().convs
        for (dataset_name, conv) in self.universal2test_mappings.items():
            self.universal2test_mappings[dataset_name] = conv.cuda()
        self.root = '/home/chenjiaqi/pj/mmsegmentation'
        self.load_path = os.path.join(self.root, load_dir)
        self.palette = torch.tensor(palette)
        self.num_classes = num_classes
        self.d_vae = get_dalle_vae(
            weight_path="/home/chenjiaqi/pj/mmsegmentation/ckp",
            device="cuda")
        self.d_vae = self.d_vae.cuda()
        #vege_dog 
        #import pdb;pdb.set_trace()

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


    def whole_inference(self, img, img_meta, rescale, **kwargs):
        class StrToBytes:
            def __init__(self, fileobj):
                self.fileobj = fileobj

            def read(self, size):
                return self.fileobj.read(size).encode()

            def readline(self, size=-1):
                return self.fileobj.readline(size).encode()

        # with open('final_project_dataset.pkl', 'r') as data_file:
        #     data_dict = pickle.load()
        """Inference with full image."""
        file_name = img_meta[0]['ori_filename'].split('.')[0] + '.pth'
        # with open(os.path.join(self.load_path, file_name)) as f:
        info = torch.load(os.path.join(self.load_path, file_name))
        indice = info['vq_indices'].cuda()
        h, w = indice.shape[-2:]
        #vege_dog 
        #import pdb;pdb.set_trace()
        rec_segmap = self.d_vae.decode(indice, img_size=[h, w])
        rec_segmap = unmap_pixels(torch.sigmoid(rec_segmap[:, :3])) * 255
        
        seg_pred = self.decode_from_segmap(torch.tensor(rec_segmap), keep_ignore_index=False, prob=False)
        # seg_pred[seg_pred == self.num_classes] = 0  # b, h, w, c
        seg_logit = F.one_hot(seg_pred.to(torch.int64), self.num_classes).squeeze(1).permute(0, 3, 1, 2).to(
            torch.float)
        # seg_logit = self.encode_decode(img, img_meta, **kwargs)
        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                # remove padding area
                resize_shape = img_meta[0]['img_shape'][:2]
                #seg_logit = seg_logit[:, :, :resize_shape[0], :resize_shape[1]]
                size = img_meta[0]['ori_shape'][:2]
            seg_logit = resize(
                seg_logit,
                size=size,
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        
        # save images
        
        #import pdb;pdb.set_trace()

        # save_image(rec_segmap / 255.0, 'work_dirs/bigseg_cityscapes_conns_swin_160k_025_dim_768_2048_wo_cpm_bs2x8_transformer_load_indice/show/' + img_meta[0]['ori_filename'].split('/')[-1])
        # print('cjq save images')
        return seg_logit
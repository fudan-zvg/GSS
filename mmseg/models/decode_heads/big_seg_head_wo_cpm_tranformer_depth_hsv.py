import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer

from torchvision.utils import save_image, make_grid
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from ..utils.dalle_d_vae import get_dalle_vae, map_pixels, unmap_pixels
import torch
from mmcv.cnn import ConvModule
from ..losses import accuracy
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.ops import resize
from ..utils.hsv import RGB_HSV
from mmseg.models.backbones.swin import SwinBlockSequence
@HEADS.register_module()
class BigSegAggHeadWoCPMTransformerDepthHSV(BaseDecodeHead):
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
                 init_cfg=[
                     dict(type='Constant', val=1.0, bias=0, layer='LayerNorm'),
                     dict(
                         type='Normal',
                         std=0.01,
                         override=dict(name='conv_seg'))],
                 norm_layer=dict(type='LN', eps=1e-6, requires_grad=True),
                 indice_channel_index=0,
                 pixel_channel_index=1,
                 interpolate_mode='bilinear',
                 palette=None,
                 indice_seg_channel=None,
                 indice_cls_channel=2048,
                 swin_num_head=16,
                 swin_depth=2,
                 swin_window_size=7,
                 **kwargs):
        super(BigSegAggHeadWoCPMTransformerDepthHSV, self).__init__(init_cfg=init_cfg, input_transform='multiple_select', channels=channels, **kwargs)
        num_inputs = len(self.in_channels)
        assert num_inputs == len(self.in_index)
        self.channels = channels
        self.palette = torch.tensor(palette)
        self.interpolate_mode = interpolate_mode
        self.img_size = img_size
        self.vocab_size = 8192
        self.indice_ignore_index = self.vocab_size
        self.pixel_ignore_index = 255
        _, self.norm = build_norm_layer(norm_layer, self.channels)
        self.pixel_channel_index = pixel_channel_index
        self.indice_channel_index = indice_channel_index
        self.indice_seg_channel = indice_seg_channel if indice_seg_channel is not None else channels
        self.indice_cls_channel = indice_cls_channel
        self.max_depth = 10
        self.min_depth = 1e-3
        self.d_vae = get_dalle_vae(
            weight_path="/home/chenjiaqi/pj/mmsegmentation/ckp",
            device="cuda")

        self.swin_num_head = swin_num_head
        self.swin_depth = swin_depth
        self.swin_window_size = swin_window_size
        # dense classificator
        self.conv_before_seg = ConvModule(
                    in_channels=self.indice_seg_channel,
                    out_channels=self.indice_cls_channel,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg)
        self.conv_seg = nn.Conv2d(self.indice_cls_channel, self.vocab_size, kernel_size=1)
        self.conv_seg_pixel = nn.Conv2d(channels, 1, kernel_size=1)
        _, self.swin_ln = build_norm_layer(dict(type='LN'), self.indice_seg_channel)

        # input translation
        self.convs = nn.ModuleList()
        self.convs_pixel = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg)
                )
            self.convs_pixel.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        # fusion blocks
        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)
        self.fusion_conv_pixel = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)
        self.count = 0
        self.mean_rmse = 0

        self.transformer_block = SwinBlockSequence(
            embed_dims=self.indice_seg_channel,
            num_heads=self.swin_num_head,
            feedforward_channels=self.indice_seg_channel * 2,
            depth=self.swin_depth,
            window_size=self.swin_window_size,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0,
            attn_drop_rate=0,
            drop_path_rate=0,
            downsample=None,
            norm_cfg=dict(type='LN'),
            with_cp=False,
            init_cfg=None)
        self.convertor = RGB_HSV()

    def forward(self, x):
        out = self.cls_seg(x)
        return out

    def feature_aggregation(self, inputs):
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            x = self.convs[idx](x)
            outs.append(
                resize(
                    input=x,
                    size=inputs[self.indice_channel_index].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        out = self.fusion_conv(torch.cat(outs, dim=1))
        return out

    def feature_aggregation_for_pixel(self, inputs):
        outs_for_pixel = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs_pixel[idx]
            outs_for_pixel.append(
                resize(
                    input=conv(x),
                    size=inputs[self.pixel_channel_index].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))
        out_for_pixel = self.fusion_conv_pixel(torch.cat(outs_for_pixel, dim=1))
        return out_for_pixel

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        inputs = self._transform_inputs(inputs)
        x = self.feature_aggregation(inputs)
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x, hw, x_down, hw_down = self.transformer_block(x, (h, w))
        x = self.swin_ln(x)
        x = x.transpose(1, 2).view(b, c, h, w)
        x = self.conv_before_seg(x)
        x_p = self.feature_aggregation_for_pixel(inputs)
        # print('cjq debug shape pixle x', x_p.shape)
        h_p, w_p = x_p.shape[-2:]
        vq_logits = self.forward(x).view(-1, self.vocab_size, h, w)
        pixel_depth = self.conv_seg_pixel(x_p).view(-1, 1, h_p, w_p)
        pixel_depth = F.sigmoid(pixel_depth) * self.max_depth

        # get vq indices from gt by dalle
        with torch.no_grad():
            if not pixel_depth.shape[-2:] == (h * 8, w * 8):
                pixel_depth = F.interpolate(pixel_depth, size=(h * 8, w * 8), mode='bilinear')
            if not gt_semantic_seg.shape[-2:] == (h * 8, w * 8):
                gt_semantic_seg = F.interpolate(gt_semantic_seg.float(),
                    size=(h * 8, w * 8), mode='bilinear')
            gt_semantic_seg[gt_semantic_seg > self.max_depth] = self.pixel_ignore_index
            gt_semantic_seg[gt_semantic_seg < self.min_depth] = self.pixel_ignore_index
            gt_semantic_seg = gt_semantic_seg.float()

            # get non-ignore gt_indice
            gt_semantic_seg_for_recon = torch.zeros_like(gt_semantic_seg).float()
            gt_semantic_seg_for_recon[gt_semantic_seg != self.pixel_ignore_index] = gt_semantic_seg[gt_semantic_seg != self.pixel_ignore_index].clone()
            gt_semantic_seg_for_recon[gt_semantic_seg == self.pixel_ignore_index] = pixel_depth[gt_semantic_seg == self.pixel_ignore_index].clone()
            gt_semantic_seg_for_recon =  self.depth_norm(gt_semantic_seg_for_recon).float()
            gt_semantic_seg_for_recon = torch.cat([gt_semantic_seg_for_recon, gt_semantic_seg_for_recon, gt_semantic_seg_for_recon], dim=1)
            gt_semantic_seg_for_recon = map_pixels(gt_semantic_seg_for_recon)
            gt_semantic_seg_indices = self.d_vae.get_codebook_indices(gt_semantic_seg_for_recon).unsqueeze(1) # % 100

            # get ignore mask
            ignore_map = torch.ones_like(gt_semantic_seg, device=gt_semantic_seg.device)
            ignore_map[gt_semantic_seg == self.pixel_ignore_index] = 0
            ignore_mask = F.max_pool2d(ignore_map.float(), kernel_size=(8, 8), stride=(8, 8))

            indice_map_mask = ignore_mask

            # get final gt indices
            masked_gt_semantic_seg_indices = gt_semantic_seg_indices.clone()
            masked_gt_semantic_seg_indices[indice_map_mask == 0] = self.indice_ignore_index

            # ignore area mask for pixel
            pixel_depth[gt_semantic_seg == self.pixel_ignore_index] \
                = gt_semantic_seg[gt_semantic_seg == self.pixel_ignore_index]

        # print('\n RMSE:', rmse)
        # save_image(map_pixels(self.depth_norm(gt_semantic_seg)), 'work_dirs/nyu_depth_pred/show_gt_final_wo_uai/gt_val_' + img_metas[0]['ori_filename'].split('/')[-1])
        # save_image(torch.cat([gt_semantic_seg[0].unsqueeze(0) / 255.0,
        #                       depth_pred.unsqueeze(0) / 255.0],
        #                      dim=0), 'work_dirs/nyu_depth_pred/show_final_pred/val_' + img_metas[0]['ori_filename'].split('/')[-1])
        # print('cjq save images')
        losses = self.losses(
            indice_logit=vq_logits,
            pixel_depth_logit=pixel_depth,
            indice_label=masked_gt_semantic_seg_indices,
            pixel_depth_label=gt_semantic_seg,
            valid_mask=(gt_semantic_seg != self.pixel_ignore_index)
        )
        return losses

    def depth_norm(self, x):
        return (x - self.min_depth) / (self.max_depth - self.min_depth)

    def depth_denorm(self, x):
        return x * (self.max_depth - self.min_depth) + self.min_depth

    def forward_test(self, inputs, img_metas, gt_semantic_seg, test_cfg):
        # return self._forward_test_recon_with_dalle(gt_semantic_seg, img_metas)
        # print('cjq debug forward test', gt_semantic_seg)
        inputs = self._transform_inputs(inputs)
        x = self.feature_aggregation(inputs)
        h, w = x.shape[-2:]
        x = F.interpolate(x, size=(int(h / 2.0), int(w / 2.0)), mode='bilinear')
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
        depth_logit = rec_segmap.mean(1).unsqueeze(0)
        depth_pred = self.depth_denorm(depth_logit)
        # depth_pred = F.sigmoid(depth_pred) * self.max_depth
        depth_pred = F.interpolate(depth_pred, size=gt_semantic_seg[0].shape[-2:], mode='bilinear')
        rmse = torch.sqrt(F.mse_loss(depth_pred, gt_semantic_seg[0]))
        self.count += 1
        self.mean_rmse += rmse
        print('\n RMSE:', self.mean_rmse.item() / self.count)
        # save_image(map_pixels(self.depth_norm(depth_pred)), 'work_dirs/nyu_depth_pred/iter_20k/val_' + img_metas[0]['ori_filename'].split('/')[-1])
        # save_image(self.depth_norm(gt_semantic_seg[0].unsqueeze(0)), 'work_dirs/nyu_depth_pred/iter_8k_tf_corr_rgb/val_com_rgb' + img_metas[0]['ori_filename'].split('/')[-1].split('.')[0] + '_gt.png')
        save_image(rec_segmap, 'work_dirs/nyu_depth_pred/iter_56k_tf_corr_rgb_corr_hsv/val_com' + img_metas[0]['ori_filename'].split('/')[-1].split('.')[0] + '_recmap.png')
        save_image(torch.cat([self.depth_norm(gt_semantic_seg[0].unsqueeze(0)),
                              self.depth_norm(depth_pred)],
                             dim=0), 'work_dirs/nyu_depth_pred/iter_56k_tf_corr_rgb_corr_hsv/val_com' + img_metas[0]['ori_filename'].split('/')[-1].split('.')[0] + '_pred_gt.png')
        print('cjq save images')
        return depth_pred

    def decode_from_segmap(self, segmap, keep_ignore_index):
        PALETTE_ = self.palette.clone().to(segmap.device) \
            if keep_ignore_index \
            else self.palette[:-1].clone().to(segmap.device)
        B, C, H, W = segmap.shape
        p = torch.Tensor.repeat(PALETTE_, B, H, W, 1, 1).permute(0, 3, 4, 1, 2)
        if keep_ignore_index:
            segmap = torch.Tensor.repeat(segmap, self.num_classes + 1, 1, 1, 1, 1).permute(1, 0, 2, 3, 4)
        else:
            segmap = torch.Tensor.repeat(segmap, self.num_classes, 1, 1, 1, 1).permute(1, 0, 2, 3, 4)
        return torch.abs(segmap - p).sum(2).argmin(1).unsqueeze(1)

    def _forward_test_recon_with_dalle(self, gt_semantic_seg, img_metas):
        assert isinstance(gt_semantic_seg, list)
        results = []
        gt_semantic_seg_item = gt_semantic_seg[0]
        gt_semantic_seg_item = self.depth_norm(gt_semantic_seg_item)

        h, s, v = gt_semantic_seg_item, torch.ones_like(gt_semantic_seg_item), torch.ones_like(gt_semantic_seg_item)
        hsv_depth = torch.cat([h, s, v], dim=0).unsqueeze(0)
        # import ipdb
        # ipdb.set_trace()
        rgb_depth = self.convertor.hsv_to_rgb(hsv_depth)
        input_depth_map = rgb_depth
        input_segmap = map_pixels(input_depth_map.float())
        input_ids = self.d_vae.get_codebook_indices(input_segmap)
        h, w = input_ids.shape[-2:]
        rec_segmap = self.d_vae.decode(input_ids, img_size=[h, w])
        rec_segmap = unmap_pixels(torch.sigmoid(rec_segmap[:, :3]))
        rec_rgb_depth = rec_segmap
        rec_hsv_depth = self.convertor.rgb_to_hsv(rec_rgb_depth)
        depth_pred = rec_hsv_depth[:,0:1,:,:]
        depth_pred = self.depth_denorm(depth_pred)
        depth_pred = F.interpolate(depth_pred, size=gt_semantic_seg[0].shape[-2:], mode='bilinear')
        rmse = torch.sqrt(F.mse_loss(depth_pred, gt_semantic_seg[0]))
        self.count += 1
        self.mean_rmse += rmse
        print('\n RMSE:', self.mean_rmse.item() / self.count)
        # print('cjq debug:', rmse)
        # we have to output an semantic prediction to run the framework
        # save_image(rec_segmap, 'work_dirs/nyu_depth_pred/rec/val_com' + img_metas[0]['ori_filename'].split('/')[-1].split('.')[0] + '_recmap.png')
        # save_image(torch.cat([self.depth_norm(gt_semantic_seg[0].unsqueeze(0)),
        #                       self.depth_norm(depth_pred)],
        #                      dim=0), 'work_dirs/nyu_depth_pred/rec/val_com' + img_metas[0]['ori_filename'].split('/')[-1].split('.')[0] + '_pred_gt.png')
        rec_segmap_for_ss = rec_segmap * 255
        seg_indices = self.decode_from_segmap(rec_segmap_for_ss, keep_ignore_index=False)
        seg_logist = F.one_hot(seg_indices.to(torch.int64), self.num_classes).squeeze(1).permute(0, 3, 1, 2).to(
            torch.float)
        seg_logist = F.interpolate(seg_logist, size=gt_semantic_seg_item.shape[-2:], mode='bilinear')
        results.append(seg_logist)

        return torch.cat(results, dim=0)

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self,
               indice_logit,
               pixel_depth_logit,
               indice_label,
               pixel_depth_label,
               valid_mask
               ):
        """Compute segmentation loss."""
        loss = dict()
        indice_logit = resize(
            input=indice_logit,
            size=indice_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        pixel_logit = resize(
            input=pixel_depth_logit,
            size=pixel_depth_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        # valid_mask = resize(
        #     input=valid_mask,
        #     size=pixel_depth_label.shape[2:],
        #     mode='nearest',
        #     align_corners=None)
        if self.sampler is not None:
            indice_weight = self.sampler.sample(indice_logit, indice_label)
            pixel_weight = self.sampler.sample(pixel_logit, pixel_depth_label)
        else:
            indice_weight = None
            pixel_weight = None
        masked_indice_seg_label = indice_label.squeeze(1)
        # pixel_depth_label = pixel_depth_label.squeeze(1)
        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                if loss_decode.loss_name == 'loss_ce':
                    loss[loss_decode.loss_name] = loss_decode(
                        indice_logit,
                        masked_indice_seg_label,
                        weight=indice_weight,
                        ignore_index=self.indice_ignore_index)
                elif loss_decode.loss_name == 'loss_pixel':
                    loss[loss_decode.loss_name] = loss_decode(
                        pixel_logit[valid_mask],
                        pixel_depth_label[valid_mask],
                        weight=pixel_weight)
            else:
                if loss_decode.loss_name == 'loss_ce':
                    loss[loss_decode.loss_name] += loss_decode(
                        indice_logit,
                        masked_indice_seg_label,
                        weight=indice_weight,
                        ignore_index=self.indice_ignore_index)
                elif loss_decode.loss_name == 'loss_pixel':
                    loss[loss_decode.loss_name] += loss_decode(
                        pixel_logit[valid_mask],
                        pixel_depth_label[valid_mask],
                        weight=pixel_weight)

        loss['rmse_aux'] = torch.sqrt(F.mse_loss(pixel_logit, pixel_depth_label, reduction='mean'))
        loss['acc_seg_indice'] = accuracy(
            indice_logit, masked_indice_seg_label, ignore_index=self.indice_ignore_index)
        return loss

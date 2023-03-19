# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on OpenAI DALL-E and lucidrains' DALLE-pytorch code bases
# https://github.com/openai/DALL-E
# https://github.com/lucidrains/DALLE-pytorch
# --------------------------------------------------------'
from math import sqrt
import os
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange
# cityscapes的颜色空间
PALETTE = torch.tensor([[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
           [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
           [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
           [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
           [0, 80, 100], [0, 0, 230], [119, 11, 32], [0, 0, 0]])

# 自己定义的颜色空间
# PALETTE = torch.tensor([[222, 222, 145], [18, 30, 7], [8, 23, 47], [30, 6, 96], [1, 13, 164], [12, 28, 191], [25, 52, 32], [29, 48, 52], [15, 51, 95], [25, 56, 167], [25, 42, 210], [27, 81, 31], [9, 88, 54], [27, 92, 113], [11, 99, 151], [26, 110, 183], [24, 130, 26], [4, 122, 75], [3, 132, 98], [26, 147, 167], [17, 132, 197], [5, 169, 28], [19, 184, 67], [0, 190, 122], [12, 167, 147], [6, 161, 196], [2, 205, 3], [5, 220, 61], [23, 225, 107], [7, 217, 157], [25, 208, 191], [74, 10, 8], [69, 30, 69], [56, 4, 98], [61, 29, 164], [60, 10, 194], [60, 52, 19], [74, 69, 52], [65, 68, 116], [81, 41, 161], [70, 60, 197], [66, 81, 14], [55, 107, 61], [76, 110, 108], [74, 104, 162], [72, 94, 197], [60, 133, 16], [69, 128, 67], [59, 148, 104], [65, 133, 154], [68, 128, 183], [79, 181, 11], [76, 170, 56], [71, 175, 103], [53, 162, 137], [53, 182, 183], [51, 229, 26], [51, 202, 51], [69, 213, 122], [63, 213, 161], [71, 203, 197], [120, 11, 31], [124, 3, 68], [131, 2, 98], [113, 1, 162], [102, 13, 209], [109, 50, 30], [126, 41, 47], [107, 46, 118], [112, 49, 147], [109, 41, 189], [103, 83, 15], [126, 99, 70], [124, 101, 104], [131, 103, 159], [128, 110, 183], [119, 148, 9], [112, 137, 50], [123, 127, 116], [107, 124, 167], [102, 148, 203], [124, 180, 15], [116, 168, 65], [104, 182, 102], [111, 164, 163], [105, 174, 191], [102, 218, 20], [126, 203, 64], [108, 215, 109], [110, 221, 157], [107, 230, 192], [160, 25, 11], [165, 12, 65], [153, 2, 117], [182, 21, 141], [160, 19, 188], [176, 58, 19], [175, 58, 56], [170, 69, 93], [176, 42, 146], [157, 44, 211], [157, 105, 2], [180, 98, 73], [182, 85, 92], [169, 93, 152], [156, 89, 202], [157, 144, 22], [180, 151, 77], [154, 146, 118], [162, 136, 143], [171, 134, 184], [170, 174, 15], [178, 180, 65], [176, 183, 120], [175, 169, 147], [181, 165, 197], [156, 227, 3], [167, 218, 61], [160, 216, 119], [164, 251, 141], [177, 201, 251], [231, 30, 13], [219, 6, 59], [211, 26, 122], [216, 16, 153], [209, 12, 192], [216, 70, 15], [215, 46, 60], [234, 61, 112], [224, 53, 157], [227, 49, 207], [221, 108, 8], [220, 93, 73], [230, 111, 113], [218, 89, 143], [231, 90, 195], [227, 144, 22], [208, 137, 49], [210, 128, 116], [225, 135, 157], [221, 135, 193], [211, 174, 18], [222, 185, 50], [229, 183, 93], [233, 162, 155], [255, 167, 205], [211, 215, 15], [232, 225, 71], [0, 0, 0], [255, 255, 255], [215, 216, 196]])


def top_k(logits, thres = 0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner


class BasicVAE(nn.Module):

    def get_codebook_indices(self, images):
        raise NotImplementedError()

    def decode(self, img_seq):
        raise NotImplementedError()

    def get_codebook_probs(self, img_seq):
        raise NotImplementedError()

    def get_image_tokens_size(self):
        pass

    def get_image_size(self):
        pass


class ResBlock(nn.Module):
    def __init__(self, chan_in, hidden_size, chan_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(chan_in, hidden_size, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_size, hidden_size, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_size, chan_out, 1)
        )

    def forward(self, x):
        return self.net(x) + x


class DiscreteVAE(BasicVAE):
    def __init__(
        self,
        image_size = 256,
        num_tokens = 512,
        codebook_dim = 512,
        num_layers = 3,
        hidden_dim = 64,
        channels = 3,
        smooth_l1_loss = False,
        temperature = 0.9,
        straight_through = False,
        kl_div_loss_weight = 0.
    ):
        super().__init__()
        # assert log2(image_size).is_integer(), 'image size must be a power of 2'
        assert num_layers >= 1, 'number of layers must be greater than or equal to 1'

        self.image_size = image_size
        self.num_tokens = num_tokens
        self.num_layers = num_layers
        self.temperature = temperature
        self.straight_through = straight_through
        self.codebook = nn.Embedding(num_tokens, codebook_dim)

        enc_layers = []
        dec_layers = []

        enc_in = channels
        dec_in = codebook_dim

        for layer_id in range(num_layers):
            enc_layers.append(nn.Sequential(nn.Conv2d(enc_in, hidden_dim, 4, stride=2, padding=1), nn.ReLU()))
            enc_layers.append(ResBlock(chan_in=hidden_dim, hidden_size=hidden_dim, chan_out=hidden_dim))
            enc_in = hidden_dim
            dec_layers.append(nn.Sequential(nn.ConvTranspose2d(dec_in, hidden_dim, 4, stride=2, padding=1), nn.ReLU()))
            dec_layers.append(ResBlock(chan_in=hidden_dim, hidden_size=hidden_dim, chan_out=hidden_dim))
            dec_in = hidden_dim

        enc_layers.append(nn.Conv2d(hidden_dim, num_tokens, 1))
        dec_layers.append(nn.Conv2d(hidden_dim, channels, 1))

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder = nn.Sequential(*dec_layers)

        self.loss_fn = F.smooth_l1_loss if smooth_l1_loss else F.mse_loss
        self.kl_div_loss_weight = kl_div_loss_weight

    def get_image_size(self):
        return self.image_size

    def get_image_tokens_size(self):
        return self.image_size // 8

    @torch.no_grad()
    @eval_decorator
    def get_codebook_indices(self, images):
        logits = self.forward(images, return_logits = True)
        codebook_indices = logits.argmax(dim = 1)
        return codebook_indices

    @torch.no_grad()
    @eval_decorator
    def get_codebook_probs(self, images):
        logits = self.forward(images, return_logits = True)
        return nn.Softmax(dim=1)(logits)

    def decode(
        self,
        img_seq
    ):
        image_embeds = self.codebook(img_seq)
        b, n, d = image_embeds.shape
        h = w = int(sqrt(n))

        image_embeds = rearrange(image_embeds, 'b (h w) d -> b d h w', h = h, w = w)
        images = self.decoder(image_embeds)
        return images

    def forward(
        self,
        img,
        return_loss = False,
        return_recons = False,
        return_logits = False,
        temp = None
    ):
        device, num_tokens, image_size, kl_div_loss_weight = img.device, self.num_tokens, self.image_size, self.kl_div_loss_weight
        assert img.shape[-1] == image_size and img.shape[-2] == image_size, f'input must have the correct image size {image_size}'

        logits = self.encoder(img)

        if return_logits:
            return logits # return logits for getting hard image indices for DALL-E training

        temp = default(temp, self.temperature)
        soft_one_hot = F.gumbel_softmax(logits, tau = temp, dim = 1, hard = self.straight_through)
        sampled = einsum('b n h w, n d -> b d h w', soft_one_hot, self.codebook.weight)
        out = self.decoder(sampled)

        if not return_loss:
            return out

        # reconstruction loss

        recon_loss = self.loss_fn(img, out)

        # kl divergence

        logits = rearrange(logits, 'b n h w -> b (h w) n')
        qy = F.softmax(logits, dim = -1)

        log_qy = torch.log(qy + 1e-10)
        log_uniform = torch.log(torch.tensor([1. / num_tokens], device = device))
        kl_div = F.kl_div(log_uniform, log_qy, None, None, 'batchmean', log_target = True)

        loss = recon_loss + (kl_div * kl_div_loss_weight)

        if not return_recons:
            return loss

        return loss, out


from dall_e import load_model


class Dalle_VAE(BasicVAE):
    def __init__(self):
        super().__init__()
        self.encoder = None
        self.decoder = None

    def load_model(self, model_dir, device):
        self.encoder = load_model(os.path.join(model_dir, "encoder.pkl"), device)
        self.decoder = load_model(os.path.join(model_dir, "decoder.pkl"), device)

    def decode(self, img_seq, img_size):
        bsz = img_seq.size()[0]
        img_seq = img_seq.view(bsz, img_size[0], img_size[1])
        z = F.one_hot(img_seq, num_classes=8192).permute(0, 3, 1, 2).float()
        return self.decoder(z).float()

    def get_codebook_indices(self, images):
        z_logits = self.encoder(images)
        return torch.argmax(z_logits, axis=1)

    def get_codebook_probs(self, images):
        z_logits = self.encoder(images)
        return nn.Softmax(dim=1)(z_logits)

    def forward(self, img_seq_prob, img_size, no_process=False):
        if no_process:
            return self.decoder(img_seq_prob.float()).float()
        else:
            bsz, seq_len, num_class = img_seq_prob.size()
            z = img_seq_prob.view(bsz, img_size, img_size, 8192)
            return self.decoder(z.permute(0, 3, 1, 2).float()).float()

def get_dalle_vae(weight_path, device):
    vae = Dalle_VAE()
    vae.load_model(model_dir=weight_path, device=device)
    return vae

logit_laplace_eps: float = 0.1

def map_pixels(x: torch.Tensor) -> torch.Tensor:
    if len(x.shape) != 4:
        raise ValueError('expected input to be 4d')
    if x.dtype != torch.float:
        raise ValueError('expected input to have type float')

    return (1 - 2 * logit_laplace_eps) * x + logit_laplace_eps

def unmap_pixels(x: torch.Tensor) -> torch.Tensor:
    if len(x.shape) != 4:
        raise ValueError('expected input to be 4d')
    if x.dtype != torch.float:
        raise ValueError('expected input to have type float')

    return torch.clamp((x - logit_laplace_eps) / (1 - 2 * logit_laplace_eps), 0, 1)

def encode_to_segmap(indice):
    PALETTE_ = PALETTE.clone().to(indice.device)
    _indice = indice.clone().detach()
    _indice[_indice > 150] = 150
    # print('cjq debug:', PALETTE_.shape)
    # print(_indice.shape)
    return PALETTE_[_indice.long()].squeeze(1).permute(0, 3, 1, 2)

def decode_from_segmap(segmap, num_classes=19):
    PALETTE_ = PALETTE[:num_classes].clone().detach().to(segmap.device)
    B, C, H, W = segmap.shape
    p = torch.Tensor.repeat(PALETTE_, B, H, W, 1, 1).permute(0, 3, 4, 1, 2)
    segmap = torch.Tensor.repeat(segmap, num_classes, 1, 1, 1, 1).permute(1, 0, 2, 3, 4)
    return torch.abs(segmap - p).sum(2).argmin(1).unsqueeze(1)

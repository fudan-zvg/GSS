# <img src="figures/dinosaur.png" width="30"> [CVPR 2023] _Generative Semantic Segmentation_
[Jiaqi Chen](), [Jiachen Lu](), [Xiafeng Zhu](), and [Li Zhang]()

## Abstract

<!-- [ABSTRACT] -->
We present _**Generative Semantic Segmentation**_ (GSS),
a generative framework for semantic segmentation.
Unlike previous methods addressing a per-pixel classification problem,
we cast semantic segmentation into an _**image-conditioned 
mask generation problem**_.
This is achieved by replacing the conventional per-pixel discriminative learning with a latent prior learning process.
Specifically, we model the variational posterior distribution of latent variables given the segmentation mask.
This is done by expressing the segmentation mask with a special type of image (dubbed as _maskige_).
This posterior distribution allows to generate segmentation masks unconditionally.
To implement semantic segmentation, we further introduce a conditioning network (_e.g._, an encoder-decoder Transformer)
optimized by minimizing the divergence between the posterior distribution of maskige (_i.e._ segmentation masks) and the latent prior distribution of input images on the training set.
Extensive experiments on standard benchmarks show that our GSS can perform competitively to prior art alternatives in the standard semantic segmentation setting,
whilst achieving a new state of the art in the more challenging cross-domain setting.
<!-- [IMAGE] -->
![GSS](figures/framework.png)
## Citation

```bibtex
@inproceedings{chen2021generative,
  title={Generative Semantic Segmentation
  author={Chen, Jiaqi and Lu, Jiachen and Zhu, Xiafeng and Zhang, Li},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}
}
```
This repository is the official implementation of the paper [Generative Semantic Segmentation]().

***

## Installation

See [installation instructions](INSTALL.md).



## Getting Started
See [Inference Demo with Pre-trained Model](demo/README.md)

See [Results](#results).

See [Preparing Datasets for GSS](datasets/README.md).

See [Getting Started](#getting-started).

See [More Usage](#more-usage).

***

# Results
In this part, we present the clean models that do not use extra detection data or tricks.
### Cityscapes dataset

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">Iterations</th>
<th valign="bottom">mIoU</th>
<th valign="bottom">mAcc</th>
<th valign="bottom">Config</th>
<th valign="bottom">Pretrain</th>
<th valign="bottom">checkpoint</th>

 <tr><td align="left">GSS-FF</td>
<td align="center">R101</td>
<td align="center">80k</td>
<td align="center">77.76</td>
<td align="center"></td>
<td align="center"><a href="">config</a></td>
<td align="center"><a href="">pretrain</a></td>
<td align="center"><a href="">model</a></td>
</tr>

 <tr><td align="left">GSS-FF</td>
<td align="center">Swin-L</td>
<td align="center">80k</td>
<td align="center">78.90</td>
<td align="center"></td>
<td align="center"><a href="">config</a></td>
<td align="center"><a href="">pretrain</a></td>
<td align="center"><a href="">model</a></td>
</tr>

 <tr><td align="left">GSS-FT-W</td>
<td align="center">ResNet</td>
<td align="center">80k</td>
<td align="center">78.46</td>
<td align="center"></td>
<td align="center"><a href="">config</a></td>
<td align="center"><a href="">pretrain</a></td>
<td align="center"><a href="">model</a></td>
</tr>

 <tr><td align="left">GSS-FT-W</td>
<td align="center">Swin-L</td>
<td align="center">80k</td>
<td align="center">80.05</td>
<td align="center"></td>
<td align="center"><a href="">config</a></td>
<td align="center"><a href="">pretrain</a></td>
<td align="center"><a href="">model</a></td>
</tr>

</tbody></table>

### ADE20K dataset

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">Iterations</th>
<th valign="bottom">mIoU</th>
<th valign="bottom">mAcc</th>
<th valign="bottom">Config</th>
<th valign="bottom">Pretrain</th>
<th valign="bottom">checkpoint</th>

 <tr><td align="left">GSS-FF</td>
<td align="center">Swin-L</td>
<td align="center">160k</td>
<td align="center">46.29</td>
<td align="center"></td>
<td align="center"><a href="">config</a></td>
<td align="center"><a href="">pretrain</a></td>
<td align="center"><a href="">model</a></td>
</tr>

 <tr><td align="left">GSS-FT-W</td>
<td align="center">Swin-L</td>
<td align="center">160k</td>
<td align="center">48.54</td>
<td align="center"></td>
<td align="center"><a href="">config</a></td>
<td align="center"><a href="">pretrain</a></td>
<td align="center"><a href="">model</a></td>
</tr>

</tbody></table>

### MSeg dataset

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">Iterations</th>
<th valign="bottom">h.mean</th>
<th valign="bottom">Config</th>
<th valign="bottom">Pretrain</th>
<th valign="bottom">checkpoint</th>

 <tr><td align="left">GSS-FF</td>
<td align="center">HRNet-W48</td>
<td align="center">160k</td>
<td align="center">52.60</td>
<td align="center"><a href="">config</a></td>
<td align="center"><a href="">pretrain</a></td>
<td align="center"><a href="">model</a></td>
</tr>

 <tr><td align="left">GSS-FF</td>
<td align="center">Swin-L</td>
<td align="center">160k</td>
<td align="center">59.49</td>
<td align="center"><a href="">config</a></td>
<td align="center"><a href="">pretrain</a></td>
<td align="center"><a href="">model</a></td>
</tr>

 <tr><td align="left">GSS-FT-W</td>
<td align="center">HRNet-W48</td>
<td align="center">160k</td>
<td align="center">55.20</td>
<td align="center"><a href="">config</a></td>
<td align="center"><a href="">pretrain</a></td>
<td align="center"><a href="">model</a></td>
</tr>

 <tr><td align="left">GSS-FT-W</td>
<td align="center">Swin-L</td>
<td align="center">160k</td>
<td align="center">61.94</td>

<td align="center"><a href="">config</a></td>
<td align="center"><a href="">pretrain</a></td>
<td align="center"><a href="">model</a></td>
</tr>

</tbody></table>
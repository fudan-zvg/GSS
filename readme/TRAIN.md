## Training
Since the pre-generated colors have already been provided, you can directly proceed to Latent prior learning stage.
### Latent posterior learning for $\mathcal{X}$ (optional)
> Note that we've carefully prepared the $\mathcal{X}$, so you can go straight to Latent prior learning to reproduce the results.

The first stage is **posterior Learning**, where the actual task performed is assigning a unique color to each semantic category. We propose using the **Maximal distance assumption** to ensure that the colors of different categories are not easily confused. To run this stage, please execute the following command:

For ADE20K dataset, you can run the following command:
```bash
python tools/posterior_learning.py --dataset ade20k
```
For MSeg dataset, you can run the following command:
```bash
python tools/posterior_learning.py --dataset mseg
```
For Cityscapes dataset, you don't need to generate color for each category, as we directlly use the deflaut visualization color of Cityscapes.

After run the before command, you will get a list of 0-255 RGB values:

```python
# -----------  Begin  ------------
[[10, 22, 26], [26, 12, 47], ..., [222, 220, 182]]
# ------------- End --------------
```

Please use the following script to validate the color assignments for each class in your generated images. If you notice that the Intersection over Union (IoU) score for a particular class is unusually low, it may be because the assigned color for that class is too similar to the colors assigned to other classes. In such cases, you can modify the color values for that class and re-run the eval command until you are satisfied with the results. The eval command is as follows:
```bash
# ADE20K
bash tools/dist_test.sh configs/gss/ade20k/dalle_reconstruction_ade20k.py ckp/non_ckp.pth 8 --eval mIoU

# MSeg
bash tools/dist_test.sh configs/gss/mseg/dalle_reconstruction_mseg.py ckp/non_ckp.pth 8 --eval mIoU

# Cityscapes
bash tools/dist_test.sh configs/gss/cityscapes/dalle_reconstruction_mseg.py ckp/non_ckp.pth 8 --eval mIoU
```

Then paste this color list into the configuration file (e.g. [configs/ade20k/gss-ff_swin-l_512x512_160k_ade20k.py](https://github.com/fudan-zvg/GSS/blob/gss/configs/gss/ade20k/gss-ff_swin-l_512x512_160k_ade20k.py)). 
If the list does not have model=dict(...) , then you can just create one:
```python
model=dict(decode_head=dict(palette=[[10, 22, 26], [26, 12, 47], ..., [222, 220, 182]]))
```
If it already has a "model=dict(...)", then you need to add a line of code to the original one:
```python
model=dict(
  ...
    decode_head_dict(
      ...
      # add your color list here
      palette=[[10, 22, 26], [26, 12, 47], ..., [222, 220, 182]]
      ...
    )
)
```

### Latent prior learning
The pre-generated colors from latent posterior learning stage have already been provided in all configs.

```shell
# train with 8 GPUs
bash tools/dist_train.sh configs/gss/<dataset><gss-ff_config_file> <num_of_GPUs>
```
For example,
```shell
# train with 8 GPUs
bash tools/dist_train.sh configs/gss/cityscapes/gss-ff_r101_768x768_80k_cityscapes.py 8
```
After undergoing Latent prior learning, one can obtain the results of GSS-FF. The first 'F' indicates that $\mathcal{X}$ is training-free, while the second 'F' signifies that $\mathcal{X}^{-1}$ is also training-free.
### Latent posterior learning for $\mathcal{X}^{-1}$
This stage is specifically designed for GSS-FT, where $\mathcal{X}^{-1}$ is a learnable module that requires training. During this stage, we load and freeze the pre-trained image encoder from Latent prior learning stage, focusing solely on training $\mathcal{X}^{-1}$.

1. **Load the pre-trained weight of image encoder**

From the Latent prior learning phase, one can utilize the intermediate checkpoint obtained (e.g., at 32k iterations) as the pre-trained image encoder weight. This weight can then be loaded into the model to commence the training of $\mathcal{X}^{-1}$. 

Additionally, we provide initialization weights, which can be downloaded to reproduce the results presented in the paper.

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Dataset</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">Iterations/Max iterations</th>
<th valign="bottom">Initial checkpoint</th>

 <tr><td align="left">Cityscapes</td>
<td align="center">Swin-L</td>
<td align="center">80k/80k</td>
<td align="center"><a href="https://drive.google.com/drive/folders/1BTvchDJtUk4rRJ0qK2rcApbHEAEK1bEZ?usp=sharing">google drive</a></td>
</tr>

<tr><td align="left">ADE20K</td>
<td align="center">Swin-L</td>
<td align="center">32k/160k</td>
<td align="center"><a href="https://drive.google.com/drive/folders/159NKXbzPa8zk9e_DCpRTY7L9VKTowLZf?usp=sharing">google drive</a></td>
</tr>

<tr><td align="left">MSeg</td>
<td align="center">Swin-L</td>
<td align="center">160k/160k</td>
<td align="center"><a href="https://drive.google.com/drive/folders/1br9IAcOHXkJsPoG0DBEwkN97U5V5liEZ?usp=sharing">google drive</a></td>
</tr>

</tbody></table>

2. **Train $\mathcal{X}^{-1}$ module**

Note that, $\mathcal{X}^{-1}$ requires training only once for each dataset and can then be applied to various image encoders, thereby conserving significant training resources.

For the $\mathcal{X}^{-1}$ of Cityscaeps, please run the following command:
```bash
bash tools/dist_train.sh configs/gss/cityscapes/gss-ft-w_swin-l_768x768_80k_40k_cityscapes.py 8 --load-from ckp/gss_ft_cityscapes_swin_init.pth
```

For the $\mathcal{X}^{-1}$ of ADE20K, please run the following command:
```bash
# train with noisy prediction
bash tools/dist_train.sh configs/gss/ade20k/gss-ft-w_swin-l_512x512_160k_ade20k.py 8 --load-from ckp/gss_ft_ade20k_swin_init.pth
# merge checkpoint
python merge_checkpoints.py --model_path work_dirs/gss-ff_swin-l_512x512_160k_ade20k/iter_160000.pth --post_model_path work_dirs/gss-ft-w_swin-l_512x512_160k_ade20k/iter_40000.pth --target_path work_dirs/gss-ft-w_swin-l_768x768_80k_40k_cityscapes/gss-ft_160k_40k_ade20k.pth --backbone_type swin
```

For the $\mathcal{X}^{-1}$ of MSeg, please run the following command:
```bash
bash tools/dist_train.sh configs/gss/mseg/gss-ft-w_swin-l_512x512_160k_40k_mseg.py 8 --load-from ckp/gss_ft_mseg_swin_init.pth
```

3. **Evaluate GSS-FT**
We can directly load the weights for evaluation.

Take Cityscapes as example:
```bash
bash tools/dist_test.sh configs/gss/mseg/gss-ft-w_swin-l_768x768_80k_40k_cityscapes.py work_dirs/gss-ft-w_swin-l_768x768_80k_40k_cityscapes/gss-ft_80k_40k_cityscapes.pth 8 --eval mIoU
```

## Training
Since the pre-generated colors have already been provided, you can directly proceed to Latent prior learning stage.
### Step 1: Latent posterior learning for $\mathcal{X}$ (Optional)
> Note that we've carefully prepared the $\mathcal{X}$, so you can go straight to next step (Latent prior learning) to reproduce the results.

The actual task Latent posterior learning for $\mathcal{X}$ performed is assigning a unique color to each semantic category. We propose using the **Maximal distance assumption** to ensure that the colors of different categories are not easily confused. To conduct this stage, please execute the following command:

For ADE20K dataset, you can run the following command:
```bash
python tools/posterior_learning.py --dataset ade20k
```
For MSeg dataset, you can run the following command:
```bash
python tools/posterior_learning.py --dataset mseg
```
For Cityscapes dataset, you don't need to generate color for each category, as we directlly use the deflaut visualization color of Cityscapes.

After running above command, you will get a list of 0-255 RGB values:

```python
# -----------  Begin  ------------
[[10, 22, 26], [26, 12, 47], ..., [222, 220, 182]]
# ------------- End --------------
```
Then, paste above color list into the configuration file for DALL-E reconstruction (e.g., [configs/ade20k/dalle_reconstruction_ade20k.py](https://github.com/fudan-zvg/GSS/blob/gss/configs/gss/ade20k/dalle_reconstruction_ade20k.py))

```python
_base_ = [
    './gss-ff_swin-l_512x512_160k_ade20k.py'
]

model=dict(
    pretrained=None,
    backbone=dict(
        _delete_=True,
        type='NoneBackbone'
    ),
    decode_head=dict(
        reconstruction_eval=True
        # add your own color list here
        # ------- begin -------
        # palette=[[10, 22, 26], [26, 12, 47], ..., [222, 220, 182]]
        # ------- end ---------
    ),
)
```

Please use the following script to validate the color assignments for each class in your generated images. 

```bash
# ADE20K
bash tools/dist_test.sh configs/gss/ade20k/dalle_reconstruction_ade20k.py ckp/non_ckp.pth 8 --eval mIoU

# Cityscapes
bash tools/dist_test.sh configs/gss/cityscapes/dalle_reconstruction_mseg.py ckp/non_ckp.pth 8 --eval mIoU

# MSeg
bash tools/dist_test.sh configs/gss/mseg/dalle_reconstruction_mseg.py ckp/non_ckp.pth 8 --eval mIoU
```

If you notice that the Intersection over Union (IoU) score for a particular class is unusually low, it may be because the assigned color for that class is too similar to the colors assigned to other classes. In such cases, you can modify the color values for that class and re-run the eval command until you are satisfied with the results. 

After that, paste the final color list into the configuration file (e.g. [configs/ade20k/gss-ff_swin-l_512x512_160k_ade20k.py](https://github.com/fudan-zvg/GSS/blob/gss/configs/gss/ade20k/gss-ff_swin-l_512x512_160k_ade20k.py)). 
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

### Step 2: Latent prior learning (Train GSS-FF)

> **GSS-FF:** The first 'F' indicates that $\mathcal{X}$ is training-free, while the second 'F' signifies that $\mathcal{X}^{-1}$ is also training-free.
> 
> The pre-generated colors have already been provided in all configs. Thus, it's fine to start from step 1.

```shell
# train GSS-FF model with 8 GPUs
bash tools/dist_train.sh configs/gss/<dataset><gss-ff_config_file> <num_of_GPUs>
```
For example,
```shell
# train GSS-FF model on Cityscapes with 8 GPUs
bash tools/dist_train.sh configs/gss/cityscapes/gss-ff_r101_768x768_80k_cityscapes.py 8
```
After undergoing Latent prior learning, one can obtain the results of GSS-FF.
### Step 3: Latent posterior learning for $\mathcal{X}^{-1}$ (Train GSS-FT)

> **GSS-FT:** The first 'F' indicates that $\mathcal{X}$ is training-free, while the second 'T' signifies that $\mathcal{X}^{-1}$ is a learnable block (in practice, a swin block) which requirs training for each dataset.
>
> Through this stage of training, you can obtain the GSS-FT.

1. **Load the pre-trained weight of image encoder**

From the Latent prior learning phase, we can utilize the final checkpoint or intermediate checkpoint (e.g., 32k iterations) obtained as the pre-trained image encoder weight. This weight can then be loaded into the model to commence the training of $\mathcal{X}^{-1}$. 

For your convenience, we provide initialization weights. You can download them and save in `$GSS/ckp/` .

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Dataset</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">Initial checkpoint</th>
<th valign="bottom">model name</th>
    
<tr><td align="left">Cityscapes</td>
<td align="center">Swin-L</td>
<td align="center">gss-ff_swin-l_768x768_80k_cityscapes_iter_80000.pth</td>
<td align="center"><a href="https://drive.google.com/drive/folders/1BTvchDJtUk4rRJ0qK2rcApbHEAEK1bEZ?usp=sharing">google drive</a></td>
</tr>

<tr><td align="left">ADE20K</td>
<td align="center">Swin-L</td>
<td align="center">gss-ff_swin-l_512x512_160k_ade20k_iter_160000.pth</td>
<td align="center"><a href="https://drive.google.com/drive/folders/1OnzGL5szxYlUnv2zmAkdw-mA-3pTNo_w?usp=sharing">google drive</a></td>
</tr>

<tr><td align="left">MSeg</td>
<td align="center">Swin-L</td>
<td align="center">gss-ff_swin-l_512x512_160k_mseg_iter_160000.pth</td>
<td align="center"><a href="https://drive.google.com/drive/folders/1br9IAcOHXkJsPoG0DBEwkN97U5V5liEZ?usp=drive_link">google drive</a></td>
</tr>

</tbody></table>

2. **Train $\mathcal{X}^{-1}$ module**

Note that, $\mathcal{X}^{-1}$ requires training only once for each dataset and can then be applied to various image encoders, thereby conserving significant training resources.

```bash
bash tools/dist_train.sh configs/gss/<dataset>/<gss-ft_config_file> <num_of_GPUs> --load-from <gss-ff_checkpoint>
```

Here are some examples:
```bash
# Train $\mathcal{X}^{-1}$ on Cityscaeps
bash tools/dist_train.sh configs/gss/cityscapes/gss-ft-w_swin-l_768x768_80k_40k_cityscapes.py 8 --load-from ckp/gss-ff_swin-l_768x768_80k_cityscapes_iter_80000.pth

# Train $\mathcal{X}^{-1}$ on ADE20K
bash tools/dist_train.sh configs/gss/ade20k/gss-ft-w_swin-l_512x512_160k_ade20k.py 8 --load-from ckp/gss-ff_swin-l_512x512_160k_ade20k_iter_160000.pth

# Trian $\mathcal{X}^{-1}$ on MSeg
bash tools/dist_train.sh configs/gss/mseg/gss-ft-w_swin-l_512x512_160k_40k_mseg.py 8 --load-from ckp/gss-ff_swin-l_512x512_160k_mseg_iter_160000.pth
```

If the GSS-FF checkpoint you are does is at intermediate iteraction (e.g. 32k iteration), it is imperative to concatenate the weights of GSS-FF from the final iteration with the weights of $\mathcal{X}^{-1}$ to obtain the ultimate model weights.
```bash
python merge_checkpoints.py --model_path <gss-ff_checkpoint> --post_model_path <X^{-1}_checkpoint> --target_path <gss-ft_checkpoint> --backbone_type <backbone_name>
```
Take the GSS-FT on ADE20K as an example: 
```bash
python merge_checkpoints.py --model_path work_dirs/gss-ff_swin-l_512x512_160k_ade20k/iter_160000.pth --post_model_path work_dirs/gss-ft-w_swin-l_512x512_160k_ade20k/iter_40000.pth --target_path work_dirs/gss-ft-w_swin-l_768x768_80k_40k_cityscapes/gss-ft_160k_40k_ade20k.pth --backbone_type swin
```

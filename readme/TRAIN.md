## Training
Since the pre-generated colors have already been provided, you can directly proceed to Latent prior learning stage.
### Latent posterior learning for $\mathcal{X}$ (optional)
Note that we've carefully prepared the $\mathcal{X}$, so you can go straight to step two to reproduce the results.

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
bash tools/dist_test.sh configs/gss/posterior_learning/dalle_reconstruction_ade20k.py ckp/non_ckp.pth 8 --eval mIoU
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
bash tools/dist_train.sh configs/gss/cityscapes/gss-ff_r101_768x768_80k_cityscapes.py 8
```

### Latent posterior learning for $\mathcal{X}^{-1}$ (will be released soon)

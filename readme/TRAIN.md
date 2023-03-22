Since the pre-generated colors have already been provided, you can directly proceed to Latent prior learning stage.
#### Efficient latent posterior learning for $\mathcal{X}$ (will be released soon)
The first stage is **posterior Learning**, where the actual task performed is assigning a unique color to each semantic category. We propose using the **Maximal distance assumption** to ensure that the colors of different categories are not easily confused. To run this stage, please execute the following command:

```bash
python tools/posterior_learning.py --num_classes 150
```
Please use the following script to validate the color assignments for each class in your generated images. If you notice that the Intersection over Union (IoU) score for a particular class is unusually low, it may be because the assigned color for that class is too similar to the colors assigned to other classes. In such cases, you can modify the color values for that class and re-run the eval command until you are satisfied with the results. The eval command is as follows:
```bash
bash tools/dist_test.sh configs/gss/posterior_learning/dalle_reconstruction_ade20k.py ckp/non_ckp.pth 8 --eval mIoU
```

#### Latent prior learning
The pre-generated colors from latent posterior learning stage have already been provided in all configs.
```shell
# train with 8 GPUs
bash tools/dist_train.sh configs/gss/cityscapes/gss-ff_r101_768x768_80k_cityscapes.py 8
# test with 8 GPUs
bash tools/dist_test.sh configs/gss/cityscapes/gss-ff_r101_768x768_80k_cityscapes.py ./ckp_dir/iter_80000.pth 8 --eval mIoU
```
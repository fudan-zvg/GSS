# Dataset preparation
It is recommended to symlink the dataset root to $GSS/data. If your folder structure is different, you may need to change the corresponding paths in config files. 

**Data folder structure**

```none
mmsegmentation
├── mmseg
├── tools
├── configs
├── data
│   ├── cityscapes
│   │   ├── leftImg8bit
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   ├── val
│   ├── ade
│   │   ├── ADEChallengeData2016
│   │   │   ├── annotations
│   │   │   │   ├── training
│   │   │   │   ├── validation
│   │   │   ├── images
│   │   │   │   ├── training
│   │   │   │   ├── validation
│   ├── mseg_dataset
│   │   ├── ADE20K
│   │   ├── Cityscapes
│   │   ├── KITTI
│   │   ├── PASCAL_VOC_2012
│   │   ├── WildDash
│   │   ├── BDD
│   │   ├── COCOPanoptic
│   │   ├── MapillaryVistasPublic
│   │   ├── ScanNet
│   │   ├── Camvid
│   │   ├── IDD
│   │   ├── PASCAL_Context
│   │   ├── SUNRGBD           
```

## Cityscapes

You can access the data [here](https://www.cityscapes-dataset.com/downloads/) once you've registered.

`**labelTrainIds.png` is utilized for Cityscapes training. MMSeg have provided a [script](https://github.com/open-mmlab/mmsegmentation/blob/1.x/tools/dataset_converters/cityscapes.py), built upon [cityscapesscripts](https://github.com/mcordts/cityscapesScripts), to generate the `**labelTrainIds.png` files.

```shell
# --nproc means 8 process for conversion, which could be omitted as well.
python tools/dataset_converters/cityscapes.py data/cityscapes --nproc 8
```

## ADE20K

You can download the training and validation sets for ADE20K from this [link](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip). Additionally, the test set is available for download [here](http://data.csail.mit.edu/places/ADEchallenge/release_test.zip).

## MSeg
Please follow 
[MSeg download instruction](https://github.com/mseg-dataset/mseg-api/blob/master/download_scripts/README.md) to download MSeg dataset


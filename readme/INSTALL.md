# Install 

## Requirements
```shell
TorchVision: 0.10.1+cu111
OpenCV: 4.7.0
MMCV: 1.3.18
MMCV Compiler: GCC 7.3
MMCV CUDA Compiler: 11.1
MMSegmentation: 0.26.0+b002c10
```
## Install steps
The following commands are tested on Ubuntu 20.04 with CUDA 11.1 and PyTorch 1.9.1+cu111.
1. Ceate a new conda environment
You are now on the project root directory, not in GSS/ directory.
```shell
conda create --name gss python=3.8.13
conda activate gss
```

2. Install pytorch, mmcv, DALL-E
```shell
pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install mmcv-full==1.3.18 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip3 install DALL-E
pip3 install einops==0.6.1
pip3 install attrs==23.1.0
pip3 install future tensorboard
pip3 install setuptools==59.5.0
```

3. Install MSeg
```shell
git clone git@github.com:mseg-dataset/mseg-api.git
cd mseg-api/
pip install -e .
```

4. Install GSS
```shell
cd ..
git clone git@github.com:fudan-zvg/GSS.git
cd GSS/
pip install -e .
```

5. Check the installation
```shell
python mmseg/utils/collect_env.py
```

## Download the DALL-E pre-trained VQVAE weights
Please run the following command:
```shell
bash tools/download_pretrain_vqvae.sh
```

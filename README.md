# Transformer-based Light Field Salient Object Detection and Its Application to Autofocus

This is the source code of our "Transformer-based Light Field Salient Object Detection and Its Application to Autofocus".

## Contents:

1. [Transformer-based Light Field Salient Object Detection](#transformer-based-light-field-salient-object-detection)<br>
  i. [Requirements](#requirements)<br>
  ii. [Data](#data)<br>
  iii. [Training TLFNet](#training-tlfnet)<br>
  iv. [Training TLFNet](#testing-tlfnet)<br>
2. [Light Field Salient Object Autofocus](#light-field-salient-object-autofocus)<br>
3. [Citation](#citation)<br>

# Transformer-based Light Field Salient Object Detection

## Requirements
* Python 3.7 <br>
* Torch 1.10.0 <br>
* Torchvision 0.11.0 <br>
* Cuda 11.8 <br>
* Tensorboard 2.11.2

## Data
* Download the [dataset](https://github.com/kerenfu/LFSOD-Survey). We organize all the data as follows:
```
data
├── train
│   ├── DUTLF-FS 
│   │      ├── allfocus
│   │      ├── mat     # focal stack
│   │      ├── contour
│   │      ├── GT
├── test
│   ├── DUTLF-FS
│   │      ├── allfocus
│   │      ├── ...
│   ├── HFUT-Lytro
│   │      ├── allfocus
│   │      └── ...
│   └── ...
```

## Training TLFNet
* Start to train with
```sh
python -m torch.distributed.launch --nproc_per_node=2 train.py 
```

## Testing TLFNet
* Download the already-trained [weigh]().
* Start to test with
```sh
python test.py  
```
# Light Field Salient Object Autofocus

# Citation
Please cite our paper if you find the work useful: 

  @ARTICLE{Jiang2024TLFNet,
  author={Jiang, Yao and Li, Xin and Fu, Keren and Zhao, Qijun},
  journal={IEEE Transactions on Image Processing}, 
  title={Transformer-Based Light Field Salient Object Detection and Its Application to Autofocus}, 
  year={2024},
  volume={33},
  pages={6647-6659}}

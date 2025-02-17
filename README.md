# Transformer-based Light Field Salient Object Detection and Its Application to Autofocus

This is the source code of our "Transformer-based Light Field Salient Object Detection and Its Application to Autofocus" ([paper](https://ieeexplore.ieee.org/abstract/document/10759590)|[project](https://github.com/jiangyao-scu/TLFNet)).


# Usage
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

## To Train
* Start to train with
```sh
python -m torch.distributed.launch --nproc_per_node=2 train.py 
```

## To Test
* Download the already-trained [weigh]().
* Start to test with
```sh
python test.py  
```


# Citation
Please cite our paper if you find the work useful: 

  @ARTICLE{Jiang2024TLFNet,
  author={Jiang, Yao and Li, Xin and Fu, Keren and Zhao, Qijun},
  journal={IEEE Transactions on Image Processing}, 
  title={Transformer-Based Light Field Salient Object Detection and Its Application to Autofocus}, 
  year={2024},
  volume={33},
  pages={6647-6659}}

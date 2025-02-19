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
* Tensorboard 2.11.2 <br>
* visdom 0.1.8.9

## Data
* Download the [dataset](https://github.com/kerenfu/LFSOD-Survey), and organize all the data as follows:
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
* Modify the "train_data_location" and "eval_data_location" in "train.py" according to the path of the data.
* Download the pre-trained [swin Transformer]() or [PVT]().
* Start to train Swin Transformer-based TLFNet with
```sh
python -m torch.distributed.launch --nproc_per_node=2 train.py --model_path path/to/save/trained/model/ --log_path path/to/save/log/ --backbone swin --pretrained_model path/of/pre-trained/swin-Transformer/ --image_size 224
```
or train PVT-based TLFNet with
```sh
python -m torch.distributed.launch --nproc_per_node=2 train.py --model_path path/to/save/trained/model/ --log_path path/to/save/log/ --backbone pvt --pretrained_model path/of/pre-trained/PVT/ --image_size 256
```

## Testing TLFNet
* We have released pre-computed saliency maps of TLFNet based on the Swin Transformer and PVT. Please retrieve the results from the following links: [TLFNet-swin]() and [TLFNet-pvt]().
* We have also released the trained weights of TLFNet. You can download them from the following links: [TLFNet-wsin]() and [TLFNet-pvt]().
* To generate saliency maps, you will need to modify the "eval_data_location" in the "test.py" according to your data's path. Then, you can generate the saliency maps with:
```sh
python test.py --save_path path/to/save/saliency maps/ --backbone swin --model_path path/of/pre-trained/TLFNet.pth/ --image_size 224
```
or 
```sh
python test.py --save_path path/to/save/saliency maps/ --backbone pvt --model_path path/of/pre-trained/TLFNet_PVT.pth/ --image_size 256
```
*It should be noted that, owing to an equipment malfunction, the original PVT-based TLFNet data was unfortunately lost. We subsequently retrained this model and obtained results that closely resemble the initial outcomes. This newly obtained experimental result does not alter the conclusions drawn in this paper.*<br>

# Light Field Salient Object Autofocus
We are building an online service for "Light Field Salient Object Autofocus". Please stay tuned for our upcoming release.

# Citation
Please cite our paper if you find the work useful: 

        @article{Jiang2024TLFNet,
        author={Jiang, Yao and Li, Xin and Fu, Keren and Zhao, Qijun},
        journal={IEEE Transactions on Image Processing}, 
        title={Transformer-Based Light Field Salient Object Detection and Its Application to Autofocus}, 
        year={2024},
        volume={33},
        pages={6647-6659}}

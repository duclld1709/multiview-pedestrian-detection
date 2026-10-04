# Multiview Detection with Shadow Transformer (and View-Coherent Data Augmentation) [[arXiv](https://arxiv.org/pdf/2108.05888.pdf)] [[paper](https://dl.acm.org/doi/abs/10.1145/3474085.3475310)]

```
@inproceedings{hou2021multiview,
  title={Multiview Detection with Shadow Transformer (and View-Coherent Data Augmentation)},
  author={Hou, Yunzhong and Zheng, Liang},
  booktitle={Proceedings of the 29th ACM International Conference on Multimedia (MM ’21)},
  year={2021}
}
```


## Overview

We release the PyTorch code for **MVDeTr**, a state-of-the-art multiview pedestrian detector. Its superior performance should be credited to transformer architectures, updated loss terms, and view-coherent data augmentations. Moreover, MVDeTr is also very efficient and can be trained on a single RTX 2080TI. 
This repo also includes a simplified version of **[MVDet](https://github.com/hou-yz/MVDet)**, which also runs on a single RTX 2080TI. 

 
## Content
- [Dependencies](#dependencies)
- [Data Preparation](#data-preparation)
- [Code Preparation](#code-preparation)
- [Training](#training)
    * [Architectures](#architectures)
    * [Loss terms](#loss-terms)
    * [Augmentations](#augmentations)


## MVDeTr Code
This repo is dedicated to the code for **MVDeTr**. 

<!-- ![alt text](https://hou-yz.github.io/images/eccv2020_mvdet_architecture.png "Architecture for MVDet") -->

## Dependencies
This code uses the following libraries
- python
- pytorch & tochvision
- numpy
- matplotlib
- pillow
- opencv-python
- kornia

## Data Preparation
By default, all datasets are in `~/Data/`. We use [MultiviewX](https://github.com/hou-yz/MultiviewX) and [Wildtrack](https://www.epfl.ch/labs/cvlab/data/data-wildtrack/) in this project. 

Your `~/Data/` folder should look like this
```
Data
├── MultiviewX/
│   └── ...
└── Wildtrack/ 
    └── ...
```

## Code Preparation
Before running the code, one should go to ```multiview_detector/models/ops``` and run ```bash mask.sh``` to build the deformable transformer (forked from [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR)). 


## Training
In order to train classifiers, please run the following,
```shell script
python main.py -d wildtrack
python main.py -d multiviewx
``` 
This should automatically return evaluation results similar to the reported 91.5\% MODA on Wildtrack dataset and 93.7\% MODA on MultiviewX dataset. 


### Architectures
This repo supports multiple architecture variants. For MVDeTr, please specify ```--world_feat deform_trans```; for a similar fully convolutional architecture like [MVDet](https://github.com/hou-yz/MVDet), please specify ```--world_feat conv```. 

### Loss terms
This fork adds **Background Recalibration Loss (BRL)** on the CornerNet heatmap focal loss for missing-annotation training.

```shell script
# default: BRL
CUDA_VISIBLE_DEVICES=0 python main.py -d wildtrack --drop_ratio drop60 --loss brl

# baseline CornerNet focal
CUDA_VISIBLE_DEVICES=0 python main.py -d wildtrack --drop_ratio drop60 --loss focal

# BRL knobs
#   --brl_pos_thr       soft-GT threshold for background (default 0.1)
#   --brl_confuse_thr   pred threshold on background → confuse (default 0.3)
#   --brl_beta          confuse term weight (default 0.1)
#   --brl_no_mirror     down-weight confuse neg instead of mirroring toward 1
```

Also: ```--use_mse 0``` keeps the focal/BRL heatmap path; ```--use_mse 1``` switches to plain MSE as in [MVDet](https://github.com/hou-yz/MVDet). 

### Offline detector pseudo-labels
Same detector cache as the MVDet fork (boxes in original image coordinates); a cache built there can be reused directly.

```shell script
python generate_pseudo_cache.py --dataset wildtrack --data_root ../Data/Wildtrack \
  --output pseudo_cache/wildtrack_yolo26s.json
python evaluate_pseudo_cache.py --dataset wildtrack --data_root ../Data/Wildtrack \
  --cache pseudo_cache/wildtrack_yolo26s.json

CUDA_VISIBLE_DEVICES=0 python main.py -d wildtrack --drop_ratio drop_60 --apply_brl 1 \
  --use_pseudo_labels --pseudo_cache pseudo_cache/wildtrack_yolo26s.json
```

Cached foot points are projected onto the training BEV only and rasterized as Gaussians (`--pseudo_sigma_m`, default 0.25 m, matching the GT kernel); evidence within `--pseudo_suppress_radius_m` (1 m) of an annotation is dropped. Since the BEV head is trained with CornerNet focal loss, the pseudo labels enter it the same way as in the TrackTacular/EarlyBird forks: the negative (and BRL easy-negative) penalty is scaled by `1 - pseudo_weight`, pseudo peaks leave the negative/confuse branches, and a score-weighted positive focal term at the peaks is added with weight `--pseudo_loss_weight` (default 0.1). Per-view heatmaps are unaffected. Logs report base and pseudo loss separately. Not compatible with `--use_mse 1`.

### Augmentations
This repo includes support for view coherent data augmentation, which applies affine transformations onto the per-view inputs, and then invert the per-view feature maps to maintain multiview coherency. 

### Pre-trained models
You can download the checkpoints at this [link](https://1drv.ms/u/s!AtzsQybTubHfhNRDo-mUXOWPd3Di4Q?e=monHmQ).

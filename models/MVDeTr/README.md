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
This repo supports multiple loss terms. For the focal loss variant as in MVDeTr, please specify ```--use_mse 0```; for the MSE loss as in [MVDet](https://github.com/hou-yz/MVDet), please specify ```----use_mse 1```. 

### Augmentations
This repo includes support for view coherent data augmentation, which applies affine transformations onto the per-view inputs, and then invert the per-view feature maps to maintain multiview coherency. 

### Pre-trained models
You can download the checkpoints at this [link](https://1drv.ms/u/s!AtzsQybTubHfhNRDo-mUXOWPd3Di4Q?e=monHmQ).


## BRL (this fork)
Training with missing annotations. Dropped labels are read from `<data_root>/drop_annotations/drop_XX/annotations_positions` (`--drop_ratio 20/45/60`); evaluation always uses the full annotations.

`--loss focal` is the original CornerNet focal loss. `--loss brl` (default) adds Background Recalibration: background pixels (GT heatmap `< --brl_pos_thr`, default 0.1) whose prediction is `>= --brl_confuse_thr` (default 0.4) are treated as possibly unlabelled people and pulled toward 1 through the positive focal branch with weight `--brl_beta` (default 0.05; `--brl_no_mirror` instead keeps them on the focal negative branch, down-weighted by `--brl_beta`). It applies to both the BEV and the per-view heatmaps. Without confuse pixels it equals the focal loss. `--use_mse 1` replaces the focal heatmap loss and therefore needs `--loss focal`.

```shell script
CUDA_VISIBLE_DEVICES=0 python main.py -d wildtrack --drop_ratio 60 --loss focal
CUDA_VISIBLE_DEVICES=0 python main.py -d wildtrack --drop_ratio 60 --loss brl
```

Keep `--cls_thres` (default 0.6) and the BRL hyper-parameters fixed across runs that are compared: the test split is the only held-out split, so it should not be used to tune them.

### Offline detector pseudo labels
Pseudo labels are an optional add-on to either base loss (`focal + pseudo`, `brl + pseudo`), applied to the BEV heatmap of the training set only; per-view heatmaps and the test set are unaffected. Without `--use_pseudo_labels` the base loss runs its original code path. The detection cache has the same format for every model in this project (boxes in original image coordinates).

```shell script
CUDA_VISIBLE_DEVICES=0 python main.py -d wildtrack --drop_ratio 60 --loss brl \
  --use_pseudo_labels --pseudo_cache <cache.json>
```

Each box's foot point is projected to the BEV grid and drawn as a Gaussian (`--pseudo_sigma_m`, default 0.5 m = 2x the GT kernel) weighted by detector score; boxes below `--pseudo_conf_threshold` (0.5) and points within `--pseudo_suppress_radius_m` (1 m) of a kept GT are dropped. Projections from different cameras are not fused, matching the MSE models. The pseudo labels act in two separately switchable ways:

| Part | Effect | Ablation flag |
|---|---|---|
| neg reweight | GT focal negative (BRL: easy-negative) term scaled by `1 - pseudo_weight` | `--pseudo_no_neg_reweight` |
| peak loss | score-weighted positive focal term at pseudo peaks, weight `--pseudo_loss_weight` (0.01), normalised by the number of GT peaks; peaks leave the negative / confuse terms | `--pseudo_no_peak_loss` |

Training logs report the base and pseudo-peak loss separately. Not compatible with `--use_mse 1`.

### Ablation
Common flags: `-d wildtrack --drop_ratio {0,20,45,60} --cls_thres 0.6`; `P` = `--use_pseudo_labels --pseudo_cache <cache.json>`. Each configuration writes to its own log dir (`logs/<dataset>/<drop>/<loss_tag>/...`).

| # | Configuration | Question | Flags |
|---|---|---|---|
| 1 | Focal | original baseline | `--loss focal` |
| 2 | BRL | effect of BRL | `--loss brl` |
| 3 | BRL, no mirror | is mirroring confuse pixels toward 1 needed? | `--loss brl --brl_no_mirror` |
| 4 | Focal + pseudo | effect of pseudo labels on focal | `--loss focal P` |
| 5 | BRL + pseudo | full method | `--loss brl P` |
| 6 | BRL + pseudo, neg reweight only | contribution of neg reweight | `--loss brl P --pseudo_no_peak_loss` |
| 7 | BRL + pseudo, peak loss only | contribution of peak loss | `--loss brl P --pseudo_no_neg_reweight` |

Rows 1, 2, 4, 5 form the 2x2 factorial (base loss x pseudo labels); rows 5-7 ablate the pseudo-label parts. Example (row 7):
```shell script
CUDA_VISIBLE_DEVICES=0 python main.py -d wildtrack --drop_ratio 60 --loss brl \
  --use_pseudo_labels --pseudo_cache <cache.json> --pseudo_no_neg_reweight
```

### Tests
- `python tests/test_brl_focal_loss.py` checks that BRL equals the focal loss when no pixel is confuse, and the confuse term otherwise.
- `python tests/test_heatmap_loss.py` checks that the losses without pseudo labels match the previous implementations exactly, and the pseudo-label formulation otherwise.

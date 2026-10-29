# Multiview Detection with Feature Perspective Transformation [[Website](https://hou-yz.github.io/publication/2020-eccv2020-mvdet)] [[arXiv](https://arxiv.org/abs/2007.07247)]

```
@inproceedings{hou2020multiview,
  title={Multiview Detection with Feature Perspective Transformation},
  author={Hou, Yunzhong and Zheng, Liang and Gould, Stephen},
  booktitle={ECCV},
  year={2020}
}
```

Please visit [link](https://github.com/hou-yz/MVDeTr) for our new work MVDeTr, a transformer-powered multiview detector that achieves new state-of-the-art!

## Overview
We release the PyTorch code for **MVDet**, a state-of-the-art multiview pedestrian detector; and **MultiviewX** dataset, a novel synthetic multiview pedestrian detection datatset.

Wildtrack             |  MultiviewX
:-------------------------:|:-------------------------:
![alt text](https://hou-yz.github.io/images/eccv2020_mvdet_wildtrack_demo.gif "Detection results on Wildtrack dataset")  |  ![alt text](https://hou-yz.github.io/images/eccv2020_mvdet_multiviewx_demo.gif "Detection results on MultiviewX dataset")

 
## Content
- [MultiviewX dataset](#multiviewx-dataset)
    * [Download MultiviewX](#download-multiviewx)
    * [Build your own version](#build-your-own-version)
- [MVDet Code](#mvdet-code)
    * [Dependencies](#dependencies)
    * [Data Preparation](#data-preparation)
    * [Training](#training)



## MultiviewX dataset
Using pedestrian models from [PersonX](https://github.com/sxzrt/Dissecting-Person-Re-ID-from-the-Viewpoint-of-Viewpoint), in Unity, we build a novel synthetic dataset **MultiviewX**. 

![alt text](https://hou-yz.github.io/images/eccv2020_mvdet_multiviewx_dataset.jpg "Visualization of MultiviewX dataset")

MultiviewX dataset covers a square of 16 meters by 25 meters. We quantize the ground plane into a 640x1000 grid. There are 6 cameras with overlapping field-of-view in MultiviewX dataset, each of which outputs a 1080x1920 resolution image. We also generate annotations for 400 frames in MultiviewX at 2 fps (same as Wildtrack). On average, 4.41 cameras are covering the same location. 

### Download MultiviewX
Please refer to this [link](https://1drv.ms/u/s!AtzsQybTubHfhYZ9Ghhahbp20OX9kA?e=Hm9Xdg) for download.

### Build your own version
Please refer to this [repo](https://github.com/hou-yz/MultiviewX) for a detailed guide & toolkits you might need.




## MVDet Code
This repo is dedicated to the code for **MVDet**. 

![alt text](https://hou-yz.github.io/images/eccv2020_mvdet_architecture.png "Architecture for MVDet")

### Dependencies
This code uses the following libraries
- python 3.7+
- pytorch 1.4+ & tochvision
- numpy
- matplotlib
- pillow
- opencv-python
- kornia
- matlab & matlabengine (required for evaluation) (see this [link](/multiview_detector/evaluation/README.md) for detailed guide)

### Data Preparation
By default, all datasets are in `~/Data/`. We use [MultiviewX](#multiviewx-dataset) and [Wildtrack](https://www.epfl.ch/labs/cvlab/data/data-wildtrack/) in this project. 

Your `~/Data/` folder should look like this
```
Data
├── MultiviewX/
│   └── ...
└── Wildtrack/ 
    └── ...
```

### Training
In order to train classifiers, please run the following,
```shell script
CUDA_VISIBLE_DEVICES=0,1 python main.py -d wildtrack
``` 
This should automatically return evaluation results similar to the reported 88.2\% MODA on Wildtrack dataset. 

### Pre-trained models
You can download the checkpoints at this [link](https://1drv.ms/u/s!AtzsQybTubHfhNRE9Iy8IjsGMXB17A?e=CCqhIQ).


## BRL and pseudo labels (this fork)
Training with missing annotations. Dropped labels are read from `<data_root>/drop_annotations/drop_XX/annotations_positions` (`--drop_ratio 20/45/60`); evaluation always uses the full annotations.

### Heatmap loss
`--loss mse` is the original `GaussianMSE`. `--loss brl` (default) is a heatmap version of Background Recalibration Loss: background pixels whose prediction is `>= --brl_confuse_thr` are treated as possibly unlabelled people and pulled toward 1 with weight `--brl_beta` (`--brl_no_mirror` instead keeps them on the background target, down-weighted by `--brl_beta`).

```shell script
CUDA_VISIBLE_DEVICES=0 python main.py -d wildtrack --drop_ratio 60 --loss mse
CUDA_VISIBLE_DEVICES=0 python main.py -d wildtrack --drop_ratio 60 --loss brl
```

### Offline detector pseudo labels
Pseudo labels are an optional add-on to either base loss (`mse + pseudo`, `brl + pseudo`), applied to the BEV map of the training set only. Without `--use_pseudo_labels` the base loss runs its original code path.

1. Detect people in every camera frame (stored in original image coordinates):
   ```shell script
   python generate_pseudo_cache.py --dataset wildtrack --data_root ~/Data/Wildtrack \
     --output pseudo_cache/wildtrack_yolo26s.json
   python evaluate_pseudo_cache.py --dataset wildtrack --data_root ~/Data/Wildtrack \
     --cache pseudo_cache/wildtrack_yolo26s.json
   ```
2. Train:
   ```shell script
   CUDA_VISIBLE_DEVICES=0 python main.py -d wildtrack --drop_ratio 60 --loss brl \
     --use_pseudo_labels --pseudo_cache pseudo_cache/wildtrack_yolo26s.json
   ```

Each box's foot point is projected to the BEV grid and drawn as a Gaussian (`--pseudo_sigma_m`, default 0.5 m) weighted by detector score; boxes below `--pseudo_conf_threshold` (0.5) and points within `--pseudo_suppress_radius_m` (1 m) of a kept GT are dropped. The pseudo labels act in two separately switchable ways:

| Part | Effect | Ablation flag |
|---|---|---|
| neg reweight | GT background term scaled by `1 - pseudo_weight` | `--pseudo_no_neg_reweight` |
| peak loss | pseudo peaks pulled toward 1, weight `--pseudo_loss_weight` (0.01); peaks leave the GT background / confuse terms | `--pseudo_no_peak_loss` |

Training logs report the base and pseudo-peak loss separately. The cache is dataset-specific; generate a separate one for MultiviewX.

### Ablation
Common flags: `-d wildtrack --drop_ratio {0,20,45,60}`; `P` = `--use_pseudo_labels --pseudo_cache <cache.json>`. Each configuration writes to its own log dir (`logs/<dataset>_frame/<drop>/<loss_tag>/...`).

| # | Configuration | Question | Flags |
|---|---|---|---|
| 1 | MSE | original baseline | `--loss mse` |
| 2 | BRL | effect of BRL | `--loss brl` |
| 3 | BRL, no mirror | is mirroring confuse pixels toward 1 needed? | `--loss brl --brl_no_mirror` |
| 4 | MSE + pseudo | effect of pseudo labels on MSE | `--loss mse P` |
| 5 | BRL + pseudo | full method | `--loss brl P` |
| 6 | BRL + pseudo, neg reweight only | contribution of neg reweight | `--loss brl P --pseudo_no_peak_loss` |
| 7 | BRL + pseudo, peak loss only | contribution of peak loss | `--loss brl P --pseudo_no_neg_reweight` |

Rows 1, 2, 4, 5 form the 2x2 factorial (base loss x pseudo labels); rows 5-7 ablate the pseudo-label parts. Example (row 7):
```shell script
CUDA_VISIBLE_DEVICES=0 python main.py -d wildtrack --drop_ratio 60 --loss brl \
  --use_pseudo_labels --pseudo_cache <cache.json> --pseudo_no_neg_reweight
```

### Tests
`python tests/test_heatmap_loss.py` checks that the losses without pseudo labels match the original implementations exactly.

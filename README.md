# Early Bird 🦅

**EarlyBird: Early-Fusion for Multi-View Tracking in the Bird's Eye View**

Torben Teepe, Philipp Wolters, Johannes Gilg, Fabian Herzog, Gerhard Rigoll

[![arxiv](https://img.shields.io/badge/arXiv-2310.13350-red)](https://arxiv.org/abs/2310.13350)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/earlybird-early-fusion-for-multi-view/multi-object-tracking-on-wildtrack)](https://paperswithcode.com/sota/multi-object-tracking-on-wildtrack?p=earlybird-early-fusion-for-multi-view)

## Usage

### Getting Started
1. Install [PyTorch](https://pytorch.org/get-started/locally/) with CUDA support
    ```shell
   pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```
2. Install [mmcv](https://mmcv.readthedocs.io/en/latest/get_started/installation.html#install-with-pip) with CUDA support
   ```shell
   pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
   ```
3. Install remaining dependencies
   ```shell
   pip install -r requirements.txt
   ```

#### Training
```shell
python main.py fit -c configs/t_fit.yml \
    -c configs/d_{multiviewx,wildtrack}.yml
```

#### Partial supervision with BEV pseudo-labels
When training on dropped annotations (`drop_ratio` 20/45/60) with the BRL configs, an offline person-detection cache can add pseudo-label supervision. The approach is the same as TrackTacular and is based on MVDet's pseudo loss. Each cached box gives a foot point, which is projected to the ground plane and then put through the same grid augmentation as the GT. Projections of one person from different cameras are fused, and the result is rasterized as soft BEV center evidence `pseudo_center_bev` / `pseudo_weight_bev`. Pseudo points within `pseudo_suppress_radius_m` of a remaining GT center are dropped. Validation and test always use full GT.

- `pseudo_mode: focal` (default): inside pseudo blobs the negative focal penalty is scaled by `1 - pseudo_weight`. Pseudo peaks also get a score-weighted positive focal term, scaled by `pseudo_loss_weight` and the same uncertainty factor as the center loss.
- `pseudo_mode: mse`: confidence-weighted MSE between `sigmoid(center)` and the pseudo target (legacy).

The cache format is shared with TrackTacular, so a cache built in either repo works in both.

```shell
cd EarlyBird
pip install -r requirements-pseudo.txt
python generate_pseudo_cache.py --dataset wildtrack --data-root /path/Wildtrack \
    --output pseudo_cache/wildtrack.json --split train --weights yolo26s.pt --device 0

# sanity checks
python check_pseudo.py loss
python check_pseudo.py data --dataset wildtrack --data-dir /path/Wildtrack \
    --pseudo-cache pseudo_cache/wildtrack.json --drop-ratio 60

# train (or: PSEUDO_CACHE_DIR=pseudo_cache ./run_brl.sh)
python main.py fit -c configs/t_fit.yml -c configs/d_wildtrack_brl.yml \
    --data.init_args.drop_ratio 60 \
    --data.init_args.pseudo_cache pseudo_cache/wildtrack.json \
    --model.pseudo_mode focal --model.pseudo_loss_weight 0.01
```

Data options: `pseudo_conf_threshold`, `pseudo_sigma_m`, `pseudo_suppress_radius_m`, `pseudo_fuse_radius_m`, `pseudo_min_views`. Model options: `pseudo_loss_weight`, `pseudo_mode`.

#### Testing
```shell
python main.py test -c model_weights/config.yaml \
    --ckpt model_weights/model-epoch=35-val_loss=6.50.ckpt
```

## Acknowledgement
- [Simple-BEV](https://simple-bev.github.io): Adam W. Harley
- [MVDeTr](https://github.com/hou-yz/MVDeTr): Yunzhong Hou

## Cite
If you use EarlyBird, please use the following BibTeX entry.

```
@InProceedings{teepe2023earlybird,
    author    = {Teepe, Torben and Wolters, Philipp and Gilg, Johannes and Herzog, Fabian and Rigoll, Gerhard},
    title     = {Early{B}ird: Early-Fusion for Multi-View Tracking in the Bird's Eye View},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) Workshops},
    month     = {January},
    year      = {2024},
    pages     = {102-111}
}
```

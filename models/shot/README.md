# SHOT

Our code is based on [MVDet](https://github.com/hou-yz/MVDet). Please see `README-mvdet.md` for installation and dataset preparation.

Trained models and logs are available at [OneDrive](https://1drv.ms/u/s!AqzBcxT1FhwG0VVogbiIfxYRBN-h?e=B8zPad).

# Our code

- `multiview_detector/models/dpersp_trans_detector.py` is the implementation of our method.
- `multiview_detector/models/dpersp_trans_detector_visualize.py` is for visualizing the feature maps in our method.


# BRL on dropped annotations (this fork)
Dropped labels are read from `<data_root>/drop_annotations/drop_XX/annotations_positions` (`--drop_ratio 20/45/60`); evaluation always uses the full annotations.

| `--loss` | Training loss (variant `default`) |
|---|---|
| `brl` (default) | `BRLGaussianMSE`: background pixels with prediction `>= --brl_confuse_thr` are pulled toward 1 with weight `--brl_beta` |
| `mse` | `GaussianMSE`, the same target as `brl` (max-pool + Gaussian, as in MVDet); the baseline for BRL |
| `shot_mse` | the original SHOT loss inside `DPerspTransDetector.forward` (avg-pool + Gaussian target, `nn.MSELoss`); for reproducing the paper |

`mse` and `shot_mse` differ only in the per-view target: the image GT is pooled from 1080x1920 to 270x480, so avg-pool lowers each peak to about 1/16 while max-pool keeps it at 1. The BEV target is identical. Compare `brl` against `mse`; `shot_mse` vs `mse` measures the effect of the target change alone. Other variants (`per`, `img_proj`, ...) always train with the criterion, so `shot_mse` equals `mse` there.

```shell script
CUDA_VISIBLE_DEVICES=0 python main.py -d wildtrack --drop_ratio 60 --loss brl
```

## Offline detector pseudo labels
Pseudo labels are an optional add-on to `--loss mse` or `--loss brl` (`mse + pseudo`, `brl + pseudo`), applied to the BEV map of the training set only. Without `--use_pseudo_labels` the base loss runs its original code path. `--loss shot_mse` bypasses the criterion, so it cannot be combined with pseudo labels (`main.py` raises an error).

1. Detect people in every camera frame (stored in original image coordinates):
   ```shell script
   python generate_pseudo_cache.py --dataset wildtrack --data_root ./data/Wildtrack \
     --output pseudo_cache/wildtrack_yolo26s.json
   python evaluate_pseudo_cache.py --dataset wildtrack --data_root ./data/Wildtrack \
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

Training logs report the base and pseudo-peak loss separately. The loss modules are the same files as in MVDet. The peak term is normalised by the BEV pixel count as in MVDet, so lambda has the same meaning relative to the BEV term; relative to the total loss it is smaller here, because SHOT weights the per-view term by `N_cam * alpha`. The cache is dataset-specific; generate a separate one for MultiviewX.

### Ablation
Common flags: `-d wildtrack --drop_ratio {0,20,45,60}`; `P` = `--use_pseudo_labels --pseudo_cache <cache.json>`. Each configuration writes to its own log dir (`logs/<dataset>_frame/<drop>/<loss_tag>/...`).

| # | Configuration | Question | Flags |
|---|---|---|---|
| 0 | SHOT original loss | reproduce the paper (reference only) | `--loss shot_mse` |
| 1 | MSE | baseline (same target as BRL) | `--loss mse` |
| 2 | BRL | effect of BRL | `--loss brl` |
| 3 | BRL, no mirror | is mirroring confuse pixels toward 1 needed? | `--loss brl --brl_no_mirror` |
| 4 | MSE + pseudo | effect of pseudo labels on MSE | `--loss mse P` |
| 5 | BRL + pseudo | full method | `--loss brl P` |
| 6 | BRL + pseudo, neg reweight only | contribution of neg reweight | `--loss brl P --pseudo_no_peak_loss` |
| 7 | BRL + pseudo, peak loss only | contribution of peak loss | `--loss brl P --pseudo_no_neg_reweight` |

Rows 1, 2, 4, 5 form the 2x2 factorial (base loss x pseudo labels); rows 5-7 ablate the pseudo-label parts. Row 0 vs row 1 measures the per-view target change alone (avg-pool vs max-pool); row 0 is not compared with BRL directly. Example (row 7):
```shell script
CUDA_VISIBLE_DEVICES=0 python main.py -d wildtrack --drop_ratio 60 --loss brl \
  --use_pseudo_labels --pseudo_cache <cache.json> --pseudo_no_neg_reweight
```

### Tests
`python tests/test_heatmap_loss.py` checks that the losses without pseudo labels match the original implementations exactly.

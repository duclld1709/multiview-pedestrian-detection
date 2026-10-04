# SHOT

Our code is based on [MVDet](https://github.com/hou-yz/MVDet). Please see `README-mvdet.md` for installation and dataset preparation.

# Our code

- `multiview_detector/models/dpersp_trans_detector.py` is the implementation of our method.
- `multiview_detector/models/dpersp_trans_detector_visualize.py` is for visualizing the feature maps in our method.


# Offline detector pseudo-labels

Same pseudo-label loss as in `../MVDet`. Detector boxes from a JSON cache are projected (box bottom-center) onto the BEV grid, rasterized as soft Gaussians, suppressed near existing GT, and added as a confidence-weighted MSE on the BEV heatmap.

```bash
python generate_pseudo_cache.py --dataset wildtrack --data_root <Wildtrack root> \
  --output pseudo_cache/wildtrack_yolo26s.json
python evaluate_pseudo_cache.py --dataset wildtrack --data_root <Wildtrack root> \
  --cache pseudo_cache/wildtrack_yolo26s.json

python main.py -d wildtrack --drop_ratio 60 --loss brl --use_pseudo_labels \
  --pseudo_cache pseudo_cache/wildtrack_yolo26s.json
```

Caches generated with the MVDet scripts work here too. Defaults are pseudo loss weight 0.01, confidence threshold 0.2, BEV Gaussian sigma 0.5 m, and GT suppression radius 1 m. Training logs report base and weighted pseudo loss separately.

Note: the default SHOT model now trains with the loss selected by `--loss` (BRL or GaussianMSE). Before this change it always used plain MSE inside `forward`, so `--loss brl` had no effect and older SHOT runs are not directly comparable.

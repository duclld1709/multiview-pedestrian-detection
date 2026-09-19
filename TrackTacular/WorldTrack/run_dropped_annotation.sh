#!/usr/bin/env bash
# Train TrackTacular on full + dropped annotations (drop_0 / drop_20 / drop_45 / drop_60).
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${MODEL:-mvdet}"  # mvdet | segnet | liftnet | bevformer

for dataset in wildtrack multiviewx; do
  for drop_ratio in 0 20 45 60; do
    echo "===== Running $dataset / model=$MODEL / drop_ratio=$drop_ratio ====="
    CUDA_VISIBLE_DEVICES=0 python world_track.py fit \
      -c configs/t_fit.yml \
      -c "configs/d_${dataset}.yml" \
      -c "configs/m_${MODEL}.yml" \
      --data.init_args.drop_ratio "$drop_ratio"
  done
done

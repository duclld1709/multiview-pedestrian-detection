#!/usr/bin/env bash
# Train EarlyBird + BRL on dropped annotations, optionally with pseudo-label supervision.
#
#   PSEUDO_CACHE_DIR=/path/to/caches ./run_brl.sh      # expects $PSEUDO_CACHE_DIR/<dataset>.json
#   PSEUDO_MODE=mse PSEUDO_W=0.01 SEEDS="1 2 3" ./run_brl.sh
set -euo pipefail
cd "$(dirname "$0")"

DATASETS="${DATASETS:-wildtrack multiviewx}"
DROP="${DROP:-60}"
SEEDS="${SEEDS:-80085}"
PSEUDO_CACHE_DIR="${PSEUDO_CACHE_DIR:-}"  # empty = BRL baseline without pseudo labels
PSEUDO_MODE="${PSEUDO_MODE:-focal}"       # focal | mse
PSEUDO_W="${PSEUDO_W:-0.01}"

for dataset in $DATASETS; do
  for seed in $SEEDS; do
    pseudo_args=()
    tag="brl"
    if [[ -n "$PSEUDO_CACHE_DIR" ]]; then
      pseudo_args=(
        --data.init_args.pseudo_cache "$PSEUDO_CACHE_DIR/${dataset}.json"
        --model.pseudo_mode "$PSEUDO_MODE"
        --model.pseudo_loss_weight "$PSEUDO_W"
      )
      tag="brl+pseudo_${PSEUDO_MODE}_w${PSEUDO_W}"
    fi
    echo "===== $dataset / drop=$DROP / $tag / seed=$seed ====="
    CUDA_VISIBLE_DEVICES=0 python main.py fit \
      -c configs/t_fit.yml \
      -c "configs/d_${dataset}_brl.yml" \
      --seed_everything "$seed" \
      --data.init_args.drop_ratio "$DROP" \
      ${pseudo_args[@]+"${pseudo_args[@]}"}
  done
done

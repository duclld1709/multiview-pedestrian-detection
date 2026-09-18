#!/usr/bin/env bash
# SHOTBRL — BRL v2 on dropped annotations (no full / drop_0).
set -e
cd "$(dirname "$0")"
for dataset in wildtrack multiviewx; do
  for drop_ratio in 20 45 60; do
    echo "===== Running $dataset / brl_v2 / drop_ratio=$drop_ratio ====="
    CUDA_VISIBLE_DEVICES=0 python main.py -d "$dataset" --drop_ratio "$drop_ratio" --loss brl_v2
  done
done
echo "===== All done ====="

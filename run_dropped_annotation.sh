#!/usr/bin/env bash
# BRL heatmap experiments on dropped annotations
set -e
cd "$(dirname "$0")"

PYTHON="${PYTHON:-$HOME/miniconda3/envs/thesis_env/bin/python}"

for dataset in wildtrack multiviewx; do
  for drop_ratio in 0 20 45 60; do
    echo "=========================================="
    echo "Running $dataset / drop_ratio=$drop_ratio / BRL"
    echo "=========================================="
    CUDA_VISIBLE_DEVICES=0 "$PYTHON" main.py -d "$dataset" --drop_ratio "$drop_ratio" --loss brl
  done
done

echo "All dropped-annotation BRL runs finished."

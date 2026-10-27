#!/usr/bin/env bash
for loss in mse brl; do
  for dataset in wildtrack multiviewx; do
    for drop_ratio in 20 45 60; do
      echo "Running $dataset / drop_ratio=$drop_ratio / $loss"
      CUDA_VISIBLE_DEVICES=0 python main.py -d "$dataset" --drop_ratio "$drop_ratio" --loss "$loss"
    done
  done
done

# Pseudo-label runs (set PSEUDO_CACHE_DIR to the folder holding <dataset>_yolo26s.json)
if [ -n "$PSEUDO_CACHE_DIR" ]; then
  for dataset in wildtrack multiviewx; do
    for drop_ratio in 20 45 60; do
      echo "Running $dataset / drop_ratio=$drop_ratio / brl + pseudo"
      CUDA_VISIBLE_DEVICES=0 python main.py -d "$dataset" --drop_ratio "$drop_ratio" --loss brl \
        --use_pseudo_labels --pseudo_cache "$PSEUDO_CACHE_DIR/${dataset}_yolo26s.json"
    done
  done
fi

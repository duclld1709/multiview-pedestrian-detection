#!/usr/bin/env bash
for loss in mse brl; do
  for dataset in wildtrack multiviewx; do
    for drop_ratio in 20 45 60; do
      echo "Running $dataset / drop_ratio=$drop_ratio / $loss"
      CUDA_VISIBLE_DEVICES=0 python main.py -d "$dataset" --drop_ratio "$drop_ratio" --loss "$loss"
    done
  done
done
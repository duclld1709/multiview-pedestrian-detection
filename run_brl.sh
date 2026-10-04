#!/usr/bin/env bash
# for dataset in wildtrack multiviewx; do
#   for drop_ratio in drop_20 drop_45 drop_60; do
#     echo "===== Running $dataset / $drop_ratio ====="
#     CUDA_VISIBLE_DEVICES=0 python main.py -d "$dataset" --drop_ratio "$drop_ratio" --cls_thres 0.4 --apply_brl True
#   done
# done
# echo "===== All done ====="
for beta in 0.05 0.08; do
  for confuse_thr in 0.3 0.4; do
    CUDA_VISIBLE_DEVICES=0 python main.py -d wildtrack --drop_ratio drop_20 --cls_thres 0.6 --apply_brl True --brl_beta $beta --brl_confuse_thr $confuse_thr
  done
done
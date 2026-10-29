#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    -d wildtrack \
    --data_path ../Data/Wildtrack \
    --drop_ratio 60 \
    --loss brl \
    --brl_pos_thr 0.1 \
    --brl_confuse_thr 0.3 \
    --brl_beta 0.1 \
    --use_pseudo_labels \
    --pseudo_cache pseudo_cache/wildtrack_yolo26s.json

#!/usr/bin/env bash
# DATA_DIR: dataset root (its basename must contain wildtrack or multiviewx)
CUDA_VISIBLE_DEVICES=0 python main.py fit \
    -c configs/t_fit.yml \
    -c configs/d_wildtrack.yml \
    --data.init_args.data_dir "${DATA_DIR:-../../Data/Wildtrack}" \
    --data.init_args.drop_ratio 60 \
    --model.heatmap_loss brl \
    --model.brl_pos_thr 0.1 \
    --model.brl_confuse_thr 0.4 \
    --model.brl_beta 0.05

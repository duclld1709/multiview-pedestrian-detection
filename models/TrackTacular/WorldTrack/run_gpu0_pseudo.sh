#!/usr/bin/env bash
# DATA_DIR: dataset root (its basename must contain wildtrack or multiviewx)
# PSEUDO_CACHE: JSON detection cache shared by all models
# The ablation uses the mvdet architecture for every run.
CUDA_VISIBLE_DEVICES=0 python world_track.py fit \
    -c configs/t_fit.yml \
    -c configs/d_wildtrack.yml \
    -c configs/m_mvdet.yml \
    --data.init_args.data_dir "${DATA_DIR:-../../Data/Wildtrack}" \
    --data.init_args.drop_ratio 60 \
    --data.init_args.pseudo_cache "${PSEUDO_CACHE:?set PSEUDO_CACHE to the detection cache json}" \
    --model.heatmap_loss brl \
    --model.brl_pos_thr 0.1 \
    --model.brl_confuse_thr 0.4 \
    --model.brl_beta 0.05 \
    --model.use_pseudo_labels true

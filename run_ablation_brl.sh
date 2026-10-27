#!/usr/bin/env bash
for dataset in wildtrack multiviewx; do
    for drop_ratio in 20 45 60; do
        for brl_pos_thr in 0.2 0.3 0.4 0.5; do
            for brl_confuse_thr in 0.1 0.4 0.5 0.6; do
                echo "Running $dataset / drop_ratio=$drop_ratio / BRL / brl_pos_thr=$brl_pos_thr / brl_confuse_thr=$brl_confuse_thr"
                CUDA_VISIBLE_DEVICES=0 python main.py -d "$dataset" --drop_ratio "$drop_ratio" --loss brl --brl_pos_thr "$brl_pos_thr" --brl_confuse_thr "$brl_confuse_thr"
            done
        done
    done
  done
done
#!/usr/bin/env bash
# Compare Focal vs BRL on dropped-annotation splits.
for dataset in "wildtrack" "multiviewx"; do
	for drop_ratio in "drop20" "drop45" "drop60"; do
		for loss in "brl"; do
			echo "Running $dataset / $drop_ratio / $loss"
			CUDA_VISIBLE_DEVICES=0 python main.py -d "$dataset" --drop_ratio "$drop_ratio" --loss "$loss" --cls_thres 0.4
		done
	done
done

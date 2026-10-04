#!/usr/bin/env bash
for dataset in "wildtrack" "multiviewx"; do
	echo "Running $dataset"
    CUDA_VISIBLE_DEVICES=0 python main.py -d $dataset
done
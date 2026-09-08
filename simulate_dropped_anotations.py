import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import os

DATASET_META = {
	"wildtrack": {
        "num_frame": 2000,
        "num_cam": 7,
    },
    "multiviewx": {
        "num_frame": 400,
        "num_cam": 6,
    },
}
 
def simulate(dataset, root, out, train_ratio, settings):
    meta = DATASET_META[dataset]
    num_frame = meta["num_frame"]
    num_cam = meta["num_cam"]
    train_nums = int(num_frame * train_ratio)
    
    annotations_dir = os.path.join(root, "annotations_positions")
    
    files = sorted(annotations_dir.glob("*.json"))
    
    for setting in settings:
        setting_root = os.path.join(out, setting)
        os.makedirs(setting_root, exist_ok=True)
        observed_dir = os.path.join(setting_root, "annotations_positions")
        hidden_dir = os.path.join(setting_root, "hidden_annotations_positions")
        os.makedirs(observed_dir, exist_ok=True)
        os.makedirs(hidden_dir, exist_ok=True)

        total_train_instances = 0
        dropped_train_instances = 0
        train_frame
        
        for file in files:
            file_name = os.path.basename(file)
            file_name = file_name.split(".")[0]
            file_name = file_name.split("_")[-1]
            file_name = file_name.split(".")[0]
        
 
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='wildtrack', choices=['wildtrack', 'multiviewx'])
    parser.add_argument('-root', type=str, help='Dataset root containing annotations_positions')
    parser.add_argument('--out', type=str, help='Output root for simulated annotations sets')
    parser.add_argument('--train_ratio', type=float, default=0.9, help='Ratio of training frames')
    parser.add_argument(
        "--settings",
        nargs="+",
        default=["normal", "easy", "hard", "extreme"],
        choices=["normal", "easy", "hard", "extreme"],
    )
    args = parser.parse_args()

    simulate(
        dataset=args.dataset,
        root=args.root,
        out=args.out,
        train_ratio=args.train_ratio,
        settings=args.settings
    )
    
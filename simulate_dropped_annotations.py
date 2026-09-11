#!/usr/bin/env python3
"""
Simulate missing-instance annotations for Wildtrack / MultiviewX.

Example:
    python simulate_dropped_anotations.py -d Wildtrack -r ./Data_temp -s drop20 drop45 drop60 --seed 1

Writes JSON-only folders under:
    {root}/{dataset}/drop_annotations/drop_20/annotations_positions/
    {root}/{dataset}/drop_annotations/drop_45/annotations_positions/
    {root}/{dataset}/drop_annotations/drop_60/annotations_positions/
"""

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np

DATASET_META = {
    "Wildtrack": {
        "num_frame": 2000,
        "num_cam": 7,
    },
    "MultiviewX": {
        "num_frame": 400,
        "num_cam": 6,
    },
}

# CLI name -> (folder name under drop_annotations/, drop ratio)
DROP_SETTINGS = {
    "drop20": ("drop_20", 0.20),
    "drop45": ("drop_45", 0.45),
    "drop60": ("drop_60", 0.60),
}

DROP_ANNOTATIONS_DIR = "drop_annotations"


def save_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def split_instances_by_ratio(
    persons: List[dict],
    drop_ratio: float,
    frame_id: int,
    seed: int = 1,
) -> Tuple[List[dict], List[dict]]:
    n = len(persons)
    if n == 0 or drop_ratio <= 0:
        return list(persons), []

    n_drop = int(round(n * drop_ratio))
    n_drop = max(0, min(n_drop, n))
    if n_drop == 0:
        return list(persons), []

    ss = np.random.SeedSequence([seed, frame_id, int(round(drop_ratio * 1000))])
    rng = np.random.default_rng(ss)
    drop_idx = set(rng.choice(n, size=n_drop, replace=False).tolist())
    observed = [p for i, p in enumerate(persons) if i not in drop_idx]
    hidden = [p for i, p in enumerate(persons) if i in drop_idx]

    assert len(observed) + len(hidden) == n
    return observed, hidden


def simulate(dataset, root, train_ratio, settings, seed=1):
    meta = DATASET_META[dataset]
    num_frame = meta["num_frame"]
    train_cutoff = int(num_frame * train_ratio)

    dataset_root = os.path.join(root, dataset)
    annotations_dir = os.path.join(dataset_root, "annotations_positions")
    if not os.path.isdir(annotations_dir):
        raise FileNotFoundError(f"Annotations directory not found: {annotations_dir}")

    files = sorted(f for f in os.listdir(annotations_dir) if f.endswith(".json"))
    if not files:
        raise FileNotFoundError(f"No .json files found in {annotations_dir}")

    drop_root = os.path.join(dataset_root, DROP_ANNOTATIONS_DIR)
    os.makedirs(drop_root, exist_ok=True)
    all_stats: Dict[str, dict] = {}

    for setting in settings:
        folder_name, drop_ratio = DROP_SETTINGS[setting]
        # JSON-only: {root}/{dataset}/drop_annotations/drop_XX/
        setting_root = os.path.join(drop_root, folder_name)
        observed_dir = os.path.join(setting_root, "annotations_positions")
        hidden_dir = os.path.join(setting_root, "hidden_annotations_positions")
        os.makedirs(observed_dir, exist_ok=True)
        os.makedirs(hidden_dir, exist_ok=True)

        total_train_instances = 0
        dropped_train_instances = 0
        train_frames = 0
        test_frames = 0
        per_frame_drop_ratios = []

        for fname in files:
            src_path = os.path.join(annotations_dir, fname)
            frame_id = int(os.path.splitext(fname)[0])
            with open(src_path, "r", encoding="utf-8") as f:
                persons = json.load(f)

            is_train = frame_id < train_cutoff
            if is_train:
                observed, hidden = split_instances_by_ratio(
                    persons, drop_ratio, frame_id, seed=seed
                )
                train_frames += 1
                total_train_instances += len(persons)
                dropped_train_instances += len(hidden)
                per_frame_drop_ratios.append(
                    (len(hidden) / len(persons)) if persons else 0.0
                )
            else:
                observed = persons
                hidden = []
                test_frames += 1

            # keep original zero-padded filename (e.g. 00000000.json)
            save_json(os.path.join(observed_dir, fname), observed)
            save_json(os.path.join(hidden_dir, fname), hidden)

        retained = total_train_instances - dropped_train_instances
        global_drop_ratio = (
            dropped_train_instances / total_train_instances
            if total_train_instances > 0
            else 0.0
        )
        mean_per_frame_drop = (
            float(np.mean(per_frame_drop_ratios)) if per_frame_drop_ratios else 0.0
        )

        stats = {
            "dataset": dataset,
            "setting": setting,
            "folder": folder_name,
            "target_drop_ratio_per_frame": drop_ratio,
            "seed": seed,
            "train_ratio": train_ratio,
            "train_cutoff_frame_id": train_cutoff,
            "num_annotation_files": len(files),
            "num_train_annotation_files": train_frames,
            "num_test_annotation_files": test_frames,
            "train_instances_full": total_train_instances,
            "train_instances_observed": retained,
            "train_instances_hidden": dropped_train_instances,
            "mean_per_frame_drop_ratio": mean_per_frame_drop,
            "global_drop_ratio": global_drop_ratio,
        }
        all_stats[setting] = stats
        save_json(os.path.join(setting_root, "stats.json"), stats)

        print(
            f"[{dataset}] {setting} -> {folder_name}: "
            f"target {100 * drop_ratio:.0f}%/frame, "
            f"mean per-frame drop {100 * mean_per_frame_drop:.2f}%, "
            f"global {retained}/{total_train_instances} kept "
            f"({100 * global_drop_ratio:.2f}% dropped) -> {setting_root}"
        )

    summary_path = os.path.join(drop_root, "summary.json")
    save_json(summary_path, all_stats)
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate dropped annotations for MultiviewX/Wildtrack dataset"
    )
    parser.add_argument(
        "-d", "--dataset", type=str, default="Wildtrack",
        choices=["Wildtrack", "MultiviewX"],
    )
    parser.add_argument(
        "-r", "--root", type=str, default="./Data_temp",
        help="Folder containing dataset (e.g. ./Data_temp)",
    )
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument(
        "-s", "--settings", nargs="+", default=["drop20", "drop45", "drop60"],
        choices=list(DROP_SETTINGS.keys()),
    )
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    simulate(
        dataset=args.dataset,
        root=args.root,
        train_ratio=args.train_ratio,
        settings=args.settings,
        seed=args.seed,
    )

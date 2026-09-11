#!/usr/bin/env python3
"""
Simulate missing-instance annotations for Wildtrack / MultiviewX.

Drop ratios are applied PER FRAME on train annotations:
- drop20: randomly drop ~20% of instances in each train frame
- drop45: randomly drop ~45% of instances in each train frame
- drop60: randomly drop ~60% of instances in each train frame

TEST frames are copied unchanged.

Example:
    python "test copy.py" \\
        --dataset wildtrack \\
        --root ~/thesis/Data/Wildtrack \\
        --out ~/thesis/Data/Wildtrack_dropped
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple
import os
import shutil
import numpy as np


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

# setting name -> per-frame drop ratio
DROP_SETTINGS = {
    "drop20": 0.20,
    "drop45": 0.45,
    "drop60": 0.60,
}


def load_frame(path: Path, expected_num_cam: int) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        persons = json.load(f)

    if not isinstance(persons, list):
        raise TypeError(f"{path}: expected a JSON list, got {type(persons).__name__}")

    for i, p in enumerate(persons):
        missing = {"personID", "positionID", "views"} - set(p.keys())
        if missing:
            raise KeyError(f"{path}: person[{i}] missing keys {sorted(missing)}")
        if not isinstance(p["views"], list):
            raise TypeError(f"{path}: person[{i}]['views'] must be a list")
        if len(p["views"]) != expected_num_cam:
            raise ValueError(
                f"{path}: person[{i}] has {len(p['views'])} views; "
                f"expected {expected_num_cam}"
            )
    return persons


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def split_instances_by_ratio(
    persons: List[dict],
    drop_ratio: float,
    seed: int,
    frame_id: int,
) -> Tuple[List[dict], List[dict]]:
    """
    Drop ~drop_ratio of instances in THIS frame only.
    n_drop = round(n * drop_ratio), clipped to [0, n].
    """
    n = len(persons)
    if n == 0 or drop_ratio <= 0:
        return list(persons), []

    n_drop = int(round(n * drop_ratio))
    n_drop = max(0, min(n_drop, n))
    if n_drop == 0:
        return list(persons), []

    # Deterministic per-frame RNG
    ss = np.random.SeedSequence([seed, frame_id, int(round(drop_ratio * 1000))])
    rng = np.random.default_rng(ss)

    drop_idx = set(rng.choice(n, size=n_drop, replace=False).tolist())
    observed = [p for i, p in enumerate(persons) if i not in drop_idx]
    hidden = [p for i, p in enumerate(persons) if i in drop_idx]

    assert len(observed) + len(hidden) == n
    return observed, hidden


def simulate(dataset, root, out, seed, train_ratio, settings):
    meta = DATASET_META[dataset]
    num_frame = meta["num_frame"]
    num_cam = meta["num_cam"]
    train_cutoff = int(num_frame * train_ratio)

    ann_dir = root / "annotations_positions"
    if not ann_dir.is_dir():
        raise FileNotFoundError(f"Missing annotation directory: {ann_dir}")

    files = sorted(ann_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No .json files found in {ann_dir}")

    # No seed_* subfolder — write directly under --out
    out.mkdir(parents=True, exist_ok=True)
    all_stats: Dict[str, dict] = {}

    for setting in settings:
        drop_ratio = DROP_SETTINGS[setting]
        setting_root = out / setting
        observed_dir = setting_root / "annotations_positions"
        hidden_dir = setting_root / "hidden_annotations_positions"
        observed_dir.mkdir(parents=True, exist_ok=True)
        hidden_dir.mkdir(parents=True, exist_ok=True)

        total_train_instances = 0
        dropped_train_instances = 0
        train_frames = 0
        test_frames = 0
        per_frame_drop_ratios = []
        per_frame_rows = []

        for src in files:
            frame_id = int(src.stem)
            persons = load_frame(src, num_cam)
            is_train = frame_id < train_cutoff

            if is_train:
                observed, hidden = split_instances_by_ratio(
                    persons, drop_ratio, seed, frame_id
                )
                train_frames += 1
                total_train_instances += len(persons)
                dropped_train_instances += len(hidden)
                frame_drop = (len(hidden) / len(persons)) if persons else 0.0
                per_frame_drop_ratios.append(frame_drop)
            else:
                observed = persons
                hidden = []
                test_frames += 1

            save_json(observed_dir / src.name, observed)
            save_json(hidden_dir / src.name, hidden)

            per_frame_rows.append({
                "frame_id": frame_id,
                "split": "train" if is_train else "test",
                "n_full": len(persons),
                "n_observed": len(observed),
                "n_hidden": len(hidden),
                "frame_drop_ratio": (
                    (len(hidden) / len(persons)) if persons and is_train else 0.0
                ),
            })

        retained = total_train_instances - dropped_train_instances
        global_drop_ratio = (
            dropped_train_instances / total_train_instances
            if total_train_instances > 0 else 0.0
        )
        mean_per_frame_drop = (
            float(np.mean(per_frame_drop_ratios)) if per_frame_drop_ratios else 0.0
        )

        stats = {
            "dataset": dataset,
            "setting": setting,
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

        with (setting_root / "per_frame_stats.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "frame_id", "split", "n_full",
                    "n_observed", "n_hidden", "frame_drop_ratio",
                ],
            )
            writer.writeheader()
            writer.writerows(per_frame_rows)

        with (setting_root / "stats.json").open("w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        print(
            f"[{dataset:10s}] {setting}: "
            f"target {100 * drop_ratio:.0f}%/frame, "
            f"mean per-frame drop {100 * mean_per_frame_drop:.2f}%, "
            f"global {retained}/{total_train_instances} kept "
            f"({100 * global_drop_ratio:.2f}% dropped)"
        )

    with (out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2)

    print(f"\nSaved to: {out}")


def main():
    parser = argparse.ArgumentParser(
        description="Write drop0/20/45/60 annotation folders inside the dataset root."
    )
    parser.add_argument("-d", "--dataset", required=True, choices=["wildtrack", "multiviewx"])
    parser.add_argument("-r", "--root", required=True, type=Path, help="Dataset root containing annotations_positions/")
    parser.add_argument("-o", "--out", required=True, type=Path, help="Output root (no seed_* subfolder)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("-s", "--settings", nargs="+", default=["drop20", "drop45", "drop60"], choices=list(DROP_SETTINGS.keys()))
    args = parser.parse_args()

    simulate(
        dataset=args.dataset,
        root=args.root.expanduser().resolve(),
        out=args.out.expanduser().resolve(),
        seed=args.seed,
        train_ratio=args.train_ratio,
        settings=args.settings,
    )
    os.shutil.copytree(args.root / "annotations_positions", args.out / "annotations_positions")
    os.shutil.copytree(args.root / "Image_subsets", args.out / "Image_subsets")
    os.shutil.copy2(args.root / "rectangles.pom", args.out / "rectangles.pom")
    if args.dataset == "MultiviewX":
        os.shutil.copytree(args.root / "matchings", args.out / "matchings")
        
        


if __name__ == "__main__":
    main()

"""Build a full FP16 cache for one dataset, independently of the demo server."""

import argparse
import json
from pathlib import Path

import cv2

from multiview_detector.models.checkpoint import load_checkpoint
from realtime_demo.datasets import CHECKPOINTS, scan_datasets
from realtime_demo.full_cache import FullFrameCache
from realtime_demo.workspace import user_cache, user_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=str.lower, choices=tuple(CHECKPOINTS), required=True)
    parser.add_argument('--fps', type=float, required=True, help='output FPS, up to the source FPS')
    parser.add_argument('--dataset-root', type=Path, help='defaults to the saved Dataset folder')
    parser.add_argument('--cache-dir', type=Path, help='defaults to the saved cache folder for this dataset')
    parser.add_argument('--config', type=Path, default=user_config(), help='settings used by the web demo')
    parser.add_argument('--source', choices=('auto', 'mp4', 'images'), default='auto')
    parser.add_argument('--input-size', type=int, nargs=2, metavar=('W', 'H'), help='override checkpoint image size')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    cv2.setNumThreads(2)
    try:
        config_path = args.config.expanduser()
        saved = json.loads(config_path.read_text(encoding='utf-8')) if config_path.is_file() else {}
        dataset_root = args.dataset_root or saved.get('dataset_root')
        if not dataset_root:
            raise ValueError('Set --dataset-root or save a Dataset folder in the web demo.')
        _, datasets, errors = scan_datasets(root, str(Path(dataset_root).expanduser().resolve()))
        if args.dataset not in datasets:
            raise ValueError(errors.get(args.dataset, f'{args.dataset} is not in this Dataset folder.'))
        dataset = datasets[args.dataset]
        source = dataset.direct if args.source == 'auto' else args.source
        if source not in dataset.sources:
            raise ValueError(f'{dataset.key} does not have the {source} source.')
        directory = args.cache_dir or saved.get('datasets', {}).get(dataset.key, {}).get('cache_dir') or user_cache()
        directory = Path(directory).expanduser().resolve()
        if (directory / 'manifest.json').is_file():
            directory = directory.parent
        size = args.input_size
        if not size:
            checkpoint = root / 'checkpoints' / CHECKPOINTS[dataset.key]
            if not checkpoint.is_file():
                checkpoint = root / 'checkpoints' / CHECKPOINTS['wavetrack']
            size = load_checkpoint(checkpoint)['img_size']
        if min(size) < 32:
            raise ValueError('Input width and height must each be at least 32 pixels.')
        cache = FullFrameCache(dataset, dataset.sources[source], directory, size, args.fps)
        print(f'{dataset.key} · {source} · {cache.duration:.1f} s · {cache.fps:g} FPS · '
              f'{cache.capacity * cache.frame_bytes / 1e9:.2f} GB of tensors plus JPEGs', flush=True)
        print(cache.directory, flush=True)
        cache.build(lambda done, total, speed: print(f'{done}/{total} frame sets ({speed:.2f}/s)', flush=True))
        print(f'Cache ready: {cache.directory}', flush=True)
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        parser.exit(1, f'{exc}\n')


if __name__ == '__main__':
    main()

"""Create an offline person-detection cache for EarlyBird BEV supervision."""

import argparse
import json
import os
from pathlib import Path

from tqdm import tqdm
from ultralytics import YOLO

from datasets.multiviewx_dataset import MultiviewX
from datasets.wildtrack_dataset import Wildtrack


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('wildtrack', 'multiviewx'), required=True)
    parser.add_argument('--data-root', required=True,
                        help='Dataset root containing Image_subsets and calibrations')
    parser.add_argument('--output', required=True, help='Destination JSON cache file')
    parser.add_argument('--weights', default='yolo26s.pt',
                        help='Ultralytics weights; accepts a local .pt path')
    parser.add_argument('--confidence', type=float, default=0.2,
                        help='Minimum confidence saved in the cache')
    parser.add_argument('--imgsz', type=int, default=1280)
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Inference batch size; lower this if GPU memory is limited')
    parser.add_argument('--device', default=None, help='Inference device, e.g. 0 or cpu')
    parser.add_argument('--split', choices=('train', 'all'), default='train',
                        help='train covers the first 90%%, matching PedestrianDataset')
    parser.add_argument('--overwrite', action='store_true', help='Replace an existing cache')
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0.0 <= args.confidence <= 1.0:
        raise ValueError('--confidence must be between 0 and 1')
    if args.batch_size < 1:
        raise ValueError('--batch-size must be at least 1')

    dataset_cls = Wildtrack if args.dataset == 'wildtrack' else MultiviewX
    base = dataset_cls(os.path.expanduser(args.data_root))
    frame_end = int(base.num_frame * 0.9) if args.split == 'train' else base.num_frame
    frame_range = range(frame_end)
    image_paths = base.get_image_fpaths(frame_range)
    image_items = [
        (frame, camera, image_paths[camera][frame])
        for frame in frame_range
        for camera in range(base.num_cam)
        if frame in image_paths[camera]
    ]
    if not image_items:
        raise FileNotFoundError(f'No images found under {args.data_root!r} for split {args.split!r}')

    output_path = Path(os.path.expanduser(args.output)).resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f'{output_path} already exists; pass --overwrite to replace it')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    detections = {}
    batches = (image_items[i:i + args.batch_size]
               for i in range(0, len(image_items), args.batch_size))
    for batch in tqdm(batches, total=(len(image_items) + args.batch_size - 1) // args.batch_size,
                      desc=f'Detecting {args.dataset} people'):
        results = model.predict(
            source=[item[2] for item in batch],
            classes=[0],
            conf=args.confidence,
            imgsz=args.imgsz,
            batch=args.batch_size,
            device=args.device,
            verbose=False,
        )
        if len(results) != len(batch):
            raise RuntimeError(f'Detector returned {len(results)} results for {len(batch)} images')
        for (frame, camera, _), result in zip(batch, results):
            rows = []
            if result.boxes is not None and len(result.boxes):
                coords = result.boxes.xyxy.cpu().tolist()
                scores = result.boxes.conf.cpu().tolist()
                rows = [list(map(float, box)) + [float(score)]
                        for box, score in zip(coords, scores)]
            detections.setdefault(str(frame), {})[str(camera)] = rows

    payload = {
        'version': 1,
        'dataset': args.dataset,
        'split': args.split,
        'weights': args.weights,
        'confidence': args.confidence,
        'detections': detections,
    }
    with output_path.open('w', encoding='utf-8') as cache_file:
        json.dump(payload, cache_file, separators=(',', ':'))
    print(f'Saved detections for {len(image_items)} camera frames to {output_path}')


if __name__ == '__main__':
    main()

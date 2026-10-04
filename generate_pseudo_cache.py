"""Generate an offline person-detection cache for SHOT pseudo supervision."""
import argparse
import json
import os

from tqdm import tqdm
from ultralytics import YOLO

from multiview_detector.datasets.MultiviewX import MultiviewX
from multiview_detector.datasets.Wildtrack import Wildtrack


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['wildtrack', 'multiviewx'], required=True)
    parser.add_argument('--data_root', required=True, help='Dataset root containing images and calibrations')
    parser.add_argument('--output', required=True, help='Output JSON detection cache')
    parser.add_argument('--weights', default='yolo26s.pt', help='Ultralytics detection weights')
    parser.add_argument('--confidence', type=float, default=0.2)
    parser.add_argument('--imgsz', type=int, default=1280)
    parser.add_argument('--device', default=None, help='Inference device, e.g. 0, cpu')
    args = parser.parse_args()

    dataset_cls = Wildtrack if args.dataset == 'wildtrack' else MultiviewX
    dataset = dataset_cls(os.path.expanduser(args.data_root))
    frame_range = range(dataset.num_frame)
    image_paths = dataset.get_image_fpaths(frame_range)
    model = YOLO(args.weights)
    detections = {}
    image_items = [(frame, camera, image_paths[camera][frame])
                   for frame in range(dataset.num_frame)
                   for camera in range(dataset.num_cam)
                   if frame in image_paths[camera]]

    for frame, camera, image_path in tqdm(image_items, desc='Detecting people'):
        prediction = model.predict(
            source=image_path, classes=[0], conf=args.confidence,
            imgsz=args.imgsz, device=args.device, verbose=False)
        boxes = prediction[0].boxes
        rows = []
        if boxes is not None and len(boxes):
            coords = boxes.xyxy.cpu().tolist()
            scores = boxes.conf.cpu().tolist()
            rows = [list(map(float, xyxy)) + [float(score)]
                    for xyxy, score in zip(coords, scores)]
        detections.setdefault(str(frame), {})[str(camera)] = rows

    output = os.path.abspath(os.path.expanduser(args.output))
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, 'w', encoding='utf-8') as cache_file:
        json.dump({'version': 1, 'dataset': args.dataset, 'detections': detections}, cache_file)
    print(f'Wrote {len(image_items)} camera frames to {output}')


if __name__ == '__main__':
    main()

"""Adapter for the legacy OpenCV UI, reading synchronized undistorted images only."""
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
from dataset import Dataset

DEFAULT_SOURCE = Path(__file__).resolve().parents[1]

class ROI(SimpleNamespace):
    def to_cell(self, x, y):
        return int(np.floor((y-self.y0)/self.cell + 1e-9)), int(np.floor((x-self.x0)/self.cell + 1e-9))

class ImageReader:
    def __init__(self, rec):
        self.rec = rec
        self.cached_frame, self.images = None, None

    def get(self, frame):
        if frame != self.cached_frame:
            images = []
            for cam, paths in zip(self.rec.cams, self.rec.dataset.paths):
                path = paths[frame]
                image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f'Cannot read image: {path}')
                if image.shape[1::-1] != cam.image_size:
                    raise ValueError(f'Image size differs from calibration: {path}')
                images.append(image)
            self.cached_frame, self.images = frame, images
        return self.images

    def release(self):
        self.cached_frame, self.images = None, None

class Recording:
    def __init__(self, source=DEFAULT_SOURCE, calib=None, kind='auto'):
        source = Path(source)
        if source.name.lower() == 'image_subsets': source = source.parent
        self.dataset = Dataset(source, calib, kind=kind)
        self.cams = self.dataset.cams
        self.frame_ids = self.dataset.frames
        self.frame_indices = {frame: i for i, frame in enumerate(self.frame_ids)}
        self.labels = self.dataset.labels
        r = self.dataset.roi
        self.roi = ROI(x0=r['x0'], y0=r['y0'], x1=r['x1'], y1=r['y1'],
                       cell=self.dataset.cell, shape=(self.dataset.rows,self.dataset.cols))
        world = self.dataset.world
        self.pitch = (world.get('pitch_x_mm', 500)/1000, world.get('pitch_y_mm', 500)/1000)

    def __len__(self): return len(self.frame_ids)
    def reader(self): return ImageReader(self)
    def __str__(self):
        return f'{self.dataset.root}: {len(self.cams)} cameras, {len(self)} shared image frames; {self.dataset.skipped} unmatched frames skipped'

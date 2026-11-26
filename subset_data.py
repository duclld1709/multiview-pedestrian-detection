"""Wildtrack-style, already rectified image subsets and raw-video rectification."""
from pathlib import Path

import cv2
import numpy as np

from calib import load_calibration


class ImageSubsetRecording:
    def __init__(self, source):
        self.session = Path(source)
        self.name = self.session.name
        self.calib_path = self.session / 'calibrations' / 'calibration.json'
        self.cams, self.roi, self.pitch = load_calibration(self.calib_path)
        self.labels = self.session / 'annotations_positions'
        self.image_dirs = [self.session / 'Image_subsets' / f'C{int(c.name[3:])}' for c in self.cams]
        if any(np.any(c.dist) for c in self.cams):
            raise ValueError('Image subsets require rectified calibration (zero distortion)')

    def image_path(self, k, frame):
        return self.image_dirs[k] / f'{frame:08d}.png'

    def __repr__(self):
        return f'image subset {self.name}: {len(self.cams)} cameras, labels {self.labels}'


def rectification_maps(cams, original_dir, output_size=None):
    """Map raw video pixels into the saved alpha=0 subset intrinsics; never remap PNGs."""
    maps = []
    for cam in cams:
        path = Path(original_dir) / f'intr_{cam.name}.xml'
        fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            raise ValueError(f'Missing original intrinsic: {path}')
        K = fs.getNode('camera_matrix').mat()
        dist = fs.getNode('distortion_coefficients').mat()
        size = tuple(int(fs.getNode(k).real()) for k in ('image_width', 'image_height'))
        fs.release()
        if K is None or dist is None or size != cam.image_size:
            raise ValueError(f'{path}: invalid intrinsic or image size {size} != {cam.image_size}')
        optimal = cv2.getOptimalNewCameraMatrix(K, dist, size, 0, size)[0]
        if np.any(cam.dist) or not np.allclose(optimal, cam.K, rtol=1e-6, atol=1e-4):
            raise ValueError(f'{cam.name}: saved calibration must match original intrinsic with alpha=0')
        target_size = size if output_size is None else tuple(output_size)
        target_K = cam.K
        if target_size != size:
            sx, sy = target_size[0] / size[0], target_size[1] / size[1]
            # Match the pixel-center convention of resizing a rectified image.
            scale = np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]])
            target_K = scale @ cam.K
        maps.append(cv2.initUndistortRectifyMap(K, dist, None, target_K, target_size, cv2.CV_32FC1))
    return maps


def tail_range(rec, seconds):
    """Reference-frame interval inside the last seconds of every camera (sync offsets included)."""
    valid = np.ones(len(rec), dtype=bool)
    for k, (cam, ts) in enumerate(zip(rec.cams, rec.ts)):
        end = ts[-1] + (np.median(np.diff(ts)) if len(ts) > 1 else 1000 / rec.fps)
        target = rec.ts[0] + (rec.offsets.get(cam.name, 0.0) if k else 0.0)
        valid &= (target >= max(0, end - seconds * 1000) - 1e-3) & (target < end)
    frames = np.flatnonzero(valid)
    if not len(frames):
        raise ValueError('Videos have no shared interval in the requested tail')
    return int(frames[0]), int(frames[-1]) + 1


def time_range(rec, start_seconds=0.0, duration_seconds=None):
    """Reference-frame interval [start, start+duration), measured from video start."""
    if not np.isfinite(start_seconds) or start_seconds < 0:
        raise ValueError('start-seconds must be finite and nonnegative')
    if duration_seconds is not None and (not np.isfinite(duration_seconds) or duration_seconds <= 0):
        raise ValueError('duration-seconds must be finite and positive')
    ts = rec.ts[0]
    if not len(ts):
        raise ValueError('Recording has no frames')
    start_ms = ts[0] + start_seconds*1000
    start = int(np.searchsorted(ts, start_ms-1e-3))
    end = len(ts) if duration_seconds is None else int(np.searchsorted(
        ts, start_ms+duration_seconds*1000-1e-3))
    if start >= end:
        raise ValueError('Requested time interval contains no video frames')
    return start, end

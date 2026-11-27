"""Discover the three supported datasets and describe their synchronized media."""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from xml.etree import ElementTree

import cv2
import numpy as np

from calib import Camera, Roi, load_calibration


CHECKPOINTS = {
    'wavetrack': 'wavetrack_lite.pt',
    'wildtrack': 'wildtrack_lite.pt',
    'multiviewx': 'multiviewx_lite.pt',
}
LABELS = {'wavetrack': 'WaveTrack', 'wildtrack': 'Wildtrack', 'multiviewx': 'MultiviewX'}


def checkpoint_dataset(name):
    """Recognize native and lite checkpoint names without loading their tensors."""
    for key, canonical in CHECKPOINTS.items():
        if name in (canonical, f'{key}_lite.pt', f'bevnet_{key}_lite.pt'):
            return key
    return None


def default_checkpoint(root, key):
    candidates = [CHECKPOINTS[key], f'{key}_lite.pt', f'bevnet_{key}_lite.pt']
    if key != 'wavetrack':
        candidates.extend((CHECKPOINTS['wavetrack'], 'wavetrack_lite.pt', 'bevnet_wavetrack_lite.pt'))
    for name in candidates:
        if (Path(root) / 'checkpoints' / name).is_file():
            return name
    raise ValueError(f'No checkpoint for {LABELS[key]}. Add one of: {", ".join(candidates)} to checkpoints/.')


BENCHMARKS = {
    'wildtrack': (('CVLab1', 'CVLab2', 'CVLab3', 'CVLab4', 'IDIAP1', 'IDIAP2', 'IDIAP3'),
                  'intrinsic_zero', .01, (-3, -9, 9, 27), 10),
    'multiviewx': (tuple(f'Camera{k}' for k in range(1, 7)), 'intrinsic', 1, (0, 0, 25, 16), 2),
}


def fingerprint(paths):
    digest = hashlib.sha256()
    for path in paths:
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


@dataclass
class Media:
    session: Path
    kind: str
    paths: tuple
    fps: float
    count: int
    original_intrinsics: Path | None = None

    def identity(self):
        """Portable media identity. Frame collections are ordered by synchronized frame ID."""
        digest = hashlib.sha256()
        for camera in self.paths:
            for path in ((camera,) if self.kind == 'mp4' else camera):
                stat = path.stat()
                relative = path.relative_to(self.session).as_posix()
                digest.update(f'{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\n'.encode())
        return digest.hexdigest()


@dataclass
class Dataset:
    key: str
    path: Path
    cams: list
    roi: Roi
    pitch: tuple
    site: dict | Path
    calibration: str
    sources: dict

    @property
    def direct(self):
        return next(iter(self.sources))


def video_properties(videos, cams):
    properties = []
    for path, cam in zip(videos, cams):
        cap = cv2.VideoCapture(str(path))
        try:
            if not cap.isOpened():
                raise ValueError(f'Cannot open {path}')
            rate = cap.get(cv2.CAP_PROP_FPS)
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            size = tuple(int(cap.get(prop)) for prop in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
            if size != cam.image_size or not math.isfinite(rate) or rate <= 0 or count <= 0:
                raise ValueError(f'{path.name}: invalid FPS, frame count or calibrated size {cam.image_size}.')
            properties.append((rate, count))
        finally:
            cap.release()
    fps = properties[0][0]
    if any(abs(rate - fps) > .01 for rate, _ in properties):
        raise ValueError('Camera MP4s must have matching constant FPS.')
    return fps, min(count for _, count in properties)


def find_videos(path, cams, names):
    for folder in (path / 'videos_rectified', path, path / 'videos'):
        if not folder.is_dir():
            continue
        files = {p.stem.casefold(): p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == '.mp4'}
        videos = tuple(files.get(c.name.casefold()) or files.get(n.casefold()) or files.get(f'c{i}')
                       for i, (c, n) in enumerate(zip(cams, names), 1))
        if not all(videos):
            continue
        sync = path / 'sync.json'
        if sync.is_file() and any(float(v) != 0 for offsets in json.loads(sync.read_text(encoding='utf-8')).values()
                                  for v in offsets.values()):
            raise ValueError('The demo requires synchronized MP4s with zero sync offsets.')
        original = path / 'calibrations/intrinsic_original'
        if folder.name == 'videos_rectified' or not all((original / f'intr_{c.name}.xml').is_file() for c in cams):
            original = None
        fps, count = video_properties(videos, cams)
        return Media(path, 'mp4', videos, fps, count, original)
    return None


def find_images(path, names, native_fps):
    folder = path / 'Image_subsets'
    if not folder.is_dir():
        return None
    available = {p.name.casefold(): p for p in folder.iterdir() if p.is_dir()}
    indexed = []
    for i, name in enumerate(names, 1):
        directory = available.get(f'c{i}') or available.get(name.casefold())
        if directory is None:
            return None
        frames = {int(p.stem): p for p in sorted(directory.iterdir()) if p.is_file() and p.stem.isdecimal()
                  and p.suffix.lower() in ('.png', '.jpg', '.jpeg')}
        indexed.append(frames)
    common = sorted(set.intersection(*(set(frames) for frames in indexed)))
    if not common:
        return None
    steps = np.diff(common)
    if len(steps) and np.any(steps != steps[0]):
        raise ValueError('Synchronized images must have a constant frame-number interval.')
    stride = int(steps[0]) if len(steps) else 1
    # MultiviewX releases index sequential captures at 2 FPS, or by their corresponding 10 FPS frame ID.
    if native_fps == 2 and stride >= 5:
        native_fps = 10
    return Media(path, 'images', tuple(tuple(frames[i] for i in common) for frames in indexed),
                 native_fps / stride, len(common))


def read_nodes(path, *names):
    if not path.is_file():
        raise ValueError(f'Missing calibration file {path}')
    try:
        storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
        try:
            return [storage.getNode(name).mat() for name in names]
        finally:
            storage.release()
    except (cv2.error, SystemError) as exc:
        raise ValueError(f'Invalid calibration file {path}') from exc


def benchmark_calibration(key, path):
    names, intrinsics, unit, bounds, image_fps = BENCHMARKS[key]
    calibration = path / 'calibrations'
    files, cams = [], []
    x0, y0, x1, y1 = bounds
    ground_center = np.array([(x0 + x1) / 2, (y0 + y1) / 2, 0.])
    for i, name in enumerate(names, 1):
        intr = calibration / intrinsics / f'intr_{name}.xml'
        extr = calibration / 'extrinsic' / f'extr_{name}.xml'
        K, dist = read_nodes(intr, 'camera_matrix', 'distortion_coefficients')
        if key == 'wildtrack':
            tree = ElementTree.parse(extr)
            rvec, tvec = [np.fromstring(tree.findtext(node + '/data') or tree.findtext(node) or '',
                                        dtype=float, sep=' ') for node in ('rvec', 'tvec')]
        else:
            rvec, tvec = read_nodes(extr, 'rvec', 'tvec')
        if K is None or K.shape != (3, 3) or rvec is None or tvec is None or rvec.size != 3 or tvec.size != 3:
            raise ValueError(f'Invalid calibration matrices for {name}')
        depth_sign = 1
        if key == 'multiviewx':
            # The released XMLs put the observed ground in the negative camera-Z
            # half-space. Preserve their pixel projections and calibrate visibility
            # on the side containing the playfield, including corrected XMLs.
            rotation, _ = cv2.Rodrigues(rvec)
            if rotation[2] @ ground_center + tvec.ravel()[2] < 0:
                depth_sign = -1
        distortion = dist if key == 'multiviewx' and dist is not None else np.zeros(5)
        cams.append(Camera(dict(name=f'Cam{i}', image_size=(1920, 1080), camera_matrix=K,
                                dist_coeffs=distortion, rvec=rvec, tvec_m=tvec * unit,
                                depth_sign=depth_sign)))
        files.extend((intr, extr))
    return cams, Roi(*bounds, .025), (1., 1.), names, image_fps, files


def ground_map(cams, roi):
    centers = np.array([cam.center[:2] for cam in cams])
    x0, y0 = np.minimum(centers.min(0), (roi.x0, roi.y0)) - 1.5
    x1, y1 = np.maximum(centers.max(0), (roi.x1, roi.y1)) + 1.5
    # Center the whole site in a square so the long Wildtrack strip fits the same canvas.
    span = max(x1 - x0, y1 - y0)
    x, y = (x0 + x1 - span) / 2, (y0 + y1 - span) / 2
    return dict(bounds=[x, x + span, y, y + span], objects=[], floor_regions=[
        dict(points=[[roi.x0, roi.y0], [roi.x1, roi.y0], [roi.x1, roi.y1], [roi.x0, roi.y1]])
    ])


def open_dataset(root, key, path):
    calib = path / 'calibrations/calibration.json'
    if calib.is_file():
        cams, roi, pitch = load_calibration(calib)
        cams.sort(key=lambda cam: int(cam.name[3:]))
        if key in BENCHMARKS:
            names, _, _, _, image_fps = BENCHMARKS[key]
        else:
            names, image_fps = tuple(c.name for c in cams), 30
        files = [p for p in (calib, calib.with_name('grid.json')) if p.is_file()]
    elif key in BENCHMARKS:
        cams, roi, pitch, names, image_fps, files = benchmark_calibration(key, path)
    else:
        raise ValueError(f'Missing {calib}')
    expected = len(BENCHMARKS[key][0]) if key in BENCHMARKS else 5
    if [cam.name for cam in cams] != [f'Cam{i}' for i in range(1, expected + 1)]:
        raise ValueError(f'Expected {expected} calibrated cameras named Cam1 through Cam{expected}.')
    if key == 'wavetrack':
        site = root / 'wavetrack-site-map.json'
    else:
        site = ground_map(cams, roi)

    sources, errors = {}, []
    for read in (lambda: find_videos(path, cams, names), lambda: find_images(path, names, image_fps)):
        try:
            media = read()
            if media is not None:
                sources[media.kind] = media
        except (ValueError, OSError) as exc:
            errors.append(str(exc))
    if not sources:
        raise ValueError(' '.join(errors) or 'No complete synchronized MP4 set or Image_subsets was found.')
    return Dataset(key, path, cams, roi, pitch, site, fingerprint(files), sources)


def scan_datasets(root, directory):
    if not isinstance(directory, str) or not directory.strip():
        raise ValueError('Enter the path to the Dataset folder.')
    directory = Path(directory.strip()).expanduser()
    if not directory.is_absolute() or not directory.is_dir():
        raise ValueError('Enter an existing absolute Dataset folder path on this machine.')
    directory = directory.resolve()
    children = {p.name.casefold(): p for p in directory.iterdir() if p.is_dir()}
    found, errors = {}, {}
    for key in LABELS:
        path = children.get(key)
        if path is None:
            continue
        # Some archives create a second folder with the same name.
        candidates = [path, *(p for p in path.iterdir() if p.is_dir() and p.name.casefold() == key)]
        for candidate in candidates:
            try:
                found[key] = open_dataset(root, key, candidate)
            except (ValueError, OSError, KeyError, TypeError, ElementTree.ParseError, cv2.error,
                    np.linalg.LinAlgError) as exc:
                errors[key] = str(exc)
            else:
                errors.pop(key, None)
                break
    return directory, found, errors

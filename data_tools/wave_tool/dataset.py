"""Image-subset IO and calibrated geometry. World coordinates are metres."""
import json
import math
import os
import re
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

POSITION_CONVENTION = 'grid-node-v1'
WILDTRACK_CAMERAS = ('CVLab1', 'CVLab2', 'CVLab3', 'CVLab4', 'IDIAP1', 'IDIAP2', 'IDIAP3')


def dataset_kind(folder, kind='auto'):
    if kind != 'auto':
        return kind
    return 'wildtrack' if (Path(folder) / 'intrinsic_zero' / 'intr_CVLab1.xml').is_file() else 'wavetrack'


def load_wildtrack_calibration(folder):
    """Official WILDTRACK uses cm and plain-text extrinsic vectors."""
    cameras = []
    for number, name in enumerate(WILDTRACK_CAMERAS, 1):
        intr = folder / 'intrinsic_zero' / f'intr_{name}.xml'
        storage = cv2.FileStorage(str(intr), cv2.FILE_STORAGE_READ)
        try:
            K = storage.getNode('camera_matrix').mat()
            if K is None or K.shape != (3, 3) or not np.isfinite(K).all():
                raise ValueError(f'{intr}: invalid camera_matrix')
        finally:
            storage.release()
        extr = folder / 'extrinsic' / f'extr_{name}.xml'
        xml = ET.parse(extr).getroot()
        vectors = {}
        for key in ('rvec', 'tvec'):
            node = xml.find(key)
            data = node.find('data') if node is not None else None
            raw = data.text if data is not None else node.text if node is not None else ''
            vector = np.asarray([float(v) for v in (raw or '').split()])
            if vector.size != 3 or not np.isfinite(vector).all():
                raise ValueError(f'{extr}: invalid {key}')
            vectors[key] = vector
        cameras.append(Camera({'name': f'Cam{number}', 'image_size': [1920, 1080],
                               'camera_matrix': K, 'dist_coeffs': np.zeros(5),
                               'rvec': vectors['rvec'], 'tvec_m': vectors['tvec'] * .01}))
    return cameras, {'roi': {'x0': -3., 'y0': -9., 'x1': 9., 'y1': 27.,
                              'cell_cm': 2.5, 'grid_shape_rows_cols': [1440, 480]}}


def load_calibration(root, calib=None, kind='auto'):
    """Read camera XMLs; grid.json supplies only grid geometry and XML units."""
    folder = Path(calib) if calib else Path(root) / 'calibrations'
    if dataset_kind(folder, kind) == 'wildtrack':
        return load_wildtrack_calibration(folder)
    if not folder.is_dir():
        raise ValueError('--calib must point to a folder containing extrinsic/, intrinsic_zero/ and grid.json')
    grid = json.loads((folder / 'grid.json').read_text(encoding='utf-8'))
    if grid.get('position_convention') != POSITION_CONVENTION:
        raise ValueError('grid.json must declare position_convention: grid-node-v1')
    unit = grid.get('extrinsic_tvec_unit')
    if unit not in ('cm', 'm'):
        raise ValueError('grid.json must declare extrinsic_tvec_unit: cm or m')
    cameras = []
    for intr in sorted((folder / 'intrinsic_zero').glob('intr_Cam*.xml')):
        name = intr.stem.removeprefix('intr_')
        if not re.fullmatch(r'Cam\d+', name):
            continue
        extr = folder / 'extrinsic' / f'extr_{name}.xml'
        if not extr.is_file():
            raise ValueError(f'Missing extrinsic: {extr}')
        def read_xml(path, matrices, scalars=()):
            storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
            try:
                if not storage.isOpened():
                    raise ValueError(f'Cannot open calibration: {path}')
                result = {}
                for key in matrices:
                    value = storage.getNode(key).mat()
                    if value is None or not np.isfinite(value).all():
                        raise ValueError(f'{path}: missing or invalid {key}')
                    result[key] = value
                for key in scalars:
                    value = storage.getNode(key).real()
                    if not np.isfinite(value) or value <= 0:
                        raise ValueError(f'{path}: missing or invalid {key}')
                    result[key] = int(value)
                return result
            finally:
                storage.release()
        i = read_xml(intr, ('camera_matrix', 'distortion_coefficients'), ('image_width', 'image_height'))
        e = read_xml(extr, ('rvec', 'tvec'))
        cameras.append(Camera({'name': name, 'image_size': [i['image_width'], i['image_height']],
                               'camera_matrix': i['camera_matrix'],
                               'dist_coeffs': i['distortion_coefficients'].ravel(),
                               'rvec': e['rvec'].ravel(),
                               'tvec_m': e['tvec'].ravel() * (0.01 if unit == 'cm' else 1.0)}))
    if not cameras:
        raise ValueError(f'No intr_Cam*.xml files in {folder / "intrinsic_zero"}')
    return sorted(cameras, key=lambda c: c.number), grid


class Camera:
    def __init__(self, data):
        self.name = data['name']
        self.number = int(re.search(r'\d+$', self.name).group())
        self.image_size = tuple(data['image_size'])
        self.K = np.asarray(data['camera_matrix'], dtype=float)
        if np.any(np.abs(data['dist_coeffs']) > 1e-10):
            raise ValueError(f'{self.name}: Image_subsets requires zero-distortion calibration')
        self.R = cv2.Rodrigues(np.asarray(data['rvec'], dtype=float))[0]
        self.t = np.asarray(data['tvec_m'], dtype=float)
        self.center = -self.R.T @ self.t
        self.Hinv = np.linalg.inv(self.K @ np.column_stack((self.R[:, :2], self.t)))

    def project(self, xyz):
        xyz = np.asarray(xyz) @ self.R.T + self.t
        uvw = xyz @ self.K.T
        return uvw[:, :2] / np.where(abs(uvw[:, 2:]) < 1e-9, np.nan, uvw[:, 2:]), xyz[:, 2] > 1e-6

    def ground(self, u, v):
        p = self.Hinv @ [u, v, 1.0]
        if abs(p[2]) < 1e-9:
            return None
        xy = p[:2] / p[2]
        return xy if self.project([[*xy, 0]])[1][0] else None

    def image_to_ground(self, pixels):
        return np.array([self.ground(*p) if self.ground(*p) is not None else [np.nan, np.nan] for p in pixels])

    def ground_to_image(self, xy):
        xy = np.atleast_2d(xy)
        uv, front = self.project(np.column_stack((xy, np.zeros(len(xy)))))
        uv[~front] = np.nan
        return uv

    def box(self, xy, radius, height):
        theta = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        ring = np.asarray(xy) + radius * np.column_stack((np.cos(theta), np.sin(theta)))
        uv, front = self.project(np.vstack([np.column_stack((ring, np.full(64, z))) for z in (0, height)]))
        if not front.all() or not np.isfinite(uv).all():
            return [-1] * 4
        low, high = uv.min(0), uv.max(0)
        w, h = self.image_size
        if high[0] < 0 or high[1] < 0 or low[0] >= w or low[1] >= h:
            return [-1] * 4
        # Wildtrack boxes may extend beyond the image boundaries.
        return [math.floor(low[0]), math.floor(low[1]), math.ceil(high[0]), math.ceil(high[1])]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class Dataset:
    def __init__(self, root, calib=None, labels=None, kind='auto'):
        self.root = Path(root).resolve()
        self.kind = dataset_kind(calib or self.root / 'calibrations', kind)
        self.read_only = self.kind == 'wildtrack'
        self.cams, data = load_calibration(self.root, calib, self.kind)
        self.world = data.get('world', {})
        self.roi = data['roi']
        self.cell = self.roi['cell_cm'] / 100
        self.rows, self.cols = self.roi['grid_shape_rows_cols']
        self.labels = Path(labels) if labels else self.root / ('annotations_positions' if self.read_only else 'annotations')
        self.paths = []
        for cam in self.cams:
            folder = self.root / 'Image_subsets' / f'C{cam.number}'
            images = {}
            for p in folder.iterdir():
                if p.suffix.lower() in ('.png', '.jpg', '.jpeg') and p.stem.isdigit():
                    frame = int(p.stem)
                    if frame in images:
                        raise ValueError(f'Duplicate frame {frame} in {folder}')
                    images[frame] = p
            self.paths.append(images)
        common = set.intersection(*(set(p) for p in self.paths))
        self.frames = sorted(common)
        self.skipped = len(set.union(*(set(p) for p in self.paths)) - common)
        if not self.frames:
            raise ValueError('No matching image frames across all calibrated cameras')

    def path(self, frame):
        return self.labels / f'{frame:08d}.json'

    def position(self, xy):
        if xy is None or not np.isfinite(xy).all():
            raise ValueError('Click a visible point on the ground')
        x, y = xy
        if not (self.roi['x0'] <= x < self.roi['x1'] and self.roi['y0'] <= y < self.roi['y1']):
            raise ValueError('Position outside the calibrated ROI')
        col = int(np.floor((x - self.roi['x0']) / self.cell + 1e-9))
        row = int(np.floor((y - self.roi['y0']) / self.cell + 1e-9))
        return row * self.cols + col

    def xy(self, position):
        if not 0 <= position < self.rows * self.cols:
            raise ValueError(f'Invalid positionID: {position}')
        row, col = divmod(position, self.cols)
        return np.array([self.roi['x0'] + col * self.cell,
                         self.roi['y0'] + row * self.cell])


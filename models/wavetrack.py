"""WaveTrack input and evaluation shared by MVDet and MVDeTr."""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from data_tools.wave_tool.dataset import Dataset


class WaveTrack:
    __name__ = 'WaveTrack'
    indexing = 'xy'
    worldcoord_unit = 1.0

    def __init__(self, root, grid_multiple=8):
        source = Dataset(root, labels=Path(root) / 'annotations_positions', kind='wavetrack')
        self.root = str(source.root)
        self.num_cam = len(source.cams)
        self.original_grid_shape = (source.rows, source.cols)
        self.worldgrid_shape = [
            ((size + grid_multiple - 1) // grid_multiple) * grid_multiple
            for size in self.original_grid_shape
        ]
        self.origin = np.array([source.roi['x0'], source.roi['y0']])
        self.cell_m = source.cell
        if not np.isclose(self.cell_m, 0.025):
            raise ValueError('WaveTrack detection expects 2.5 cm grid cells')
        if [cam.number for cam in source.cams] != list(range(1, self.num_cam + 1)):
            raise ValueError('WaveTrack camera IDs must run consecutively from Cam1')
        if len({cam.image_size for cam in source.cams}) != 1:
            raise ValueError('WaveTrack cameras must use the same image size')
        self.img_shape = list(reversed(source.cams[0].image_size))
        self.intrinsic_matrices = [cam.K for cam in source.cams]
        self.extrinsic_matrices = [np.column_stack((cam.R, cam.t)) for cam in source.cams]
        self.worldcoord_from_worldgrid_mat = np.array([
            [self.cell_m, 0, self.origin[0]],
            [0, self.cell_m, self.origin[1]],
            [0, 0, 1],
        ])
        swap_xy = np.eye(3)[[1, 0, 2]]
        self.worldgrid2worldcoord_mat = self.worldcoord_from_worldgrid_mat @ swap_xy
        self.world_indexing_from_xy_mat = self.img_xy_from_xy_mat = np.eye(3)
        self.world_indexing_from_ij_mat = self.img_xy_from_ij_mat = swap_xy
        self.roi_grid_max = (np.array([source.roi['x1'], source.roi['y1']]) - self.origin) / self.cell_m

        self.annotation_paths = {}
        for path in sorted(source.labels.glob('*.json')):
            if path.stem.isascii() and path.stem.isdigit():
                frame = int(path.stem)
                if frame in self.annotation_paths:
                    raise ValueError(f'Duplicate WaveTrack annotation for frame {frame}')
                self.annotation_paths[frame] = path
        self.labeled_frames = sorted(self.annotation_paths)
        self.num_frame = len(self.labeled_frames)
        if self.num_frame < 2:
            raise ValueError(f'WaveTrack needs at least two labeled frames in {source.labels}')
        missing = set(self.labeled_frames).difference(source.frames)
        if missing:
            raise ValueError(f'WaveTrack frame {min(missing)} is missing an image in at least one camera')
        self.image_paths = source.paths

    def get_image_fpaths(self, frame_ids):
        return {cam: {frame: paths[frame] for frame in frame_ids}
                for cam, paths in enumerate(self.image_paths)}

    def get_worldgrid_from_pos(self, pos):
        rows, cols = self.original_grid_shape
        if np.any(np.asarray(pos) < 0) or np.any(np.asarray(pos) >= rows * cols):
            raise ValueError(f'WaveTrack positionID outside the original grid: {pos}')
        row, col = np.divmod(pos, cols)
        return np.array([col, row])

    def get_worldgrid_from_worldcoord(self, coords):
        origin = self.origin.reshape((2,) + (1,) * (np.ndim(coords) - 1))
        return (coords - origin) / self.cell_m

    def contains_grid_points(self, positions):
        return ((positions[:, 0] >= 0) & (positions[:, 0] < self.roi_grid_max[0]) &
                (positions[:, 1] >= 0) & (positions[:, 1] < self.roi_grid_max[1]))

    def read_annotations(self, frame):
        path = self.annotation_paths[frame]
        people = json.loads(path.read_text(encoding='utf-8-sig'))
        height, width = self.img_shape
        bounds = ('xmin', 'ymin', 'xmax', 'ymax')
        for person in people:
            if not isinstance(person['positionID'], int):
                raise ValueError(f'{path}: expected an integer positionID')
            self.get_worldgrid_from_pos(person['positionID'])
            views = sorted(person['views'], key=lambda view: view['viewNum'])
            if [view['viewNum'] for view in views] != list(range(self.num_cam)):
                raise ValueError(f'{path}: expected one annotation view per camera')
            for view in views:
                x0, y0, x1, y1 = (view[key] for key in bounds)
                clipped = (max(0, x0), max(0, y0), min(width - 1, x1), min(height - 1, y1))
                if clipped[0] > clipped[2] or clipped[1] > clipped[3]:
                    clipped = (-1, -1, -1, -1)
                view.update(zip(bounds, clipped))
            person['views'] = views
        return people

    def evaluate(self, predictions, frame_ids):
        """Return CLEAR detection metrics in percent over every supplied test frame."""
        predictions = predictions[self.contains_grid_points(predictions[:, 1:3])]
        threshold = 0.5 / self.cell_m
        total_gt = total_det = matches = 0
        precision_sum = 0.0
        for frame in frame_ids:
            gt = np.array([self.get_worldgrid_from_pos(person['positionID'])
                           for person in self.read_annotations(frame)]).reshape(-1, 2)
            det = predictions[predictions[:, 0] == frame, 1:3]
            total_gt += len(gt)
            total_det += len(det)
            if len(gt) and len(det):
                distances = np.linalg.norm(gt[:, None] - det[None, :], axis=-1)
                valid = distances < threshold
                rows, cols = linear_sum_assignment(np.where(valid, distances, 1e6))
                assigned = distances[rows, cols]
                assigned = assigned[assigned < threshold]
                matches += len(assigned)
                precision_sum += (1 - assigned / threshold).sum()

        recall = matches / max(1, total_gt)
        precision = matches / max(1, total_det)
        errors = total_gt + total_det - 2 * matches
        moda = max(0, 1 - errors / total_gt) if total_gt else 0
        modp = precision_sum / max(1, matches)
        return tuple(100 * value for value in (recall, precision, moda, modp))

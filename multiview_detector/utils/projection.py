"""World-aligned BEV grid and calibrated image sampling."""
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class BevGrid:
    """Model BEV grid; larger than the labelling ROI so people just outside it are still covered."""
    x0: float = -1.8
    y0: float = -1.6
    x1: float = 1.8
    y1: float = 3.6
    cell: float = 0.05

    @classmethod
    def covering(cls, rois, cell=0.05, margin=0.5):
        """Build one world-aligned grid that covers one or more calibration ROIs plus a margin."""
        rois = list(rois)
        if not rois:
            raise ValueError("at least one ROI is required to build a BEV grid")
        # Snap outward to cell boundaries so every requested ROI point stays inside the output.
        x0 = np.floor((min(r.x0 for r in rois) - margin) / cell + 1e-9) * cell
        y0 = np.floor((min(r.y0 for r in rois) - margin) / cell + 1e-9) * cell
        x1 = np.ceil((max(r.x1 for r in rois) + margin) / cell - 1e-9) * cell
        y1 = np.ceil((max(r.y1 for r in rois) + margin) / cell - 1e-9) * cell
        return cls(float(x0), float(y0), float(x1), float(y1), float(cell))

    @property
    def shape(self):  # (rows=Y, cols=X)
        return int(round((self.y1 - self.y0) / self.cell)), int(round((self.x1 - self.x0) / self.cell))

    def centers(self):
        rows, cols = self.shape
        X = self.x0 + (np.arange(cols) + 0.5) * self.cell
        Y = self.y0 + (np.arange(rows) + 0.5) * self.cell
        return np.meshgrid(X, Y)  # each (rows, cols)

    def to_rc(self, XY):
        XY = np.asarray(XY, dtype=np.float64).reshape(-1, 2)
        return np.column_stack([(XY[:, 1] - self.y0) / self.cell - 0.5, (XY[:, 0] - self.x0) / self.cell - 0.5])

    def to_xy(self, rc):
        rc = np.asarray(rc, dtype=np.float64).reshape(-1, 2)
        return np.column_stack([self.x0 + (rc[:, 1] + 0.5) * self.cell, self.y0 + (rc[:, 0] + 0.5) * self.cell])


def projection_grid(cam, grid):
    """Normalised sampling coords (rows, cols, 2) in [-1, 1] (align_corners=True) + validity mask."""
    X, Y = grid.centers()
    P = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
    uv, front = cam.project(P)
    w, h = cam.image_size
    valid = front & (uv[:, 0] >= 0) & (uv[:, 0] <= w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= h - 1)
    g = np.column_stack([uv[:, 0] / (w - 1) * 2 - 1, uv[:, 1] / (h - 1) * 2 - 1])
    g[~valid] = -2.0  # outside -> zero padding
    rows, cols = grid.shape
    return g.reshape(rows, cols, 2).astype(np.float32), valid.reshape(rows, cols).astype(np.float32)


def camera_grids(cams, grid):
    """Stacked projection grids (N, rows, cols, 2) for a list of cameras."""
    return torch.from_numpy(np.stack([projection_grid(c, grid)[0] for c in cams]))

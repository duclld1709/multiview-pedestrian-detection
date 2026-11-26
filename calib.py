"""Camera calibration loading and image <-> ground-plane (Z=0) projection."""
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Roi:
    x0: float
    y0: float
    x1: float
    y1: float
    cell: float  # metres
    grid_shape: tuple = None
    node_positions: bool = False

    @property
    def shape(self):
        if self.grid_shape is not None:
            return self.grid_shape
        rows = int(round((self.y1 - self.y0) / self.cell))
        cols = int(round((self.x1 - self.x0) / self.cell))
        return rows, cols

    def contains(self, X, Y):
        return (X >= self.x0) & (X < self.x1) & (Y >= self.y0) & (Y < self.y1)

    def to_cell(self, X, Y):
        """World (m) -> grid cell (row, col), following calibration.json's convention."""
        quantize = np.rint if self.node_positions else np.floor
        col = quantize((np.asarray(X) - self.x0) / self.cell).astype(int)
        row = quantize((np.asarray(Y) - self.y0) / self.cell).astype(int)
        return row, col


class Camera:
    MIN_DEPTH = 0.05

    def __init__(self, cfg):
        self.name = cfg["name"]
        self.image_size = tuple(cfg["image_size"])  # (w, h)
        self.K = np.array(cfg["camera_matrix"], dtype=np.float64)
        self.dist = np.array(cfg["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
        self.rvec = np.array(cfg["rvec"], dtype=np.float64).reshape(3, 1)
        self.tvec = np.array(cfg["tvec_m"], dtype=np.float64).reshape(3, 1)
        self.depth_sign = cfg.get("depth_sign", 1)
        if self.depth_sign not in (-1, 1):
            raise ValueError("Camera depth_sign must be +1 or -1")
        R, _ = cv2.Rodrigues(self.rvec)
        self.R = R
        self.center = (-R.T @ self.tvec).ravel()
        # Homography world ground plane (X, Y, 1) -> undistorted pixel
        self.H = self.K @ np.column_stack([R[:, 0], R[:, 1], self.tvec.ravel()])
        self.H_inv = np.linalg.inv(self.H)
        # The distortion polynomial is only fitted inside the image: rays far outside it (e.g. points right next
        # to the camera) project to absurd pixels, so they are treated like points behind the camera.
        w, h = self.image_size
        corners = self.undistort([(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)])
        self.max_ray = 1.5 * np.hypot(*((corners - self.K[:2, 2]) / np.diag(self.K)[:2]).T).max()

    @property
    def forward(self):
        return self.depth_sign * self.R[2]

    def undistort(self, uv):
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 1, 2)
        crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, 1e-6)
        return cv2.undistortPoints(uv, self.K, self.dist, R=None, P=self.K, criteria=crit).reshape(-1, 2)

    def image_to_ground(self, uv):
        """Distorted pixel(s) (N,2) -> ground points (N,2) in metres; NaN if ray misses the floor."""
        und = self.undistort(uv)
        pts = np.column_stack([und, np.ones(len(und))]) @ self.H_inv.T
        w = pts[:, 2:3]
        XY = pts[:, :2] / w
        # Ray must hit the floor in front of the camera
        bad = (np.abs(w[:, 0]) < 1e-12)
        dirs = np.column_stack([XY, np.zeros(len(XY))]) - self.center
        bad |= (dirs @ self.forward) <= 0
        XY[bad] = np.nan
        return XY

    def visible(self, P):
        """Points (N,3) in front of the camera and within the field of view the distortion model is valid for."""
        Pc = P @ self.R.T + self.tvec.ravel()
        z = self.depth_sign * Pc[:, 2]
        return (z > self.MIN_DEPTH) & (np.hypot(Pc[:, 0], Pc[:, 1]) <= self.max_ray * np.maximum(z, 0))

    def project(self, P):
        """World points (N,3) in metres -> distorted pixels (N,2) and mask of points the camera can project."""
        P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
        uv, _ = cv2.projectPoints(P, self.rvec, self.tvec, self.K, self.dist)
        return uv.reshape(-1, 2), self.visible(P)

    def ground_to_image(self, XY, Z=0.0):
        """Ground points (N,2) in metres -> distorted pixels (N,2). Points the camera can't project -> NaN."""
        XY = np.asarray(XY, dtype=np.float64).reshape(-1, 2)
        uv, ok = self.project(np.column_stack([XY, np.full(len(XY), Z)]))
        uv[~ok] = np.nan
        return uv


def load_calibration(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    grid_path = Path(path).with_name('grid.json')
    if grid_path.is_file():
        with open(grid_path, encoding='utf-8') as f:
            grid_cfg = json.load(f)
        cfg.update({k: grid_cfg[k] for k in ('roi', 'world', 'position_convention') if k in grid_cfg})
    r = cfg["roi"]
    nodes = cfg.get('position_convention') == 'grid-node-v1'
    roi = Roi(r["x0"], r["y0"], r["x1"], r["y1"], r["cell_cm"] / 100.0,
              tuple(r['grid_shape_rows_cols']) if nodes else None, nodes)
    cams = [Camera(c) for c in cfg["cameras"]]
    pitch = (cfg["world"]["pitch_x_mm"] / 1000.0, cfg["world"]["pitch_y_mm"] / 1000.0)
    return cams, roi, pitch

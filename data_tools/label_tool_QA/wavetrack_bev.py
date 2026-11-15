"""WaveTrack's five-camera geometry in the MVDeTr scene conventions."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import kinds


CELL_M = 0.025
KEY = "wavetrack"


class WaveTrackKind(kinds.Kind):
    """Read the authoritative JSON K/R/t (metres), in C1..C5 image order."""

    def read_calibration(self, root: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
        from labeling import wavetrack_calibration

        manifest = wavetrack_calibration(Path(root))
        if manifest is None:
            raise ValueError(f"WaveTrack calibration.json not found under {root}")
        cameras = {item["name"]: item for item in manifest["cameras"]}
        if not {f"Cam{i}" for i in range(1, 6)} <= set(cameras) or any(
                name not in cameras and not name.startswith("VirtualCam") for name in self.names):
            raise ValueError("WaveTrack camera names do not match the BEV rig")
        intrinsics, extrinsics = [], []
        for name in self.names:
            # The requested virtual view receives black pixels. Reusing valid camera
            # geometry keeps projection matrices finite without claiming a real camera.
            item = cameras.get(name, cameras["Cam5"])
            K = np.asarray(item["camera_matrix"], dtype=float)
            R = np.asarray(item["R"], dtype=float)
            t = np.asarray(item["tvec_m"], dtype=float).reshape(3, 1)
            if K.shape != (3, 3) or R.shape != (3, 3) or not np.isfinite(K).all() or not np.isfinite(R).all() \
                    or not np.isfinite(t).all() or abs(np.linalg.det(R)-1) > .01:
                raise ValueError(f"Invalid BEV calibration for {name}")
            if list(map(int, item["image_size"])) != [1920, 1080]:
                raise ValueError(f"WaveTrack BEV expects 1920x1080 images: {name}")
            intrinsics.append(K)
            extrinsics.append(np.hstack([R, t]))
        return intrinsics, extrinsics


def register(manifest: dict | None = None, views: int = 5) -> WaveTrackKind:
    """Register 5-camera scene. Pad the ground grid to MVDeTr's stride of four."""
    roi = (manifest or {}).get("roi") or {"x0": 0, "x1": 7.82, "y0": 0, "y1": 6.02, "cell_cm": 2.5}
    cell = float(roi.get("cell_cm", 2.5)) / 100
    if abs(cell-CELL_M) > 1e-9:
        raise ValueError("WaveTrack BEV requires a 2.5 cm ground cell")
    width = math.ceil((float(roi["x1"])-float(roi["x0"])) / CELL_M)
    height = math.ceil((float(roi["y1"])-float(roi["y0"])) / CELL_M)
    if views not in (5, 6, 7):
        raise ValueError("WaveTrack padding supports 5, 6 or 7 views")
    # DeformTransWorldFeat downsamples the reduced world grid once more and
    # builds positional embeddings with integer division. Its two shapes only
    # agree when the full world grid is a multiple of eight.
    stride = 8 if views in (6, 7) else 4
    grid_x = math.ceil(width / stride) * stride
    grid_y = math.ceil(height / stride) * stride
    key = KEY if views == 5 else f"{KEY}_pad{views}"
    kind = WaveTrackKind(key, "WaveTrack", views, grid_x, grid_y,
                         (float(roi["x0"]), float(roi["y0"])), "m", "xy", "intrinsic_zero",
                         [f"Cam{i}" for i in range(1, 6)] + [f"VirtualCam{i}" for i in range(6, views+1)],
                         "opencv", False, 0,
                         aliases=["wavetrack_synced_2fps"], source="label_tool/wavetrack_bev.py")
    return kinds.register(kind, replace=True)


register()

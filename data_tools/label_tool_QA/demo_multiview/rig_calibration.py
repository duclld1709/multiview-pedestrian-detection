"""Calibration computed from the Unity camera rig a live stream sends with each frame.

Unity reports each camera (position, right, up, forward, vertical FOV or focal length in pixels) in its own
coordinates; the scene kind's ``live.unity_axes`` maps them to the dataset world frame. PersonX (MvxLiveStream):
MultiviewX world (X, Y, Z) = rig-local Unity (x, -z, y) in metres, as in tools/mvxsim_to_dataset.py. Warehouse
(MultiviewLiveStream): (X, Y, Z) = Unity (x, z, y). The OpenCV camera frame is (right, -up, forward); when the axis
change keeps Unity's handedness that basis is a reflection, so like the original MultiviewX files the stored
rotation is the proper matrix -R with t' = -t (negative depth, identical pixels). Files are written in the kind's
calibration layout and unit (Warehouse: centimetres, Wildtrack-style folders).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import kinds

IMAGE_W, IMAGE_H = 1920, 1080
UNITY_TO_WORLD = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
PERSON_RADIUS, PERSON_HEIGHT = 0.16, 1.8   # cylinder used for rectangles.pom in mvxsim_to_dataset.py
AREA_CENTRE = np.array([12.5, 8.0, 0.0])   # every camera of the rig looks at the 25 x 16 m area


def camera_matrices(camera: dict, axes=UNITY_TO_WORLD) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """K, R, t (metres) with positive depth in front of the camera (R may be improper, see module docstring)."""
    axes = np.asarray(axes, dtype=float)
    right, up, forward = (axes @ np.asarray(camera[key], dtype=float) for key in ("right", "up", "forward"))
    centre = axes @ np.asarray(camera["position"], dtype=float)
    R = np.stack([right, -up, forward])
    focal = float(camera.get("focal_px") or 0) or (IMAGE_H / 2) / np.tan(np.radians(float(camera["vfov_deg"])) / 2)
    K = np.array([[focal, 0, IMAGE_W / 2], [0, focal, IMAGE_H / 2], [0, 0, 1]])
    return K, R, -R @ centre


def unity_axes(kind: str):
    return ((kinds.get(kind).live or {}).get("unity_axes")) or UNITY_TO_WORLD


def rig_key(cameras: list[dict]) -> str:
    rounded = [{key: np.round(np.asarray(camera[key], dtype=float), 4).tolist()
                for key in ("position", "right", "up", "forward", "vfov_deg")} for camera in cameras]
    return hashlib.sha1(json.dumps(rounded).encode()).hexdigest()[:12]


def write_calibration(root: Path, cameras: list[dict], kind: str | None = None) -> Path:
    """Write the rig as calibration files under root: the kind's layout and unit, or (no kind / MultiviewX)
    calibrations/intrinsic|extrinsic/*_CameraN.xml in metres, read by the MultiviewX dataset classes."""
    import cv2
    if kind and kind != "multiviewx":
        target = kinds.get(kind)
        if len(cameras) != target.cameras:
            raise ValueError(f"{target.name} needs {target.cameras} cameras, the Unity rig sent {len(cameras)}")
        cams = []
        for camera in cameras:
            K, R, t = camera_matrices(camera, unity_axes(kind))
            if np.linalg.det(R) < 0:
                R, t = -R, -t
            cams.append({"K": K, "R": R, "t": t})
        target.write_calibration(cams, Path(root))
        (Path(root) / "cameras.json").write_text(json.dumps({"source": "Unity live rig", "kind": kind,
                                                              "cameras": cameras}, indent=1))
        return Path(root)
    for folder in ("intrinsic", "extrinsic"):
        (root / "calibrations" / folder).mkdir(parents=True, exist_ok=True)
    for index, camera in enumerate(cameras, 1):
        K, R, t = camera_matrices(camera)
        rvec, _ = cv2.Rodrigues(-R)
        storage = cv2.FileStorage(str(root / "calibrations" / "intrinsic" / f"intr_Camera{index}.xml"), cv2.FILE_STORAGE_WRITE)
        storage.write("camera_matrix", K)
        storage.write("distortion_coefficients", np.zeros((1, 5)))
        storage.release()
        storage = cv2.FileStorage(str(root / "calibrations" / "extrinsic" / f"extr_Camera{index}.xml"), cv2.FILE_STORAGE_WRITE)
        storage.write("rvec", rvec)
        storage.write("tvec", (-t).reshape(3, 1))
        storage.release()
    (root / "cameras.json").write_text(json.dumps({"source": "MvxLiveStream rig", "cameras": cameras}, indent=1))
    return root


def depth_sign(R: np.ndarray, t: np.ndarray) -> float:
    """+1 when points in front of the camera have positive depth; MultiviewX-style files give -1."""
    return 1.0 if (R @ AREA_CENTRE + t)[2] > 0 else -1.0


def read_matrices(root: Path, kind: str) -> list[tuple]:
    """(K, R, t in metres, depth sign) per camera of a calibration folder in the kind's layout."""
    target = kinds.get(kind)
    intrinsics, extrinsics = target.read_calibration(Path(root))
    centre = np.array([sum(target.x_m) / 2, sum(target.y_m) / 2, 0.0])
    matrices = []
    for K, Rt in zip(intrinsics, extrinsics):
        R, t = Rt[:, :3], Rt[:, 3] / target.unit_per_m
        matrices.append((K, R, t, 1.0 if (R @ centre + t)[2] > 0 else -1.0))
    return matrices


def read_calibration(root: Path, count: int) -> list[tuple]:
    """(K, R, t, depth sign) per camera, as the dataset classes load the files."""
    import cv2
    matrices = []
    for index in range(1, count + 1):
        storage = cv2.FileStorage(str(root / "calibrations" / "intrinsic" / f"intr_Camera{index}.xml"), cv2.FILE_STORAGE_READ)
        K = storage.getNode("camera_matrix").mat()
        storage.release()
        storage = cv2.FileStorage(str(root / "calibrations" / "extrinsic" / f"extr_Camera{index}.xml"), cv2.FILE_STORAGE_READ)
        rvec, tvec = storage.getNode("rvec").mat().squeeze(), storage.getNode("tvec").mat().squeeze()
        storage.release()
        R, t = cv2.Rodrigues(rvec)[0], np.asarray(tvec, dtype=float)
        matrices.append((K, R, t, depth_sign(R, t)))
    return matrices


def project(matrices: tuple, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World points [N, 3] -> pixels [N, 2] and a mask of points in front of the camera."""
    K, R, t, sign = matrices
    camera = points @ R.T + t
    front = camera[:, 2] * sign > 1e-6
    pixels = camera @ K.T
    return pixels[:, :2] / pixels[:, 2:3], front


def reprojection_error(matrices: list[tuple], probes: list[dict]) -> float | None:
    """Largest pixel distance between Unity's own projection of probe points and the written calibration."""
    worst = None
    for probe in probes:
        world = np.asarray(probe["world"], dtype=float).reshape(1, 3)
        pixels = probe.get("pixels")
        if pixels is None:   # MvxLiveStream: flat px = [u0, v0, u1, v1, ...] with a visible flag per camera
            flat = probe.get("px", [])
            pixels = [flat[2 * i:2 * i + 2] if visible else None for i, visible in enumerate(probe.get("visible", []))]
        for camera, pixel in enumerate(pixels):
            if pixel is None or camera >= len(matrices):
                continue
            projected, front = project(matrices[camera], world)
            if front[0]:
                error = float(np.hypot(*(projected[0] - np.asarray(pixel, dtype=float))))
                worst = error if worst is None else max(worst, error)
    return worst


def person_boxes(matrices: list[tuple], x: float, y: float) -> dict[int, list[int]]:
    """Bounding box of a 0.16 m x 1.8 m person cylinder at ground point (x, y) metres, per camera that sees it."""
    corners = np.array([[x + dx * PERSON_RADIUS, y + dy * PERSON_RADIUS, z]
                        for z in (0.0, PERSON_HEIGHT) for dx, dy in ((1, 1), (-1, 1), (1, -1), (-1, -1))])
    boxes = {}
    for camera, matrix in enumerate(matrices):
        pixels, front = project(matrix, corners)
        if not front.all():
            continue
        left, top = np.floor(pixels.min(axis=0))
        right, bottom = np.ceil(pixels.max(axis=0))
        if right <= 0 or bottom <= 0 or left >= IMAGE_W or top >= IMAGE_H:
            continue
        boxes[camera] = [int(max(0, left)), int(max(0, top)), int(min(IMAGE_W - 1, right)), int(min(IMAGE_H - 1, bottom))]
    return boxes

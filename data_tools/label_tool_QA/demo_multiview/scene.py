"""Custom scenes for the demo: N synchronized videos or an image folder, seen through a calibration folder or through
cameras estimated without calibration (CalibFree), then mapped onto the BEV grid of the chosen checkpoint.

A checkpoint is tied to its training rig in two ways: its weights expect a fixed number of cameras (inference.py
reads it from the state dict) and its BEV grid has a fixed size and origin (Wildtrack 12 x 36 m, MultiviewX
25 x 16 m, Warehouse 16.5 x 58 m: the scene kinds of kinds.py). A scene therefore keeps its cameras in their own world frame ("rig world", metres, z up) and a
*placement* that puts the checkpoint's grid rectangle somewhere on that ground. The calibration handed to the
checkpoint is the rig expressed in the grid's frame, written in the dataset layout its loader reads.
"""

from __future__ import annotations

import json
import math
import re
import sys
import threading
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

import kinds

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
VIDEO_TYPES = {".mp4", ".avi", ".mkv", ".mov", ".m4v", ".webm", ".mpg", ".mpeg", ".wmv"}
IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MODEL_IMAGE = (1920, 1080)          # both trainers' calibrations and boxes live in 1920 x 1080 pixels
WILDTRACK_NAMES = kinds.WILDTRACK_NAMES


def natural_key(text: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


# ---------------------------------------------------------------------------------------------------- grid cells

def cells_to_metres(kind: str, x: float, y: float) -> tuple[float, float]:
    """Demo grid cells (2.5 cm along world X, Y) -> metres in the dataset world frame."""
    return kinds.get(kind).cells_to_metres(x, y)


def metres_to_cells(kind: str, x: float, y: float) -> tuple[float, float]:
    return kinds.get(kind).metres_to_cells(x, y)


def position_metres(kind: str, position: int) -> tuple[float, float]:
    """positionID of the kind's annotations -> metres in its world frame."""
    return kinds.get(kind).position_metres(position)


# ---------------------------------------------------------------------------------------------------- calibration in

def calibration_dir(path: Path) -> Path | None:
    for candidate in (path, path / "calibrations"):
        if (candidate / "extrinsic").is_dir():
            return candidate
    return None


def _storage_matrix(path: Path, node: str):
    import cv2
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    try:
        value = storage.getNode(node)
        return None if value.empty() else value.mat()
    finally:
        storage.release()


def _read_extrinsic(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        rvec, tvec = _storage_matrix(path, "rvec"), _storage_matrix(path, "tvec")
    except Exception:   # Wildtrack writes plain-text nodes that OpenCV does not parse as matrices
        rvec = tvec = None
    if rvec is None or tvec is None:
        root = ET.parse(path).getroot()
        rvec = np.array(root.find("rvec").text.split(), dtype=float)
        tvec = np.array(root.find("tvec").text.split(), dtype=float)
    return np.asarray(rvec, dtype=float).reshape(3), np.asarray(tvec, dtype=float).reshape(3)


def _cams_from_json(path: Path) -> list[dict] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    items = data.get("cameras", data) if isinstance(data, dict) else data
    if not isinstance(items, list) or not items or not all(isinstance(item, dict) and {"K", "R", "t"} <= set(item)
                                                           for item in items):
        return None
    return [{"K": np.asarray(item["K"], float), "R": np.asarray(item["R"], float),
             "t": np.asarray(item["t"], float).reshape(3), "dist": np.asarray(item.get("dist", np.zeros(5)), float),
             "name": item.get("name", f"C{index + 1}")} for index, item in enumerate(items)]


def load_rig(path: str | Path, unit: str = "auto") -> dict:
    """Cameras of a calibration folder (Wildtrack, MultiviewX or any intr_*/extr_* OpenCV layout) or of a JSON file
    with K/R/t per camera (CalibFree cams_*.json). Translations are returned in metres; ``dist`` holds the distortion
    coefficients of the raw images when the folder has them (Wildtrack: intrinsic_original)."""
    import cv2
    path = Path(path)
    if path.is_file() and path.suffix.lower() == ".json":
        cams = _cams_from_json(path)
        if not cams:
            raise ValueError("The JSON file has no list of cameras with K, R and t.")
        return {"cams": cams, "layout": "json", "unit": "m", "native_kind": None, "path": str(path)}
    root = calibration_dir(path) if path.is_dir() else None
    if root is None:
        raise ValueError("No calibration found: pick a folder with extrinsic/ and intrinsic*/ (or its parent).")
    extrinsics = {file.stem[len("extr_"):]: file for file in (root / "extrinsic").glob("extr_*.xml")}
    if not extrinsics:
        raise ValueError(f"{root / 'extrinsic'} has no extr_*.xml files.")
    if set(WILDTRACK_NAMES) <= set(extrinsics):
        layout, names = "wildtrack", WILDTRACK_NAMES
    elif all(re.fullmatch(r"Camera\d+", name) for name in extrinsics):
        layout, names = "multiviewx", sorted(extrinsics, key=natural_key)
    else:
        layout, names = "generic", sorted(extrinsics, key=natural_key)
    folders = [root / name for name in ("intrinsic_original", "intrinsic", "intrinsic_zero")]
    folders += sorted(folder for folder in root.glob("intrinsic*") if folder.is_dir() and folder not in folders)
    cams = []
    for name in names:
        intrinsic = next((folder / f"intr_{name}.xml" for folder in folders if (folder / f"intr_{name}.xml").is_file()),
                         None)
        if intrinsic is None:
            raise ValueError(f"Missing intr_{name}.xml for extr_{name}.xml")
        K = np.asarray(_storage_matrix(intrinsic, "camera_matrix"), dtype=float)
        dist = _storage_matrix(intrinsic, "distortion_coefficients")
        rvec, tvec = _read_extrinsic(extrinsics[name])
        cams.append({"K": K, "R": cv2.Rodrigues(rvec)[0], "t": tvec, "name": name,
                     "dist": np.zeros(5) if dist is None else np.asarray(dist, dtype=float).ravel()})
    # The dataset this calibration belongs to (its dataset_info.json, else names + camera count): its world is
    # then that kind's grid world.
    native = kinds.detect(root.parent, register_new=False) if root.name == "calibrations" else None
    if native is None:
        native = {"wildtrack": "wildtrack", "multiviewx": "multiviewx"}.get(layout)
    if native and len(cams) != kinds.get(native).cameras:
        native = None
    if unit == "auto":
        if native:
            unit = kinds.get(native).unit
        elif layout == "wildtrack":
            unit = "cm"
        else:   # surveillance cameras stand a few metres up and up to tens of metres away
            reach = float(np.median([np.linalg.norm(cam["R"].T @ cam["t"]) for cam in cams]))
            unit = "mm" if reach > 5000 else "cm" if reach > 150 else "m"
    scale = {"m": 1.0, "cm": 0.01, "mm": 0.001}[unit]
    for cam in cams:
        cam["t"] = cam["t"] * scale
    return {"cams": cams, "layout": layout, "unit": unit, "native_kind": native, "path": str(root)}


def describe_rig(rig: dict) -> dict:
    distorted = any(np.abs(cam["dist"]).max() > 1e-9 for cam in rig["cams"])
    return {"layout": rig["layout"], "cameras": len(rig["cams"]), "unit": rig["unit"], "native_kind": rig["native_kind"],
            "names": [cam["name"] for cam in rig["cams"]], "distortion": distorted, "path": rig.get("path")}


# ---------------------------------------------------------------------------------------------------- placement

@dataclass
class Placement:
    """Where the checkpoint's BEV rectangle lies on the rig ground: its centre (rig metres), the direction of the
    grid's +x axis (degrees from rig +x, counter-clockwise) and whether the grid is a mirror image (MultiviewX labels
    live in a mirrored world)."""
    cx: float
    cy: float
    yaw_deg: float = 0.0
    mirror: bool = False

    @staticmethod
    def identity(kind: str) -> "Placement":
        area = kinds.get(kind)
        return Placement(sum(area.x_m) / 2, sum(area.y_m) / 2, 0.0, False)

    def as_dict(self) -> dict:
        return {key: (round(value, 4) if isinstance(value, float) else value) for key, value in asdict(self).items()}


def placement_transform(kind: str, placement: Placement) -> tuple[np.ndarray, np.ndarray]:
    """(B, b) with X_rig = B @ X_grid + b, X_grid in metres of the dataset world frame (z up in both)."""
    area = kinds.get(kind)
    centre = np.array([sum(area.x_m) / 2, sum(area.y_m) / 2, 0.0])
    angle = math.radians(placement.yaw_deg)
    rotation = np.array([[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
    B = rotation @ np.diag([1.0, -1.0 if placement.mirror else 1.0, 1.0])
    return B, np.array([placement.cx, placement.cy, 0.0]) - B @ centre


def grid_to_rig(kind: str, placement: Placement, xy: np.ndarray) -> np.ndarray:
    B, b = placement_transform(kind, placement)
    xy = np.atleast_2d(np.asarray(xy, dtype=float))
    return (np.column_stack([xy, np.zeros(len(xy))]) @ B.T + b)[:, :2]


def rig_to_grid(kind: str, placement: Placement, xy: np.ndarray) -> np.ndarray:
    B, b = placement_transform(kind, placement)
    xy = np.atleast_2d(np.asarray(xy, dtype=float))
    return ((np.column_stack([xy, np.zeros(len(xy))]) - b) @ np.linalg.inv(B).T)[:, :2]


def area_corners(kind: str, placement: Placement) -> list[list[float]]:
    (x0, x1), (y0, y1) = kinds.get(kind).x_m, kinds.get(kind).y_m
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return grid_to_rig(kind, placement, np.array(corners)).round(3).tolist()


def frame_intrinsics(K: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """K in the pixels of the frames. A calibration made at another resolution (e.g. 1920 x 1080 cameras recorded
    to 1280 x 720 videos) shows as a principal point far from the frame centre; it is rescaled to the frame."""
    width, height = size
    sx, sy = width / (2 * K[0, 2]), height / (2 * K[1, 2])
    # Real principal points sit up to ~20 % off centre (Wildtrack); a resolution change moves both axes by the same
    # factor, far from 1.
    if abs(sx - sy) < 0.08 * max(sx, sy) and abs(sx - 1) > 0.3:
        return np.diag([sx, sy, 1.0]) @ K
    return K


def grid_cameras(cams: list[dict], kind: str, placement: Placement, image_sizes: list[tuple[int, int]]) -> list[dict]:
    """The rig in the grid's frame with intrinsics scaled to 1920 x 1080. Rotations stay proper (Rodrigues needs it):
    a mirrored placement stores -R, -t like the released MultiviewX files (negative depth, identical pixels)."""
    B, b = placement_transform(kind, placement)
    out = []
    for cam, (width, height) in zip(cams, image_sizes):
        scale = np.diag([MODEL_IMAGE[0] / width, MODEL_IMAGE[1] / height, 1.0])
        R = cam["R"] @ B
        t = cam["R"] @ b + cam["t"]
        if np.linalg.det(R) < 0:
            R, t = -R, -t
        out.append({"K": scale @ frame_intrinsics(cam["K"], (width, height)), "R": R, "t": t})
    return out


def export_calibration(cams: list[dict], kind: str, placement: Placement, image_sizes: list[tuple[int, int]],
                       out_root: Path) -> tuple[Path, list[dict]]:
    """Write <out_root>/calibrations in the layout the checkpoint's dataset class reads; returns the root and the
    grid-frame cameras (metres)."""
    grid = grid_cameras(cams, kind, placement, image_sizes)
    out_root.mkdir(parents=True, exist_ok=True)
    if kinds.get(kind).builtin:   # the writer the CalibFree experiments used for these two datasets
        if str(PROJECT / "CalibFree") not in sys.path:
            sys.path.insert(0, str(PROJECT / "CalibFree"))
        from calibfree.datasets import get_spec
        get_spec(kind).write_calibration(grid, str(out_root))
    else:
        kinds.get(kind).write_calibration(grid, out_root)
    (out_root / "placement.json").write_text(json.dumps({"kind": kind, "placement": placement.as_dict()}, indent=1))
    return out_root, grid


def projection_matrices(grid_cams: list[dict], kind: str) -> list[tuple]:
    """(K, R, t, depth sign) per camera, the tuples rig_calibration.person_boxes / project take. The sign says which
    side of the camera is in front: the side its optical axis tilts down to the ground on."""
    return [(cam["K"], cam["R"], cam["t"], 1.0 if (cam["R"].T @ np.array([0.0, 0.0, 1.0]))[2] < 0 else -1.0)
            for cam in grid_cams]


# ---------------------------------------------------------------------------------------------------- ground views

def camera_centre(cam: dict) -> np.ndarray:
    return -cam["R"].T @ cam["t"]


def _ray_sign(cam: dict) -> float:
    """+1 when rays R^T K^-1 [u, v, 1] point out of the camera (positive depth), -1 for MultiviewX-style files."""
    K, R = cam["K"], cam["R"]
    ray = R.T @ np.linalg.inv(K) @ np.array([K[0, 2], K[1, 2], 1.0])
    return 1.0 if ray[2] < 0 else -1.0   # surveillance cameras look down at the ground


def footprints(cams: list[dict], image_sizes: list[tuple[int, int]], max_range: float = 45.0) -> list[dict]:
    """Ground area each camera sees (polygon in rig xy, clipped at max_range) and its centre, for the BEV editor."""
    out = []
    for cam, (width, height) in zip(cams, image_sizes):
        centre = camera_centre(cam)
        steps = np.linspace(0, 1, 13)
        border = ([(width * s, 0) for s in steps] + [(width, height * s) for s in steps[1:]]
                  + [(width * (1 - s), height) for s in steps[1:]] + [(0, height * (1 - s)) for s in steps[1:-1]])
        K = frame_intrinsics(cam["K"], (width, height))
        rays = (cam["R"].T @ np.linalg.inv(K) @ np.column_stack([border, np.ones(len(border))]).T).T
        rays *= _ray_sign(cam)
        points = []
        for ray in rays:
            horizontal = np.linalg.norm(ray[:2]) + 1e-12
            reach = max_range / horizontal
            if ray[2] < -1e-9:
                reach = min(reach, -centre[2] / ray[2])
            points.append((centre + reach * ray)[:2])
        out.append({"centre": centre.round(3).tolist(), "polygon": np.round(points, 3).tolist()})
    return out


def _inside(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    x, y = points[:, 0], points[:, 1]
    inside = np.zeros(len(points), bool)
    px, py = polygon[:, 0], polygon[:, 1]
    for i in range(len(polygon)):
        x1, y1, x2, y2 = px[i - 1], py[i - 1], px[i], py[i]
        crosses = (y1 > y) != (y2 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            inside ^= crosses & (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1)
    return inside


def coverage(footprint_list: list[dict], step: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Ground sample points and how many cameras see each one."""
    polygons = [np.asarray(item["polygon"]) for item in footprint_list]
    every = np.concatenate(polygons)
    lo, hi = every.min(0), every.max(0)
    xs, ys = np.arange(lo[0], hi[0], step), np.arange(lo[1], hi[1], step)
    grid = np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2)
    counts = np.sum([_inside(grid, polygon) for polygon in polygons], axis=0)
    return grid, counts


def auto_placement(kind: str, footprint_list: list[dict], people: np.ndarray | None = None,
                   mirror: bool = False) -> Placement:
    """Rectangle of the checkpoint's size that holds the most evidence: detected people when there are enough of
    them (calibration-free scenes), else the ground seen by the most cameras. Searched over heading (5 degrees) and
    centre (0.5 m) with a summed-area table per heading."""
    area = kinds.get(kind)
    size = np.array([area.x_m[1] - area.x_m[0], area.y_m[1] - area.y_m[0]])
    if people is not None and len(people) >= 50:
        points, weights = np.asarray(people, float), np.ones(len(people))
    else:
        points, counts = coverage(footprint_list)
        keep = counts >= 2 if (counts >= 2).any() else counts >= 1
        points, weights = points[keep], counts[keep].astype(float) ** 2
    if not len(points):
        return Placement(0.0, 0.0, 0.0, mirror)
    cell = 0.5
    best = (-1.0, 0.0, np.zeros(2))
    for yaw in np.arange(0, 180, 5.0):
        angle = math.radians(yaw)
        local = points @ np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        lo = local.min(0) - size
        bins = np.ceil((local.max(0) + size - lo) / cell).astype(int)
        hist, _, _ = np.histogram2d(local[:, 0], local[:, 1], bins=bins, weights=weights,
                                    range=[[lo[0], lo[0] + bins[0] * cell], [lo[1], lo[1] + bins[1] * cell]])
        table = np.pad(hist.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        w, h = (size / cell).round().astype(int)
        if w >= table.shape[0] or h >= table.shape[1]:
            continue
        sums = table[w:, h:] - table[:-w, h:] - table[w:, :-h] + table[:-w, :-h]
        i, j = np.unravel_index(int(np.argmax(sums)), sums.shape)
        if sums[i, j] > best[0]:
            local_centre = lo + (np.array([i, j]) + size / cell / 2) * cell
            centre = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]) @ local_centre
            best = (float(sums[i, j]), float(yaw), centre)
    return Placement(float(best[2][0]), float(best[2][1]), best[1], mirror)


# ---------------------------------------------------------------------------------------------------- sources

class ImageFolderSource:
    """Image_subsets/C1..CN of a dataset or a recorded PersonX sequence; annotations are optional."""

    kind_label = "images"

    def __init__(self, path: str | Path, fps: float = 2.0):
        root = Path(path)
        if not (root / "Image_subsets").is_dir():
            found = next((candidate for candidate in [root, *root.glob("*"), *root.glob("*/*")]
                          if candidate.is_dir() and (candidate / "Image_subsets").is_dir()), None)
            if found is None:
                raise ValueError("No Image_subsets/C1..CN folder found there.")
            root = found
        self.root, self.fps = root, float(fps)
        folders = sorted((folder for folder in (root / "Image_subsets").iterdir()
                          if folder.is_dir() and re.fullmatch(r"C\d+", folder.name)), key=lambda f: natural_key(f.name))
        if not folders:
            raise ValueError("Image_subsets has no C1..CN camera folders.")
        self.files = [{int(file.stem): file for file in folder.iterdir()
                       if file.suffix.lower() in IMAGE_TYPES and file.stem.isdigit()} for folder in folders]
        self.frames = sorted(set.intersection(*(set(files) for files in self.files)))
        if not self.frames:
            raise ValueError("The camera folders share no frame numbers.")
        self.num_cams = len(folders)
        annotations = root / "annotations_positions"
        self.annotations = {int(file.stem): file for file in annotations.glob("*.json") if file.stem.isdigit()} \
            if annotations.is_dir() else {}
        from PIL import Image
        with Image.open(self.files[0][self.frames[0]]) as image:
            self.sizes = [image.size] * self.num_cams

    def time(self, frame: int) -> float:
        return self.frames.index(frame) / self.fps

    def read(self, frame: int) -> list[np.ndarray]:
        from PIL import Image
        images = []
        for files in self.files:
            with Image.open(files[frame]) as image:
                images.append(np.asarray(image.convert("RGB")))
        return images

    def ground_truth(self, frame: int) -> list[tuple[int, int]] | None:
        """(positionID, personID) of the frame, in the dataset's own grid; None without annotations."""
        if frame not in self.annotations:
            return None
        people = json.loads(self.annotations[frame].read_text(encoding="utf-8"))
        return [(int(person["positionID"]), int(person["personID"])) for person in people
                if "positionID" in person and "personID" in person]

    def describe(self) -> dict:
        return {"type": "images", "path": str(self.root), "cameras": self.num_cams, "frames": len(self.frames),
                "first": self.frames[0], "last": self.frames[-1], "fps": self.fps, "size": list(self.sizes[0]),
                "annotations": len(self.annotations), "frame_ids": self.frames}

    def close(self):
        pass


class VideoSource:
    """N video files played on one clock: sample j is time start + j / sample_fps, camera k shows its frame at
    that time + offsets[k]. Reading forward decodes sequentially; jumps seek."""

    kind_label = "videos"

    def __init__(self, files: list[str], offsets: list[float] | None = None, sample_fps: float = 5.0,
                 start: float = 0.0, end: float | None = None, gt: dict | None = None):
        import cv2
        if not files:
            raise ValueError("Choose at least one video.")
        self.paths = [Path(file) for file in files]
        self.offsets = [float(value) for value in (offsets or [0.0] * len(files))]
        if len(self.offsets) != len(self.paths):
            raise ValueError("One offset per video is needed.")
        self.captures, self.positions, self.rates, self.sizes, lengths = [], [], [], [], []
        for path in self.paths:
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                raise ValueError(f"Cannot open {path}")
            rate = capture.get(cv2.CAP_PROP_FPS) or 25.0
            self.captures.append(capture)
            self.positions.append(0)
            self.rates.append(rate)
            self.sizes.append((int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))
            lengths.append(capture.get(cv2.CAP_PROP_FRAME_COUNT) / rate)
        self.num_cams = len(self.paths)
        self.sample_fps, self.start = float(sample_fps), max(0.0, float(start))
        duration = min(length - offset for length, offset in zip(lengths, self.offsets))
        self.end = min(float(end), duration) if end else duration
        if self.end <= self.start:
            raise ValueError("The chosen time range is empty for these videos and offsets.")
        self.frames = list(range(int((self.end - self.start) * self.sample_fps)))
        self.gt = gt or {}
        self.lock = threading.Lock()

    def time(self, frame: int) -> float:
        return self.start + frame / self.sample_fps

    def _read_at(self, camera: int, index: int) -> np.ndarray:
        import cv2
        capture = self.captures[camera]
        position = self.positions[camera]
        if not position <= index <= position + 240:   # decoding forward beats an H.264 seek (~0.3 s per camera)
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        else:
            for _ in range(index - position):
                capture.grab()
        ok, image = capture.read()
        if not ok:
            raise ValueError(f"{self.paths[camera].name}: no frame at {index}")
        self.positions[camera] = index + 1
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def read(self, frame: int) -> list[np.ndarray]:
        t = self.time(frame)
        with self.lock:
            return [self._read_at(camera, max(0, round((t + self.offsets[camera]) * self.rates[camera])))
                    for camera in range(self.num_cams)]

    def ground_truth(self, frame: int) -> list[tuple[int, int]] | None:
        """Annotations attached by a preset (Wildtrack: annotated frame f is video frame 6 f + offset)."""
        folder = self.gt.get("annotations")
        if not folder:
            return None
        units = (self.time(frame) - self.gt["zero"]) * self.gt["ids_per_second"]
        annotated = round(units)
        if abs(units - annotated) > 0.01 or annotated % self.gt.get("step", 1):
            return None
        file = Path(folder) / self.gt["pattern"].format(annotated)
        if not file.is_file():
            return None
        people = json.loads(file.read_text(encoding="utf-8"))
        return [(int(person["positionID"]), int(person["personID"])) for person in people]

    def describe(self) -> dict:
        return {"type": "videos", "files": [str(path) for path in self.paths], "cameras": self.num_cams,
                "frames": len(self.frames), "first": 0, "last": len(self.frames) - 1, "fps": self.sample_fps,
                "offsets": self.offsets, "start": self.start, "end": round(self.end, 3), "size": list(self.sizes[0]),
                "annotations": bool(self.gt.get("annotations"))}

    def close(self):
        with self.lock:
            for capture in self.captures:
                capture.release()


def probe_videos(files: list[str]) -> list[dict]:
    import cv2
    out = []
    for file in files:
        capture = cv2.VideoCapture(str(file))
        if not capture.isOpened():
            out.append({"path": str(file), "error": "cannot open"})
            continue
        rate = capture.get(cv2.CAP_PROP_FPS) or 0.0
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        out.append({"path": str(file), "name": Path(file).name, "fps": round(rate, 4),
                    "duration": round(count / rate, 2) if rate else None,
                    "size": [int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))]})
        capture.release()
    return out


# The Wildtrack release ships the raw 60 fps videos next to the annotated frames. Annotated frame f (one every
# 5 ids, 2 fps) is video frame 6 f + offset of that camera; measured by matching the undistorted video frames
# against Image_subsets (frames 5 and 100, all seven cameras).
WILDTRACK_VIDEO_OFFSETS = [9, 9, 9, 10, 12, 13, 16]
WILDTRACK_VIDEO_FPS = 60000 / 1001


def wildtrack_preset(files: list[str]) -> dict | None:
    names = [Path(file).name.lower() for file in files]
    if sorted(names) != [f"cam{i}.mp4" for i in range(1, 8)]:
        return None
    folder = Path(files[0]).parent
    ordered = [str(folder / f"cam{i}.mp4") for i in range(1, 8)]
    calibration = calibration_dir(folder)
    zero = WILDTRACK_VIDEO_OFFSETS[0] / WILDTRACK_VIDEO_FPS
    preset = {"name": "wildtrack", "label": "Wildtrack: 7 synchronized raw videos",
              "files": ordered, "offsets": [(offset - WILDTRACK_VIDEO_OFFSETS[0]) / WILDTRACK_VIDEO_FPS
                                            for offset in WILDTRACK_VIDEO_OFFSETS],
              "start": zero, "sample_fps": WILDTRACK_VIDEO_FPS / 6,
              "calibration": str(calibration) if calibration else None, "undistort": True}
    if (folder / "annotations_positions").is_dir():
        preset["gt"] = {"annotations": str(folder / "annotations_positions"), "zero": zero,
                        "ids_per_second": WILDTRACK_VIDEO_FPS / 6, "step": 5, "pattern": "{:08d}.json",
                        "kind": "wildtrack"}
    return preset


class Undistorter:
    """Per-camera remap of raw frames to the pinhole model of K (the K the calibration gives the detector).
    The calibration is brought to the pixels of the source frames (frame_intrinsics)."""

    def __init__(self, cams: list[dict], sizes: list[tuple[int, int]]):
        import cv2
        self.maps = []
        for cam, (width, height) in zip(cams, sizes):
            if np.abs(cam["dist"]).max() < 1e-9:
                self.maps.append(None)
                continue
            K = frame_intrinsics(cam["K"], (width, height))
            self.maps.append(cv2.initUndistortRectifyMap(K, cam["dist"], None, K, (width, height), cv2.CV_16SC2))

    def __call__(self, images: list[np.ndarray]) -> list[np.ndarray]:
        import cv2
        return [image if maps is None else cv2.remap(image, *maps, cv2.INTER_LINEAR)
                for image, maps in zip(images, self.maps)]


# ---------------------------------------------------------------------------------------------------- file browser

def long_path(path) -> str:
    """Path string Windows opens beyond 260 characters."""
    import os
    text = os.path.abspath(str(path))
    prefix = "\\\\?\\"
    return prefix + text if os.name == "nt" and not text.startswith(prefix) else text


def list_folder(path: str | None, extensions: set[str] | None = None) -> dict:
    """Folder listing for the browser's picker (the server is local: the browser cannot reveal real paths)."""
    import os
    import string
    drives = [f"{letter}:\\" for letter in string.ascii_uppercase if os.path.exists(f"{letter}:\\")] \
        if os.name == "nt" else ["/"]
    if not path:
        return {"path": "", "parent": None, "drives": drives,
                "entries": [{"name": drive, "path": drive, "dir": True} for drive in drives]}
    folder = Path(os.path.abspath(os.path.expanduser(path)))
    if os.path.isfile(long_path(folder)):
        folder = folder.parent
    if not os.path.isdir(long_path(folder)):
        raise ValueError(f"Folder not found: {path}")
    entries = []
    try:   # \?\ paths: training runs nest folders past Windows' 260-character limit (MVDeTr logs reach 280)
        children = list(os.scandir(long_path(folder)))
    except PermissionError as error:
        raise ValueError(f"Access denied: {folder}") from error
    for child in children:
        try:
            is_dir = child.is_dir()
        except OSError:
            continue
        if child.name.startswith(("$", ".")):
            continue
        if not is_dir and extensions and Path(child.name).suffix.lower() not in extensions:
            continue
        item = {"name": child.name, "path": str(folder / child.name), "dir": is_dir}
        if not is_dir:
            try:
                item["size"] = child.stat().st_size
            except OSError:
                pass
        entries.append(item)
    entries.sort(key=lambda item: (not item["dir"], natural_key(item["name"])))
    parent = str(folder.parent) if folder.parent != folder else ""
    return {"path": str(folder), "parent": parent, "drives": drives, "entries": entries}

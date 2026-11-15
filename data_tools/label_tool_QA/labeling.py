"""Pseudo-label engine of the multi-view labeling tool.

One frame goes through three sources of evidence:

1. a multi-view BEV checkpoint (MVDeTr / MVDet, loaded through demo_multiview/inference.py) gives ground points;
2. every ground point is drawn into each camera as the box of a person cylinder (radius 0.20 m, height 1.8 m: the
   Wildtrack annotation boxes are exactly such projections, median IoU 0.91 with the released ones);
3. a per-image person detector (YOLO26) checks those boxes view by view: the number of views that confirm a point
   sets its status, and YOLO boxes that no model point explains, but that triangulate to one ground point across
   >= 2 independent cameras, become suggested people. Moving points toward the YOLO evidence (``refine``) is the
   reviewer's "snap" action; doing it automatically did not help on Wildtrack (9.2 -> 9.5 cm).

Positions are metres in the calibration's world frame (z up, ground z = 0). The browser repeats ``project_box`` and
``match_views`` (app.js), so both sides must stay identical: a 16-gon cylinder, the same visibility rule and the same
greedy IoU matching.
"""

from __future__ import annotations

import json
import math
import shutil
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sources import Dataset, IMAGE_TYPES, camera_folders, natural_key, numbered_images

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEMO = HERE / "demo_multiview"
if not DEMO.is_dir():
    DEMO = PROJECT / "demo_multiview"
if str(DEMO) not in sys.path:
    sys.path.append(str(DEMO))   # appended: label_tool/server.py must win over demo_multiview/server.py

import scene as scenes   # noqa: E402  (demo_multiview)
import wavetrack_bev   # noqa: E402  (register WaveTrack's 5-camera BEV scene)

YOLO_DIR = HERE
SEGMENTS = 16
GRID_STEP = 0.025            # Wildtrack / MultiviewX positionID grid (2.5 cm)
POS_WIDTH = {"wildtrack": 480, "multiviewx": 1000}


@dataclass
class Params:
    radius: float = 0.20         # person cylinder, fitted to the Wildtrack annotation boxes
    height: float = 1.80
    iou_gate: float = 0.30       # projected box <-> YOLO box
    yolo_conf: float = 0.25
    yolo_imgsz: int = 1280
    suggest_conf: float = 0.40   # unexplained YOLO boxes that may become suggested people
    suggest_views: int = 2
    suggest_iou: float = 0.50
    suggest_margin: float = 0.0  # suggested people must stand this far inside the BEV area (m)
    prior_sd: float = 0.10       # model point uncertainty (m) when fused with the YOLO boxes
    along_sd: float = 0.35       # foot pixel -> distance along the camera ray (m), see view_rows
    foot_bias: float = 0.15      # the lowest box pixel is the near side of the feet (m)
    outlier: float = 0.50        # bearings farther than this from the point are ignored (m)
    low_score: float = 0.40      # model points between low_score and the checkpoint threshold need YOLO support
    auto_refine: bool = False    # move model points toward the YOLO evidence (no gain on Wildtrack: 9.2 -> 9.5 cm)
    box_mode: str = "projected"  # label boxes: "projected" (Wildtrack style) or "yolo" (tight box when matched)

    @classmethod
    def from_dict(cls, data: dict | None) -> "Params":
        params = cls()
        for key, value in (data or {}).items():
            if hasattr(params, key) and value is not None:
                kind = type(getattr(params, key))
                setattr(params, key, value if kind is bool and isinstance(value, bool) else
                        str(value).lower() in ("1", "true", "yes") if kind is bool else kind(value))
        return params

    def as_dict(self) -> dict:
        return dict(self.__dict__)


# ---------------------------------------------------------------------------------------------------- geometry

@dataclass
class Camera:
    index: int                   # position in the rig (Wildtrack viewNum)
    name: str
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    width: int
    height: int
    folder: Path | None = None

    def to_camera(self, points: np.ndarray) -> np.ndarray:
        return points @ self.R.T + self.t

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        camera = self.to_camera(points)
        pixels = camera @ self.K.T
        return pixels[:, :2] / pixels[:, 2:3], camera[:, 2]

    def ground(self, u: float, v: float) -> tuple[float, float] | None:
        """Ground point (z = 0) seen at pixel (u, v), or None when the ray misses the ground in front."""
        ray = self.R.T @ np.linalg.solve(self.K, np.array([u, v, 1.0]))
        centre = -self.R.T @ self.t
        if abs(ray[2]) < 1e-9:
            return None
        depth = -centre[2] / ray[2]
        if depth <= 0:
            return None
        point = centre + depth * ray
        return float(point[0]), float(point[1])

    def public(self) -> dict:
        return {"index": self.index, "name": self.name, "K": self.K.tolist(), "R": self.R.tolist(),
                "t": self.t.tolist(), "width": self.width, "height": self.height,
                "centre": (-self.R.T @ self.t).tolist()}


_ANGLES = np.linspace(0, 2 * np.pi, SEGMENTS, endpoint=False)


def project_box(cam: Camera, x: float, y: float, radius: float, height: float) -> list[float] | None:
    """Clipped image box of the person cylinder at (x, y), or None when the camera does not usefully see it:
    behind the camera, under 12 px tall, or less than 30 % of the box inside the image."""
    ring = np.column_stack([x + radius * np.cos(_ANGLES), y + radius * np.sin(_ANGLES)])
    points = np.vstack([np.column_stack([ring, np.zeros(SEGMENTS)]), np.column_stack([ring, np.full(SEGMENTS, height)])])
    pixels, depth = cam.project(points)
    if (depth <= 0.1).any():
        return None
    left, top = pixels.min(axis=0)
    right, bottom = pixels.max(axis=0)
    area = (right - left) * (bottom - top)
    cl, ct, cr, cb = max(left, 0.0), max(top, 0.0), min(right, cam.width - 1.0), min(bottom, cam.height - 1.0)
    if cr <= cl or cb <= ct or bottom - top < 12 or (cr - cl) * (cb - ct) < 0.3 * area:
        return None
    return [round(cl, 1), round(ct, 1), round(cr, 1), round(cb, 1)]


def iou(a, b) -> float:
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def match_views(cams: list[Camera], persons: list[dict], yolo: dict, params: Params) -> list[dict]:
    """Per person and camera: projected box, matched YOLO index and IoU. Greedy one-to-one matching per camera by
    decreasing IoU (ties: person order, then YOLO order). Returns one dict {camera index: view} per person."""
    views = [{} for _ in persons]
    for cam in cams:
        boxes = yolo.get(str(cam.index), [])
        pairs = []
        for p, person in enumerate(persons):
            box = project_box(cam, person["x"], person["y"], person.get("r", params.radius), person.get("h", params.height))
            if box is None:
                continue
            views[p][cam.index] = {"proj": box, "yolo": None, "iou": 0.0}
            for y, det in enumerate(boxes):
                overlap = iou(box, det)
                if overlap >= params.iou_gate:
                    pairs.append((-overlap, p, y))
        pairs.sort()
        used_p, used_y = set(), set()
        for negative, p, y in pairs:
            if p in used_p or y in used_y:
                continue
            used_p.add(p)
            used_y.add(y)
            views[p][cam.index].update(yolo=y, iou=round(-negative, 3))
    return views


def label_box(view: dict, override: str | list | None, yolo_boxes: list, params: Params) -> list[float] | None:
    if isinstance(override, list):   # drawn or resized by hand in this camera
        return [round(float(v), 1) for v in override]
    mode = override or ("yolo" if params.box_mode == "yolo" else "proj")
    if mode == "off":
        return None
    if mode == "yolo" and view.get("yolo") is not None:
        return [round(v, 1) for v in yolo_boxes[view["yolo"]][:4]]
    return view["proj"]


def view_rows(cam: Camera, det: list, proj_height: float | None, params: Params) -> list[tuple]:
    """Least-squares rows (a, value, weight, tolerance) meaning a . p = value for the ground point p of a YOLO box.

    The box centre gives the bearing (the ground line from the camera through the person), accurate to ~8 cm on
    Wildtrack; the foot pixel gives the distance along that line, which is not (sd ~0.5 m, and ~0.15 m short because
    the lowest pixel is the near side of the feet). A box cut at the image side gives no bearing; a box cut at the
    bottom, or much shorter than the projected person (lower body occluded), gives no distance."""
    x1, y1, x2, y2, conf = det[:5]
    width, tall = x2 - x1, y2 - y1
    if tall < 24:
        return []
    u = (x1 + x2) / 2
    centre = (-cam.R.T @ cam.t)[:2]
    ground = cam.ground(u, y2)
    if ground is None:
        return []
    ray = np.asarray(ground) - centre
    distance = float(np.linalg.norm(ray))
    if distance < 1e-6:
        return []
    d = ray / distance
    n = np.array([-d[1], d[0]])
    rows = []
    if x1 > 2 and x2 < cam.width - 3:
        side = cam.ground(u + 0.1 * width + 2, y2)
        sd = max(0.03, math.hypot(side[0] - ground[0], side[1] - ground[1])) if side else 0.1
        rows.append((n, float(n @ centre), conf / sd ** 2, params.outlier))
    if y2 < cam.height - 3 and not (proj_height and tall < 0.7 * proj_height):
        lower = cam.ground(u, y2 + 0.04 * tall)
        sd = max(params.along_sd, math.hypot(lower[0] - ground[0], lower[1] - ground[1]) if lower else 1.0)
        rows.append((d, float(d @ centre) + distance + params.foot_bias, conf / sd ** 2, 1.5))
    return rows


def solve_rows(rows: list[tuple], prior=None, prior_sd: float = 0.1) -> np.ndarray | None:
    A, b = np.zeros((2, 2)), np.zeros(2)
    if prior is not None:
        A += np.eye(2) / prior_sd ** 2
        b += np.asarray(prior, dtype=float) / prior_sd ** 2
    for a, value, weight, _ in rows:
        A += weight * np.outer(a, a)
        b += weight * value * a
    if np.linalg.cond(A) > 1e8:
        return None
    return np.linalg.solve(A, b)


def agreeing(rows: list[tuple], point) -> list[tuple]:
    return [row for row in rows if abs(row[0] @ point - row[1]) <= row[3]]


def refine(cams: list[Camera], person: dict, view: dict, yolo: dict, params: Params, prior: bool = True) -> tuple:
    """Least-squares ground point from the model point (prior) and the bearings / feet of its matched YOLO boxes,
    ignoring rows that disagree with the current point. Returns (x, y, number of views used)."""
    by_index = {cam.index: cam for cam in cams}
    p0 = np.array([person["x"], person["y"]])
    rows, used = [], 0
    for index, item in view.items():
        if item.get("yolo") is None:
            continue
        det = yolo[str(index)][item["yolo"]]
        mine = agreeing(view_rows(by_index[index], det, item["proj"][3] - item["proj"][1], params), p0)
        if mine:
            rows += mine
            used += 1
    point = solve_rows(rows, p0 if prior else None, params.prior_sd) if rows else None
    if point is None:
        return person["x"], person["y"], 0
    return float(point[0]), float(point[1]), used


def in_area(area: list[float] | None, x: float, y: float, margin: float = 0.0) -> bool:
    if not area:
        return True
    return area[0] - margin <= x <= area[1] + margin and area[2] - margin <= y <= area[3] + margin


SAME_BEARING = math.cos(math.radians(15))


def independent(support: list) -> bool:
    """At least two supporting views whose bearings differ by 15 degrees or more (a real triangulation)."""
    normals = [item[4][0][0] for item in support]
    return any(abs(normals[i] @ normals[j]) <= SAME_BEARING
               for i in range(len(normals)) for j in range(i + 1, len(normals)))


def contradictions(cams: list[Camera], point, support: set, others: list, yolo: dict, params: Params) -> int:
    """Views that see the point unoccluded (no known person's box overlaps it, no box touches the image border)
    and still have no YOLO box there: evidence that nobody stands at the point."""
    count = 0
    for cam in cams:
        if cam.index in support:
            continue
        box = project_box(cam, point[0], point[1], params.radius, params.height)
        if box is None or box[0] <= 1 or box[1] <= 1 or box[2] >= cam.width - 2 or box[3] >= cam.height - 2:
            continue
        if any((other := project_box(cam, x, y, params.radius, params.height)) and iou(box, other) > 0.05
               for x, y in others):
            continue
        if not any(iou(box, det) >= params.iou_gate for det in yolo.get(str(cam.index), [])):
            count += 1
    return count


def suggest(cams: list[Camera], persons: list[dict], views: list[dict], yolo: dict, params: Params,
            area: list[float] | None) -> list[dict]:
    """People only YOLO sees. Every pair of unexplained boxes from two cameras is triangulated (bearings + feet);
    a candidate needs the person cylinder at that point to overlap the boxes with IoU >= suggest_iou in at least
    suggest_views cameras. Candidates are taken greedily (most cameras, then confidence), each box at most once."""
    taken = {(index, item["yolo"]) for view in views for index, item in view.items() if item.get("yolo") is not None}
    boxes = []
    for cam in cams:
        for y, det in enumerate(yolo.get(str(cam.index), [])):
            if (cam.index, y) not in taken and det[4] >= params.suggest_conf:
                rows = view_rows(cam, det, None, params)
                if len(rows) == 2:
                    boxes.append((cam, y, det, rows))
    existing = [(p["x"], p["y"]) for p in persons]

    def supporters(point):
        best = {}
        for cam, y, det, rows in boxes:
            if abs(rows[0][0] @ point - rows[0][1]) > params.outlier:   # bearing first: cheap
                continue
            box = project_box(cam, point[0], point[1], params.radius, params.height)
            if box is not None and (overlap := iou(box, det)) >= params.suggest_iou and                     overlap > best.get(cam.index, (0,))[0]:
                best[cam.index] = (overlap, cam, y, det, rows)
        return list(best.values())

    candidates = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            if a[0].index == b[0].index or abs(a[3][0][0] @ b[3][0][0]) > SAME_BEARING:
                continue   # same camera, or two cameras looking along the same line (Wildtrack C1/C4)
            point = solve_rows(a[3] + b[3])
            if point is None or not in_area(area, *point, -params.suggest_margin) or                     len(agreeing(a[3] + b[3], point)) < 4:
                continue
            support = supporters(point)
            if len(support) < params.suggest_views:
                continue
            refined = solve_rows([row for item in support for row in item[4]])
            if refined is not None:
                point, support = refined, supporters(refined)
            if len(support) >= params.suggest_views and in_area(area, *point, -params.suggest_margin) and                     independent(support):
                candidates.append((len(support), sum(item[3][4] for item in support), point, support))
    candidates.sort(key=lambda item: (-item[0], -item[1]))
    used, out = set(), []
    for count, total, point, support in candidates:
        keys = {(item[1].index, item[2]) for item in support}
        if keys & used or any(math.hypot(point[0] - x, point[1] - y) < 0.6 for x, y in existing):
            continue
        if contradictions(cams, point, {item[1].index for item in support}, existing, yolo, params) >= count:
            continue
        used |= keys
        existing.append(tuple(point))
        out.append({"x": float(point[0]), "y": float(point[1]), "score": round(total / count, 3), "origin": "yolo",
                    "status": "suggested", "support": count})
    return out


def fuse(cams: list[Camera], points: list[tuple], yolo: dict, params: Params, threshold: float,
         area: list[float] | None) -> list[dict]:
    """Model ground points [(x, y, score)] in metres + YOLO boxes per camera -> proposed people."""
    persons = [{"x": float(x), "y": float(y), "score": round(float(s), 3), "origin": "model"} for x, y, s in points]
    for _ in range(2 if params.auto_refine else 0):   # refine, then re-match at the refined positions
        views = match_views(cams, persons, yolo, params)
        for person, view in zip(persons, views):
            person["x"], person["y"], _ = refine(cams, person, view, yolo, params)
    views = match_views(cams, persons, yolo, params)
    kept = []
    for person, view in zip(persons, views):
        visible = len(view)
        support = sum(1 for item in view.values() if item.get("yolo") is not None)
        person["support"], person["visible"] = support, visible
        if person["score"] < threshold:
            if support < 2:
                continue      # weak model point that YOLO does not confirm
            person["status"], person["origin"] = "suggested", "model-low"   # labelled only once accepted
        else:
            person["status"] = "ok" if support >= 2 or (visible and support == visible) else "check"
        kept.append(person)
    views = match_views(cams, kept, yolo, params)
    suggested = suggest(cams, kept, views, yolo, params, area)
    people = kept + suggested
    for number, person in enumerate(people, 1):
        person["uid"] = f"p{number}"
        person["h"] = params.height
        person["x"], person["y"] = round(person["x"], 4), round(person["y"], 4)
        person["auto"] = [person["x"], person["y"]]
        person.setdefault("verified", False)
        person.setdefault("override", {})
    return people


# ---------------------------------------------------------------------------------------------------- data sources

def find_calibration(root: Path) -> Path | None:
    for candidate in (root / "calibrations", root):
        if scenes.calibration_dir(candidate):
            return candidate
        if wavetrack_calibration(candidate):
            return candidate
    # WaveTrackNew keeps both image variants beside one shared calibration.
    if root.name.lower() in ("image_subsets", "image_subsets_raw"):
        candidate = root.parent / "calibrations"
        if scenes.calibration_dir(candidate) or wavetrack_calibration(candidate):
            return candidate
    return None


def wavetrack_calibration(path: Path) -> dict | None:
    """WaveTrack's manifest declares metres, source resolution, ROI and named cameras.

    Its XML extrinsics are centimetres; do not use the generic unit heuristic for this rig.
    Other JSON calibration formats continue through the existing scene loader.
    """
    from annotation_import import xml_manifest
    xml = xml_manifest(path)
    if xml is not None:
        return xml
    file = path if path.is_file() else path / "calibration.json"
    if not file.is_file() and path.is_dir():
        file = path / "calibrations" / "calibration.json"
    if not file.is_file():
        return None
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    items = data.get("cameras") if isinstance(data, dict) else None
    if not isinstance(items, list) or not items or not all(
            isinstance(c, dict) and {"name", "camera_matrix", "R", "tvec_m", "image_size"} <= c.keys()
            for c in items):
        return None
    return {**data, "path": str(file)}


def load_cameras(calib_path: Path, folders: list[Path], dataset: Dataset | None = None) -> tuple[list[Camera], dict]:
    """Cameras for the image folders. Wildtrack's Image_subsets are already undistorted, so its intrinsic_zero
    matrices are used; other calibrations keep their distortion and the images are undistorted when served."""
    from PIL import Image
    manifest = wavetrack_calibration(calib_path)
    if manifest:
        return load_wavetrack_cameras(manifest, folders, dataset)
    rig = scenes.load_rig(calib_path)
    root = scenes.calibration_dir(Path(calib_path)) if Path(calib_path).is_dir() else None
    if len(rig["cams"]) != len(folders):
        raise ValueError(f"The calibration has {len(rig['cams'])} cameras but the dataset has {len(folders)} camera "
                         "folders.")
    cams, distortion = [], []
    for index, (cam, folder) in enumerate(zip(rig["cams"], folders)):
        K, dist = cam["K"], cam["dist"]
        zero = root / "intrinsic_zero" / f"intr_{cam['name']}.xml" if root else None
        if rig["layout"] == "wildtrack" and zero and zero.is_file():
            K, dist = np.asarray(scenes._storage_matrix(zero, "camera_matrix"), dtype=float), np.zeros(5)
        if dataset:
            width, height = dataset.size(folder)
        else:
            first = min(numbered_images(folder).items())[1]
            with Image.open(first) as image:
                width, height = image.size
        cams.append(Camera(index, cam["name"], scenes.frame_intrinsics(K, (width, height)), cam["R"], cam["t"],
                           width, height, folder))
        distortion.append(np.asarray(dist, dtype=float))
    info = scenes.describe_rig(rig)
    info["distortion"] = any(np.abs(d).max() > 1e-9 for d in distortion)
    info["calib_root"] = str(root.parent if root and root.name == "calibrations" else root) if root else None
    return cams, {"info": info, "dist": distortion}


def load_wavetrack_cameras(manifest: dict, folders: list[Path], dataset: Dataset | None) -> tuple[list[Camera], dict]:
    import re
    from PIL import Image

    def camera_number(name):
        match = re.fullmatch(r"(?:Cam(?:era)?|C)0*(\d+)(?:_.*)?", name, re.IGNORECASE)
        if not match:
            raise ValueError(f"Cannot match WaveTrack camera name: {name}")
        return int(match[1])

    def pixels_of(folder):
        if dataset is None or dataset.mode != "images":
            return "raw"
        # Only a paired Image_subsets / Image_subsets_raw export identifies
        # processed pixels by name. Older WaveTrack subsets without a raw sibling
        # still contain distorted source images and must be rectified once.
        if any(parent.name.lower() == "image_subsets_raw" for parent in folder.parents):
            return "raw"
        if any(parent.name.lower() == "image_subsets" and
               (parent.parent / "Image_subsets_raw").is_dir() for parent in folder.parents):
            return "undistorted"
        return "raw"

    pixel_states = {pixels_of(folder) for folder in folders}
    if len(pixel_states) != 1:
        raise ValueError("WaveTrack camera folders mix raw and undistorted images")
    source_pixels = "undistorted" if manifest.get("xml_grid") else pixel_states.pop()

    bev_kind = wavetrack_bev.register(manifest) if len(manifest["cameras"]) == 5 and \
        {item["name"] for item in manifest["cameras"]} == {f"Cam{i}" for i in range(1, 6)} else None
    by_number = {}
    for item in manifest["cameras"]:
        number = camera_number(item["name"])
        if number in by_number:
            raise ValueError(f"Duplicate WaveTrack calibration camera: {item['name']}")
        by_number[number] = item
    if len(by_number) != len(folders) or {camera_number(f.name) for f in folders} != set(by_number):
        raise ValueError("WaveTrack camera folders do not match the named cameras in calibration.json.")
    cams, distortion = [], []
    for index, folder in enumerate(folders):
        item = by_number[camera_number(folder.name)]
        if dataset:
            width, height = dataset.size(folder)
        else:
            with Image.open(min(numbered_images(folder).items())[1]) as image:
                width, height = image.size
        cw, ch = map(float, item["image_size"])
        K = np.diag([width / cw, height / ch, 1.0]) @ np.asarray(item["camera_matrix"], dtype=float)
        cams.append(Camera(index, item["name"], K, np.asarray(item["R"], dtype=float),
                           np.asarray(item["tvec_m"], dtype=float).reshape(3), width, height, folder))
        source_dist = np.asarray(item.get("dist_coeffs", np.zeros(5)), dtype=float).ravel()
        distortion.append(np.zeros_like(source_dist) if source_pixels == "undistorted" else source_dist)
    roi = manifest.get("roi") or {}
    area = [float(roi[k]) for k in ("x0", "x1", "y0", "y1")] if all(
        k in roi for k in ("x0", "x1", "y0", "y1")) else None
    info = {"layout": "wavetrack", "cameras": len(cams), "unit": "m", "native_kind": bev_kind.key if bev_kind else None,
            "names": [c.name for c in cams], "distortion": any(np.any(np.abs(d) > 1e-9) for d in distortion),
            "path": manifest["path"], "calib_root": str(Path(manifest["path"]).parent.parent), "area": area,
            "source_pixels": source_pixels,
            "intrinsic_variant": "intrinsic_zero" if source_pixels == "undistorted" else "intrinsic_original"}
    if bev_kind:
        info["bev_grid"] = {"x": bev_kind.grid_x, "y": bev_kind.grid_y, "cell_m": wavetrack_bev.CELL_M}
    return cams, {"info": info, "dist": distortion}


def area_for(kind: str | None, cams: list[Camera]) -> list[float]:
    """[x0, x1, y0, y1] of the BEV map: the checkpoint grid, else the cameras' ground footprints."""
    area = grid_area(kind)
    if area:
        return area
    prints = scenes.footprints([{"K": c.K, "R": c.R, "t": c.t} for c in cams], [(c.width, c.height) for c in cams],
                               max_range=30.0)
    points = np.array([point for item in prints for point in item["polygon"]])
    low, high = np.percentile(points, 5, axis=0), np.percentile(points, 95, axis=0)
    return [float(low[0]), float(high[0]), float(low[1]), float(high[1])]


def grid_area(kind: str | None) -> list[float] | None:
    if hasattr(scenes, "KINDS"):
        area = scenes.KINDS.get(kind)
        return [*area["x"], *area["y"]] if area else None
    if kind is None:
        return None
    try:
        spec = scenes.kinds.get(kind)
    except (KeyError, ValueError):
        return None
    return [*spec.x_m, *spec.y_m]


def position_id(kind: str | None, x: float, y: float) -> int | None:
    if kind not in POS_WIDTH:
        return None
    x0, x1, y0, y1 = grid_area(kind)
    i, j = round((x - x0) / GRID_STEP), round((y - y0) / GRID_STEP)
    width, height = POS_WIDTH[kind], round((y1 - y0) / GRID_STEP)
    if not (0 <= i < width and 0 <= j < height):
        return None
    return int(i + j * width)


def position_metres(kind: str, position: int) -> tuple[float, float]:
    if hasattr(scenes, "position_metres"):
        return scenes.position_metres(kind, position)
    return scenes.POSITION_METRES[kind](position)


def read_ground_truth(root: Path, kind: str | None, frame: int, folder: Path | None = None) -> list[dict] | None:
    """Wildtrack-format positions of one frame, from ``folder`` (default ``<root>/annotations_positions``)."""
    path = (folder or root / "annotations_positions") / f"{frame:08d}.json"
    if kind not in POS_WIDTH or not path.is_file():
        return None
    people = []
    for person in json.loads(path.read_text(encoding="utf-8")):
        try:
            x, y = position_metres(kind, int(person["positionID"]))
        except (KeyError, TypeError, ValueError):
            continue
        boxes = {}
        for view in person.get("views", []):
            if view.get("xmin", -1) >= 0 and view.get("xmax", -1) > view["xmin"]:
                boxes[int(view["viewNum"])] = [view["xmin"], view["ymin"], view["xmax"], view["ymax"]]
        people.append({"id": int(person.get("personID", -1)), "x": x, "y": y, "boxes": boxes})
    return people


# ---------------------------------------------------------------------------------------------------- evaluation

def match_ground(a: list[tuple], b: list[tuple], radius: float = 0.5) -> list[tuple[int, int, float]]:
    if not a or not b:
        return []
    from scipy.optimize import linear_sum_assignment
    distance = np.hypot(np.array([p[0] for p in a])[:, None] - np.array([q[0] for q in b])[None],
                        np.array([p[1] for p in a])[:, None] - np.array([q[1] for q in b])[None])
    rows, cols = linear_sum_assignment(np.where(distance <= radius, distance, 1e6))
    return [(r, c, float(distance[r, c])) for r, c in zip(rows, cols) if distance[r, c] <= radius]


def score_frame(persons: list[dict], gt: list[dict], cams: list[Camera] | None = None,
                boxes: list[dict] | None = None) -> dict:
    """TP/FP/FN at 0.5 m (MODA), summed ground error of the matches, and IoU of label boxes with GT boxes."""
    pairs = match_ground([(p["x"], p["y"]) for p in persons], [(g["x"], g["y"]) for g in gt])
    result = {"tp": len(pairs), "fp": len(persons) - len(pairs), "fn": len(gt) - len(pairs), "gt": len(gt),
              "dist": sum(d for *_, d in pairs), "iou_sum": 0.0, "iou_n": 0}
    if boxes is not None and cams is not None:
        indices = {cam.index for cam in cams}
        for p, g, _ in pairs:
            for index, gt_box in gt[g]["boxes"].items():
                if index in indices and boxes[p].get(index):
                    result["iou_sum"] += iou(boxes[p][index], gt_box)
                    result["iou_n"] += 1
    return result


def summarise(rows: list[dict]) -> dict:
    total = {key: sum(row[key] for row in rows) for key in ("tp", "fp", "fn", "gt", "dist", "iou_sum", "iou_n")}
    gt, tp = total["gt"], total["tp"]
    return {"frames": len(rows), "tp": tp, "fp": total["fp"], "fn": total["fn"], "gt": gt,
            "moda": None if not gt else round(100 * (1 - (total["fp"] + total["fn"]) / gt), 2),
            "precision": None if not tp + total["fp"] else round(100 * tp / (tp + total["fp"]), 2),
            "recall": None if not gt else round(100 * tp / gt, 2),
            "error_cm": None if not tp else round(100 * total["dist"] / tp, 2),
            "box_iou": None if not total["iou_n"] else round(total["iou_sum"] / total["iou_n"], 3)}


# ---------------------------------------------------------------------------------------------------- detectors

class Detectors:
    """The BEV checkpoint (demo_multiview RunnerPool) and the YOLO person detector, loaded once."""

    def __init__(self):
        self.pool = None
        self.yolo, self.yolo_path = None, None
        self.lock = threading.Lock()

    def bev(self, spec: dict, kind: str, calib_root: Path, images: list, threshold: float) -> tuple[list, float]:
        from inference import RunnerPool
        runner_kind = kind
        wave_roi = None
        if kind == "wavetrack" and spec.get("backend") == "mvdetr" and spec.get("num_cam") in (6, 7):
            from PIL import Image
            views = int(spec["num_cam"])
            if len(images) != 5:
                raise ValueError("WaveTrack requires five real camera images before BEV padding")
            manifest = wavetrack_calibration(calib_root)
            runner_kind = wavetrack_bev.register(manifest, views).key
            wave_roi = manifest["roi"]
            images = [*images, *[Image.new("RGB", images[0].size, (0, 0, 0)) for _ in range(views-5)]]
        with self.lock:
            if self.pool is None:
                self.pool = RunnerPool(max_loaded=1, fp16=True)
            runner = self.pool.get(spec, runner_kind, calib_root)
        default = runner.threshold = float(spec.get("threshold", runner.threshold))
        runner.threshold = min(threshold, default)
        _, points = runner.predict(images)
        mapped = [(*scenes.cells_to_metres(kind, x, y), score) for x, y, score in points]
        if wave_roi is not None:
            mapped = [p for p in mapped if float(wave_roi["x0"]) <= p[0] <= float(wave_roi["x1"])
                      and float(wave_roi["y0"]) <= p[1] <= float(wave_roi["y1"])]
        return mapped, default

    def detect(self, weights: str, images: list[np.ndarray], params: Params) -> list[list]:
        import torch
        from ultralytics import YOLO
        use_cuda = torch.cuda.is_available()
        with self.lock:
            if self.yolo is None or self.yolo_path != weights:
                self.yolo, self.yolo_path = YOLO(weights), weights
        results = self.yolo.predict(images, classes=[0], conf=params.yolo_conf, imgsz=params.yolo_imgsz,
                                    device=0 if use_cuda else "cpu", half=use_cuda, verbose=False)
        out = []
        for result in results:
            boxes = result.boxes
            if boxes is None or not len(boxes):
                out.append([])
                continue
            rows = np.column_stack([boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()])
            out.append([[round(float(v), 1) for v in row[:4]] + [round(float(row[4]), 3)] for row in rows])
        return out


def yolo_weights() -> list[dict]:
    found = []
    for path in sorted(YOLO_DIR.glob("*.pt")):
        found.append({"name": path.stem, "path": str(path)})
    found.sort(key=lambda item: (item["name"] != "yolo26s", item["name"]))
    return found


# ---------------------------------------------------------------------------------------------------- export

def link_ids(frames: list[int], positions: dict[int, list[tuple]], step: int) -> dict[int, list[int]]:
    """Person IDs across the labelled frames: Hungarian matching of consecutive frames. ``step`` is the dataset's
    frame-number step (Wildtrack 5 = 0.5 s); the gate is 0.5 m + 0.75 m per step and frames more than 6 steps
    apart are not linked."""
    from scipy.optimize import linear_sum_assignment
    ids, next_id, previous = {}, 0, None
    for frame in frames:
        points = positions[frame]
        current = [-1] * len(points)
        if previous is not None and points:
            gap = (frame - previous[0]) / max(step, 1)
            old = previous[1]
            if old and gap <= 6:
                gate = 0.5 + 0.75 * gap
                distance = np.hypot(np.array([p[0] for p in points])[:, None] - np.array([q[0] for q in old])[None],
                                    np.array([p[1] for p in points])[:, None] - np.array([q[1] for q in old])[None])
                rows, cols = linear_sum_assignment(np.where(distance <= gate, distance, 1e6))
                for r, c in zip(rows, cols):
                    if distance[r, c] <= gate:
                        current[r] = ids[previous[0]][c]
        for k, value in enumerate(current):
            if value < 0:
                current[k] = next_id
                next_id += 1
        ids[frame] = current
        previous = (frame, points)
    return ids


def frame_boxes(cams: list[Camera], persons: list[dict], yolo: dict, params: Params) -> list[dict]:
    views = match_views(cams, persons, yolo, params)
    out = []
    for person, view in zip(persons, views):
        boxes = {}
        for index, item in view.items():
            box = label_box(item, person.get("override", {}).get(str(index)), yolo.get(str(index), []), params)
            if box is not None:
                boxes[index] = box
        for cam in cams:   # a box drawn by hand where the cylinder projects outside the image
            manual = person.get("override", {}).get(str(cam.index))
            if isinstance(manual, list) and cam.index not in view:
                boxes[cam.index] = label_box({}, manual, [], params)
        out.append(boxes)
    return out


def export_dataset(project, out_dir: Path, copy_images: bool = True, only_reviewed: bool = False) -> dict:
    """Wildtrack-layout dataset: annotations_positions/*.json (all rig cameras, -1 where unlabelled), YOLO txt per
    labelled camera, Image_subsets (copied), calibrations (copied), label_project.json and export_summary.json."""
    cams, params, kind = project.selected_cams(), project.params, project.kind
    frames = [f for f in project.frames if project.state[f]["status"] == "done" and
              (project.state[f].get("reviewed") or not only_reviewed)]
    if not frames:
        raise ValueError("No processed frames to export.")
    out_dir.mkdir(parents=True, exist_ok=True)
    from annotation_import import encode_position
    annotation_grid = project.config.get("annotation_grid")
    kept, skipped = {}, 0
    for frame in frames:
        persons = [p for p in project.state[frame]["persons"] if p.get("status") != "suggested" or p.get("verified")]
        inside = [p for p in persons if (encode_position(annotation_grid, p["x"], p["y"]) is not None if annotation_grid else
                   position_id(kind, p["x"], p["y"]) is not None if kind in POS_WIDTH else
                  in_area(project.area, p["x"], p["y"]) if project.calibration["layout"] == "wavetrack" else True)]
        skipped += len(persons) - len(inside)
        kept[frame] = inside
    ids = link_ids(frames, {f: [(p["x"], p["y"]) for p in kept[f]] for f in frames}, project.step)
    (out_dir / "annotations_positions").mkdir(exist_ok=True)
    rows = []
    for frame in frames:
        persons, yolo = kept[frame], project.state[frame]["yolo"]
        boxes = frame_boxes(project.active_cams(frame), persons, yolo, params)
        records = []
        for person, pid, box in zip(persons, ids[frame], boxes):
            views = []
            for cam in project.all_cams:
                b = box.get(cam.index)
                views.append({"viewNum": cam.index, "xmin": int(round(b[0])) if b else -1,
                              "ymin": int(round(b[1])) if b else -1, "xmax": int(round(b[2])) if b else -1,
                              "ymax": int(round(b[3])) if b else -1})
            record = {"personID": int(pid), "positionID": (encode_position(annotation_grid, person["x"], person["y"]) if annotation_grid else
                                                       position_id(kind, person["x"], person["y"])), "views": views}
            if annotation_grid or record["positionID"] is None:
                record["position_m"] = [person["x"], person["y"]]
            records.append(record)
        (out_dir / "annotations_positions" / f"{frame:08d}.json").write_text(json.dumps(records, indent=1))
        for cam in cams:
            folder = out_dir / "labels_yolo" / f"C{cam.index + 1}"
            folder.mkdir(parents=True, exist_ok=True)
            lines = [f"0 {(b[0] + b[2]) / 2 / cam.width:.6f} {(b[1] + b[3]) / 2 / cam.height:.6f} "
                     f"{(b[2] - b[0]) / cam.width:.6f} {(b[3] - b[1]) / cam.height:.6f}"
                     for b in (box.get(cam.index) for box in boxes) if b]
            (folder / f"{frame:08d}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        gt = project.ground_truth(frame)
        if gt is not None:
            rows.append(score_frame(persons, gt, cams, boxes))
    if copy_images:
        for cam in project.all_cams:
            folder = out_dir / "Image_subsets" / f"C{cam.index + 1}"
            folder.mkdir(parents=True, exist_ok=True)
            files = project.dataset.files.get(cam.folder.name, {})
            for frame in frames:
                source = files.get(frame)
                target = folder / f"{frame:08d}{source.suffix.lower() if source else '.jpg'}"
                if source is None or (project.undistort and np.abs(project.dist[cam.index]).max() > 1e-9):
                    import cv2
                    if not cv2.imwrite(str(target), project.undistorted(cam.index, frame)[:, :, ::-1]):
                        raise ValueError(f"Could not write exported image: {target}")
                elif not target.exists():
                    shutil.copy2(source, target)
    calib = project.calib_dir()
    if calib and not (out_dir / "calibrations").exists():
        shutil.copytree(calib, out_dir / "calibrations")
    if copy_images and project.calibration["layout"] == "wavetrack":
        export_wavetrack_calibration(project, out_dir / "calibrations")
    summary = {"frames": frames, "people": sum(len(kept[f]) for f in frames), "outside_grid": skipped,
               "reviewed": sum(1 for f in frames if project.state[f].get("reviewed")),
               "cameras": [cam.index + 1 for cam in cams], "params": params.as_dict(),
               "source": project.dataset.public(),
               "vs_ground_truth": summarise(rows) if rows else None, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out_dir / "export_summary.json").write_text(json.dumps(summary, indent=1))
    project.save(out_dir / "label_project.json")
    return {**summary, "path": str(out_dir)}


def export_wavetrack_calibration(project, folder: Path):
    """Export calibration in the pixels of exported frames, without applying distortion twice on reload."""
    import cv2
    manifest = wavetrack_calibration(Path(project.config["calib"]))
    manifest.pop("path", None)
    by_name = {item["name"]: item for item in manifest["cameras"]}
    for cam in project.all_cams:
        item = by_name[cam.name]
        dist = np.zeros_like(project.dist[cam.index]) if project.undistort else project.dist[cam.index]
        P = cam.K @ np.column_stack([cam.R, cam.t])
        item.update(camera_matrix=cam.K.tolist(), image_size=[cam.width, cam.height], dist_coeffs=dist.tolist(),
                    P=P.tolist(), H_ground_to_undistorted_image=P[:, [0, 1, 3]].tolist())
        for subdir in ("intrinsic_original", "intrinsic_zero"):
            (folder / subdir).mkdir(parents=True, exist_ok=True)
            storage = cv2.FileStorage(str(folder / subdir / f"intr_{cam.name}.xml"), cv2.FILE_STORAGE_WRITE)
            storage.write("image_width", cam.width)
            storage.write("image_height", cam.height)
            storage.write("camera_matrix", cam.K)
            storage.write("distortion_coefficients", dist if subdir == "intrinsic_original" else np.zeros_like(dist))
            storage.release()
    (folder / "calibration.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")

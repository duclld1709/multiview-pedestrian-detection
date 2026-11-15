"""Editable WaveTrack floor-plan overlay, separate from the model's BEV ROI."""
import hashlib
import json
import math
import uuid
from pathlib import Path

import numpy as np


MAP_DIR = Path(__file__).resolve().parent / "projects" / "site_maps"


def shared_path(calibration_path: str) -> Path:
    key = hashlib.sha256(str(Path(calibration_path).resolve()).lower().encode("utf-8")).hexdigest()[:20]
    return MAP_DIR / f"{key}.json"


def default_bounds(area: list[float], cams: list) -> list[float]:
    xs, ys = [area[0], area[1]], [area[2], area[3]]
    for cam in cams:
        center = -np.asarray(cam.R).T @ np.asarray(cam.t)
        if np.isfinite(center[:2]).all():
            xs.append(float(center[0]))
            ys.append(float(center[1]))
    return [math.floor((min(xs) - 1.5) * 2) / 2,
            math.ceil((max(xs) + 1.5) * 2) / 2,
            math.floor((min(ys) - 1.5) * 2) / 2,
            math.ceil((max(ys) + 1.5) * 2) / 2]


def _point(value, bounds, label):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} phải là tọa độ [X, Y] theo mét.")
    x, y = (_number(v, label) for v in value)
    if not all(map(math.isfinite, (x, y))) or not (bounds[0] <= x <= bounds[1] and bounds[2] <= y <= bounds[3]):
        raise ValueError(f"{label} nằm ngoài khung bản đồ.")
    return [round(x, 4), round(y, 4)]


def _number(value, label):
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} phải là số hữu hạn.") from error
    if not math.isfinite(number):
        raise ValueError(f"{label} phải là số hữu hạn.")
    return number


def normalize(value: dict | None, area: list[float], cams: list) -> dict:
    value = value or {}
    if not isinstance(value, dict):
        raise ValueError("Bản đồ phải là một object JSON.")
    raw_bounds = value.get("bounds") or default_bounds(area, cams)
    if not isinstance(raw_bounds, (list, tuple)) or len(raw_bounds) != 4:
        raise ValueError("Khung bản đồ cần [Xmin, Xmax, Ymin, Ymax].")
    bounds = [_number(v, "Khung bản đồ") for v in raw_bounds]
    if not (bounds[0] < bounds[1] and bounds[2] < bounds[3]):
        raise ValueError("Khung bản đồ không hợp lệ.")
    if bounds[1] - bounds[0] > 100 or bounds[3] - bounds[2] > 100:
        raise ValueError("Khung bản đồ tối đa 100 m mỗi chiều.")
    if bounds[0] > area[0] or bounds[1] < area[1] or bounds[2] > area[2] or bounds[3] < area[3]:
        raise ValueError("Khung bản đồ phải chứa toàn bộ ROI gán nhãn.")
    bounds = [round(v, 4) for v in bounds]
    flip_vertical = value.get("flip_vertical", False)
    if not isinstance(flip_vertical, bool):
        raise ValueError("Lật dọc sơ đồ phải là true hoặc false.")
    flip_horizontal = value.get("flip_horizontal", False)
    if not isinstance(flip_horizontal, bool):
        raise ValueError("Lật ngang sơ đồ phải là true hoặc false.")
    objects = value.get("objects", [])
    if not isinstance(objects, list) or len(objects) > 300:
        raise ValueError("Bản đồ chứa tối đa 300 vật thể.")
    clean, ids = [], set()
    for index, item in enumerate(objects):
        if not isinstance(item, dict):
            raise ValueError("Mỗi vật thể phải là một object JSON.")
        identifier = str(item.get("id", ""))
        if not identifier or len(identifier) > 64 or not all(c.isalnum() or c in "_-" for c in identifier) or identifier in ids:
            raise ValueError("ID vật thể trống, trùng hoặc không hợp lệ.")
        ids.add(identifier)
        kind = item.get("kind")
        if kind not in ("wall", "pillar"):
            raise ValueError("Vật thể phải là vách hoặc cột.")
        occludes = item.get("occludes", True)
        if not isinstance(occludes, bool):
            raise ValueError("Trạng thái chặn tầm nhìn phải là true hoặc false.")
        name = str(item.get("name") or ("Vách" if kind == "wall" else "Cột") + f" {index + 1}").strip()[:80]
        height = _number(item.get("height", 2.8), "Chiều cao vật thể")
        if not 0.1 <= height <= 10:
            raise ValueError("Chiều cao vật thể phải từ 0,1 đến 10 m.")
        if kind == "wall":
            points = item.get("points")
            if not isinstance(points, list) or not 2 <= len(points) <= 40:
                raise ValueError("Vách cần từ 2 đến 40 điểm.")
            points = [_point(point, bounds, "Điểm vách") for point in points]
            if all(math.dist(points[0], point) < 0.01 for point in points[1:]):
                raise ValueError("Vách cần hai điểm khác nhau.")
            width = _number(item.get("width", 0.12), "Bề dày vách")
            if not 0.02 <= width <= 2:
                raise ValueError("Bề dày vách phải từ 0,02 đến 2 m.")
            wall = {"id": identifier, "kind": kind, "name": name, "points": points,
                    "width": round(width, 4), "height": round(height, 4)}
            if not occludes:
                wall["occludes"] = False
            if item.get("role") == "door":
                wall["role"] = "door"
            if item.get("estimated") is True:
                wall["estimated"] = True
            clean.append(wall)
        else:
            center = _point(item.get("center"), bounds, "Tâm cột")
            radius = _number(item.get("radius", 0), "Bán kính cột")
            if not 0.02 <= radius <= 5:
                raise ValueError("Bán kính cột phải từ 0,02 đến 5 m.")
            pillar = {"id": identifier, "kind": kind, "name": name, "center": center,
                      "radius": round(radius, 4), "height": round(height, 4)}
            if not occludes:
                pillar["occludes"] = False
            clean.append(pillar)
    result = {"version": 1, "bounds": bounds, "flip_vertical": flip_vertical,
              "flip_horizontal": flip_horizontal, "objects": clean}
    regions = value.get("floor_regions", [])
    if not isinstance(regions, list) or len(regions) > 30:
        raise ValueError("Sơ đồ chứa tối đa 30 vùng sàn.")
    if regions:
        floors = []
        for region in regions:
            if not isinstance(region, dict) or not isinstance(region.get("points"), list) or not 3 <= len(region["points"]) <= 100:
                raise ValueError("Vùng sàn cần từ 3 đến 100 điểm biên.")
            points = [_point(p, bounds, "Biên sàn") for p in region["points"]]
            twice_area = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(points, points[1:] + points[:1]))
            if abs(twice_area) < 0.02:
                raise ValueError("Vùng sàn phải có diện tích lớn hơn 0,01 m².")
            floors.append({"name": str(region.get("name", "Vùng sàn"))[:80], "points": points})
        result["floor_regions"] = floors
    return result


def read(path: Path, fallback: dict | None, area: list[float], cams: list) -> dict:
    if path.is_file():
        try:
            return normalize(json.loads(path.read_text(encoding="utf-8")), area, cams)
        except (OSError, ValueError, TypeError):
            pass
    return normalize(fallback, area, cams)


def write(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)

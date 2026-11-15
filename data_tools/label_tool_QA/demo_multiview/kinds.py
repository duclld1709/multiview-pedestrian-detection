"""Scene families ("kinds") of the demo: one per ground grid a checkpoint can be trained on.

A kind is everything the demo needs to know about a scene family, in one place:

* how many cameras its checkpoints take, in which order (calibration file names C1..CN);
* its 2.5 cm ground grid: cells along world X and Y, origin and unit of the calibration (cm or m), and how
  positionID numbers the cells (``positionID = gx + grid_x * gy``);
* how its checkpoints lay out the BEV heatmap (``indexing``: "ij" rows = world X like Wildtrack, "xy" rows = world Y
  like MultiviewX), which the model's dataset class needs;
* how the BEV map is drawn (``transpose``: world Y across, used when the area is longer along Y);
* optionally the Unity project that simulates it live (``live``).

Built in: Wildtrack and MultiviewX. More come from ``kinds.json`` next to this file (Warehouse is there),
``--kinds-config`` files, or a dataset folder's ``dataset_info.json`` (the Unity recorders write one), so adding a
scene is a JSON entry, not code. Everything is in metres outside this module except the grid cells.
"""

from __future__ import annotations

import copy
import json
import re
import threading
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CELL_M = 0.025                       # every kind uses a 2.5 cm ground grid (the demo's 40 cells per metre)
UNITS = {"m": 1.0, "cm": 100.0, "mm": 1000.0}
IMAGE_SHAPE = (1080, 1920)


def natural_key(text: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


@dataclass
class Kind:
    key: str
    name: str
    cameras: int
    grid_x: int                      # cells along world X (positionID % grid_x)
    grid_y: int                      # cells along world Y (positionID // grid_x)
    origin: tuple = (0.0, 0.0)       # world (X, Y) of cell (0, 0), calibration unit
    unit: str = "m"                  # unit of the calibration translations and the origin
    indexing: str = "xy"             # heatmap rows: "xy" -> world Y (MultiviewX), "ij" -> world X (Wildtrack)
    intrinsic: str = "intrinsic"     # calibrations/<intrinsic>/intr_<name>.xml
    names: list = field(default_factory=list)   # calibration names of C1..CN (extr_<name>.xml)
    extrinsic: str = "opencv"        # how extr_*.xml stores rvec / tvec: "opencv" matrices or "text" (Wildtrack)
    transpose: bool | None = None    # draw world Y across the map (default: when the area is longer along Y)
    default_frame: int = 0
    aliases: list = field(default_factory=list)   # other names checkpoints / logs use for this dataset
    datasets: list = field(default_factory=list)  # dataset folders opened at start-up when they exist
    live: dict | None = None         # Unity live simulation of this scene (see kinds.json)
    builtin: bool = False
    source: str = ""

    def __post_init__(self):
        self.origin = tuple(float(value) for value in self.origin)
        if self.unit not in UNITS:
            raise ValueError(f"{self.key}: unit must be one of {sorted(UNITS)}")
        if self.indexing not in ("ij", "xy"):
            raise ValueError(f"{self.key}: indexing must be 'ij' or 'xy'")
        if not self.names:
            self.names = [f"Camera{index}" for index in range(1, self.cameras + 1)]
        if len(self.names) != self.cameras:
            raise ValueError(f"{self.key}: {len(self.names)} calibration names for {self.cameras} cameras")
        if self.transpose is None:
            self.transpose = self.grid_y > self.grid_x

    # ------------------------------------------------------------------ geometry

    @property
    def unit_per_m(self) -> float:
        return UNITS[self.unit]

    @property
    def cell(self) -> float:
        """Grid cell in calibration units."""
        return CELL_M * self.unit_per_m

    @property
    def x_m(self) -> tuple[float, float]:
        x0 = self.origin[0] / self.unit_per_m
        return x0, x0 + self.grid_x * CELL_M

    @property
    def y_m(self) -> tuple[float, float]:
        y0 = self.origin[1] / self.unit_per_m
        return y0, y0 + self.grid_y * CELL_M

    @property
    def map_size(self) -> tuple[int, int]:
        """(width, height) of the BEV map in cells, as map_point lays it out."""
        return (self.grid_y, self.grid_x) if self.transpose else (self.grid_x, self.grid_y)

    def cells_to_metres(self, x: float, y: float) -> tuple[float, float]:
        return self.x_m[0] + x * CELL_M, self.y_m[0] + y * CELL_M

    def metres_to_cells(self, x: float, y: float) -> tuple[float, float]:
        return (x - self.x_m[0]) / CELL_M, (y - self.y_m[0]) / CELL_M

    def position_cells(self, position: int) -> tuple[int, int]:
        return position % self.grid_x, position // self.grid_x

    def position_metres(self, position: int) -> tuple[float, float]:
        return self.cells_to_metres(*self.position_cells(position))

    def worldgrid_shape(self) -> list[int]:
        """[rows, cols] of the checkpoints' full-resolution BEV."""
        return [self.grid_x, self.grid_y] if self.indexing == "ij" else [self.grid_y, self.grid_x]

    # ------------------------------------------------------------------ calibration files

    def intrinsic_file(self, root: Path, index: int) -> Path:
        return Path(root) / "calibrations" / self.intrinsic / f"intr_{self.names[index]}.xml"

    def extrinsic_file(self, root: Path, index: int) -> Path:
        return Path(root) / "calibrations" / "extrinsic" / f"extr_{self.names[index]}.xml"

    def has_calibration(self, root: Path) -> bool:
        return all(self.intrinsic_file(root, i).is_file() and self.extrinsic_file(root, i).is_file()
                   for i in range(self.cameras))

    def read_calibration(self, root: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """K [3x3] and [R | t] [3x4] per camera in calibration units, as the dataset classes of the trainers load
        them (what the checkpoints' projection layers are built from)."""
        import cv2
        intrinsics, extrinsics = [], []
        for index in range(self.cameras):
            storage = cv2.FileStorage(str(self.intrinsic_file(root, index)), cv2.FILE_STORAGE_READ)
            K = storage.getNode("camera_matrix").mat()
            storage.release()
            rvec, tvec = read_extrinsic(self.extrinsic_file(root, index))
            R = cv2.Rodrigues(rvec)[0]
            intrinsics.append(np.asarray(K, dtype=float))
            extrinsics.append(np.hstack([R, tvec.reshape(3, 1)]))
        return intrinsics, extrinsics

    def write_calibration(self, cams_m: list[dict], root: Path):
        """Cameras {K, R, t (metres)} -> root/calibrations in this kind's layout (what its dataset class reads)."""
        import cv2
        intr_dir, extr_dir = Path(root) / "calibrations" / self.intrinsic, Path(root) / "calibrations" / "extrinsic"
        intr_dir.mkdir(parents=True, exist_ok=True)
        extr_dir.mkdir(parents=True, exist_ok=True)
        for index, cam in enumerate(cams_m):
            storage = cv2.FileStorage(str(self.intrinsic_file(root, index)), cv2.FILE_STORAGE_WRITE)
            storage.write("camera_matrix", np.asarray(cam["K"], np.float64))
            storage.write("distortion_coefficients", np.zeros((1, 5)))
            storage.release()
            rvec = cv2.Rodrigues(np.asarray(cam["R"], np.float64))[0]
            tvec = np.asarray(cam["t"], np.float64).reshape(3, 1) * self.unit_per_m
            path = self.extrinsic_file(root, index)
            if self.extrinsic == "text":
                path.write_text('<?xml version="1.0"?>\n<opencv_storage>\n'
                                f"<rvec>{' '.join(f'{v:.10f}' for v in rvec.ravel())}</rvec>\n"
                                f"<tvec>{' '.join(f'{v:.10f}' for v in tvec.ravel())}</tvec>\n</opencv_storage>\n")
            else:
                storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
                storage.write("rvec", rvec)
                storage.write("tvec", tvec)
                storage.release()

    # ------------------------------------------------------------------ export

    def public(self) -> dict:
        """What the browser needs to draw and click the map."""
        live = None
        if self.live:
            live = {"label": self.live.get("label") or self.name, "scene": self.live.get("scene")}
        return {"key": self.key, "name": self.name, "cameras": self.cameras, "grid_x": self.grid_x,
                "grid_y": self.grid_y, "x_m": list(self.x_m), "y_m": list(self.y_m), "transpose": self.transpose,
                "map": list(self.map_size), "builtin": self.builtin, "live": live}

    def signature(self) -> tuple:
        return (self.cameras, self.grid_x, self.grid_y, round(self.x_m[0], 3), round(self.y_m[0], 3))


def read_extrinsic(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """rvec, tvec of an extr_*.xml written as OpenCV matrices (MultiviewX) or as plain text nodes (Wildtrack)."""
    rvec = tvec = None
    try:
        import cv2
        storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
        try:
            rvec, tvec = storage.getNode("rvec").mat(), storage.getNode("tvec").mat()
        finally:
            storage.release()
    except Exception:   # Wildtrack writes plain-text nodes that OpenCV does not parse as matrices
        rvec = tvec = None
    if rvec is None or tvec is None:
        root = ET.parse(path).getroot()
        rvec = np.array(root.find("rvec").text.split(), dtype=float)
        tvec = np.array(root.find("tvec").text.split(), dtype=float)
    return np.asarray(rvec, dtype=float).reshape(3), np.asarray(tvec, dtype=float).reshape(3)


# ---------------------------------------------------------------------------------------------------- registry

WILDTRACK_NAMES = ["CVLab1", "CVLab2", "CVLab3", "CVLab4", "IDIAP1", "IDIAP2", "IDIAP3"]
BUILTIN = [
    Kind("wildtrack", "Wildtrack", 7, 480, 1440, (-300.0, -900.0), "cm", "ij", "intrinsic_zero", WILDTRACK_NAMES,
         "text", True, 1800, builtin=True, source="built-in"),
    Kind("multiviewx", "MultiviewX", 6, 1000, 640, (0.0, 0.0), "m", "xy", "intrinsic", [], "opencv", False, 360,
         builtin=True, source="built-in",
         live={"label": "PersonX · MultiviewX rig (Unity 2018)", "unity_axes": [[1, 0, 0], [0, 0, -1], [0, 1, 0]]}),
]

_LOCK = threading.RLock()
KINDS: dict[str, Kind] = {}

FIELDS = {"key", "name", "cameras", "grid", "calibration", "transpose", "default_frame", "aliases", "datasets", "live"}


def from_config(item: dict, source: str = "") -> Kind:
    """One entry of kinds.json (see the file for the fields)."""
    unknown = set(item) - FIELDS - {"comment", "_comment"}
    if unknown:
        raise ValueError(f"kind {item.get('key')}: unknown fields {sorted(unknown)}")
    grid, calibration = item.get("grid", {}), item.get("calibration", {})
    cell = float(grid.get("cell", 2.5 if grid.get("unit") == "cm" else 0.025))
    unit = grid.get("unit", "m")
    if abs(cell / UNITS[unit] - CELL_M) > 1e-9:
        raise ValueError(f"kind {item.get('key')}: the demo needs a 2.5 cm grid cell (got {cell} {unit})")
    return Kind(key=slug(item["key"]), name=item.get("name", item["key"]), cameras=int(item["cameras"]),
                grid_x=int(grid["x"]), grid_y=int(grid["y"]), origin=tuple(grid.get("origin", (0.0, 0.0))),
                unit=unit, indexing=grid.get("indexing", "xy"), intrinsic=calibration.get("intrinsic", "intrinsic"),
                names=list(calibration.get("names") or []), extrinsic=calibration.get("extrinsic", "opencv"),
                transpose=item.get("transpose"), default_frame=int(item.get("default_frame", 0)),
                aliases=[alias.lower() for alias in item.get("aliases", [])],
                datasets=list(item.get("datasets", [])), live=item.get("live"), source=source)


def register(kind: Kind, replace: bool = False) -> Kind:
    with _LOCK:
        if kind.key in KINDS and not replace:
            return KINDS[kind.key]
        KINDS[kind.key] = kind
        return kind


def load_config(path: str | Path):
    path = Path(path)
    items = json.loads(path.read_text(encoding="utf-8"))
    items = items.get("kinds", items) if isinstance(items, dict) else items
    if not isinstance(items, list):
        raise ValueError(f"{path}: expected a list of kinds")
    for item in items:
        kind = from_config(item, str(path))
        if kind.key in KINDS and KINDS[kind.key].builtin:   # extend a built-in (live, datasets, aliases) only
            base = KINDS[kind.key]
            base.datasets += [d for d in kind.datasets if d not in base.datasets]
            base.aliases += [a for a in kind.aliases if a not in base.aliases]
            if kind.live:
                base.live = {**(base.live or {}), **kind.live}
            continue
        register(kind, replace=True)


def get(key: str) -> Kind:
    try:
        return KINDS[key]
    except KeyError:
        raise ValueError(f"Unknown scene kind '{key}' (known: {', '.join(KINDS)})") from None


def all_kinds() -> list[Kind]:
    return list(KINDS.values())


def by_cameras(count: int) -> list[Kind]:
    return [kind for kind in KINDS.values() if kind.cameras == count]


def match_text(text: str, cameras: int | None = None) -> str | None:
    """Kind named in a log / path / manifest: the longest key or alias found ("wildtrack_warehouse" is the
    warehouse, not Wildtrack)."""
    lowered = text.lower()
    best, length = None, 0
    for kind in KINDS.values():
        if cameras is not None and kind.cameras != cameras:
            continue
        for name in [kind.key, *kind.aliases]:
            if name and name in lowered and len(name) > length:
                best, length = kind.key, len(name)
    return best


def dataset_info(root: Path) -> dict | None:
    file = Path(root) / "dataset_info.json"
    if not file.is_file():
        return None
    try:
        info = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return info if isinstance(info, dict) and "worldgrid_shape" in info and "num_cam" in info else None


def from_dataset_info(root: Path, info: dict) -> Kind:
    """A kind described by a recorder's dataset_info.json (worldgrid_shape [H, W], positionID = gx + W * gy,
    grid_origin_cm, grid_cell_cm)."""
    rows, cols = (int(v) for v in info["worldgrid_shape"])
    unit = "cm" if "grid_cell_cm" in info or "grid_origin_cm" in info else info.get("unit", "m")
    origin = info.get("grid_origin_cm", info.get("grid_origin", (0.0, 0.0)))
    calibration = Path(root) / "calibrations"
    names = sorted((file.stem[len("extr_"):] for file in (calibration / "extrinsic").glob("extr_*.xml")),
                   key=natural_key)
    intrinsic = next((name for name in ("intrinsic_zero", "intrinsic", "intrinsic_original")
                      if (calibration / name).is_dir()), "intrinsic")
    sample = calibration / "extrinsic" / f"extr_{names[0]}.xml" if names else None
    text = sample is not None and "type_id" not in sample.read_text(encoding="utf-8", errors="replace")
    key = slug(info.get("name") or Path(root).name)
    return Kind(key=key, name=info.get("name") or Path(root).name, cameras=int(info["num_cam"]), grid_x=cols,
                grid_y=rows, origin=tuple(origin), unit=unit, indexing=info.get("indexing", "xy"),
                intrinsic=intrinsic, names=names[:int(info["num_cam"])], extrinsic="text" if text else "opencv",
                default_frame=0, source=str(Path(root) / "dataset_info.json"))


def detect(root: Path, register_new: bool = True) -> str | None:
    """Kind of a dataset folder: its dataset_info.json (matched to a registered kind by grid, else registered as a
    new kind), else the calibration file names + camera count (Wildtrack CVLab/IDIAP, MultiviewX Camera1..6)."""
    root = Path(root)
    info = dataset_info(root)
    if info is not None:
        candidate = from_dataset_info(root, info)
        same = [kind for kind in KINDS.values() if kind.signature() == candidate.signature()]
        if same:
            named = match_text(str(root) + " " + json.dumps(info), candidate.cameras)
            return named if named in {kind.key for kind in same} else same[0].key
        if not register_new:
            return None
        while candidate.key in KINDS:
            candidate.key += "-2"
        register(candidate)
        return candidate.key
    count = len(list((root / "calibrations" / "extrinsic").glob("extr_*.xml")))
    return next((kind.key for kind in KINDS.values() if kind.cameras == count and kind.has_calibration(root)), None)


def reset(config_paths: list[str | Path] | None = None):
    """Built-ins, kinds.json next to this file, then extra config files."""
    with _LOCK:
        KINDS.clear()
        for kind in BUILTIN:
            KINDS[kind.key] = copy.deepcopy(kind)
        default = HERE / "kinds.json"
        for path in [default] + [Path(p) for p in (config_paths or [])]:
            if path.is_file():
                load_config(path)


reset()

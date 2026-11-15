"""A labeling project: dataset + calibration + chosen cameras and frames, the auto-label job and the review state.

Everything is kept in ``<out_dir>/label_project.json`` (rewritten after each change), so a project can be closed and
reopened, and the export reads the same state the browser shows.
"""

from __future__ import annotations

import json
import random
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from pathlib import Path

import numpy as np

import labeling as lb
import site_map
from model_registry import model_catalog, describe_model

def probe(root: str, calib: str | None = None, videos: dict | None = None, timeline_fps: float = 2.0) -> dict:
    """What the setup dialog needs to know about a dataset folder before a project is created."""
    root_path = Path(root)
    if not root_path.is_dir():
        raise ValueError(f"Folder not found: {root}")
    dataset = lb.Dataset(root_path, videos, timeline_fps)
    folders = dataset.folders
    calib_path = Path(calib) if calib else lb.find_calibration(root_path)
    if calib_path is None:
        raise ValueError("No calibration found next to the camera sources; pick the calibration folder.")
    cams, extra = lb.load_cameras(calib_path, folders, dataset)
    frames = dataset.frames
    kind = extra["info"]["native_kind"]
    has_gt = (root_path / "annotations_positions").is_dir() and kind in lb.POS_WIDTH
    return {"root": str(root_path), "calib": str(calib_path), "kind": kind, "calibration": extra["info"],
            "source": dataset.public(),
            "cameras": [{"index": c.index, "name": c.name, "folder": c.folder.name, "width": c.width,
                         "height": c.height} for c in cams],
            "frames": {"count": len(frames), "first": frames[0] if frames else None,
                       "last": frames[-1] if frames else None, "step": int(np.median(np.diff(frames))) if len(frames) > 1 else 1},
            "ground_truth": has_gt, "gt_supported": kind in lb.POS_WIDTH,
            "gt_dir": str(root_path / "annotations_positions") if has_gt else ""}


def manual_box(value) -> list[float] | None:
    """A per-camera box drawn or resized by hand: [x1, y1, x2, y2] in pixels of the (undistorted) image."""
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = (round(float(v), 1) for v in value)
    except (TypeError, ValueError):
        return None
    return [x1, y1, x2, y2] if x2 > x1 and y2 > y1 else None


def pick_frames(available: list[int], count: int, strategy: str, start: int | None, end: int | None,
                seed: int = 0) -> list[int]:
    if strategy == "dataset":
        start = end = None
    pool = [f for f in available if (start is None or f >= start) and (end is None or f <= end)]
    if not pool:
        raise ValueError("No frames in the chosen range.")
    if strategy in ("all", "dataset"):
        return pool
    count = max(1, min(count, len(pool)))
    if strategy == "first":
        return pool[:count]
    if strategy == "random":
        return sorted(random.Random(seed).sample(pool, count))
    picks = np.linspace(0, len(pool) - 1, count).round().astype(int)
    return sorted({pool[i] for i in picks})


class Project:
    def __init__(self, config: dict, models: dict[str, dict], detectors: lb.Detectors):
        self.config, self.models, self.detectors = config, models, detectors
        self.config.setdefault("image_processing_version", 2)
        self.root = Path(config["root"])
        self.out_dir = Path(config["out_dir"])
        self.dataset = lb.Dataset(self.root, config.get("videos"), config.get("timeline_fps", 2.0))
        self.config.update(self.dataset.settings())
        self.all_cams, extra = lb.load_cameras(Path(config["calib"]), self.dataset.folders, self.dataset)
        self.dist, self.calibration = extra["dist"], extra["info"]
        self.kind = self.calibration["native_kind"]
        self.cameras = [i for i in config["cameras"] if 0 <= i < len(self.all_cams)] or \
            [c.index for c in self.all_cams]
        self.params = lb.Params.from_dict(config.get("params"))
        self.frames = list(config["frames"])
        numbers = sorted(self.dataset.files[self.all_cams[0].folder.name]) if self.dataset.mode == "images" else []
        self.step = int(np.median(np.diff(numbers))) if len(numbers) > 1 else 1   # dataset frame-number step
        self.undistort = bool(config.get("undistort", True)) and self.calibration["distortion"]
        self.pixel_alignment_warning = None
        self.area = self.calibration.get("area") or lb.area_for(self.kind, self.all_cams)
        self.site_map_path = (Path(config["site_map_file"]) if config.get("site_map_file") else
                              site_map.shared_path(self.calibration["path"])) if self.kind == "wavetrack" else None
        self.site_map = site_map.read(self.site_map_path, config.get("site_map"), self.area, self.all_cams) \
            if self.site_map_path else None
        self.state = {f: {"status": "pending", "persons": [], "yolo": {}, "reviewed": False} for f in self.frames}
        self.lock = threading.RLock()
        self.job = {"running": False, "done": 0, "total": len(self.frames), "error": None, "frame": None,
                    "started": None, "finished": None}
        self._images = OrderedDict()
        # Browser cache entries must never be reused across different sources/projects.
        self.image_version = uuid.uuid4().hex
        self._undistort_maps = {}
        self._save_timer = None
        self._gt: dict[int, list[dict] | None] = {}
        self._model_runtime = {engine: {"state": "ready", "last_frame": None, "outputs": None,
                                        "ms": None, "error": None} for engine in ("bev", "yolo")}

    # ------------------------------------------------------------------------------------------------ setup

    def selected_cams(self) -> list[lb.Camera]:
        return [self.all_cams[i] for i in self.cameras]

    def active_cams(self, frame: int) -> list[lb.Camera]:
        return [cam for cam in self.selected_cams() if self.dataset.active(cam.folder, frame)]

    def model_catalog(self) -> list[dict]:
        return model_catalog(self.models, self.kind, len(self.all_cams), self.calibration["layout"])

    def _validate_model(self, model_id):
        if not model_id:
            return None
        spec = self.models.get(model_id)
        if spec is None:
            raise ValueError(f"Model {model_id} không còn trong danh sách đã đăng ký.")
        item = describe_model(spec, self.kind, len(self.all_cams), self.calibration["layout"])
        if not item["compatible"]:
            raise ValueError(f"{item['method']}: " + " ".join(item["reasons"]))
        return spec

    def model_spec(self, strict=False) -> dict | None:
        try:
            return self._validate_model(self.config.get("model_id"))
        except ValueError:
            if strict:
                raise
            return None

    def set_model(self, model_id):
        with self.lock:
            if self.job["running"]:
                raise ValueError("Chờ auto-label hoàn tất trước khi đổi model.")
            self._validate_model(model_id)
            if self.config.get("model_id") == model_id:
                return
            self.config["model_id"] = model_id
            self._model_runtime["bev"] = {"state": "ready", "last_frame": None, "outputs": None,
                                          "ms": None, "error": None}
        self.save()

    def yolo_weights(self):
        available = lb.yolo_weights()
        return self.config.get("yolo_weights") or (available[0]["path"] if available else str(lb.HERE / "yolo26s.pt"))

    def model_status(self):
        catalog = self.model_catalog()
        selected = self.config.get("model_id")
        item = next((row for row in catalog if row["id"] == selected), None)
        with self.lock:
            bev, yolo = (dict(self._model_runtime[k]) for k in ("bev", "yolo"))
        bev.update(model_id=selected, model=item)
        if not selected:
            bev.update(state="not_selected", message="Chưa chọn model BEV; auto-label dùng YOLO đa view.")
        elif item is None or not item["compatible"]:
            bev.update(state="blocked", message=" ".join(item["reasons"]) if item else
                       "Model đã chọn không còn trong danh sách đăng ký.")
        else:
            bev["message"] = bev["error"] or {"ready": "Tương thích; weight sẽ được nạp khi auto-label.",
                "running": "Đang nạp/chạy model BEV.", "loaded": "Model đã chạy thành công trong phiên này."}.get(bev["state"], "")
            if item["experimental"]:
                bev["message"] = f"{bev['message']} {item['warning']}".strip()
        yolo.update(weights=self.yolo_weights(), message=yolo["error"] or {
            "ready": "Sẽ nạp khi auto-label; kết quả đã lưu vẫn có thể xem.",
            "running": "Đang nạp/chạy YOLO.", "loaded": "YOLO đã chạy thành công trong phiên này."}.get(yolo["state"], ""))
        return {"registered": len(catalog), "compatible": sum(row["compatible"] for row in catalog),
                "bev": bev, "yolo": yolo}

    def _model_call(self, engine, frame, callback):
        began = time.perf_counter()
        with self.lock:
            self._model_runtime[engine].update(state="running", last_frame=frame, outputs=None, error=None)
        try:
            result = callback()
        except Exception as error:
            with self.lock:
                self._model_runtime[engine].update(state="error", error=f"{type(error).__name__}: {error}",
                                                   ms=round(1000*(time.perf_counter()-began)))
            raise
        count = len(result[0]) if engine == "bev" else sum(len(rows) for rows in result)
        with self.lock:
            self._model_runtime[engine].update(state="loaded", outputs=count,
                                               ms=round(1000*(time.perf_counter()-began)))
        return result

    def model_calib_root(self, spec: dict) -> Path:
        # A checkpoint trained on estimated cameras keeps them on its own dataset (as the demo does).
        if spec.get("calib_root") and self.calibration["layout"] == spec["dataset"]:
            return Path(spec["calib_root"])
        return Path(self.calibration["calib_root"])

    def calib_dir(self) -> Path | None:
        path = Path(self.config["calib"])
        if self.calibration["layout"] == "wavetrack":
            return Path(self.calibration["path"]).parent
        return path if path.is_dir() else None

    def gt_dir(self) -> Path:
        return Path(self.config.get("gt_dir") or self.root / "annotations_positions")

    def gt_supported(self) -> bool:
        return self.kind in lb.POS_WIDTH   # positionID can only be decoded on a known BEV grid

    def ground_truth(self, frame: int) -> list[dict] | None:
        if frame not in self._gt:
            self._gt[frame] = lb.read_ground_truth(self.root, self.kind, frame, self.gt_dir())
        return self._gt[frame]

    def set_gt_dir(self, path: str | None) -> dict:
        """Point the comparison at another annotations_positions folder (empty: the dataset's own)."""
        folder = Path(path) if path else self.root / "annotations_positions"
        if not self.gt_supported():
            raise ValueError(f"Comparing needs a Wildtrack/MultiviewX grid to decode positionID (dataset is {self.kind}).")
        if (folder / "annotations_positions").is_dir():   # the dataset folder was picked instead of its annotations
            folder = folder / "annotations_positions"
        if not folder.is_dir():
            raise ValueError(f"Folder not found: {folder}")
        if not any(folder.glob("*.json")):
            raise ValueError(f"No *.json annotation files in {folder}.")
        with self.lock:
            self.config["gt_dir"] = str(folder) if path else None
            self._gt.clear()
        self.schedule_save()
        return self.public()

    def gt_diff(self, frame: int) -> dict | None:
        """Current labels against the original ones of one frame: TP/FP/FN at 0.5 m and the mean ground error."""
        gt = self.ground_truth(frame)
        state = self.state[frame]
        if gt is None or state["status"] != "done":
            return None
        current = [p for p in state["persons"] if p.get("status") != "suggested" or p.get("verified")]
        row = lb.score_frame(current, gt)
        return {"tp": row["tp"], "fp": row["fp"], "fn": row["fn"], "gt": row["gt"],
                "error_cm": round(100 * row["dist"] / row["tp"], 1) if row["tp"] else None}

    # ------------------------------------------------------------------------------------------------ images

    def undistorted(self, camera: int, frame: int) -> np.ndarray:
        """RGB frame as the labels see it (undistorted when the calibration has distortion)."""
        import cv2
        image = self.dataset.read(self.all_cams[camera].folder, frame)[:, :, ::-1]
        if self.undistort and np.abs(self.dist[camera]).max() > 1e-9:
            cam = self.all_cams[camera]
            if camera not in self._undistort_maps:
                self._undistort_maps[camera] = cv2.initUndistortRectifyMap(
                    cam.K, self.dist[camera], None, cam.K, (cam.width, cam.height), cv2.CV_16SC2)
            image = cv2.remap(image, *self._undistort_maps[camera], cv2.INTER_LINEAR)
        return np.ascontiguousarray(image)

    def _path(self, camera: int, frame: int) -> Path:
        key = ("files", camera)
        files = self._images.get(key)
        if files is None:
            files = self._images[key] = lb.numbered_images(self.all_cams[camera].folder)
        return files[frame]

    def jpeg(self, camera: int, frame: int, width: int | None = None) -> bytes:
        import cv2
        key = (camera, frame, width)
        with self.lock:
            if key in self._images:
                self._images.move_to_end(key)
                return self._images[key]
        image = self.undistorted(camera, frame)
        if width and width < image.shape[1]:
            image = cv2.resize(image, (width, round(image.shape[0] * width / image.shape[1])),
                               interpolation=cv2.INTER_AREA)
        ok, data = cv2.imencode(".jpg", image[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
        data = data.tobytes()
        with self.lock:
            self._images[key] = data
            while len(self._images) > 120:
                self._images.popitem(last=False)
        return data

    # ------------------------------------------------------------------------------------------------ auto-label

    def start(self, frames: list[int] | None = None) -> dict:
        with self.lock:
            if self.job["running"]:
                return self.job_state()
            self.model_spec(strict=True)
            todo = frames or [f for f in self.frames if self.state[f]["status"] != "done"]
            self.job.update(running=True, done=0, total=len(todo), error=None, started=time.time(), finished=None)
        threading.Thread(target=self._run, args=(todo,), daemon=True).start()
        return self.job_state()

    def _run(self, todo: list[int]):
        try:
            spec = self.model_spec(strict=True)
            weights = self.yolo_weights()
            for frame in todo:
                with self.lock:
                    self.job["frame"] = frame
                    self.state[frame]["status"] = "running"
                self.state[frame].update(self.auto_label(frame, spec, weights))
                with self.lock:
                    self.state[frame]["status"] = "done"
                    self.job["done"] += 1
                self.schedule_save()
        except Exception as error:   # surface the error in the browser instead of killing the thread silently
            traceback.print_exc()
            with self.lock:
                self.job["error"] = f"{type(error).__name__}: {error}"
                for frame in todo:
                    if self.state[frame]["status"] == "running":
                        self.state[frame]["status"] = "error"
        finally:
            with self.lock:
                self.job.update(running=False, frame=None, finished=time.time())
            self.save()

    def auto_label(self, frame: int, spec: dict | None, weights: str) -> dict:
        started = time.perf_counter()
        images = [self.undistorted(cam.index, frame) for cam in self.all_cams]
        points, threshold = [], 1.0
        if spec:
            from PIL import Image
            low = min(self.params.low_score, float(spec.get("threshold", 0.5)))
            points, threshold = self._model_call("bev", frame, lambda: self.detectors.bev(
                spec, self.kind, self.model_calib_root(spec), [Image.fromarray(image) for image in images], low))
        cams = self.active_cams(frame)
        detections = self._model_call("yolo", frame, lambda: self.detectors.detect(
            weights, [images[cam.index] for cam in cams], self.params)) if cams else []
        yolo = {str(cam.index): [] for cam in self.selected_cams()}
        yolo.update({str(cam.index): rows for cam, rows in zip(cams, detections)})
        persons = lb.fuse(cams, points, yolo, self.params, threshold, self.area)
        model_only = [{"x": x, "y": y} for x, y, s in points if s >= threshold]
        return {"persons": persons, "yolo": yolo, "reviewed": False,
                "auto": {"model": model_only, "fused": [{"x": p["x"], "y": p["y"], "status": p["status"]}
                                                        for p in persons],
                         "threshold": threshold, "model_id": spec["id"] if spec else None, "yolo_weights": weights,
                         "experimental_black_views": int(spec["num_cam"])-5 if spec and
                         self.kind == "wavetrack" and spec.get("backend") == "mvdetr" and
                         spec.get("num_cam") in (6, 7) else 0,
                         "weight_dataset": spec.get("dataset") if spec else None,
                         "ms": round(1000 * (time.perf_counter() - started))}}

    def job_state(self) -> dict:
        with self.lock:
            job = dict(self.job)
        job["eta_s"] = None
        if job["running"] and job["done"] and job["started"]:
            job["eta_s"] = round((time.time() - job["started"]) / job["done"] * (job["total"] - job["done"]))
        return job

    # ------------------------------------------------------------------------------------------------ review

    def frame_payload(self, frame: int) -> dict:
        with self.lock:
            state = json.loads(json.dumps(self.state[frame]))
        gt = self.ground_truth(frame)
        return {"frame": frame, **state, "gt": gt,
                "sources": {str(cam.index): self.dataset.frame_source(cam.folder, frame) for cam in self.all_cams},
                "inactive_cameras": [cam.index for cam in self.all_cams if not self.dataset.active(cam.folder, frame)]}

    def update_frame(self, frame: int, body: dict) -> dict:
        persons = []
        for person in body.get("persons", []):
            persons.append({"uid": str(person["uid"]), "x": float(person["x"]), "y": float(person["y"]),
                            "h": float(person.get("h", self.params.height)),
                             **({"r": float(person["r"])} if "r" in person else {}), "score": person.get("score"),
                            "origin": person.get("origin", "manual"), "status": person.get("status", "manual"),
                            "verified": bool(person.get("verified")), "auto": person.get("auto"),
                            "support": person.get("support"),
                            "override": {str(k): manual_box(v) or v for k, v in (person.get("override") or {}).items()
                                         if manual_box(v) or v in ("proj", "yolo", "off")}})
        with self.lock:
            state = self.state[frame]
            if state["status"] != "done":
                raise ValueError("This frame has not been auto-labelled yet.")
            state["persons"] = persons
            if "reviewed" in body:
                state["reviewed"] = bool(body["reviewed"])
        self.schedule_save()
        return self.summary(frame)

    def snap(self, frame: int, person: dict) -> dict:
        """Move one person to the inverse-variance mean of its point and its matched YOLO feet."""
        cams, yolo = self.active_cams(frame), self.state[frame]["yolo"]
        view = lb.match_views(cams, [person], yolo, self.params)[0]
        moved, used = dict(person), 0
        for _ in range(2):
            x, y, feet = lb.refine(cams, moved, view, yolo, self.params, prior=False)
            if not feet:
                break
            used = feet
            moved["x"], moved["y"] = x, y
            view = lb.match_views(cams, [moved], yolo, self.params)[0]
        return {"x": round(moved["x"], 4), "y": round(moved["y"], 4), "feet": used}

    def summary(self, frame: int) -> dict:
        state = self.state[frame]
        persons = state["persons"]
        return {"frame": frame, "status": state["status"], "reviewed": state.get("reviewed", False),
                "people": sum(1 for p in persons if p.get("status") != "suggested" or p.get("verified")),
                "check": sum(1 for p in persons if p.get("status") == "check" and not p.get("verified")),
                "suggested": sum(1 for p in persons if p.get("status") == "suggested" and not p.get("verified")),
                "diff": self.gt_diff(frame)}

    def metrics(self) -> dict | None:
        """Pseudo labels against the dataset's ground truth (when it has one): model points, auto labels, current."""
        rows = {"model": [], "auto": [], "current": []}
        for frame in self.frames:
            state = self.state[frame]
            if state["status"] != "done":
                continue
            gt = self.ground_truth(frame)
            if gt is None:   # a GT folder may cover only part of the frames
                continue
            auto = state.get("auto", {})
            rows["model"].append(lb.score_frame(auto.get("model", []), gt))
            rows["auto"].append(lb.score_frame([p for p in auto.get("fused", []) if p["status"] != "suggested"], gt))
            current = [p for p in state["persons"] if p.get("status") != "suggested" or p.get("verified")]
            cams = self.active_cams(frame)
            boxes = lb.frame_boxes(cams, current, state["yolo"], self.params)
            rows["current"].append(lb.score_frame(current, gt, cams, boxes))
        if not rows["current"]:
            return None
        return {key: lb.summarise(value) for key, value in rows.items()}

    # ------------------------------------------------------------------------------------------------ persistence

    def public(self) -> dict:
        spec = self.model_spec()
        return {"config": self.config, "kind": self.kind, "calibration": self.calibration, "area": self.area,
                "site_map": self.site_map,
                "image_processing": {"source_pixels": self.calibration.get("source_pixels"),
                                     "intrinsic_variant": self.calibration.get("intrinsic_variant"),
                                     "applied_undistort": self.undistort,
                                     "warning": self.pixel_alignment_warning},
                "image_version": self.image_version,
                "source": self.dataset.public(),
                "cameras": [cam.public() for cam in self.selected_cams()],
                "all_cameras": [c.public() for c in self.all_cams],
                "params": self.params.as_dict(), "step": self.step,
                "model": {"id": spec["id"], "method": spec["method"], "name": spec["name"],
                          "moda": None if self.kind == "wavetrack" and spec.get("dataset") != "wavetrack"
                          else spec.get("test_moda"),
                          "experimental": self.kind == "wavetrack" and spec.get("dataset") != "wavetrack",
                          "black_views": max(0, int(spec.get("num_cam", 5))-5) if self.kind == "wavetrack" else 0,
                          "weight_dataset": spec.get("dataset")} if spec else None,
                "models": self.model_catalog(), "model_status": self.model_status(),
                "frames": [self.summary(f) for f in self.frames], "job": self.job_state(),
                "gt_supported": self.gt_supported(), "gt_dir": str(self.gt_dir()),
                "has_gt": any(self.ground_truth(f) is not None for f in self.frames)}

    def schedule_save(self, delay: float = 1.0):
        with self.lock:
            if self._save_timer:
                self._save_timer.cancel()
            self._save_timer = threading.Timer(delay, self.save)
            self._save_timer.daemon = True
            self._save_timer.start()

    def set_site_map(self, value: dict) -> dict:
        if self.site_map_path is None:
            raise ValueError("Bản đồ hiện trường chỉ áp dụng cho rig WaveTrack.")
        clean = site_map.normalize(value, self.area, self.all_cams)
        with self.lock:
            site_map.write(self.site_map_path, clean)
            self.site_map = clean
            self.config["site_map"] = clean
            self.save()
        return clean

    def save(self, path: Path | None = None):
        path = path or self.out_dir / "label_project.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock:
            data = {"version": 1, "config": {**self.config, "params": self.params.as_dict()},
                    "state": {str(f): s for f, s in self.state.items()}}
            text = json.dumps(data)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path, models: dict, detectors: lb.Detectors) -> "Project":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        saved_pixel_version = int(data["config"].get("image_processing_version") or 0)
        project = cls(data["config"], models, detectors)
        for key, state in data.get("state", {}).items():
            frame = int(key)
            if frame in project.state:
                if state.get("status") == "running":
                    state["status"] = "pending"
                project.state[frame] = state
        legacy = project.calibration.get("source_pixels") == "undistorted" and \
            saved_pixel_version < 2 and \
            any(state.get("status") == "done" for state in project.state.values())
        if legacy:
            project.config["image_processing_version"] = 1
            project.pixel_alignment_warning = ("Project này có nhãn tạo trước khi sửa lỗi khử méo hai lần. "
                                               "Không dùng nhãn cũ làm ground truth; hãy chạy lại model và gán nhãn "
                                               "trên các frame đã xử lý với ảnh nguồn hiện tại.")
        return project

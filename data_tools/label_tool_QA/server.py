"""Multi-view labeling tool: BEV checkpoint + YOLO pseudo labels, reviewed and corrected in the browser.

    python label_tool/server.py            # http://127.0.0.1:8770
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import labeling as lb
from project import Project, pick_frames, probe
from model_registry import model_catalog

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
DEFAULT_ROOTS = []
MAX_BODY = 8 * 1024 * 1024


def load_models() -> dict[str, dict]:
    from inference import load_model_specs
    return {spec["id"]: spec for spec in load_model_specs(None)}


def best_model(models: dict[str, dict], kind: str | None, cameras: int) -> dict | None:
    """Highest test MODA among the checkpoints of this dataset family and camera count."""
    best = next((row for row in model_catalog(models, kind, cameras) if row["best"]), None)
    return models[best["id"]] if best else None


def default_root() -> str:
    for path in DEFAULT_ROOTS:
        found = path if (path / "Image_subsets").is_dir() else next(path.rglob("Image_subsets"), None) \
            if path.is_dir() else None
        if found:
            return str(found if found.name != "Image_subsets" else found.parent)
    return ""


class App:
    def __init__(self, models: dict[str, dict]):
        self.models = models
        self.detectors = lb.Detectors()
        self.project: Project | None = None
        self.lock = threading.Lock()

    def model_list(self, kind: str | None, cameras: int, layout: str | None = None) -> list[dict]:
        return model_catalog(self.models, kind, cameras, layout)

    def probe(self, body: dict) -> dict:
        info = probe(body["root"], body.get("calib") or None, body.get("videos"), body.get("timeline_fps", 2.0))
        info["models"] = self.model_list(info["kind"], len(info["cameras"]), info["calibration"]["layout"])
        info["yolo"] = lb.yolo_weights()
        name = Path(info["root"]).name.lower().replace(" ", "_")
        if info["source"]["mode"] == "videos" and all(c["read_mode"] == "timestamp" for c in info["source"]["cameras"]):
            name += "_synced"
        info["out_dir"] = str(HERE / "projects" / name)
        info["params"] = lb.Params().as_dict()
        from sync_audio import load_profile
        info["sync_profile"] = load_profile(Path(info["root"]), HERE / "projects" / "wavetrack_sync")
        return info

    def create(self, body: dict) -> dict:
        info = probe(body["root"], body.get("calib") or None, body.get("videos"), body.get("timeline_fps", 2.0))
        dataset = lb.Dataset(Path(info["root"]), body.get("videos"), body.get("timeline_fps", 2.0))
        available = dataset.frames
        frames = pick_frames(available, int(body.get("count") or 10), body.get("strategy", "even"),
                             body.get("start"), body.get("end"), int(body.get("seed", 0)))
        cameras = sorted({int(c) for c in body.get("cameras", [])}) or [c["index"] for c in info["cameras"]]
        config = {"root": info["root"], "calib": info["calib"], "cameras": cameras, "frames": frames,
                  "model_id": body.get("model_id") or None, "yolo_weights": body.get("yolo_weights") or None,
                  "params": body.get("params") or {}, "out_dir": body.get("out_dir") or str(HERE / "projects" / "default"),
                  "undistort": bool(body.get("undistort", True)), "gt_dir": None,
                  "image_processing_version": 2, **dataset.settings()}
        project_file = Path(config["out_dir"]) / "label_project.json"
        if project_file.exists() and not body.get("overwrite"):
            raise ValueError(f"{project_file} already exists: open it, or tick 'Overwrite'.")
        project = Project(config, self.models, self.detectors)
        project.model_spec(strict=True)
        if body.get("gt_dir") and project.gt_supported():
            project.set_gt_dir(body["gt_dir"])
        project.save()
        with self.lock:
            self.project = project
        project.start()
        return project.public()

    def import_annotations(self, body: dict) -> dict:
        from annotation_import import create_import_project
        with self.lock:
            if self.project and self.project.job['running']:
                raise ValueError('Wait for the current auto-label job before importing.')
            project = create_import_project(body, self.models, self.detectors)
            self.project = project
        return project.public()

    def source_preview(self, body: dict) -> dict:
        import cv2
        from sources import read_video_frame
        dataset = lb.Dataset(Path(body["root"]), body.get("videos"), body.get("timeline_fps", 2.0))
        frames = []
        join_folder = body.get("joins")
        if join_folder:
            folder = next((f for f in dataset.folders if f.name == join_folder), None)
            if folder is None or dataset.mode != "videos":
                raise ValueError("Camera video folder not found.")
            clips = dataset.videos[folder.name]["clips"]
            for before, after in zip(clips, clips[1:]):
                for clip, number, side in ((before, before["frames"] - 1, "frame cuối"), (after, 0, "frame đầu")):
                    frames.append((folder, read_video_frame(folder / clip["file"], number),
                                   f"Nối tại {after['start_s']:.3f} s · {clip['file']} · {side}"))
        else:
            frames = [(folder, dataset.read(folder, dataset.frames[0]), "") for folder in dataset.folders]
        images = []
        for folder, image, caption in frames:
            height, width = image.shape[:2]
            image = cv2.resize(image, (480, max(1, round(height * 480 / width))))
            ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                raise ValueError(f"Cannot encode preview for {folder.name}.")
            images.append({"folder": folder.name, "offset_s": dataset.videos.get(folder.name, {}).get("offset_s", 0),
                           "caption": caption,
                           "image": "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")})
        return {"images": images}

    def open(self, body: dict) -> dict:
        path = Path(body["path"])
        if path.is_dir():
            path = path / "label_project.json"
        project = Project.load(path, self.models, self.detectors)
        with self.lock:
            self.project = project
        return project.public()

    def need(self) -> Project:
        if self.project is None:
            raise ValueError("No project open.")
        return self.project


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            if not self.path.startswith(("/api/image", "/api/job")):
                super().log_message(fmt, *args)

        def send_body(self, body: bytes, content_type: str, status: int = 200, cache: bool = False):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "private, max-age=600" if cache else "no-store")
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, value, status: int = 200):
            self.send_body(json.dumps(value).encode(), "application/json", status)

        def read_json(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                raise ValueError("Request too large")
            return json.loads(self.rfile.read(length) or b"{}")

        def handle_api(self, method: str):
            url = urlsplit(self.path)
            parts = [p for p in url.path.split("/") if p][1:]
            query = {k: v[-1] for k, v in parse_qs(url.query).items()}
            body = self.read_json() if method in ("POST", "PUT") else {}
            head = parts[0] if parts else ""
            if method == "GET" and head == "state":
                return self.send_json({"project": app.project.public() if app.project else None,
                                       "default_root": default_root()})
            if method == "GET" and head == "fs":
                import scene
                return self.send_json(scene.list_folder(query.get("path") or None,
                                                        {".json"} if query.get("files") else set()))
            if method == "POST" and head == "probe":
                return self.send_json(app.probe(body))
            if method == "POST" and head == "source-preview":
                return self.send_json(app.source_preview(body))
            if method == "POST" and head == "import-annotations":
                return self.send_json(app.import_annotations(body))
            if method == "POST" and head == "create":
                return self.send_json(app.create(body))
            if method == "POST" and head == "open":
                return self.send_json(app.open(body))
            project = app.need()
            if method == "GET" and head == "job":
                return self.send_json({"job": project.job_state(), "frames": [project.summary(f) for f in project.frames],
                                       "model_status": project.model_status()})
            if method == "GET" and head == "models":
                return self.send_json({"models": project.model_catalog(), "model_status": project.model_status()})
            if method == "POST" and head == "model":
                project.set_model(body.get("model_id") or None)
                return self.send_json(project.public())
            if method == "POST" and head == "run":
                frames = [int(f) for f in body.get("frames", [])] or None
                return self.send_json(project.start(frames))
            if head == "frame" and len(parts) >= 2:
                frame = int(parts[1])
                if frame not in project.state:
                    raise ValueError("Unknown frame")
                if method == "GET":
                    return self.send_json(project.frame_payload(frame))
                if method == "PUT":
                    return self.send_json(project.update_frame(frame, body))
                if method == "POST" and len(parts) == 3 and parts[2] == "snap":
                    return self.send_json(project.snap(frame, body["person"]))
                if method == "POST" and len(parts) == 3 and parts[2] == "relabel":
                    return self.send_json(project.start([frame]))
            if method == "GET" and head == "image" and len(parts) == 3:
                version = query.get("v")
                if version and version != project.image_version:
                    return self.send_json({"error": "Project changed; reload the current project images."}, 409)
                width = int(query["w"]) if query.get("w") else None
                return self.send_body(project.jpeg(int(parts[2]), int(parts[1]), width), "image/jpeg", cache=bool(version))
            if method == "GET" and head == "metrics":
                return self.send_json(project.metrics())
            if method == "POST" and head == "gt":
                return self.send_json(project.set_gt_dir((body.get("path") or "").strip() or None))
            if method == "POST" and head == "export":
                out = Path(body.get("path") or project.out_dir / "export")
                return self.send_json(lb.export_dataset(project, out, bool(body.get("copy_images", True)),
                                                        bool(body.get("only_reviewed", False))))
            if method == "POST" and head == "params":
                for key, value in body.items():
                    if hasattr(project.params, key) and key in ("box_mode", "iou_gate", "radius"):
                        setattr(project.params, key, type(getattr(project.params, key))(value))
                project.schedule_save()
                return self.send_json(project.params.as_dict())
            if method == "PUT" and head == "site-map":
                return self.send_json(project.set_site_map(body))
            self.send_json({"error": "Not found"}, 404)

        def dispatch(self, method: str):
            try:
                if self.path.startswith("/api/"):
                    return self.handle_api(method)
                if method != "GET":
                    return self.send_json({"error": "Not found"}, 404)
                name = urlsplit(self.path).path.lstrip("/") or "index.html"
                path = (STATIC / name).resolve()
                if STATIC not in path.parents or not path.is_file():
                    return self.send_json({"error": "Not found"}, 404)
                self.send_body(path.read_bytes(), mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            except (ValueError, KeyError, FileNotFoundError) as error:
                self.send_json({"error": str(error)}, 400)
            except Exception as error:
                traceback.print_exc()
                self.send_json({"error": f"{type(error).__name__}: {error}"}, 500)

        def do_GET(self):
            self.dispatch("GET")

        def do_POST(self):
            self.dispatch("POST")

        def do_PUT(self):
            self.dispatch("PUT")

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--open", default=None, help="label_project.json (or its folder) to open at start")
    args = parser.parse_args()
    mimetypes.add_type("text/javascript", ".js")
    app = App(load_models())
    if args.open:
        app.open({"path": args.open})
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    print(f"Label tool on http://{args.host}:{args.port}  ({len(app.models)} checkpoints)")
    server.serve_forever()


if __name__ == "__main__":
    main()

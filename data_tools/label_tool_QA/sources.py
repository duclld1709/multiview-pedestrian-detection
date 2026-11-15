"""Image datasets and camera-folder videos on a shared, configurable timeline."""

from __future__ import annotations

import math
import re
from pathlib import Path

IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_TYPES = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}


def natural_key(text: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def camera_folders(root: Path) -> list[Path]:
    base = root / "Image_subsets" if (root / "Image_subsets").is_dir() else root
    # Some exports keep the dataset wrapper as Image_subsets/Image_subsets/C*.
    # Prefer the prepared image set when both prepared and raw siblings exist.
    if (base / "Image_subsets").is_dir() and not any(
            child.is_dir() and any(file.is_file() and file.suffix.lower() in IMAGE_TYPES
                                   for file in child.iterdir())
            for child in base.iterdir() if child.name != "Image_subsets"):
        base = base / "Image_subsets"
    folders = [child for child in base.iterdir() if child.is_dir() and any(
        file.is_file() and file.suffix.lower() in IMAGE_TYPES for file in child.iterdir())]
    return sorted(folders, key=lambda folder: natural_key(folder.name))


def numbered_images(folder: Path) -> dict[int, Path]:
    return {int(file.stem): file for file in folder.iterdir()
            if file.is_file() and file.suffix.lower() in IMAGE_TYPES and file.stem.isdigit()}


def video_info(path: Path) -> dict:
    import cv2
    capture = cv2.VideoCapture(str(path))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        width, height = capture.get(cv2.CAP_PROP_FRAME_WIDTH), capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if not capture.isOpened() or not all(math.isfinite(v) and v > 0 for v in (fps, count, width, height)):
            raise ValueError(f"Cannot read video metadata: {path}")
        return {"file": path.name, "fps": fps, "frames": int(count), "width": int(width), "height": int(height)}
    finally:
        capture.release()


def read_video_frame(path: Path, number: int):
    import cv2
    capture = cv2.VideoCapture(str(path))
    try:
        if number:
            capture.set(cv2.CAP_PROP_POS_FRAMES, number)
        ok, image = capture.read()
        if not ok:
            raise ValueError(f"Cannot decode source frame {number} from {path}.")
        return image
    finally:
        capture.release()


def read_video_time(path: Path, seconds: float):
    """Nearest presentation timestamp, including variable-frame-rate phone recordings."""
    import av
    def oriented(frame):
        import numpy as np
        # PyAV rotation is counter-clockwise, per its VideoFrame API.
        image = frame.to_ndarray(format="bgr24")
        return np.ascontiguousarray(np.rot90(image, round(frame.rotation / 90)))
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        origin = float((stream.start_time or 0) * stream.time_base)
        target = origin + seconds
        container.seek(int(target / stream.time_base), stream=stream, backward=True)
        previous = None
        for frame in container.decode(stream):
            timestamp = float(frame.pts * frame.time_base)
            if timestamp >= target:
                chosen = previous if previous is not None and target-float(previous.pts*previous.time_base) < timestamp-target else frame
                return oriented(chosen)
            previous = frame
        if previous is not None:
            return oriented(previous)
    raise ValueError(f"Cannot decode {seconds:.3f} s from {path}.")


class Dataset:
    """Images keep their original frame IDs. Videos use time = frame / timeline_fps * rate + offset_s.

    Open a separate capture per read: background labeling and HTTP image requests may run concurrently.
    Only top-level camera videos are considered; intrinsic calibration footage is never included.
    """

    def __init__(self, root: Path, videos: dict | None = None, timeline_fps: float = 2.0):
        self.root = root
        self.folders = camera_folders(root)
        self.mode = "images" if self.folders else "videos"
        self.videos, self.options, self.files = {}, {}, {}
        if self.mode == "images":
            for folder in self.folders:
                self.files[folder.name] = numbered_images(folder)
                if not self.files[folder.name]:
                    raise ValueError(f"No numbered images in {folder} (expected 00000000.jpg, ...).")
            self.frames = sorted(set.intersection(*(set(files) for files in self.files.values())))
        else:
            self.timeline_fps = float(timeline_fps)
            if not math.isfinite(self.timeline_fps) or self.timeline_fps <= 0:
                raise ValueError("Timeline FPS must be a positive finite number.")
            self.folders = sorted([p for p in root.iterdir() if p.is_dir() and
                                   re.fullmatch(r"Cam\d+(?:_.*)?", p.name, re.IGNORECASE) and
                                   any(f.is_file() and f.suffix.lower() in VIDEO_TYPES for f in p.iterdir())],
                                  key=lambda p: natural_key(p.name))
            ends = []
            for folder in self.folders:
                paths = sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_TYPES],
                               key=lambda p: natural_key(p.name))
                self.options[folder.name] = [p.name for p in paths]
                chosen = (videos or {}).get(folder.name, {})
                filenames = chosen.get("files", [chosen.get("file") or paths[0].name])
                if not isinstance(filenames, list) or not filenames or not all(isinstance(f, str) for f in filenames):
                    raise ValueError(f"Choose at least one video for {folder.name}.")
                if len(set(filenames)) != len(filenames):
                    raise ValueError(f"Duplicate clips in {folder.name}: each video can be used only once.")
                clips, durations = [], []
                rate = float(chosen.get("rate", 1))
                read_mode = chosen.get("read_mode", "frame")
                if not math.isfinite(rate) or not 0.99 <= rate <= 1.01:
                    raise ValueError(f"Invalid synchronization clock rate for {folder.name}.")
                if read_mode not in ("frame", "timestamp"):
                    raise ValueError("Unknown video read mode.")
                for filename in filenames:
                    if filename not in self.options[folder.name]:
                        raise ValueError(f"Video {filename!r} is not in {folder}.")
                    clip = video_info(folder / filename)
                    # A clip occupies [start, end); its successor starts immediately at end.
                    start = math.fsum(durations)
                    duration = clip["frames"] / clip["fps"]
                    if read_mode == "timestamp":
                        import av
                        with av.open(str(folder / filename)) as container:
                            stream = container.streams.video[0]
                            if stream.duration is not None:
                                duration = float(stream.duration * stream.time_base)
                    durations.append(duration)
                    clips.append({**clip, "start_s": start, "end_s": math.fsum(durations),
                                  "duration_s": durations[-1]})
                info = clips[0]
                offset = float(chosen.get("offset_s", 0))
                if not math.isfinite(offset) or offset < 0:
                    raise ValueError(f"Start time for {folder.name} must be finite and non-negative.")
                duration = math.fsum(durations)
                if offset >= duration:
                    raise ValueError(f"Start time is past the end of {folder / filename}.")
                end_s = float(chosen.get("end_s", duration))
                if not math.isfinite(end_s) or not offset < end_s <= duration + 1e-6:
                    raise ValueError(f"Invalid synchronized end time for {folder.name}.")
                self.videos[folder.name] = {"file": info["file"], "fps": info["fps"],
                                          "width": info["width"], "height": info["height"],
                                          "frames": sum(c["frames"] for c in clips), "clips": clips,
                                          "duration_s": duration, "offset_s": offset,
                                          "rate": rate, "end_s": end_s, "read_mode": read_mode}
                ends.append((end_s - offset) / rate * self.timeline_fps)
            # Keep every sample before the longest stream ends; shorter cameras become black.
            self.frames = range(math.ceil(max(ends) - 1e-9)) if ends else []
        if not self.folders:
            raise ValueError("No camera sources found (expected Image_subsets/C1..CN, image folders, "
                             "or Cam01_Name folders containing videos).")
        if not self.frames:
            raise ValueError("No common frames across all camera sources.")

    def settings(self) -> dict:
        if self.mode == "images":
            return {}
        return {"videos": {name: {"file": info["file"], "offset_s": info["offset_s"],
                                  **({"rate": info["rate"], "end_s": info["end_s"], "read_mode": info["read_mode"]}
                                     if info["read_mode"] == "timestamp" or info["rate"] != 1 or info["end_s"] != info["duration_s"] else {}),
                                  **({"files": [c["file"] for c in info["clips"]]} if len(info["clips"]) > 1 else {})}
                           for name, info in self.videos.items()}, "timeline_fps": self.timeline_fps}

    def public(self) -> dict:
        return {"mode": self.mode, **self.settings(),
                "cameras": [{"folder": name, "choices": self.options[name], **info}
                            for name, info in self.videos.items()]}

    def size(self, folder: Path) -> tuple[int, int]:
        if self.mode == "videos":
            info = self.videos[folder.name]
            return info["width"], info["height"]
        from PIL import Image
        with Image.open(self.files[folder.name][min(self.files[folder.name])]) as image:
            return image.size

    def locate(self, folder: Path, frame: int) -> tuple[dict, int] | None:
        """The clip and local source frame at this timeline sample; None after this camera ends."""
        info = self.videos[folder.name]
        time = frame / self.timeline_fps * info["rate"] + info["offset_s"]
        if time >= info["end_s"] - 1e-9:
            return None
        for clip in info["clips"]:
            # Tolerate floating-point roundoff when the sample is exactly on a join.
            if clip["start_s"] - 1e-9 <= time < clip["end_s"] - 1e-9:
                number = math.floor((time - clip["start_s"]) * clip["fps"] + 1e-7)
                return clip, max(0, min(clip["frames"] - 1, number))
        return None

    def source_frame(self, folder: Path, frame: int) -> int | None:
        source = self.locate(folder, frame)
        return source[1] if source else None

    def active(self, folder: Path, frame: int) -> bool:
        return frame in self.files[folder.name] if self.mode == "images" else self.locate(folder, frame) is not None

    def frame_source(self, folder: Path, frame: int) -> dict:
        if self.mode == "images":
            return {"folder": folder.name, "file": self.files[folder.name][frame].name,
                    "frame": frame, "active": True}
        source = self.locate(folder, frame)
        if source is None:
            return {"folder": folder.name, "file": None, "frame": None, "active": False}
        clip, number = source
        info = self.videos[folder.name]
        return {"folder": folder.name, "file": clip["file"], "frame": number, "active": True,
                "time_s": (frame / self.timeline_fps * info["rate"] + info["offset_s"] - clip["start_s"])
                if info["read_mode"] == "timestamp" else number / clip["fps"]}

    def read(self, folder: Path, frame: int):
        """Return BGR pixels. Honor the file's encoded orientation, as OpenCV's default decoder does."""
        import cv2
        if self.mode == "images":
            path = self.files[folder.name][frame]
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        else:
            info = self.videos[folder.name]
            source = self.locate(folder, frame)
            if source is None:
                import numpy as np
                return np.zeros((info["height"], info["width"], 3), dtype=np.uint8)
            clip, number = source
            path = folder / clip["file"]
            if info["read_mode"] == "timestamp":
                time = frame / self.timeline_fps * info["rate"] + info["offset_s"] - clip["start_s"]
                image = read_video_time(path, max(0, time))
            else:
                image = read_video_frame(path, number)
            # Every clip in a camera uses the first clip's pixel coordinates and calibration.
            if image.shape[:2] != (info["height"], info["width"]):
                image = cv2.resize(image, (info["width"], info["height"]), interpolation=cv2.INTER_LINEAR)
        if image is None:
            raise ValueError(f"Cannot decode frame {frame} from {path}.")
        return image

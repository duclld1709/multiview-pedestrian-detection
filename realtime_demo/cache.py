"""Disk-backed, fully normalized FP16 inputs and rectified JPEG display frames."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import hashlib
import json
import math
import shutil
import time
from uuid import uuid4

import cv2
import numpy as np

from subset_data import rectification_maps

CACHE_FPS = 15
MAX_CACHE_BYTES = 3_000_000_000
MEAN = np.array([.485, .456, .406], dtype=np.float32)[None, None, :]
STD = np.array([.229, .224, .225], dtype=np.float32)[None, None, :]
NORMALIZED_LUT = ((np.arange(256, dtype=np.float32)[None, :] / 255 - MEAN.reshape(3, 1)) / STD.reshape(3, 1)).astype(np.float16)


def normalized_tensor(bgr, size):
    if (bgr.shape[1], bgr.shape[0]) != tuple(size):
        bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
    rgb = bgr[:, :, ::-1].transpose(2, 0, 1)
    # Exactly the same FP16 values as float32 ImageNet normalization, computed once.
    return NORMALIZED_LUT[np.arange(3)[:, None, None], rgb]


def encode_preview(rectified):
    preview = cv2.resize(rectified, (640, 360), interpolation=cv2.INTER_AREA)
    ok, jpeg = cv2.imencode('.jpg', preview, [cv2.IMWRITE_JPEG_QUALITY, 77])
    if not ok:
        raise ValueError('JPEG encoding failed')
    return jpeg.tobytes()


def video_metadata(session, cams):
    videos = [session / f'cam{int(cam.name[3:])}.mp4' for cam in cams]
    props = []
    for path in videos:
        cap = cv2.VideoCapture(str(path))
        try:
            if not cap.isOpened():
                raise ValueError(f'Cannot open {path}')
            props.append((cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
        finally:
            cap.release()
    fps = props[0][0]
    count = min(n for _, n in props)
    if fps <= 0 or count <= 0 or any(not math.isfinite(rate) or abs(rate - fps) > .01 for rate, _ in props):
        raise ValueError('This demo requires nonempty recordings at matching constant camera FPS.')
    sync = session / 'sync.json'
    if sync.exists() and any(float(v) != 0 for value in json.loads(sync.read_text()).values() for v in value.values()):
        raise ValueError('This demo requires zero camera sync offsets.')
    return videos, fps, count


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(f'.{path.name}.{uuid4().hex}.tmp')
    try:
        temporary.write_text(json.dumps(value, indent=2), encoding='utf-8', newline='\n')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_npy_header(file, shape, dtype):
    file.seek(0)
    np.lib.format.write_array_header_1_0(file, dict(
        shape=shape, descr=np.dtype(dtype).str, fortran_order=False))


class FrameCache:
    def __init__(self, session, cams, directory, size, seconds=180):
        self.session, self.cams = Path(session), list(cams)
        self.size, self.seconds = tuple(size), seconds
        self.videos, self.source_fps, self.source_count = video_metadata(self.session, self.cams)
        self.fps = min(CACHE_FPS, self.source_fps)
        self.frame_shape = (len(self.cams), 3, size[1], size[0])
        self.frame_bytes = math.prod(self.frame_shape) * np.dtype(np.float16).itemsize
        available_frames = math.ceil(self.source_count * self.fps / self.source_fps)
        self.capacity = min(int(seconds * self.fps), available_frames, MAX_CACHE_BYTES // self.frame_bytes)
        if self.capacity < 1:
            raise ValueError('Cache needs room and duration for at least one frame set.')
        self.count = self.capacity
        self.signature = dict(version=2, size=list(size), frames=self.capacity, fps=self.fps,
                              source_fps=self.source_fps, max_bytes=MAX_CACHE_BYTES,
                              format='normalized RGB CHW float16',
                              calibration=hashlib.sha256((self.session / 'calibrations/calibration.json').read_bytes()).hexdigest(),
                              cameras=[c.name for c in cams],
                              videos=[dict(path=str(p.resolve()), size=p.stat().st_size, mtime=p.stat().st_mtime_ns) for p in self.videos])
        for name in ('grid.json',):
            p = self.session / 'calibrations' / name
            if p.exists():
                self.signature[name] = hashlib.sha256(p.read_bytes()).hexdigest()
        intrinsics = self.session / 'calibrations/intrinsic_original'
        self.signature['intrinsics'] = [hashlib.sha256((intrinsics / f'intr_{c.name}.xml').read_bytes()).hexdigest() for c in cams]
        key = hashlib.sha256(json.dumps(self.signature, sort_keys=True).encode()).hexdigest()[:12]
        self.directory = Path(directory) / f'{self.session.name}_{size[0]}x{size[1]}_{key}'
        self.size_bytes = 0
        self.array = None
        self.indices = None
        self.ready = False
        self.open_existing()

    def open_existing(self):
        meta = self.directory / 'manifest.json'
        if not meta.exists():
            return
        value = json.loads(meta.read_text(encoding='utf-8'))
        if value.get('signature') != self.signature or not value.get('ready'):
            return
        count = value['completed']
        array = np.load(self.directory / 'inputs.npy', mmap_mode='r', allow_pickle=False)
        if not 0 < count <= self.capacity or array.shape != (count, *self.frame_shape) or array.dtype != np.float16:
            raise ValueError('Cache tensor shape/dtype mismatch')
        indices = np.load(self.directory / 'offsets.npy', mmap_mode='r', allow_pickle=False)
        if indices.shape != (count + 1, len(self.cams)) or indices.dtype != np.int64:
            raise ValueError('Cache JPEG index mismatch')
        for k in range(len(self.cams)):
            if (self.directory / f'camera{k}.jpgs').stat().st_size != int(indices[-1, k]):
                raise ValueError('Incomplete JPEG cache')
        self.size_bytes = sum(path.stat().st_size for path in self.directory.iterdir() if path.is_file())
        self.count = count
        self.array, self.indices, self.ready = array, indices, True

    def _source_frame(self, index):
        return int(index * self.source_fps / self.fps)

    def build(self, progress):
        if self.ready:
            progress(self.count, self.count, 0)
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        cache_volume = self.directory.resolve()
        progress_path = self.directory / 'progress.json'
        start = 0
        if progress_path.exists():
            value = json.loads(progress_path.read_text())
            if value.get('signature') == self.signature:
                start = int(value['completed'])
        metadata = dict(signature=self.signature, completed=self.capacity, ready=True)
        # Allow for the progress file and an atomic metadata write at the same time.
        metadata_bytes = 2 * len(json.dumps(metadata, indent=2).encode('utf-8'))
        mode = 'r+b' if start else 'w+b'
        camera_count = len(self.cams)
        ends = np.zeros(camera_count, dtype=np.int64)
        maps = rectification_maps(self.cams, self.session / 'calibrations/intrinsic_original')
        begun = time.perf_counter()
        readers, blobs = [], []

        def prepare_camera(k, source_frame, skipped):
            reader = readers[k]
            for _ in range(skipped):
                if not reader.grab():
                    raise ValueError(f'Cannot advance camera {k} to frame {source_frame}')
            ok, raw = reader.read()
            if not ok:
                raise ValueError(f'Cannot decode camera {k}, frame {source_frame}')
            expected_ms = source_frame * 1000 / self.source_fps
            if abs(reader.get(cv2.CAP_PROP_POS_MSEC) - expected_ms) > 1000 / self.source_fps:
                raise ValueError(f'Camera timestamp mismatch at frame {source_frame}')
            rectified = cv2.remap(raw, *maps[k], cv2.INTER_LINEAR)
            return normalized_tensor(rectified, self.size), encode_preview(rectified)

        with ExitStack() as stack:
            tensor = stack.enter_context((self.directory / 'inputs.npy').open(mode))
            offsets = stack.enter_context((self.directory / 'offsets.npy').open(mode))
            # Stream complete frame sets, then finalize the NPY shapes before publishing.
            write_npy_header(tensor, (self.capacity, *self.frame_shape), np.float16)
            write_npy_header(offsets, (self.capacity + 1, camera_count), np.int64)
            if start:
                tensor.truncate(tensor.seek(start * self.frame_bytes, 1))
                offsets.seek(start * ends.nbytes, 1)
                ends = np.frombuffer(offsets.read(ends.nbytes), dtype=np.int64)
                offsets.truncate(offsets.tell())
            else:
                offsets.write(ends.tobytes())
            for k, path in enumerate(self.videos):
                cap = cv2.VideoCapture(str(path))
                stack.callback(cap.release)
                if 0 < start < self.capacity and not cap.set(cv2.CAP_PROP_POS_FRAMES, self._source_frame(start)):
                    raise ValueError(f'Cannot resume camera {k} at frame {self._source_frame(start)}')
                readers.append(cap)
                blob = stack.enter_context((self.directory / f'camera{k}.jpgs').open(mode))
                blob.truncate(blob.seek(int(ends[k])))
                blobs.append(blob)
            pool = stack.enter_context(ThreadPoolExecutor(max_workers=camera_count))

            def checkpoint(completed):
                for file in (tensor, offsets, *blobs):
                    file.flush()
                atomic_json(progress_path, dict(signature=self.signature, completed=completed))

            completed = start
            next_source_frame = self._source_frame(start)
            for index in range(start, self.capacity):
                source_frame = self._source_frame(index)
                skipped = source_frame - next_source_frame
                futures = [pool.submit(prepare_camera, k, source_frame, skipped) for k in range(camera_count)]
                prepared = [future.result() for future in futures]
                next_ends = ends + [len(jpeg) for _, jpeg in prepared]
                frame_bytes = self.frame_bytes + ends.nbytes + int((next_ends - ends).sum())
                current_bytes = tensor.tell() + offsets.tell() + int(ends.sum())
                if current_bytes + frame_bytes + metadata_bytes > MAX_CACHE_BYTES:
                    break
                if shutil.disk_usage(cache_volume).free < frame_bytes + 384 * 1024**2:
                    checkpoint(completed)
                    raise ValueError('Cache paused: less than 384 MiB would remain on disk.')
                for (array, jpeg), blob in zip(prepared, blobs):
                    tensor.write(array.tobytes())
                    blob.write(jpeg)
                ends = next_ends
                offsets.write(ends.tobytes())
                next_source_frame = source_frame + 1
                completed = index + 1
                if completed % 30 == 0:
                    checkpoint(completed)
                    progress(completed, self.capacity, (completed - start) / max(.001, time.perf_counter() - begun))
            if not completed:
                raise ValueError('Cache byte limit is too small for one frame set.')
            checkpoint(completed)
            write_npy_header(tensor, (completed, *self.frame_shape), np.float16)
            write_npy_header(offsets, (completed + 1, camera_count), np.int64)
        atomic_json(self.directory / 'manifest.json', dict(signature=self.signature, completed=completed, ready=True))
        self.open_existing()
        progress(completed, completed, (completed - start) / max(.001, time.perf_counter() - begun))

    def batch(self, start, count, cameras):
        if not self.ready:
            raise ValueError('Cache is not ready')
        # Fully preprocessed: runtime performs only cache reads, stacking and H2D transfer.
        return np.ascontiguousarray(self.array[start:start + count, cameras])

    def jpeg(self, frame, camera):
        if not self.ready or not 0 <= frame < self.count or not 0 <= camera < len(self.cams):
            raise ValueError('Invalid cached image')
        start, end = (int(v) for v in self.indices[frame:frame + 2, camera])
        with (self.directory / f'camera{camera}.jpgs').open('rb') as f:
            f.seek(start)
            return f.read(end - start)

"""Resumable full-dataset FP16 caches."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np

from realtime_demo.cache import FrameCache, atomic_json, write_npy_header
from realtime_demo.datasets import fingerprint
from realtime_demo.video_source import direct_source


class CacheCancelled(Exception):
    pass


@contextmanager
def build_lock(path):
    with path.open('a+b') as handle:
        handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError('A cache builder is already writing to this directory.') from exc
        yield


class FullFrameCache(FrameCache):
    def __init__(self, dataset, media, directory, size, fps, *, exact=False, identity=None):
        if not math.isfinite(fps) or fps <= 0 or fps > media.fps:
            raise ValueError(f'Cache FPS must be greater than zero and at most {media.fps:g}.')
        fps = float(fps)
        self.media, self.cams, self.size = media, dataset.cams, tuple(size)
        self.source_fps, self.source_count = media.fps, media.count
        self.fps, self.duration = fps, media.count / media.fps
        self.frame_shape = (len(self.cams), 3, size[1], size[0])
        self.frame_bytes = math.prod(self.frame_shape) * np.dtype(np.float16).itemsize
        self.capacity = self.count = math.ceil(self.duration * fps)
        original = media.original_intrinsics
        self.signature = dict(version=3, dataset=dataset.key, size=list(size), fps=fps, frames=self.capacity,
                              source=media.kind, media=identity or media.identity(), source_fps=media.fps,
                              source_frames=media.count, calibration=dataset.calibration,
                              original_intrinsics=fingerprint(
                                  [original / f'intr_{c.name}.xml' for c in self.cams]) if original else None,
                              format='normalized RGB CHW float16')
        key = hashlib.sha256(json.dumps(self.signature, sort_keys=True).encode()).hexdigest()[:12]
        self.directory = (Path(directory) if exact else
                          Path(directory) / f'{dataset.key}_{size[0]}x{size[1]}_{key}')
        self.size_bytes = 0
        self.array = self.indices = None
        self.ready = False
        self.open_existing()

    def open_existing(self):
        super().open_existing()
        if self.ready and self.count != self.capacity:
            raise ValueError('Incomplete full-dataset cache.')

    def build(self, progress, cancelled=None):
        self.directory.mkdir(parents=True, exist_ok=True)
        with build_lock(self.directory / '.build.lock'):
            self.open_existing()
            if not self.ready:
                self._write_frames(progress, cancelled)
                self.open_existing()
            progress(self.count, self.count, 0)

    def _write_frames(self, progress, cancelled):
        progress_path = self.directory / 'progress.json'
        start = 0
        if progress_path.is_file():
            status = json.loads(progress_path.read_text(encoding='utf-8'))
            if status.get('signature') == self.signature:
                start = status['completed']
        if not 0 <= start <= self.capacity:
            raise ValueError('Invalid cache build progress.')
        if shutil.disk_usage(self.directory).free < (self.capacity - start) * self.frame_bytes:
            raise ValueError('Insufficient disk space for the full FP16 cache. Choose a lower FPS or another cache folder.')

        camera_count = len(self.cams)
        mode = 'r+b' if start else 'w+b'
        begun = time.perf_counter()
        last_progress = begun
        progress(start, self.capacity, 0)
        with ExitStack() as stack:
            tensor = stack.enter_context((self.directory / 'inputs.npy').open(mode))
            offsets = stack.enter_context((self.directory / 'offsets.npy').open(mode))
            write_npy_header(tensor, (self.capacity, *self.frame_shape), np.float16)
            write_npy_header(offsets, (self.capacity + 1, camera_count), np.int64)
            ends = np.zeros(camera_count, dtype=np.int64)
            if start:
                tensor.truncate(tensor.seek(start * self.frame_bytes, 1))
                offsets.seek(start * ends.nbytes, 1)
                ends = np.frombuffer(offsets.read(ends.nbytes), dtype=np.int64)
                offsets.truncate(offsets.tell())
            else:
                offsets.write(ends.tobytes())
            blobs = [stack.enter_context((self.directory / f'camera{k}.jpgs').open(mode)) for k in range(camera_count)]
            for blob, end in zip(blobs, ends):
                blob.truncate(blob.seek(int(end)))
            reader = direct_source(self.media, self.cams, self.size)
            stack.callback(reader.close)
            pool = stack.enter_context(ThreadPoolExecutor(max_workers=camera_count))

            def checkpoint(completed):
                for file in (tensor, offsets, *blobs):
                    file.flush()
                atomic_json(progress_path, dict(signature=self.signature, completed=completed))

            for index in range(start, self.capacity):
                if cancelled is not None and cancelled():
                    checkpoint(index)
                    progress(index, self.capacity, 0)
                    raise CacheCancelled()
                frame = min(self.source_count - 1, self._source_frame(index))
                futures = [pool.submit(reader.prepared, frame, k) for k in range(camera_count)]
                inputs = [job.result() for job in futures]
                jpeg_sizes = [len(jpeg) for _, jpeg in inputs]
                if shutil.disk_usage(self.directory).free < self.frame_bytes + sum(jpeg_sizes) + ends.nbytes:
                    checkpoint(index)
                    raise ValueError('Insufficient disk space. Free space and retry the same cache to resume.')
                for (array, jpeg), blob in zip(inputs, blobs):
                    tensor.write(array.tobytes())
                    blob.write(jpeg)
                ends = ends + jpeg_sizes
                offsets.write(ends.tobytes())
                completed = index + 1
                if completed % 30 == 0:
                    checkpoint(completed)
                now = time.perf_counter()
                if completed == start + 1 or now - last_progress >= .5:
                    progress(completed, self.capacity, (completed - start) / max(.001, now - begun))
                    last_progress = now
            checkpoint(self.capacity)
        atomic_json(self.directory / 'manifest.json', dict(signature=self.signature, completed=self.capacity, ready=True))
        progress(self.capacity, self.capacity, (self.capacity - start) / max(.001, time.perf_counter() - begun))


def available_cache(dataset, directory, size):
    """Select a compatible completed cache, preferring the highest FPS on the primary source."""
    directory = Path(directory)
    exact = (directory / 'manifest.json').is_file()
    candidates = [directory] if exact else directory.glob(f'{dataset.key}_*')
    ranked = []
    for path in candidates:
        try:
            value = json.loads((path / 'manifest.json').read_text(encoding='utf-8'))
            signature = value['signature']
            media = dataset.sources[signature['source']]
            if (value.get('ready') and signature['version'] == 3 and signature['size'] == list(size)
                    and signature['dataset'] == dataset.key):
                fps = float(signature['fps'])
                if math.isfinite(fps) and 0 < fps <= media.fps:
                    ranked.append((media.kind != dataset.direct, -fps, path, media.kind))
        except (OSError, ValueError, TypeError, KeyError):
            continue

    @lru_cache(maxsize=None)
    def media_identity(kind):
        return dataset.sources[kind].identity()

    for _, fps, path, kind in sorted(ranked):
        try:
            cache = FullFrameCache(dataset, dataset.sources[kind], path, size, -fps, exact=True,
                                   identity=media_identity(kind))
            if cache.ready:
                return cache
        except (OSError, ValueError, TypeError, KeyError):
            continue
    return None

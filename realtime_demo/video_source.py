"""Decode and prepare the synchronized MP4s on demand."""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import cv2
import numpy as np

from realtime_demo.cache import encode_preview, normalized_tensor, video_metadata
from subset_data import rectification_maps


class Mp4Source:
    def __init__(self, session, cams, size, fused_remap=False, *, media=None):
        if media is None:
            self.videos, self.fps, self.count = video_metadata(session, cams)
            original = session / 'calibrations/intrinsic_original'
        else:
            self.videos, self.fps, self.count = media.paths, media.fps, media.count
            original = media.original_intrinsics
        self.source_fps, self.source_count = self.fps, self.count
        self.cams = cams
        self.size = size
        # Keep enough pixels for the camera preview when testing smaller model inputs.
        rectified_size = (max(size[0], 640), max(size[1], 360)) if fused_remap else None
        self.maps = rectification_maps(cams, original, rectified_size) if original else None
        self._setup_readers()

    def _setup_readers(self):
        cams = self.cams
        self.readers = [None] * len(cams)
        self.last_rectified = [None] * len(cams)
        self.previews = [OrderedDict() for _ in cams]
        self.locks = [Lock() for _ in cams]
        self.ready = True

    def _rectified(self, frame, camera):
        previous = self.last_rectified[camera]
        if previous is not None and previous[0] == frame:
            return previous[1]
        cap = self.readers[camera]
        if cap is None:
            threads = getattr(cv2, 'CAP_PROP_N_THREADS', None)
            params = [threads, 2] if threads is not None else []
            cap = cv2.VideoCapture(str(self.videos[camera]), cv2.CAP_FFMPEG, params)
            if not cap.isOpened():
                cap.release()
                raise ValueError(f'Cannot open {self.videos[camera]}')
            self.readers[camera] = cap
        next_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if 0 <= frame - next_frame <= self.fps:
            for _ in range(frame - next_frame):
                if not cap.grab():
                    raise ValueError(f'Cannot advance camera {camera} to frame {frame}')
        elif not cap.set(cv2.CAP_PROP_POS_FRAMES, frame):
            raise ValueError(f'Cannot seek camera {camera} to frame {frame}')
        ok, raw = cap.read()
        if not ok or int(cap.get(cv2.CAP_PROP_POS_FRAMES)) != frame + 1:
            raise ValueError(f'Cannot decode camera {camera}, frame {frame}')
        rectified = cv2.remap(raw, *self.maps[camera], cv2.INTER_LINEAR) if self.maps else raw
        self.last_rectified[camera] = frame, rectified
        return rectified

    def prepared(self, frame, camera):
        with self.locks[camera]:
            rectified = self._rectified(frame, camera)
            return normalized_tensor(rectified, self.size), encode_preview(rectified)

    def _preview(self, rectified, frame, camera):
        jpeg = encode_preview(rectified)
        previews = self.previews[camera]
        previews[frame] = jpeg
        previews.move_to_end(frame)
        if len(previews) > 16:
            previews.popitem(last=False)
        return jpeg

    def batch(self, start, count, cameras):
        if not 0 <= start < start + count <= self.count:
            raise ValueError('Invalid source frames')
        width, height = self.size
        batch = np.empty((count, len(cameras), 3, height, width), dtype=np.float16)

        def prepare_camera(item):
            column, camera = item
            with self.locks[camera]:
                if not self.ready:
                    raise ValueError('Video source is closed')
                for offset in range(count):
                    frame = start + offset
                    rectified = self._rectified(frame, camera)
                    batch[offset, column] = normalized_tensor(rectified, self.size)
                    if frame not in self.previews[camera]:
                        self._preview(rectified, frame, camera)

        with ThreadPoolExecutor(max_workers=len(cameras)) as pool:
            list(pool.map(prepare_camera, enumerate(cameras)))
        return batch

    def jpeg(self, frame, camera):
        if not 0 <= frame < self.count or not 0 <= camera < len(self.videos):
            raise ValueError('Invalid source image')
        with self.locks[camera]:
            if not self.ready:
                raise ValueError('Video source is closed')
            cached = self.previews[camera].get(frame)
            if cached is not None:
                return cached
            return self._preview(self._rectified(frame, camera), frame, camera)

    def close(self):
        self.ready = False
        for camera, lock in enumerate(self.locks):
            with lock:
                cap = self.readers[camera]
                if cap is not None:
                    cap.release()
                    self.readers[camera] = None
                self.last_rectified[camera] = None
                self.previews[camera].clear()


class ImageSource(Mp4Source):
    def __init__(self, media, cams, size):
        self.cams, self.size = cams, size
        self.videos, self.fps, self.count = media.paths, media.fps, media.count
        self.source_fps, self.source_count = self.fps, self.count
        self._setup_readers()

    def _rectified(self, frame, camera):
        path = self.videos[camera][frame]
        # imdecode accepts Unicode paths on Windows through Python's file API.
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.shape[1::-1] != self.cams[camera].image_size:
            raise ValueError(f'Invalid image or unexpected calibrated size: {path}')
        return image


def direct_source(media, cams, size, fused_remap=False):
    if media.kind == 'images':
        return ImageSource(media, cams, size)
    return Mp4Source(media.session, cams, size, fused_remap, media=media)

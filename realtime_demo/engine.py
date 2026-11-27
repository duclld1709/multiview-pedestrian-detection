"""Single-owner GPU worker and pause-only timeline/camera configuration."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from functools import lru_cache
import json
import math
from pathlib import Path
import threading
import time
from uuid import uuid4

import cv2
import numpy as np
import torch

from calib import load_calibration
from cylinders import cylinder_box
from multiview_detector.models.checkpoint import load_bevnet, load_checkpoint
from multiview_detector.utils.heatmap import decode
from multiview_detector.utils.projection import camera_grids
from realtime_demo.cache import CACHE_FPS, MAX_CACHE_BYTES, FrameCache, atomic_json
from realtime_demo.datasets import CHECKPOINTS, checkpoint_dataset, default_checkpoint
from realtime_demo.device import inference_device
from realtime_demo.full_cache import CacheCancelled, FullFrameCache, available_cache
from realtime_demo.video_source import Mp4Source, direct_source
from site_layout import SiteMap


class Conflict(ValueError):
    pass


class Engine:
    def __init__(self, root, session, cache_dir, size=None, seconds=22, latency_ms=350, *, dataset=None):
        self.root, self.session = Path(root), Path(session) if session else None
        self.dataset = dataset
        self.cache_dir, self.requested_size = Path(cache_dir), size
        self.cache_entry = None
        self.cache_build = None
        self.cache_cancel = None
        self.seconds, self.latency_ms = seconds, latency_ms
        self.cams, self.roi, self.pitch, self.site_map = [], None, (), None
        if dataset:
            self.cams, self.roi, self.pitch = dataset.cams, dataset.roi, dataset.pitch
            self.site_map = SiteMap(self.cams, self.roi, self.pitch, dataset.site, wavetrack=dataset.key == 'wavetrack')
        elif self.session:
            self.cams, self.roi, self.pitch = load_calibration(self.session / 'calibrations/calibration.json')
            self.cams.sort(key=lambda c: int(c.name[3:]))
            self.site_map = SiteMap(self.cams, self.roi, self.pitch, self.root / 'wavetrack-site-map.json')
        self.lock = threading.RLock()
        self.changed = threading.Condition(self.lock)
        self.phase, self.message = 'idle', 'Choose a checkpoint and load the model to begin.'
        self.model = self.grid = self.grids = self.cache = None
        self.video = None
        self.source_kind = 'cache'
        self.mp4_options = dict(fused_remap=False, prefetch=True)
        self.scenario_path = self.session / 'outputs/realtime_demo_scenarios.json' if self.session else None
        self.scenarios = (json.loads(self.scenario_path.read_text(encoding='utf-8'))
                          if self.scenario_path and self.scenario_path.exists() else [])
        self.active_scenario = None
        self.selected = list(range(len(self.cams)))
        self.next_frame, self.batch_size, self.generation = 0, 1, 0
        self.latest, self.benchmark_rows = None, []
        self.cache_progress = dict(completed=0, total=0, fps=0)
        self.metrics = dict(processing_fps=0, delivered_fps=0, infer_ms=0, cache_ms=0, post_ms=0, batch_ms=0)
        self.worker = None
        self.checkpoint = None
        self.mode = 'realtime'
        self.playhead_frame = 0.0
        self.clock_started = None
        self.run_start_frame = 0
        self.processed_frames = 0

    @property
    def input_source(self):
        return self.cache if self.source_kind == 'cache' else self.video

    @property
    def cache_media(self):
        if not self.dataset:
            return None
        kind = self.source_kind
        if isinstance(self.cache, FullFrameCache) and kind == 'cache':
            kind = self.cache.media.kind
        return self.dataset.sources.get(kind, self.dataset.sources[self.dataset.direct])

    @staticmethod
    def _frame_bounds(source, scenario):
        if source is None:
            return 0, 0
        if scenario is None:
            return 0, source.count
        if not source.ready:
            raise ValueError('Prepare the cache or select Direct MP4.')
        duration = source.duration if isinstance(source, FullFrameCache) else source.count / source.fps
        if scenario['end_s'] > duration + 1e-9:
            raise ValueError('The current source does not cover this scenario. Select another source.')
        start = math.ceil(scenario['start_s'] * source.fps - 1e-9)
        end = min(source.count, math.ceil(scenario['end_s'] * source.fps - 1e-9))
        if start >= end:
            raise ValueError('This scenario contains no frames at the current source frame rate.')
        return start, end

    def frame_bounds(self):
        return self._frame_bounds(self.input_source, self.active_scenario)

    def _clock_end(self):
        source = self.input_source
        if self.active_scenario:
            return min(self.frame_bounds()[1], self.active_scenario['end_s'] * source.fps)
        return min(source.count, source.duration * source.fps) if isinstance(source, FullFrameCache) else (
            source.count if source else 0)

    def _clock_frame(self, now=None):
        position = self.playhead_frame
        if self.clock_started is not None:
            position += ((time.perf_counter() if now is None else now) - self.clock_started) * self.input_source.fps
        start, _ = self.frame_bounds()
        return min(float(self._clock_end()), max(float(start), position))

    def _freeze_clock(self):
        if self.mode == 'realtime':
            self.playhead_frame = self._clock_frame()
            self.clock_started = None
            self.next_frame = (self.frame_bounds()[1] if self.playhead_frame >= self._clock_end()
                               else int(self.playhead_frame))

    def _launch(self, phase, action):
        with self.lock:
            if self.phase not in ('idle', 'paused', 'error'):
                raise Conflict('Pause inference and wait for the current task to finish.')
            self.phase = phase
            self.message = ''
            def guarded():
                try:
                    action()
                except Exception as exc:
                    with self.lock:
                        if self.clock_started is not None:
                            self._freeze_clock()
                        self.phase, self.message = 'error', f'{type(exc).__name__}: {exc}'
                        self.changed.notify_all()
                    import traceback
                    traceback.print_exc()
            self.worker = threading.Thread(target=guarded, daemon=True)
            self.worker.start()

    def checkpoints(self):
        files = [p.name for p in sorted((self.root / 'checkpoints').glob('*.pt'))
                 if p.is_file() and not p.name.startswith('mvdetr')]
        preferred = (self.default_checkpoint, CHECKPOINTS['wavetrack'])
        return [name for name in dict.fromkeys((*preferred, *files)) if name in files]

    @property
    def default_checkpoint(self):
        return default_checkpoint(self.root, self.dataset.key if self.dataset else 'wavetrack')

    def load(self, name):
        if not self.session or name not in self.checkpoints():
            raise ValueError('Unknown checkpoint')
        def work():
            device = inference_device()
            ckpt = load_checkpoint(self.root / 'checkpoints' / name)
            origin = checkpoint_dataset(name)
            transfer = bool(self.dataset and origin != self.dataset.key)
            model, grid = load_bevnet(ckpt, device, roi=self.roi if self.dataset else None,
                                     view_count=len(self.cams), foreign=transfer)
            size = tuple(self.requested_size or ckpt['img_size'])
            grids = camera_grids(self.cams, grid).to(device)
            kind = self.source_kind
            cache = self._open_source('cache', size) if self.dataset or kind == 'cache' else None
            if self.dataset and kind == 'cache' and cache is None:
                kind = self.dataset.direct
            source = cache if kind == 'cache' else self._open_source(kind, size)
            try:
                start, _ = self._frame_bounds(source, self.active_scenario)
            except ValueError:
                if kind != 'cache':
                    source.close()
                raise
            with self.lock:
                old_video = self.video
                self.model, self.grid = model, grid
                self.device, self.thresh = device, ckpt.get('thresh', .4)
                self.grids = grids
                self.source_kind = kind
                self.cache, self.video = cache, None if kind == 'cache' else source
                self.checkpoint = name
                self._update_cache_progress()
                self._reset_position(start)
                self.benchmark_rows = []
                self.selected = list(range(len(self.cams)))
                self.message = 'Model loaded.'
                if transfer:
                    self.message = f'Using {name} with {self.dataset.key} calibration.'
                elif not source.ready:
                    self.message = 'No compatible legacy cache found. Select Direct MP4.'
            if old_video is not None:
                old_video.close()
            with self.lock:
                self.phase = 'paused'
                self.changed.notify_all()
        self._launch('loading', work)

    def _open_source(self, kind, size, *, cache_dir=None, fused_remap=None):
        if fused_remap is None:
            fused_remap = self.mp4_options['fused_remap']
        if getattr(self, 'dataset', None):
            if kind == 'cache':
                if cache_dir is None and self.cache_entry:
                    preferred = available_cache(self.dataset, self.cache_entry, size)
                    if preferred:
                        return preferred
                return available_cache(self.dataset, cache_dir or self.cache_dir, size)
            return direct_source(self.dataset.sources[kind], self.cams, size, fused_remap)
        if kind == 'mp4':
            return Mp4Source(self.session, self.cams, size, fused_remap)
        return FrameCache(self.session, self.cams, self.cache_dir, size, self.seconds)

    def _update_cache_progress(self):
        cache = self.cache
        self.cache_progress = dict(completed=cache.count if cache and cache.ready else 0,
                                   total=cache.count if cache else 0, fps=0)

    def select_source(self, kind):
        dataset = getattr(self, 'dataset', None)
        choices = tuple(dataset.sources) if dataset else ('mp4',)
        if kind != 'cache' and kind not in choices:
            raise ValueError('This dataset does not have that video source.')
        with self.lock:
            if self.phase not in ('idle', 'paused', 'error'):
                raise Conflict('Pause inference before changing the video source.')
            if kind == self.source_kind:
                return
            if self.model is None:
                self.source_kind = kind
                return
            old = self.input_source
            source = self.cache if kind == 'cache' else None
            if source is None or (dataset and kind == 'cache'):
                source = self._open_source(kind, old.size)
            if source is None:
                raise ValueError('No compatible full cache found. Generate or scan one in Settings.')
            try:
                start, end = self._frame_bounds(source, self.active_scenario)
                if end <= start:
                    raise ValueError('The source contains no frames to play.')
            except ValueError:
                if kind != 'cache':
                    source.close()
                raise
            old_video = self.video
            if kind == 'cache':
                self.cache = source
                self.video = None
            else:
                self.video = source
            self.source_kind = kind
            self._update_cache_progress()
            self._reset_position(start)
            self.benchmark_rows = []
            self.phase = 'paused'
            self.message = 'Video source is ready.' if source.ready else 'No compatible cache found.'
            if old_video is not None:
                old_video.close()

    def set_cache_directory(self, directory):
        with self.lock:
            if not self.dataset or self.phase not in ('idle', 'paused', 'error'):
                raise Conflict('Pause inference before changing the cache folder.')
            if self.model is None:
                self.cache_dir = directory
                self.cache_entry = None
                return
            cache = self._open_source('cache', self.input_source.size, cache_dir=directory)
            if self.source_kind == 'cache':
                kind = 'cache' if cache else self.dataset.direct
                source = cache or self._open_source(kind, self.input_source.size)
                try:
                    start, _ = self._frame_bounds(source, self.active_scenario)
                except ValueError:
                    if kind != 'cache':
                        source.close()
                    raise
                self.source_kind = kind
                self.video = None if cache else source
                self._reset_position(start)
                self.benchmark_rows = []
            self.cache_dir, self.cache = directory, cache
            self.cache_entry = None
            self.cache_build = None
            self._update_cache_progress()
            self.message = 'Full cache found.' if cache else 'No compatible full cache in this folder.'

    def generate_cache(self, directory, fps, save_location):
        with self.lock:
            if self.phase != 'paused' or self.model is None:
                raise Conflict('Load a model and pause inference before generating a cache.')
            media = self.cache_media
            if media is None:
                raise ValueError('Select a dataset to generate a full cache.')
            if type(fps) not in (int, float) or not 0 < fps <= media.fps:
                raise ValueError(f'Enter a cache FPS above zero and at most {media.fps:g}.')
            size = self.input_source.size
            cancel = threading.Event()
            self.cache_cancel = cancel
            self.cache_build = 'building'
            self.cache_progress = dict(completed=0, total=math.ceil(media.count / media.fps * fps), fps=0)

            def progress(completed, total, speed):
                with self.lock:
                    self.cache_progress = dict(completed=completed, total=total, fps=speed)

            def work():
                try:
                    cache = FullFrameCache(self.dataset, media, directory, size, fps)
                    cache.build(progress, cancel.is_set)
                    save_location(cache.directory)
                except CacheCancelled:
                    status, message = 'cancelled', 'Cache generation cancelled. Generate again with the same settings to resume.'
                except Exception as exc:
                    status, message = 'error', str(exc)
                else:
                    status, message = 'ready', f'Full cache ready at {fps:g} FPS.'
                    with self.lock:
                        if self.source_kind == 'cache':
                            try:
                                start, _ = self._frame_bounds(cache, self.active_scenario)
                            except ValueError:
                                self.active_scenario = None
                                start = 0
                                message += ' The active scenario was cleared for this FPS.'
                            self._reset_position(start)
                            self.benchmark_rows = []
                        self.cache_dir, self.cache_entry, self.cache = directory, cache.directory, cache
                with self.changed:
                    self.cache_build, self.message = status, message
                    self.cache_cancel = None
                    self.phase = 'paused'
                    self.changed.notify_all()

            self._launch('caching', work)

    def cancel_cache(self):
        with self.lock:
            if self.cache_cancel is None:
                raise Conflict('There is no cache generation in progress.')
            self.cache_cancel.set()
            self.cache_build = 'cancelling'

    def configure_mp4(self, fused_remap, prefetch):
        if type(fused_remap) is not bool or type(prefetch) is not bool:
            raise ValueError('Both input options must be on or off.')
        options = dict(fused_remap=fused_remap, prefetch=prefetch)
        with self.lock:
            if self.phase not in ('idle', 'paused'):
                raise Conflict('Pause inference before changing input options.')
            if options == self.mp4_options:
                return
            if self.source_kind == 'mp4' and self.video is not None and fused_remap != self.mp4_options['fused_remap']:
                old_video = self.video
                replacement = self._open_source('mp4', old_video.size, fused_remap=fused_remap)
                try:
                    self._frame_bounds(replacement, self.active_scenario)
                except ValueError:
                    replacement.close()
                    raise
                self.video = replacement
                self.generation += 1
                self.latest = None
                old_video.close()
            self.mp4_options = options
            self.benchmark_rows = []
            self.metrics = dict.fromkeys(self.metrics, 0)
            self.message = 'Input options applied.'

    def _reset_position(self, frame):
        self.next_frame, self.playhead_frame = frame, float(frame)
        self.clock_started = None
        self.processed_frames, self.run_start_frame = 0, frame
        self.generation += 1
        self.latest = None
        self.metrics = dict(processing_fps=0, delivered_fps=0, infer_ms=0, cache_ms=0, post_ms=0, batch_ms=0)

    def _sync(self):
        if self.device.type != 'cpu':
            getattr(torch, self.device.type).synchronize()

    def _forward(self, start, count, selected, array=None):
        t = time.perf_counter()
        if array is None:
            array = self.input_source.batch(start, count, selected)
        cpu = torch.from_numpy(array)
        x = cpu.to(self.device)
        grids = self.grids[selected][None].expand(count, -1, -1, -1, -1)
        self._sync()
        cache_ms = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        accelerated = self.device.type != 'cpu'
        with torch.inference_mode(), torch.autocast(self.device.type, enabled=accelerated):
            heat = self.model(x if accelerated else x.float(), grids).float().sigmoid()[:, 0]
        self._sync()
        infer_ms = (time.perf_counter() - t) * 1000
        return heat, cache_ms, infer_ms

    def _detections(self, heat):
        # One GPU-to-CPU copy per batch; decode small heatmaps without repeated CUDA synchronizations.
        heat = heat.cpu()
        output = []
        for h in heat:
            people = []
            maximum = 100 if getattr(self, 'dataset', None) and self.dataset.key != 'wavetrack' else 20
            for X, Y, score in decode(h, self.grid, self.thresh, min_dist=.3, max_det=maximum):
                boxes = {}
                for k in self.selected:
                    box = cylinder_box(self.cams[k], np.array([X, Y]), .25, 1.8)[0]
                    if box is not None:
                        boxes[str(k)] = [float(v) for v in box]
                people.append(dict(x=X, y=Y, score=score, boxes=boxes,
                                   inside=bool(self.roi.contains(X, Y))))
            output.append(people)
        return output

    def _prefetch_enabled(self):
        return self.mp4_options['prefetch']

    def _benchmark_samples(self, begin, end, bs, selected):
        """Measure pipeline cycles after warm-up, including exposed input waits.

        As in playback, prepare only one upcoming batch during the current
        forward/postprocess cycle. No playback FPS pacing is applied here.
        """
        source = self.input_source
        context = ThreadPoolExecutor(max_workers=1) if self._prefetch_enabled() else nullcontext()
        samples, pending = [], None
        with context as pool:
            for repetition in range(4):  # one excluded warm-up, three measured cycles
                sample_start = min(begin + repetition * bs, end - bs)
                t = time.perf_counter()
                array = None
                if pool is not None:
                    array = (pending.result() if pending is not None
                             else source.batch(sample_start, bs, selected))
                    pending = None
                    if repetition < 3:
                        upcoming = min(begin + (repetition + 1) * bs, end - bs)
                        pending = pool.submit(source.batch, upcoming, bs, selected)
                heat, _, _ = self._forward(sample_start, bs, selected, array)
                self._detections(heat)
                if repetition:
                    samples.append((time.perf_counter() - t) * 1000)
        return samples

    def benchmark(self):
        def work():
            source = self.input_source
            rows = []
            selected = list(self.selected)
            start, end = self.frame_bounds()
            begin = min(self.next_frame, max(start, end - 32))
            for bs in (1, 2, 4, 8):
                try:
                    if bs > end - start:
                        rows.append(dict(batch=bs, error='Not enough frames in the selected interval'))
                    else:
                        samples = self._benchmark_samples(begin, end, bs, selected)
                        median = float(np.median(samples))
                        p95 = float(np.percentile(samples, 95))
                        rows.append(dict(batch=bs, fps=bs * 1000 / median, median_ms=median,
                                         p95_ms=p95, input_span_ms=(bs - 1) * 1000 / source.fps,
                                         cameras=len(selected), samples=3, error=None))
                except torch.OutOfMemoryError:
                    if self.device.type != 'cpu':
                        getattr(torch, self.device.type).empty_cache()
                    rows.append(dict(batch=bs, error='Out of memory'))
                rows[-1].update(source=self.source_kind, prefetch=self._prefetch_enabled())
                with self.lock:
                    self.benchmark_rows = list(rows)
                    self.message = f'Finished benchmarking batch size {bs}.'
                print('BENCHMARK', rows[-1], flush=True)
            valid = [r for r in rows if not r['error']]
            if not valid:
                raise ValueError('No batch size fits GPU memory.')
            # Prefer real-time input throughput within the latency budget, then smallest delay.
            realtime = [r for r in valid if r['fps'] >= source.fps and r['p95_ms'] <= self.latency_ms]
            eligible = [r for r in valid if r['p95_ms'] <= self.latency_ms]
            best = min(realtime, key=lambda r: r['p95_ms']) if realtime else max(eligible, key=lambda r: r['fps']) if eligible else min(valid, key=lambda r: r['p95_ms'])
            if self.source_kind == 'cache':
                atomic_json(self.cache.directory / 'benchmark_latest.json', dict(
                    checkpoint=self.checkpoint, cameras=selected, selected_batch=best['batch'],
                    latency_budget_ms=self.latency_ms, measurements=rows,
                    source=self.source_kind, prefetch=self._prefetch_enabled(),
                    device=getattr(torch, self.device.type).get_device_name() if self.device.type != 'cpu' else 'cpu'))
            with self.lock:
                self.batch_size = best['batch']
                self.phase = 'paused'
                self.message = (f"Selected batch size {best['batch']} at {best['fps']:.1f} FPS "
                                f"with a {self.latency_ms:g} ms latency budget.")
        with self.lock:
            self._require_source()
            self._launch('benchmarking', work)

    def _require_source(self):
        source = self.input_source
        if self.model is None or source is None or not source.ready:
            raise ValueError('The model and video source must be ready.')

    def _scenario(self, identifier):
        for scenario in self.scenarios:
            if scenario['id'] == identifier:
                return scenario
        raise ValueError('Scenario not found.')

    def _write_scenarios(self, scenarios):
        self.scenario_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(self.scenario_path, scenarios)
        self.scenarios = scenarios

    def _require_scenario_edit(self):
        if self.phase != 'paused':
            raise Conflict('Pause inference before editing scenarios.')
        self._require_source()

    def save_scenario(self, name, start_s, end_s, identifier=None):
        with self.lock:
            self._require_scenario_edit()
            source = self.input_source
            if not isinstance(name, str) or not 0 < len(name.strip()) <= 80:
                raise ValueError('The scenario name must be 1 to 80 characters long.')
            if (type(start_s) not in (int, float) or type(end_s) not in (int, float)
                    or not 0 <= start_s < end_s <= source.source_count / source.source_fps + 1e-9):
                raise ValueError('Choose valid start and end times within the video.')
            if identifier is not None:
                self._scenario(identifier)
            scenario = dict(id=identifier if identifier is not None else uuid4().hex,
                            name=name.strip(), start_s=start_s, end_s=end_s)
            if self.active_scenario is not None and identifier == self.active_scenario['id']:
                start, _ = self._frame_bounds(source, scenario)
            else:
                start = None
            if identifier is None:
                scenarios = [*self.scenarios, scenario]
            else:
                scenarios = [scenario if item['id'] == identifier else item for item in self.scenarios]
            self._write_scenarios(scenarios)
            if start is not None:
                self._activate_scenario(scenario, start)

    def delete_scenario(self, identifier):
        with self.lock:
            self._require_scenario_edit()
            self._scenario(identifier)
            self._write_scenarios([s for s in self.scenarios if s['id'] != identifier])
            if self.active_scenario and self.active_scenario['id'] == identifier:
                self._activate_scenario(None, min(self.next_frame, self.input_source.count - 1))

    def select_scenario(self, identifier):
        with self.lock:
            self._require_scenario_edit()
            if identifier is None:
                frame = min(self.next_frame, self.input_source.count - 1)
                self._activate_scenario(None, frame)
            else:
                scenario = self._scenario(identifier)
                start, _ = self._frame_bounds(self.input_source, scenario)
                self._activate_scenario(scenario, start)

    def _activate_scenario(self, scenario, frame):
        self.active_scenario = scenario
        self._reset_position(frame)
        self.benchmark_rows = []
        self.message = 'Scenario activated.' if scenario else 'Free playback enabled.'

    def configure(self, frame=None, cameras=None, batch=None, mode=None):
        with self.lock:
            if self.phase != 'paused':
                raise Conflict('Pause inference before seeking or changing cameras or batch size.')
            self._require_source()
            if frame is not None:
                start, end = self.frame_bounds()
                if type(frame) is not int or not start <= frame < end:
                    raise ValueError('The frame is outside the playback range.')
            if cameras is not None:
                if getattr(self.model, 'fusion', None) == 'concat':
                    raise ValueError('This checkpoint uses a fixed set of cameras.')
                if not isinstance(cameras, list) or not cameras or any(type(k) is not int or not 0 <= k < len(self.cams) for k in cameras) or len(set(cameras)) != len(cameras):
                    raise ValueError('Select at least one valid camera.')
            if batch is not None and (type(batch) is not int or batch not in (1, 2, 4, 8)):
                raise ValueError('Batch size must be 1, 2, 4, or 8.')
            if mode is not None and mode not in ('sequential', 'realtime'):
                raise ValueError('Select Realtime or Sequential playback.')
            if mode is not None and mode != self.mode:
                self.mode = mode
                self.playhead_frame = float(self.next_frame)
            position = self.next_frame if frame is None else frame
            playhead = self.playhead_frame if frame is None else float(frame)
            if cameras is not None:
                self.selected = sorted(cameras)
                self.benchmark_rows = []
            if batch is not None:
                self.batch_size = batch
            self._reset_position(position)
            self.playhead_frame = playhead
            self.run_start_frame = int(playhead)
            self.message = 'Settings updated. Resume inference from the selected position.'

    def resume(self):
        with self.lock:
            self._require_source()
            if self.phase != 'paused':
                raise Conflict('The demo is not ready to resume.')
            if self.next_frame >= self.frame_bounds()[1]:
                raise ValueError('The playback interval has ended. Seek back before resuming.')
            if self.mode == 'realtime':
                self.clock_started = time.perf_counter()
                self.run_start_frame = int(self.playhead_frame)
                self.processed_frames = 0
            self._launch('running', self._run_realtime if self.mode == 'realtime' else self._run)

    def pause(self):
        with self.lock:
            if self.phase == 'running':
                self._freeze_clock()
                self.phase = 'pausing'
                self.message = 'Waiting for the current batch to finish…'
                self.changed.notify_all()
            elif self.phase not in ('paused', 'pausing'):
                raise Conflict('Inference is not running.')

    def _run_realtime(self):
        self._run_with_prefetch(self._realtime_loop)

    def _run(self):
        self._run_with_prefetch(self._sequential_loop)

    def _run_with_prefetch(self, loop):
        context = ThreadPoolExecutor(max_workers=1) if self._prefetch_enabled() else nullcontext()
        with context as pool:
            message = loop(pool)
            with self.changed:
                # Keep controls locked until any in-flight source read finishes.
                self.phase = 'pausing'
        with self.changed:
            self.phase, self.message = 'paused', message
            self.changed.notify_all()

    def _realtime_loop(self, pool):
        """Infer due frames, with at most one upcoming source frame being prepared."""
        end = self._clock_end()
        _, frame_end = self.frame_bounds()
        source = self.input_source
        source_fps = source.fps
        pending = None
        gpu_cycle_s = 0.0
        began = time.perf_counter()
        delivered = deque([(began, 0)], maxlen=60)
        published, last_input = 0, -1
        while True:
            with self.changed:
                if self.phase != 'running':
                    return 'Paused. You can seek or change cameras.'
                position = self._clock_frame()
                if position >= end:
                    self._freeze_clock()
                    return 'Playback interval ended.'
                due = pending[0] if pending is not None else last_input + 1
                if position < due:
                    self.changed.wait(timeout=max(.001, (due - position) / source_fps))
                    continue
                frame = pending[0] if pending is not None else int(position)
                selected = list(self.selected)
            t = time.perf_counter()
            if pool is None:
                gpu_begin = t
                heat, cache_ms, infer_ms = self._forward(frame, 1, selected)
            else:
                array = pending[1].result() if pending is not None else source.batch(frame, 1, selected)
                pending = None
                with self.changed:
                    if self.phase != 'running':
                        return 'Paused. You can seek or change cameras.'
                    # Aim for the playhead at the end of the current GPU cycle. A prepared
                    # frame can be consumed only once its timestamp is due.
                    position = self._clock_frame()
                    upcoming = max(frame + 1, int(position + gpu_cycle_s * source_fps))
                    if position < end and upcoming < frame_end:
                        pending = upcoming, pool.submit(source.batch, upcoming, 1, selected)
                gpu_begin = time.perf_counter()
                heat, cache_ms, infer_ms = self._forward(frame, 1, selected, array)
                cache_ms += (gpu_begin - t) * 1000
            post = time.perf_counter()
            people = self._detections(heat)[0]
            finished = time.perf_counter()
            measured_cycle = min(.25, finished - gpu_begin)
            gpu_cycle_s = .7 * gpu_cycle_s + .3 * measured_cycle if gpu_cycle_s else measured_cycle
            post_ms = (finished - post) * 1000
            batch_ms = (finished - t) * 1000
            with self.lock:
                if self.phase != 'running':
                    return 'Paused. You can seek or change cameras.'
                if self.active_scenario is None and self._clock_frame() >= end:
                    self._freeze_clock()
                    return 'Playback interval ended.'
                now = time.perf_counter()
                published += 1
                self.processed_frames += 1
                delivered.append((now, published))
                delivered_fps = (delivered[-1][1] - delivered[0][1]) / max(.000001, now - delivered[0][0])
                self.latest = dict(frame=frame, time_s=frame / source_fps, people=people,
                                   generation=self.generation, cameras=selected)
                self.next_frame = frame + 1
                self.metrics = dict(processing_fps=1000 / batch_ms, delivered_fps=delivered_fps,
                                    infer_ms=infer_ms, cache_ms=cache_ms, post_ms=post_ms, batch_ms=batch_ms)
                last_input = frame

    def _sequential_loop(self, pool):
        _, end = self.frame_bounds()
        source = self.input_source
        source_fps = source.fps
        pending = None
        deadline = time.perf_counter()
        delivered = deque([(deadline, 0)], maxlen=60)
        published = 0
        while True:
            with self.lock:
                if self.phase != 'running' or self.next_frame >= end:
                    if self.next_frame >= end:
                        return 'Inference finished for this playback interval.'
                    return 'Paused. You can seek or change cameras.'
                start = self.next_frame
                count = min(self.batch_size, end - start)
                selected = list(self.selected)
            t = time.perf_counter()
            if pool is None:
                heat, cache_ms, infer_ms = self._forward(start, count, selected)
            else:
                array = pending.result() if pending is not None else source.batch(start, count, selected)
                pending = None
                with self.changed:
                    if self.phase != 'running':
                        return 'Paused. You can seek or change cameras.'
                    upcoming = start + count
                    if upcoming < end:
                        pending = pool.submit(source.batch, upcoming, min(self.batch_size, end - upcoming), selected)
                gpu_begin = time.perf_counter()
                heat, cache_ms, infer_ms = self._forward(start, count, selected, array)
                cache_ms += (gpu_begin - t) * 1000
            post = time.perf_counter()
            detections = self._detections(heat)
            post_ms = (time.perf_counter() - post) * 1000
            batch_ms = (time.perf_counter() - t) * 1000
            for j, people in enumerate(detections):
                with self.changed:
                    # All frames are processed in order; pause discards unpublished batch tail.
                    wait = deadline - time.perf_counter()
                    if wait > 0:
                        self.changed.wait_for(lambda: self.phase != 'running', timeout=wait)
                    if self.phase != 'running':
                        return 'Paused. You can seek or change cameras.'
                    now = time.perf_counter()
                    published += 1
                    delivered.append((now, published))
                    delivered_fps = (delivered[-1][1] - delivered[0][1]) / max(.000001, delivered[-1][0] - delivered[0][0])
                    frame = start + j
                    self.latest = dict(frame=frame, time_s=frame / source_fps, people=people,
                                       generation=self.generation, cameras=selected)
                    self.next_frame = frame + 1
                    self.metrics = dict(processing_fps=count * 1000 / batch_ms, delivered_fps=delivered_fps,
                                        infer_ms=infer_ms, cache_ms=cache_ms, post_ms=post_ms, batch_ms=batch_ms)
                    deadline = max(deadline, now) + 1 / source_fps

    def jpeg(self, generation, frame, camera):
        with self.lock:
            start, end = self.frame_bounds()
            source = self.input_source
            if generation != self.generation or not start <= frame < end or source is None:
                raise ValueError('Stale image request or frame outside playback range')
        return source.jpeg(frame, camera)

    def state(self):
        with self.lock:
            cache = self.cache
            source = self.input_source
            start, end = self.frame_bounds()
            fps = source.fps if source else CACHE_FPS
            clock_end = self._clock_end()
            playhead = self._clock_frame() if source and self.mode == 'realtime' else min(self.next_frame, clock_end)
            return dict(phase=self.phase, message=self.message, checkpoint=self.checkpoint,
                        mode=self.mode, effective_batch=1 if self.mode == 'realtime' else self.batch_size,
                        playhead_s=playhead / fps,
                        clock_running=self.mode == 'realtime' and self.clock_started is not None,
                        processed_frames=self.processed_frames,
                        skipped_frames=max(0, int(playhead) - self.run_start_frame - self.processed_frames) if self.mode == 'realtime' else 0,
                        cache_ready=bool(cache and cache.ready), cache_progress=dict(self.cache_progress),
                        cache_build=self.cache_build,
                        cache_fps=cache.fps if cache and cache.ready else None,
                        cache_size_gb=(cache.size_bytes if cache.ready else MAX_CACHE_BYTES) / 1e9 if cache else 0,
                        cache_path=str(cache.directory) if cache else None,
                        camera_toggles=getattr(self.model, 'fusion', None) != 'concat',
                        input_size=list(source.size) if source else None,
                        total_frames=source.count if source else 0, source_fps=fps,
                        source=self.source_kind, source_ready=bool(source and source.ready),
                        mp4_options=dict(self.mp4_options),
                        video_duration_s=source.source_count / source.source_fps if source else 0,
                        timeline_start=start, timeline_end=end, timeline_end_s=clock_end / fps,
                        scenarios=list(self.scenarios), active_scenario=self.active_scenario,
                        next_frame=self.next_frame, selected=list(self.selected), batch=self.batch_size,
                        generation=self.generation, latest=self.latest, metrics=dict(self.metrics),
                        benchmark=list(self.benchmark_rows), latency_budget_ms=self.latency_ms,
                        device=str(self.device) if self.model is not None else None)

    def scene(self):
        if self.site_map is None:
            return dict(cameras=[], checkpoints=[], default_checkpoint=None)
        roi = self.roi
        XY = np.array([[roi.x0, roi.y0], [roi.x1, roi.y0], [roi.x1, roi.y1], [roi.x0, roi.y1], [roi.x0, roi.y0]])
        cameras = []
        for k, cam in enumerate(self.cams):
            uv = cam.ground_to_image(XY)
            camera = dict(id=k, name=cam.name, size=list(cam.image_size), center=cam.center.tolist(),
                          K=cam.K.tolist(), R=cam.R.tolist(), depth_sign=cam.depth_sign,
                          max_ray=float(cam.max_ray), min_depth=cam.MIN_DEPTH,
                          roi=[[float(x), float(y)] if np.isfinite([x, y]).all() else None for x, y in uv])
            if cam.dist.size == 5 and np.any(cam.dist):
                camera['dist'] = cam.dist.ravel().tolist()
            cameras.append(camera)
        site_map = self.site_map
        return dict(cameras=cameras, roi=vars(roi), site=site_map.site, pitch=self.pitch,
                    map=dict(extent=site_map.extent, padding=site_map.PREVIEW_PADDING, scale=site_map.scale),
                    checkpoints=self.checkpoints(), default_checkpoint=self.default_checkpoint, seconds=self.seconds)

    def site_map_png(self, roi, routes, mask):
        if roi not in (0, 1) or routes not in (0, 1) or not 0 < mask < 1 << len(self.cams):
            raise ValueError('Invalid site map options')
        active = [number for i, number in enumerate(self.site_map.numbers) if mask & (1 << i)]
        return self._render_map(self.site_map, roi, routes, tuple(active))

    @staticmethod
    @lru_cache(maxsize=12)
    def _render_map(site_map, roi, routes, active):
        image = site_map.preview(roi, routes, active)
        ok, png = cv2.imencode('.png', image)
        if not ok:
            raise ValueError('Cannot encode site map')
        return png.tobytes()

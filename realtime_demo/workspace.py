"""Persist local paths and replace inference engines at a safe frame boundary."""

import json
import os
from pathlib import Path
from threading import Lock, RLock

from realtime_demo.cache import atomic_json
from realtime_demo.datasets import CHECKPOINTS, LABELS, scan_datasets
from realtime_demo.engine import Conflict, Engine


def user_config():
    if os.name == 'nt':
        return Path(os.environ.get('LOCALAPPDATA', Path.home())) / 'BEVLab/realtime.json'
    return Path(os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config')) / 'bev-lab/realtime.json'


def user_cache():
    if os.name == 'nt':
        return Path(os.environ.get('LOCALAPPDATA', Path.home())) / 'BEVLab/full-cache'
    return Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache')) / 'bev-lab'


class Workspace:
    def __init__(self, root, args):
        self.root, self.args = root, args
        self.config_path = args.config
        self.default_cache = args.cache_dir or (root / 'realtime_demo/cache_data' if args.session else user_cache())
        self.mutations = Lock()
        self.lock = RLock()
        self.transition = False
        self.revision = 0
        self.engine = self._new_engine()
        self.datasets, self.errors, self.active = {}, {}, None
        self.config = {}
        if args.session:
            if args.prepare:
                self.engine.load(args.checkpoint or self.engine.default_checkpoint)
            return
        if self.config_path.is_file():
            try:
                self.config = json.loads(self.config_path.read_text(encoding='utf-8'))
                if not isinstance(self.config, dict):
                    raise ValueError('Settings must be a JSON object.')
            except (ValueError, OSError) as exc:
                self.config = {}
                self.engine.message = f'Could not read saved settings. {exc}'
        saved = args.dataset_root or self.config.get('dataset_root')
        if saved:
            try:
                directory, datasets, errors = scan_datasets(root, str(saved))
                active = self._choose_dataset(datasets)
                self.engine = self._new_engine(datasets.get(active))
                self.datasets, self.errors, self.active = datasets, errors, active
                if active:
                    self.engine.load(args.checkpoint or self.engine.default_checkpoint)
                else:
                    self.engine.message = 'No supported dataset was found in this folder.'
                if args.dataset_root:
                    self._save(dict(self.config, dataset_root=str(directory), active=active))
            except (ValueError, OSError, KeyError) as exc:
                self.engine.message = str(exc)

    def _choose_dataset(self, datasets):
        preferred = self.active or self.config.get('active')
        return preferred if preferred in datasets else next(iter(datasets), None)

    def _new_engine(self, dataset=None):
        settings = self.config.get('datasets', {}).get(dataset.key, {}) if dataset else {}
        folder = Path(settings.get('cache_dir', self.default_cache))
        session = dataset.path if dataset else self.args.session
        engine = Engine(self.root, session, folder, self.args.input_size, self.args.seconds,
                        self.args.latency_ms, dataset=dataset)
        if dataset:
            source = settings.get('source', 'cache')
            engine.source_kind = source if source in ('cache', *dataset.sources) else dataset.direct
            entry = settings.get('cache_entry')
            if entry:
                engine.cache_entry = Path(entry)
        elif not session:
            engine.message = 'Choose your Dataset folder to begin.'
        return engine

    def _save(self, settings):
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(self.config_path, settings)
        with self.lock:
            self.config = settings

    def _save_dataset_settings(self, **changes):
        entries = self.config.get('datasets', {})
        settings = dict(entries.get(self.active, {}), **changes)
        self._save(dict(self.config, datasets=dict(entries, **{self.active: settings})))

    def state(self):
        with self.lock:
            engine = self.engine
            value = engine.state()
            dataset = self.datasets.get(self.active)
            media = engine.cache_media
            if self.transition:
                value.update(phase='switching', message='Switching dataset…', clock_running=False)
            value.update(
                dataset=self.active or ('wavetrack' if self.args.session else None),
                datasets=[dict(id=key, name=LABELS[key]) for key in self.datasets],
                dataset_errors=dict(self.errors),
                dataset_root=self.config.get('dataset_root') or str(self.args.dataset_root or ''),
                needs_dataset_folder=not self.args.session and not self.datasets,
                legacy=bool(self.args.session),
                scene_revision=self.revision,
                sources=['cache', *dataset.sources] if dataset else (['cache', 'mp4'] if self.args.session else []),
                can_fuse_remap=bool(self.args.session or (
                    dataset and 'mp4' in dataset.sources and dataset.sources['mp4'].original_intrinsics)),
                cache_directory=str(engine.cache_dir) if dataset else None,
                cache_media=dict(kind=media.kind, fps=media.fps, total=media.count,
                                 cameras=len(dataset.cams)) if media else None,
                using_wave_weights=bool(self.active and self.active != 'wavetrack'
                                        and value['checkpoint'] == CHECKPOINTS['wavetrack']),
            )
            return value

    def scene(self):
        with self.lock:
            return dict(self.engine.scene(), scene_revision=self.revision)

    def current_engine(self, revision=None):
        with self.lock:
            if revision is not None and revision != self.revision:
                raise ValueError('Stale scene request.')
            return self.engine

    def scan(self, directory):
        directory, datasets, errors = scan_datasets(self.root, directory)
        active = self._choose_dataset(datasets)
        self._replace(datasets, errors, active, str(directory))

    def refresh(self):
        saved = self.config.get('dataset_root')
        if not saved or self.args.session:
            return
        directory, datasets, errors = scan_datasets(self.root, saved)
        key = self._choose_dataset(datasets)
        before, after = self.engine.dataset, datasets.get(key)
        same = before and after and (before.key, before.path, before.calibration, before.sources) == (
            after.key, after.path, after.calibration, after.sources)
        if same or (before is None and after is None):
            with self.lock:
                self.datasets, self.errors = datasets, errors
        else:
            self._replace(datasets, errors, key, str(directory))

    def select(self, key):
        if key not in self.datasets:
            raise ValueError('Choose a detected dataset.')
        if key != self.active:
            self._replace(self.datasets, self.errors, key, self.config['dataset_root'])

    def _replace(self, datasets, errors, key, directory):
        old = self.engine
        if old.phase not in ('idle', 'paused', 'error', 'running', 'pausing'):
            raise Conflict('Wait for the current task to finish before switching datasets.')
        replacement = self._new_engine(datasets.get(key))
        running = old.phase == 'running'
        with self.lock:
            self.transition = True
        try:
            if running:
                old.pause()
            if old.worker is not None:
                old.worker.join()
            replacement.generation = old.generation + 1
            replacement.mode = old.mode
            replacement.mp4_options = dict(old.mp4_options)
            if key:
                replacement.load(replacement.default_checkpoint)
                replacement.worker.join()
                if replacement.phase == 'error':
                    raise ValueError(replacement.message)
            else:
                replacement.message = 'No supported dataset was found in this folder.'
            with self.lock:
                self._save(dict(self.config, dataset_root=directory, active=key))
                self.engine, self.active = replacement, key
                self.datasets, self.errors = datasets, errors
                self.revision += 1
            if old.video is not None:
                old.video.close()
            if running and replacement.input_source is not None:
                replacement.resume()
        except Exception:
            if self.engine is old:
                if replacement.video is not None:
                    replacement.video.close()
                if running and old.phase == 'paused' and old.next_frame < old.frame_bounds()[1]:
                    old.resume()
            raise
        finally:
            with self.lock:
                self.transition = False

    def select_source(self, kind):
        with self.lock:
            self.engine.select_source(kind)
            if self.active is not None:
                self._save_dataset_settings(source=kind)

    @staticmethod
    def _cache_folder(directory):
        if not isinstance(directory, str) or not directory.strip():
            raise ValueError('Enter a cache folder path.')
        folder = Path(directory.strip()).expanduser()
        if not folder.is_absolute() or (folder.exists() and not folder.is_dir()):
            raise ValueError('Enter an absolute cache folder path on this machine.')
        return folder.resolve()

    def cache_directory(self, directory):
        folder = self._cache_folder(directory)
        with self.lock:
            self.engine.set_cache_directory(folder)
            self._save_dataset_settings(cache_dir=str(folder), cache_entry=None, source=self.engine.source_kind)

    def generate_cache(self, directory, fps):
        folder = self._cache_folder(directory)

        def save_location(entry):
            self._save_dataset_settings(cache_dir=str(folder), cache_entry=str(entry))

        self.engine.generate_cache(folder, fps, save_location)

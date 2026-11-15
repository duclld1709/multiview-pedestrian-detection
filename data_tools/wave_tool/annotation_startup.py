"""Choose image, calibration and annotation folders on every startup."""
import json
import os
from pathlib import Path

from dataset import atomic_json, dataset_kind
from recording import DEFAULT_SOURCE

SETTINGS_PATH = Path(__file__).with_name('.settings.json')
PATH_FIELDS = (
    ('source', 'Images (dataset root or Image_subsets)'),
    ('calib', 'Calibration (extrinsic, intrinsic_zero; WaveTrack also needs grid.json)'),
    ('annotations_dir', 'Annotations (QA / WaveTrack / WILDTRACK annotations_positions)'),
)


def ask_startup_paths(initial, error=''):
    import tkinter as tk
    from tkinter import filedialog, simpledialog

    try:
        root = tk.Tk()
    except tk.TclError:
        if error:
            print(error)
        try:
            return {key: input(f'{label} [{initial[key]}] (Ctrl+C to cancel): ').strip()
                    or str(initial[key]) for key, label in PATH_FIELDS}
        except (KeyboardInterrupt, EOFError):
            return None
    root.withdraw()

    class StartupDialog(simpledialog.Dialog):
        def body(self, master):
            self.entries = {}
            for row, (key, label) in enumerate(PATH_FIELDS):
                tk.Label(master, text=label, anchor='w').grid(row=row * 2, column=0, sticky='w')
                entry = tk.Entry(master, width=85)
                entry.insert(0, str(initial[key]))
                entry.grid(row=row * 2 + 1, column=0, sticky='ew', pady=(0, 10))
                self.entries[key] = entry
                tk.Button(master, text='Browse...', command=lambda k=key: self.browse(k)).grid(
                    row=row * 2 + 1, column=1, padx=(8, 0))
            if error:
                tk.Label(master, text=error, fg='red', wraplength=650, justify='left').grid(
                    row=6, column=0, columnspan=2, sticky='w')
            return self.entries['source']

        def browse(self, key):
            entry = self.entries[key]
            folder = filedialog.askdirectory(parent=self, initialdir=entry.get(), mustexist=True)
            if folder:
                entry.delete(0, tk.END)
                entry.insert(0, folder)

        def apply(self):
            self.result = {key: entry.get() for key, entry in self.entries.items()}

    try:
        return StartupDialog(root, title='WaveTrack startup paths').result
    finally:
        root.destroy()


def choose_startup_paths(args, settings_path=None):
    """Return accepted paths; cancellation leaves settings and arguments untouched."""
    settings_path = Path(settings_path) if settings_path else SETTINGS_PATH
    try:
        settings = json.loads(settings_path.read_text(encoding='utf-8'))
        if not isinstance(settings, dict):
            settings = {}
    except (OSError, ValueError):
        settings = {}
    source = Path(args.source or settings.get('source') or DEFAULT_SOURCE)
    dataset_root = source.parent if source.name.lower() == 'image_subsets' else source
    kind = getattr(args, 'dataset', 'auto')
    reuse_settings = not args.source or not settings.get('source') or Path(settings['source']).resolve() == source.resolve()
    calib_default = args.calib or (settings.get('calib') if reuse_settings else None) or dataset_root / 'calibrations'
    labels_name = 'annotations_positions' if dataset_kind(calib_default, kind) == 'wildtrack' else 'annotations'
    initial = {
        'source': str(source),
        'calib': str(calib_default),
        'annotations_dir': str(args.labels_dir or (settings.get('annotations_dir') if reuse_settings else None)
                               or dataset_root / labels_name),
    }
    error = ''
    while True:
        values = ask_startup_paths(initial, error)
        if values is None:
            return None
        initial = values
        try:
            paths = {}
            for key, label in PATH_FIELDS:
                value = values[key].strip().strip('"')
                if not value:
                    raise ValueError(f'{label}: please choose a folder.')
                path = Path(os.path.expandvars(value)).expanduser().resolve()
                if not path.is_dir():
                    raise ValueError(f'{label}: folder does not exist: {path}')
                paths[key] = path
            source = paths['source']
            images = source if source.name.lower() == 'image_subsets' else source / 'Image_subsets'
            if not images.is_dir() or not any(p.is_dir() for p in images.glob('C[0-9]*')):
                raise ValueError(f'Image folder must contain Image_subsets/C1, C2, ...: {source}')
            calib = paths['calib']
            if not ((dataset_kind(calib, kind) == 'wildtrack' or (calib / 'grid.json').is_file())
                    and (calib / 'extrinsic').is_dir()
                    and (calib / 'intrinsic_zero').is_dir()):
                raise ValueError(f'Calibration folder must contain extrinsic/, intrinsic_zero/ and grid.json: {calib}')
        except (OSError, ValueError) as exc:
            error = str(exc)
            continue
        atomic_json(settings_path, {**settings, **{key: str(path) for key, path in paths.items()}})
        return paths

"""Import label_tools grid-node detections into an independent QA review project."""
import importlib.util
import json
import math
from pathlib import Path

import cv2


def xml_manifest(path):
    folder = Path(path).parent if Path(path).is_file() else Path(path)
    if not (folder / 'grid.json').is_file():
        return None
    # Reuse the exact XML reader used by label_tool without importing its GUI.
    spec = importlib.util.spec_from_file_location('_label_tool_xml',
        Path(__file__).with_name('_xml_dataset.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cams, grid = module.load_calibration(folder.parent, folder)
    return {**grid, 'path': str(folder / 'grid.json'), 'xml_grid': True,
            'cameras': [{'name': c.name, 'image_size': c.image_size,
                         'camera_matrix': c.K.tolist(), 'R': c.R.tolist(),
                         'rvec': cv2.Rodrigues(c.R)[0].ravel().tolist(),
                         'tvec_m': c.t.tolist(), 'dist_coeffs': [0.] * 5} for c in cams]}


def encode_position(grid, x, y):
    roi = grid['roi']
    if not (math.isfinite(x) and math.isfinite(y) and
            roi['x0'] <= x < roi['x1'] and roi['y0'] <= y < roi['y1']):
        return None
    cell = roi['cell_cm'] / 100
    col = math.floor((x - roi['x0']) / cell + 1e-9)
    row = math.floor((y - roi['y0']) / cell + 1e-9)
    rows, cols = roi['grid_shape_rows_cols']
    return row * cols + col if 0 <= row < rows and 0 <= col < cols else None


def create_import_project(body, models, detectors):
    from project import Project
    import labeling as lb
    root = Path(body['root']).resolve()
    calib = Path(body.get('calib') or root / 'calibrations').resolve()
    manifest = xml_manifest(calib)
    if manifest is None:
        raise ValueError('Import requires extrinsic/, intrinsic_zero/ and grid.json.')
    labels = Path(body.get('labels_dir') or root / 'annotations').resolve()
    if not str(body.get('out_dir') or '').strip():
        raise ValueError('Choose a new QA project output folder.')
    out = Path(body['out_dir']).resolve()
    if out == root or out == labels or labels in out.parents:
        raise ValueError('Choose a separate QA project folder outside the source annotations.')
    if (out / 'label_project.json').exists():
        raise ValueError('Project already exists. Open it or choose a new output folder.')
    dataset = lb.Dataset(root)
    if dataset.mode != 'images':
        raise ValueError('Import requires the same Image_subsets used by label_tool.')
    available = set(dataset.frames)
    grid = manifest['roi']
    rows, cols = grid['grid_shape_rows_cols']
    cell = grid['cell_cm'] / 100
    states = {}
    for path in sorted(labels.glob('*.json')):
        if not path.stem.isdigit():
            continue
        frame = int(path.stem)
        if frame not in available:
            raise ValueError(f'{path.name}: frame is missing from the selected images.')
        if frame in states:
            raise ValueError(f'Duplicate annotation frame: {frame}')
        meta = labels / '.editor' / path.name
        settings = json.loads(meta.read_text(encoding='utf-8')) if meta.exists() else {}
        if settings.get('position_convention') != 'grid-node-v1':
            raise ValueError(f'{path.name}: migrate legacy annotations with label_tools/scripts/migrate_grid_labels.py first.')
        radius, height = float(settings.get('radius', .25)), float(settings.get('height', 1.8))
        if not all(math.isfinite(v) and v > 0 for v in (radius, height)):
            raise ValueError(f'{path.name}: invalid cylinder dimensions.')
        records = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(records, list):
            raise ValueError(f'{path.name}: expected a list of people.')
        persons = []
        for i, record in enumerate(records):
            pid = record.get('positionID')
            if isinstance(pid, bool) or not isinstance(pid, int) or not 0 <= pid < rows * cols:
                raise ValueError(f'{path.name}: invalid positionID {pid}.')
            row, col = divmod(pid, cols)
            x, y = grid['x0'] + col * cell, grid['y0'] + row * cell
            persons.append({'uid': f'import-{frame}-{i}', 'x': x, 'y': y, 'r': radius, 'h': height,
                            'origin': 'manual', 'status': 'manual', 'verified': False,
                            'score': None, 'auto': None, 'override': {}})
        states[frame] = {'status': 'done', 'persons': persons, 'reviewed': False, 'yolo': {}}
    if not states:
        raise ValueError(f'No annotation JSON files found in {labels}.')
    config = {'root': str(root), 'calib': str(calib), 'out_dir': str(out),
              'frames': sorted(states), 'cameras': list(range(len(dataset.folders))),
              'model_id': None, 'params': {'radius': .25, 'height': 1.8}, 'undistort': False,
              'image_processing_version': 2, 'annotation_source': str(labels),
              'annotation_grid': {k: manifest[k] for k in ('roi', 'position_convention')}}
    project = Project(config, models, detectors)
    project.state.update(states)
    project.save()
    return project

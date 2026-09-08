import os
from collections import deque


_KAGGLE_SEARCH_ROOTS = ('/kaggle/input', '/kaggle/working')
_SKIP_DIRS = {'Image_subsets', 'annotations_positions', '.git', '__pycache__'}

# Structural signatures used to distinguish full or partially annotated copies
# of the two supported datasets. The contents of annotations_positions may be
# partial; only the directory and the original camera calibration layout matter.
_DATASET_SIGNATURES = {
    'wildtrack': (
        'Image_subsets',
        'annotations_positions',
        os.path.join('calibrations', 'intrinsic_zero', 'intr_CVLab1.xml'),
        os.path.join('calibrations', 'extrinsic', 'extr_CVLab1.xml'),
    ),
    'multiviewx': (
        'Image_subsets',
        'annotations_positions',
        os.path.join('calibrations', 'intrinsic', 'intr_Camera1.xml'),
        os.path.join('calibrations', 'extrinsic', 'extr_Camera1.xml'),
    ),
}


def _is_dataset_root(path, required_paths):
    return os.path.isdir(path) and all(
        os.path.exists(os.path.join(path, relative_path))
        for relative_path in required_paths
    )


def _find_dataset_root(search_root, required_paths, max_depth=4):
    """Search shallowly without descending into large image/annotation folders."""
    if not os.path.isdir(search_root):
        return None

    queue = deque([(os.path.abspath(search_root), 0)])
    while queue:
        current, depth = queue.popleft()
        if _is_dataset_root(current, required_paths):
            return current
        if depth >= max_depth:
            continue

        try:
            entries = sorted(os.scandir(current), key=lambda entry: entry.name.lower())
        except OSError:
            continue
        for entry in entries:
            if entry.is_dir(follow_symlinks=False) and entry.name not in _SKIP_DIRS:
                queue.append((entry.path, depth + 1))
    return None


def detect_dataset_root(root, max_depth=4):
    """Return ``(dataset_name, dataset_root)`` detected below ``root``.

    The supplied path may be the dataset root itself or a shallow parent such
    as a Kaggle input directory containing an extra dataset/version folder.
    """
    requested_root = os.path.abspath(os.path.expanduser(os.fspath(root)))
    if not os.path.isdir(requested_root):
        raise FileNotFoundError(f'Dataset path does not exist or is not a directory: {requested_root}')

    direct_matches = [
        (dataset_name, requested_root)
        for dataset_name, required_paths in _DATASET_SIGNATURES.items()
        if _is_dataset_root(requested_root, required_paths)
    ]
    if len(direct_matches) == 1:
        return direct_matches[0]

    matches = []
    for dataset_name, required_paths in _DATASET_SIGNATURES.items():
        resolved_root = _find_dataset_root(requested_root, required_paths, max_depth=max_depth)
        if resolved_root is not None:
            matches.append((dataset_name, resolved_root))

    if not matches:
        expected = '; '.join(
            f'{name}: {", ".join(paths)}'
            for name, paths in _DATASET_SIGNATURES.items()
        )
        raise FileNotFoundError(
            f'Could not detect a Wildtrack or MultiviewX dataset under {requested_root}. '
            f'Expected one of these layouts: {expected}'
        )
    if len(matches) > 1:
        found = ', '.join(f'{name} at {path}' for name, path in matches)
        raise ValueError(
            f'Dataset path is ambiguous because it contains multiple datasets: {found}. '
            'Pass --data_path pointing to one dataset root.'
        )
    return matches[0]


def resolve_dataset_root(root, dataset_name, required_paths):
    """Resolve a local dataset path, falling back to Kaggle-mounted datasets."""
    requested_root = os.path.abspath(os.path.expanduser(os.fspath(root)))
    if _is_dataset_root(requested_root, required_paths):
        return requested_root

    for search_root in _KAGGLE_SEARCH_ROOTS:
        resolved_root = _find_dataset_root(search_root, required_paths)
        if resolved_root is not None:
            return resolved_root

    required = ', '.join(required_paths)
    raise FileNotFoundError(
        f'Could not find the {dataset_name} dataset. Checked {requested_root} and '
        f'{", ".join(_KAGGLE_SEARCH_ROOTS)}. Expected a directory containing: {required}'
    )

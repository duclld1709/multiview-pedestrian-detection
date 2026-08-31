import os
from collections import deque


_KAGGLE_SEARCH_ROOTS = ('/kaggle/input', '/kaggle/working')
_SKIP_DIRS = {'Image_subsets', 'annotations_positions', '.git', '__pycache__'}


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

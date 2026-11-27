"""Download and unpack the demo datasets without changing existing datasets."""

import argparse
from importlib.util import find_spec
from pathlib import Path
import stat
import subprocess
import sys
from zipfile import BadZipFile, ZipFile


DRIVE_FOLDER = 'https://drive.google.com/drive/folders/1k4qLz_JNe6sHdKfp_rqpPhl8zgZ1ENSf'
DATASETS = ('WaveTrack', 'Wildtrack', 'MultiviewX')


def ready(path):
    # The demo also accepts releases wrapped as Wildtrack/Wildtrack/.
    return direct_ready(path) or direct_ready(path / path.name)


def direct_ready(path):
    """Recognize an extracted dataset with calibration and camera media."""
    calibration = path / 'calibrations'
    if not calibration.is_dir():
        return False
    calibrated = (calibration / 'calibration.json').is_file() or (
        (calibration / 'extrinsic').is_dir() and
        any((calibration / name).is_dir() for name in ('intrinsic_zero', 'intrinsic')))
    images = path / 'Image_subsets'
    count = {'wavetrack': 5, 'wildtrack': 7, 'multiviewx': 6}.get(path.name.casefold())
    if count is None:
        return False
    camera_dirs = {p.name.casefold(): p for p in images.iterdir() if p.is_dir()} if images.is_dir() else {}
    has_images = all(any(f.is_file() and f.suffix.lower() in ('.png', '.jpg', '.jpeg')
                         for f in camera_dirs.get(f'c{i}', path / '__missing_camera__').glob('*'))
                     for i in range(1, count + 1))
    native_names = {'wildtrack': ('CVLab1', 'CVLab2', 'CVLab3', 'CVLab4', 'IDIAP1', 'IDIAP2', 'IDIAP3'),
                    'multiviewx': tuple(f'Camera{i}' for i in range(1, 7))}
    names = native_names.get(path.name.casefold(), tuple(f'Cam{i}' for i in range(1, count + 1)))
    has_video = False
    for folder in (path, path / 'videos', path / 'videos_rectified'):
        if folder.is_dir():
            files = {p.stem.casefold() for p in folder.iterdir() if p.is_file() and p.suffix.lower() == '.mp4'}
            has_video |= all(any(n in files for n in (f'cam{i}', f'c{i}', native.casefold()))
                             for i, native in enumerate(names, 1))
    return calibrated and (has_images or has_video)


def extract_archives(staging):
    """Unpack both direct dataset ZIPs and ZIPs containing more ZIPs."""
    for _ in range(32):
        archives = sorted(p for p in staging.rglob('*') if p.is_file() and p.suffix.lower() == '.zip')
        if not archives:
            return
        for archive in archives:
            destination = archive.with_suffix('')
            destination.mkdir(parents=True, exist_ok=True)
            root = destination.resolve()
            print(f'Extracting {archive}', flush=True)
            with ZipFile(archive) as zipped:
                for member in zipped.infolist():
                    target = (root / member.filename.replace('\\', '/')).resolve()
                    if not target.is_relative_to(root) or stat.S_ISLNK(member.external_attr >> 16):
                        raise ValueError(f'Unsafe ZIP entry: {member.filename}')
                zipped.extractall(root)
            # CRC/decompression failures above leave the ZIP available for retry.
            archive.unlink()
    raise ValueError('ZIP nesting exceeds 32 levels.')


def install_available(staging, dataset_root):
    if not staging.is_dir():
        return
    for name in DATASETS:
        target = dataset_root / name
        if ready(target):
            continue
        candidates = [p for p in staging.rglob('*')
                      if p.is_dir() and p.name.casefold() == name.casefold() and direct_ready(p)]
        if not candidates:
            continue
        if target.exists():
            raise ValueError(f'{target} exists but is incomplete. Move it aside before retrying; no files were overwritten.')
        source = min(candidates, key=lambda p: len(p.parts))
        dataset_root.mkdir(parents=True, exist_ok=True)
        # Both paths are under the explicitly selected destination, on one filesystem.
        if not source.resolve().is_relative_to(staging.resolve()) or not target.resolve().is_relative_to(dataset_root.resolve()):
            raise ValueError('Dataset path is outside the selected destination.')
        source.rename(target)
        print(f'Ready: {target}', flush=True)


def setup(destination):
    destination = Path(destination).expanduser().resolve()
    dataset_root = destination / 'Dataset'
    staging = destination / '.mvpd-dataset-download'
    if all(ready(dataset_root / name) for name in DATASETS):
        print(f'Datasets already exist. Dataset folder: {dataset_root}')
        return dataset_root
    for name in DATASETS:
        target = dataset_root / name
        if target.exists() and not ready(target):
            raise ValueError(f'{target} exists but is incomplete. Move it aside before retrying; no files were overwritten.')
    # Resume an interrupted extraction/install before downloading anything again.
    if staging.is_dir():
        extract_archives(staging)
        install_available(staging, dataset_root)
    if not all(ready(dataset_root / name) for name in DATASETS):
        if find_spec('gdown') is None:
            raise ValueError('Install gdown first: python -m pip install "gdown>=5.2,<6"')
        staging.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, '-m', 'gdown', '--folder', DRIVE_FOLDER,
                        '--continue', '-O', str(staging)], check=True)
        extract_archives(staging)
        install_available(staging, dataset_root)
    missing = [name for name in DATASETS if not ready(dataset_root / name)]
    if missing:
        raise ValueError(f'Missing or invalid datasets: {", ".join(missing)}. Downloaded files remain at {staging}.')
    print(f'Dataset folder: {dataset_root}\nUse this absolute path in Settings > Dataset > Scan.')
    return dataset_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dest', type=Path, default=Path('.'),
                        help='parent of Dataset/ (default: current working directory)')
    args = parser.parse_args()
    try:
        setup(args.dest)
    except (OSError, ValueError, BadZipFile, subprocess.CalledProcessError) as exc:
        parser.exit(1, f'Dataset setup failed: {exc}\n')


if __name__ == '__main__':
    main()

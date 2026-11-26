"""Build the portable source, launch scripts and checkpoint distribution."""

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from realtime_demo.datasets import CHECKPOINTS

DISTRIBUTION_FOLDER = 'MVPD_Demo'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('dist') / f'{DISTRIBUTION_FOLDER}.zip')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    required = root / 'checkpoints' / CHECKPOINTS['wavetrack']
    if not required.is_file():
        parser.error(f'Missing required checkpoint {required}')
    source_files = [root / name for name in (
        'calib.py', 'cylinders.py', 'site_layout.py', 'subset_data.py', 'wavetrack-site-map.json',
        'requirements-demo.txt', 'setup_demo.py',
        'map_visualization/physics.mjs')]
    package_files = [p for p in (root / 'multiview_detector').rglob('*.py') if p.name != 'legacy.py']
    demo = root / 'realtime_demo'
    demo_files = [p for p in demo.glob('*.py')] + [p for p in (demo / 'web').iterdir() if p.is_file()]
    weights = [root / 'checkpoints' / name for name in CHECKPOINTS.values()]
    readme = demo / 'README.md'
    if not readme.is_file():
        readme = root / 'README.md'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.output, 'w', ZIP_DEFLATED) as archive:
        archive.write(readme, f'{DISTRIBUTION_FOLDER}/README.md')
        for path in sorted([*source_files, *package_files, *demo_files, *(p for p in weights if p.is_file())]):
            archive.write(path, f'{DISTRIBUTION_FOLDER}/{path.relative_to(root).as_posix()}')
    print(args.output.resolve())


if __name__ == '__main__':
    main()

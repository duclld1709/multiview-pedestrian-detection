"""Create the Windows/Linux demo virtual environment."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', default='auto', help='uv PyTorch backend, for example auto, cpu or cu124')
    args = parser.parse_args()
    uv = shutil.which('uv')
    if not uv:
        parser.error('Install uv and make it available on PATH. See https://docs.astral.sh/uv/getting-started/installation/')
    root = Path(__file__).resolve().parent
    environment = root / '.venv-mvpd'
    python = environment / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')

    def install(backend):
        subprocess.run([uv, 'pip', 'install', '--python', python, '--torch-backend', backend,
                        '--reinstall-package', 'torch', '--reinstall-package', 'torchvision',
                        '-r', root / 'requirements-demo.txt'], cwd=root, check=True)

    try:
        if not python.is_file():
            subprocess.run([uv, 'venv', '--python', '3.11', '--allow-existing', environment], cwd=root, check=True)
        install(args.backend)
        probe = [python, '-B', '-m', 'realtime_demo.verify_device']
        result = subprocess.run(probe, cwd=root)
        if args.backend == 'auto' and result.returncode == 2:
            print('The selected GPU backend failed the inference check. Installing the CPU build.', flush=True)
            install('cpu')
            result = subprocess.run(probe, cwd=root)
        result.check_returncode()
    except subprocess.CalledProcessError as exc:
        parser.exit(exc.returncode, 'Setup stopped. See the error above.\n')
    print(f'MVPD environment ready at {environment}', flush=True)


if __name__ == '__main__':
    main()

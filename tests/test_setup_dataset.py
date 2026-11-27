import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import BadZipFile, ZipFile

import setup_dataset as installer


def dataset_zip(name, prefix=''):
    data = io.BytesIO()
    with ZipFile(data, 'w') as archive:
        archive.writestr(f'{prefix}{name}/calibrations/calibration.json', '{}')
        count = {'WaveTrack': 5, 'Wildtrack': 7, 'MultiviewX': 6}[name]
        for i in range(1, count + 1):
            archive.writestr(f'{prefix}{name}/Image_subsets/C{i}/00000000.png', b'fixture')
    return data.getvalue()


class DatasetSetupTests(unittest.TestCase):
    def test_nested_archives_normalize_and_delete_zips(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            staging = parent / '.mvpd-dataset-download'
            staging.mkdir()
            with ZipFile(staging / 'bundle.zip', 'w') as outer:
                for name in installer.DATASETS:
                    outer.writestr(f'AIP491_Dataset/{name}.zip', dataset_zip(name, prefix=f'{name}/'))
            with patch.object(installer.subprocess, 'run') as download:
                root = installer.setup(parent)
                download.assert_not_called()
            self.assertEqual(root, parent / 'Dataset')
            self.assertTrue(all(installer.ready(root / name) for name in installer.DATASETS))
            self.assertFalse(list(staging.rglob('*.zip')))
            with patch.object(installer.subprocess, 'run') as download:
                installer.setup(parent)
                download.assert_not_called()

    def test_folder_download_command_and_direct_zips(self):
        with tempfile.TemporaryDirectory() as temp:
            def download(command, check):
                self.assertTrue(check)
                self.assertIn('--folder', command)
                self.assertIn('--continue', command)
                staging = Path(command[-1])
                for name in installer.DATASETS:
                    (staging / f'{name}.zip').write_bytes(dataset_zip(name))
            with patch.object(installer, 'find_spec', return_value=object()), \
                    patch.object(installer.subprocess, 'run', side_effect=download) as run:
                root = installer.setup(temp)
                self.assertEqual(run.call_count, 1)
                self.assertTrue(all(installer.ready(root / name) for name in installer.DATASETS))

    def test_invalid_zip_is_kept(self):
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp) / 'broken.zip'
            archive.write_bytes(b'broken')
            with self.assertRaises(BadZipFile):
                installer.extract_archives(Path(temp))
            self.assertTrue(archive.exists())

    def test_zip_cannot_escape_destination(self):
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp) / 'unsafe.zip'
            with ZipFile(archive, 'w') as zipped:
                zipped.writestr('../outside.txt', 'outside')
            with self.assertRaises(ValueError):
                installer.extract_archives(Path(temp))
            self.assertTrue(archive.exists())
            self.assertFalse((Path(temp) / 'outside.txt').exists())

    def test_existing_partial_dataset_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            staging = parent / '.mvpd-dataset-download'
            staging.mkdir()
            (staging / 'WaveTrack.zip').write_bytes(dataset_zip('WaveTrack'))
            target = parent / 'Dataset' / 'WaveTrack'
            target.mkdir(parents=True)
            marker = target / 'keep.txt'
            marker.write_text('keep')
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                installer.setup(parent)
            self.assertEqual(marker.read_text(), 'keep')


if __name__ == '__main__':
    unittest.main()

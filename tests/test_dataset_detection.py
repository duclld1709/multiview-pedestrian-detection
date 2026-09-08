import tempfile
import unittest
from pathlib import Path

from multiview_detector.datasets.path_utils import detect_dataset_root


DATASET_FILES = {
    'wildtrack': (
        'calibrations/intrinsic_zero/intr_CVLab1.xml',
        'calibrations/extrinsic/extr_CVLab1.xml',
    ),
    'multiviewx': (
        'calibrations/intrinsic/intr_Camera1.xml',
        'calibrations/extrinsic/extr_Camera1.xml',
    ),
}


def make_dataset(root, dataset_name):
    root.mkdir(parents=True)
    (root / 'Image_subsets').mkdir()
    (root / 'annotations_positions').mkdir()
    for relative_path in DATASET_FILES[dataset_name]:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


class DatasetDetectionTest(unittest.TestCase):
    def test_detects_partial_wildtrack_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / 'wildtrack-partial-40-percent'
            make_dataset(root, 'wildtrack')

            dataset_name, resolved_root = detect_dataset_root(root)

            self.assertEqual(dataset_name, 'wildtrack')
            self.assertEqual(Path(resolved_root), root.resolve())

    def test_detects_nested_multiviewx_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            parent = Path(temp_dir) / 'kaggle-export'
            root = parent / 'version-2' / 'multiviewx-partial'
            make_dataset(root, 'multiviewx')

            dataset_name, resolved_root = detect_dataset_root(parent)

            self.assertEqual(dataset_name, 'multiviewx')
            self.assertEqual(Path(resolved_root), root.resolve())

    def test_rejects_unknown_layout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(FileNotFoundError, 'Could not detect'):
                detect_dataset_root(temp_dir)

    def test_rejects_parent_containing_both_datasets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            parent = Path(temp_dir)
            make_dataset(parent / 'wildtrack', 'wildtrack')
            make_dataset(parent / 'multiviewx', 'multiviewx')

            with self.assertRaisesRegex(ValueError, 'multiple datasets'):
                detect_dataset_root(parent)


if __name__ == '__main__':
    unittest.main()

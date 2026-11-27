from pathlib import Path
import tempfile
import unittest

from realtime_demo.datasets import checkpoint_dataset, default_checkpoint


class CheckpointSelectionTests(unittest.TestCase):
    def test_native_lite_and_foreign_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / 'checkpoints'
            folder.mkdir()
            for name in ('wavetrack_lite.pt', 'wildtrack_lite.pt'):
                (folder / name).touch()
            self.assertEqual(default_checkpoint(temp, 'wavetrack'), 'wavetrack_lite.pt')
            self.assertEqual(default_checkpoint(temp, 'wildtrack'), 'wildtrack_lite.pt')
            self.assertEqual(default_checkpoint(temp, 'multiviewx'), 'wavetrack_lite.pt')
            self.assertEqual(checkpoint_dataset('wildtrack_lite.pt'), 'wildtrack')
            self.assertEqual(checkpoint_dataset('wavetrack_lite.pt'), 'wavetrack')
            (folder / 'bevnet_wildtrack.pt').touch()
            self.assertEqual(default_checkpoint(temp, 'wildtrack'), 'bevnet_wildtrack.pt')

    def test_missing_checkpoint_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, 'No checkpoint for Wildtrack'):
                default_checkpoint(temp, 'wildtrack')


if __name__ == '__main__':
    unittest.main()

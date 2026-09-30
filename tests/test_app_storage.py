import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app_storage


class AppStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / 'Redirected Documents' / 'data files'
        mocked = patch.object(app_storage, 'DATA_ROOT', self.data)
        mocked.start()
        self.addCleanup(mocked.stop)

    def test_read_and_write_use_same_location_from_another_directory(self):
        previous = Path.cwd()
        try:
            os.chdir(self.root)
            path = app_storage.data_path('Files/crop_regions.json')
            self.assertEqual(path, self.data / 'Files' / 'crop_regions.json')
            path.write_text('{"saved": true}')
            self.assertEqual(app_storage.data_path('Files\\crop_regions.json').read_text(),
                             '{"saved": true}')
        finally:
            os.chdir(previous)

    def test_absolute_legacy_paths_are_relocated(self):
        self.assertEqual(app_storage.data_path(r'C:\Users\Someone\OldApp\Files\camera.json'),
                         self.data / 'Files' / 'camera.json')
        self.assertEqual(app_storage.data_path(self.data / 'Files' / 'camera.json'),
                         self.data / 'Files' / 'camera.json')

    def test_parent_escape_is_rejected(self):
        with self.assertRaises(ValueError):
            app_storage.data_path('../outside.json')

    def test_migration_moves_without_overwriting_or_moving_config(self):
        code = self.root / 'code'
        (code / 'Files').mkdir(parents=True)
        (code / 'Files' / 'camera.json').write_text('old calibration')
        (code / 'Files' / 'crops.json').write_text('old crops')
        (code / 'config.json').write_text('code configuration')
        app_storage.data_path('Files/crops.json').write_text('new crops')
        app_storage.initialize_storage(code)
        self.assertEqual(app_storage.data_path('Files/camera.json').read_text(), 'old calibration')
        self.assertEqual(app_storage.data_path('Files/crops.json').read_text(), 'new crops')
        self.assertFalse((self.data / 'config.json').exists())
        self.assertTrue((code / 'config.json').exists())
        self.assertFalse((code / 'Files').exists())
        self.assertEqual((self.data / '.migration_conflicts/project/Files/crops.json').read_text(),
                         'old crops')
        app_storage.data_path('Files/camera.json').unlink()
        app_storage.initialize_storage(code)
        self.assertFalse(app_storage.data_path('Files/camera.json').exists())

    def test_migration_includes_archives_helpers_and_previous_documents(self):
        code = self.root / 'code'
        for name in ('Dataset_Capture/Camera1/Original/frame.jpg',
                     'temp_calibration_images/camera_2/calib.png', 'unet/result_mask.png'):
            file = code / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(b'image')
        (code / 'unet/predictor.py').write_text('code')
        (code / 'unet/models').mkdir()
        (code / 'unet/models/model.h5').write_bytes(b'model')
        previous = self.data.parent / 'MAS Unichela'
        (previous / 'Files').mkdir(parents=True)
        (previous / 'Files/camera.json').write_text('calibration')
        self.assertEqual(app_storage.migrate_storage(code), 4)
        self.assertTrue((self.data / 'Dataset_capture/Camera1/Original/frame.jpg').exists())
        self.assertTrue((self.data / 'unet/result_mask.png').exists())
        self.assertFalse((code / 'Dataset_Capture').exists())
        self.assertFalse(previous.exists())
        self.assertTrue((code / 'unet/predictor.py').exists())
        self.assertTrue((code / 'unet/models/model.h5').exists())
        self.assertEqual(app_storage.migrate_storage(code), 0)


if __name__ == '__main__':
    unittest.main()

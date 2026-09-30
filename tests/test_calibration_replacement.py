"""Headless checks for replacement, failure rollback and the new-calibration flow."""
import ast
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from CalibrateAPP.calibration_store import replace_calibration_files


class CalibrationReplacementTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.lens = self.root / 'intrinsics.json'
        self.surface = self.root / 'extrinsics.json'
        self.lens.write_text('{"lens": "old"}')
        self.surface.write_text('{"surface": "old"}')

    def test_success_replaces_both_files(self):
        replace_calibration_files({self.lens: {'lens': 'new'},
                                   self.surface: {'surface': 'new'}})
        self.assertEqual(json.loads(self.lens.read_text()), {'lens': 'new'})
        self.assertEqual(json.loads(self.surface.read_text()), {'surface': 'new'})
        self.assertEqual(list(self.root.glob('*.tmp')), [])

    def test_second_file_failure_restores_old_pair(self):
        replace = os.replace
        failed = False
        def fail_surface(source, destination):
            nonlocal failed
            if Path(destination) == self.surface and not failed:
                failed = True
                raise PermissionError('file locked')
            return replace(source, destination)
        with patch('CalibrateAPP.calibration_store.os.replace', side_effect=fail_surface):
            with self.assertRaises(PermissionError):
                replace_calibration_files({self.lens: {'lens': 'new'},
                                           self.surface: {'surface': 'new'}})
        self.assertEqual(json.loads(self.lens.read_text()), {'lens': 'old'})
        self.assertEqual(json.loads(self.surface.read_text()), {'surface': 'old'})
        self.assertEqual(list(self.root.glob('*.tmp')), [])

    def test_invalid_data_does_not_change_old_pair(self):
        with self.assertRaises(ValueError):
            replace_calibration_files({self.lens: {'lens': 'new'},
                                       self.surface: {'invalid': float('nan')}})
        self.assertEqual(json.loads(self.lens.read_text()), {'lens': 'old'})
        self.assertEqual(json.loads(self.surface.read_text()), {'surface': 'old'})
        self.assertEqual(list(self.root.glob('*.tmp')), [])

    def methods_without_camera_dependencies(self):
        tree = ast.parse((ROOT / 'CalibrateAPP/calibration_ui.py').read_text(encoding='utf-8-sig'))
        app = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CalibrationApp')
        methods = [n for n in app.body if isinstance(n, ast.FunctionDef)
                   and n.name in ('begin_new_calibration', '_save_calibration')]
        module = ast.Module(body=methods, type_ignores=[])
        namespace = {'Path': Path, 'json': json, 'replace_calibration_files': replace_calibration_files,
                     'TEMP_ROOT': self.root, 'NUM_CAPTURES': 20, 'AMBER': 'amber',
                     'CONFIG': {'cameras': [{'index': 2}, {'index': 3}]},
                     'project_path': lambda path: str(self.root / path)}
        exec(compile(module, 'calibration flow methods', 'exec'), namespace)
        return namespace

    def test_new_calibration_resets_capture_state_without_deleting_saved_pair(self):
        methods = self.methods_without_camera_dependencies()
        app = MagicMock()
        app.camera_running = True
        app.processing_capture = app.processing_verification = False
        app.stage = 'complete'
        app.camera_index = 2
        methods['begin_new_calibration'](app)
        self.assertEqual(app.stage, 'capture')
        self.assertEqual(app.captured_images, [])
        self.assertIsNone(app.pending_intrinsics)
        self.assertIsNone(app.camera_matrix)
        self.assertEqual(json.loads(self.lens.read_text()), {'lens': 'old'})
        self.assertEqual(json.loads(self.surface.read_text()), {'surface': 'old'})

    def test_new_pair_invalidates_only_selected_camera_regions(self):
        methods = self.methods_without_camera_dependencies()
        region = self.root / 'Files/region_homographies.json'
        region.parent.mkdir()
        region.write_text(json.dumps({'cameras': {'1': {'1': {'old': True}},
                                                 '2': {'1': {'keep': True}}}}))
        app = MagicMock()
        app.camera_index = 2
        app._calibration_path.return_value = self.lens
        app._extrinsics_path.return_value = self.surface
        methods['_save_calibration'](app, {'lens': 'new'}, {'surface': 'new'})
        store = json.loads(region.read_text())
        self.assertEqual(store['cameras']['1'], {})
        self.assertEqual(store['cameras']['2'], {'1': {'keep': True}})


if __name__ == '__main__':
    unittest.main()

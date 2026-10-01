"""Headless regression checks for one/two-camera configuration and routing."""
import ast
import copy
from concurrent.futures import Future
import datetime
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import tkinter as tk
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]


def load_dashboard(count):
    tree = ast.parse((ROOT / 'main_1366.py').read_text(encoding='utf-8-sig'))
    original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'IndustrialDashboard')
    names = {'start_video_stream', 'update_video_feed', '_refresh_inspection_availability',
             'start_detect_thread', '_handle_serial_button', '_simulate_detection',
             '_select_crop_camera', 'maximize_camera'}
    cls = ast.ClassDef(name=original.name, bases=[], keywords=[],
                      body=[n for n in original.body if isinstance(n, ast.FunctionDef) and n.name in names],
                      decorator_list=[])
    namespace = {'CAMERA_COUNT': count, 'ACTIVE_CAMERAS': [
        {'index': 2, 'calibration_file': 'Files/lens0.json'},
        {'index': 3, 'calibration_file': 'Files/lens1.json'}][:count],
        'CONFIG': {'preview': {'interval_ms': 100}}, 'tk': tk, 'time': time,
        'threading': MagicMock(), 'ThreadPoolExecutor': MagicMock(),
        'CameraHandler': MagicMock(), 'project_path': lambda path: str(ROOT / path),
        'os': os, 'datetime': datetime, 'cv2': MagicMock(),
        'extract_rotated_crop': MagicMock(), 'CROP_OUTPUT_SIZE': (2208, 552), 'CROP_RATIO': 4}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
                 'camera mode dashboard', 'exec'), namespace)
    return namespace['IndustrialDashboard'], namespace


class CameraModeTests(unittest.TestCase):
    def make_app(self, count, ready=True):
        cls, namespace = load_dashboard(count)
        app = cls()
        app.camera1 = MagicMock(calibration_available=ready)
        app.camera2 = MagicMock(calibration_available=ready) if count == 2 else None
        app.detect_btn_L = MagicMock()
        app.detect_btn_R = MagicMock()
        app.detect_btn_L.__getitem__.return_value = tk.NORMAL
        app.detect_btn_R.__getitem__.return_value = tk.NORMAL
        app.set_pass_fail = MagicMock()
        app.update_progress = MagicMock()
        app.inspection_busy = False
        app.active_size = 'M'
        app._crop_cameras = MagicMock(return_value={'1': [{}, {}], '2': [{}, {}]})
        app._show_error_popup = MagicMock()
        app._run_detection = MagicMock()
        app._send_serial_status = MagicMock()
        app.root = MagicMock()
        return app, namespace

    def test_available_camera_needs_no_second_camera(self):
        app, _ = self.make_app(1)
        app._refresh_inspection_availability()
        app.detect_btn_L.config.assert_called_once_with(state=tk.NORMAL)
        app.detect_btn_R.config.assert_called_once_with(state=tk.DISABLED)
        app.set_pass_fail.assert_called_once_with('LIVE')

    def test_missing_single_camera_calibration_blocks_inspection(self):
        app, _ = self.make_app(1, ready=False)
        app._refresh_inspection_availability()
        app.detect_btn_L.config.assert_called_once_with(state=tk.DISABLED)
        app.set_pass_fail.assert_called_once_with('SETUP')
        self.assertNotIn('right', app.update_progress.call_args.args[1])

    def test_single_camera_inspection_keeps_two_crop_requirement(self):
        app, namespace = self.make_app(1)
        self.assertTrue(app.start_detect_thread('L'))
        namespace['threading'].Thread.assert_called_once_with(
            target=app._run_detection, args=('L',), daemon=True)
        app, namespace = self.make_app(1)
        app._crop_cameras.return_value = {'1': [{}, None]}
        self.assertFalse(app.start_detect_thread('L'))
        namespace['threading'].Thread.assert_not_called()

    def test_left_serial_trigger_works_and_inactive_right_is_rejected(self):
        app, _ = self.make_app(1)
        app._handle_serial_button('R')
        app._send_serial_status.assert_called_once_with('R_NOT_READY')
        self.assertFalse(app.inspection_busy)
        app._send_serial_status.reset_mock()
        app._handle_serial_button('L')
        app._send_serial_status.assert_called_once_with('L_ACK')
        self.assertTrue(app.inspection_busy)

    def test_two_camera_mode_keeps_both_inspections(self):
        app, _ = self.make_app(2)
        app._refresh_inspection_availability()
        app.detect_btn_L.config.assert_called_once_with(state=tk.NORMAL)
        app.detect_btn_R.config.assert_called_once_with(state=tk.NORMAL)
        self.assertTrue(app.start_detect_thread('R'))

    def test_startup_opens_only_active_cameras_and_resumes_without_reopening(self):
        for count in (1, 2):
            with self.subTest(count=count):
                app, namespace = self.make_app(count)
                app.camera1 = app.camera2 = None
                app.calibration_page = None
                app.update_video_feed = MagicMock()
                executor = namespace['ThreadPoolExecutor'].return_value
                def submit(factory, index, path):
                    future = Future()
                    future.set_result(MagicMock(camera_index=index))
                    return future
                executor.submit.side_effect = submit
                app.start_video_stream()
                self.assertEqual(executor.submit.call_count, count)
                opened = [call.args[1] for call in executor.submit.call_args_list]
                self.assertEqual(opened, [2, 3][:count])
                app.root.after.call_args.args[1]()
                self.assertIsNotNone(app.camera1)
                self.assertEqual(app.camera2 is None, count == 1)
                app.update_video_feed.assert_called_once()
                self.assertEqual(executor.submit.call_count, count)

    def test_live_preview_does_not_read_missing_second_camera(self):
        app, _ = self.make_app(1)
        frame = object()
        app.camera1.get_raw_frame_with_ret.return_value = (True, frame)
        app.video_streaming = True
        app.video_paused = False
        app.maximized_camera = None
        app._display_video_frame = MagicMock()
        app.update_video_feed()
        app._display_video_frame.assert_called_once_with(frame, 1)

    def test_calibration_pages_build_only_active_camera_selectors(self):
        pages = [('CalibrateAPP/calibration_ui.py', 'CalibrationApp', '_create_ui'),
                 ('CalibrateAPP/calibration_ui.py', 'CalibrationCheckApp', '_create_ui'),
                 ('CalibrateAPP/region_calibration_ui.py', 'RegionCalibrationApp', '_create_camera_section')]
        for count in (1, 2):
            for filename, class_name, method_name in pages:
                with self.subTest(count=count, page=class_name):
                    tree = ast.parse((ROOT / filename).read_text(encoding='utf-8-sig'))
                    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
                    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
                    start = next(i for i, n in enumerate(method.body) if isinstance(n, ast.Assign)
                                 and any(isinstance(t, ast.Attribute) and t.attr == 'btn_cam0' for t in n.targets))
                    end = next(i for i in range(start, len(method.body))
                               if isinstance(method.body[i], ast.If)
                               and 'len(CAMERA_IDS)' in ast.unparse(method.body[i].test))
                    app = MagicMock()
                    namespace = {'self': app, 'CAMERA_IDS': [2, 3][:count],
                                 'cam_buttons': MagicMock(), 'camera_row': MagicMock(),
                                 'GREEN': 'green', 'BLUE': 'blue', 'tk': tk}
                    exec(compile(ast.Module(body=method.body[start:end + 1], type_ignores=[]),
                                 'camera selector controls', 'exec'), namespace)
                    self.assertEqual(app._button.call_count, count)
                    self.assertEqual(app.btn_cam1 is None, count == 1)

    def test_single_camera_extracts_and_saves_both_crops(self):
        app, namespace = self.make_app(1)
        app.camera1.get_raw_frame_with_ret.return_value = (True, object())
        app.camera1.get_undistorted_frame.return_value = SimpleNamespace(shape=(3456, 4608, 3))
        crops = [object(), object()]
        namespace['extract_rotated_crop'].side_effect = crops
        namespace['cv2'].imwrite.return_value = True
        app._present_detection_result = MagicMock()
        app._publish_crop_progress = MagicMock()
        app._detect_crops = MagicMock(return_value=(crops, []))
        with tempfile.TemporaryDirectory() as folder:
            namespace['project_path'] = lambda path: str(Path(folder) / path)
            app._simulate_detection('L')
        self.assertEqual(namespace['extract_rotated_crop'].call_count, 2)
        self.assertEqual(namespace['cv2'].imwrite.call_count, 4)
        self.assertEqual(app._detect_crops.call_args.args[0], 1)
        self.assertEqual(app._detect_crops.call_args.args[2], crops)


class CameraConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / 'config.json').read_text())
        cv = SimpleNamespace(aruco=SimpleNamespace(
            DICT_4X4_100=0, getPredefinedDictionary=lambda _: SimpleNamespace(
                bytesList=SimpleNamespace(shape=(100, 4, 4)))))
        namespace = {}
        with patch.dict(sys.modules, {'cv2': cv}):
            exec(compile((ROOT / 'config_validation.py').read_text(encoding='utf-8-sig'),
                         'config_validation.py', 'exec'), namespace)
        self.validate = namespace['validate_config']

    def test_one_and_two_camera_modes_and_old_configuration_are_valid(self):
        for count in (1, 2):
            config = copy.deepcopy(self.config)
            config['camera_count'] = count
            self.validate(config)
        self.config.pop('camera_count')
        self.validate(self.config)

    def test_single_camera_list_and_sam_flags_are_supported(self):
        self.config['camera_count'] = 1
        self.config['cameras'] = self.config['cameras'][:1]
        self.config['sam_detection']['send_crops'] = {'1': [True, True]}
        self.validate(self.config)

    def test_configuration_loader_activates_first_camera_and_preserves_second(self):
        for count in (1, 2):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as folder:
                config = copy.deepcopy(self.config)
                config['camera_count'] = count
                (Path(folder) / 'config.json').write_text(json.dumps(config))
                storage = SimpleNamespace(DATA_ROOT=Path(folder), data_path=lambda path: Path(folder) / path,
                                          initialize_storage=lambda: None)
                cv = SimpleNamespace(setNumThreads=lambda threads: None)
                validator = SimpleNamespace(validate_config=self.validate)
                namespace = {'__file__': str(Path(folder) / 'app_config.py')}
                with patch.dict(sys.modules, {'cv2': cv, 'config_validation': validator,
                                              'app_storage': storage}):
                    exec(compile((ROOT / 'app_config.py').read_text(encoding='utf-8-sig'),
                                 'app_config.py', 'exec'), namespace)
                self.assertEqual(namespace['CAMERA_COUNT'], count)
                self.assertEqual(len(namespace['ACTIVE_CAMERAS']), count)
                self.assertEqual(len(namespace['CONFIG']['cameras']), 2)
                self.assertEqual(namespace['ACTIVE_CAMERAS'][0], config['cameras'][0])

    def test_invalid_count_or_missing_camera_is_rejected(self):
        for count in (0, 3, True, 1.5, '1'):
            config = copy.deepcopy(self.config)
            config['camera_count'] = count
            with self.subTest(count=count), self.assertRaises(ValueError):
                self.validate(config)
        self.config['camera_count'] = 2
        self.config['cameras'] = self.config['cameras'][:1]
        with self.assertRaises(ValueError):
            self.validate(self.config)


if __name__ == '__main__':
    unittest.main()

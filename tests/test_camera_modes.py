"""Automatic inspection retains camera setup while enforcing one active camera."""
import copy
import unittest
from unittest.mock import MagicMock, patch
from concurrent.futures import Future
import numpy as np
from app_config import CONFIG
from config_validation import validate_config
from main_1366 import IndustrialDashboard


class AutomaticCameraTests(unittest.TestCase):
    def test_second_active_camera_is_rejected(self):
        cfg = copy.deepcopy(CONFIG)
        cfg['camera_count'] = 2
        with self.assertRaisesRegex(ValueError, 'camera_count = 1'):
            validate_config(cfg)

    def test_result_tabs_do_not_stop_hand_frame_delivery(self):
        app = IndustrialDashboard.__new__(IndustrialDashboard)
        app.root = MagicMock()
        app.video_streaming, app.video_paused = True, False
        app.camera1 = MagicMock()
        raw = np.zeros((10, 20, 3), np.uint8)
        app.camera1.stream.read_snapshot.return_value = (raw, 17)
        app.result_view = MagicMock()
        app._submit_automatic_frame = MagicMock()
        app._display_video_frame = MagicMock()
        app.update_video_feed()
        app._submit_automatic_frame.assert_called_once_with(raw, 17)
        app._display_video_frame.assert_not_called()
        app.result_view.update_live_preview.assert_called_once_with(raw)

    def test_missing_calibration_still_allows_setup(self):
        app = IndustrialDashboard.__new__(IndustrialDashboard)
        app.camera1 = MagicMock(calibration_available=False)
        app.set_pass_fail, app.update_progress = MagicMock(), MagicMock()
        app._refresh_inspection_availability()
        app.set_pass_fail.assert_called_once_with('SETUP')

    def test_startup_uses_existing_camera_handler_and_config(self):
        app = IndustrialDashboard.__new__(IndustrialDashboard)
        app.root = MagicMock()
        app.camera1 = app.camera2 = None
        app.update_progress = app.set_pass_fail = MagicMock()
        app.update_video_feed = app._refresh_inspection_availability = MagicMock()
        camera = MagicMock()
        camera.stream.resolution_warning = None
        future = Future()
        future.set_result(camera)
        with patch('main_1366.ThreadPoolExecutor') as executor:
            executor.return_value.submit.return_value = future
            app.start_video_stream()
            self.assertEqual(executor.return_value.submit.call_count, 1)
            args = executor.return_value.submit.call_args.args
            self.assertEqual(args[1], CONFIG['cameras'][0]['index'])
            self.assertIn('camera_calibration_0.json', args[2])


if __name__ == '__main__':
    unittest.main()

import tkinter as tk
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from main_1366 import IndustrialDashboard


class InspectionAvailabilityTests(unittest.TestCase):
    def make_app(self, left_ready, right_ready):
        app = IndustrialDashboard.__new__(IndustrialDashboard)
        app.camera1 = SimpleNamespace(calibration_available=left_ready)
        app.camera2 = SimpleNamespace(calibration_available=right_ready)
        app.detect_btn_L = MagicMock()
        app.detect_btn_R = MagicMock()
        app.set_pass_fail = MagicMock()
        app.update_progress = MagicMock()
        return app

    def test_only_uncalibrated_camera_inspection_is_disabled(self):
        app = self.make_app(False, True)

        app._refresh_inspection_availability()

        app.detect_btn_L.config.assert_called_once_with(state=tk.DISABLED)
        app.detect_btn_R.config.assert_called_once_with(state=tk.NORMAL)
        app.set_pass_fail.assert_called_once_with("SETUP")
        message = app.update_progress.call_args.args[1]
        self.assertIn("left", message)
        self.assertIn("inspection is disabled", message)

    def test_both_buttons_enable_after_calibration_reload(self):
        app = self.make_app(True, True)

        app._refresh_inspection_availability()

        app.detect_btn_L.config.assert_called_once_with(state=tk.NORMAL)
        app.detect_btn_R.config.assert_called_once_with(state=tk.NORMAL)
        app.set_pass_fail.assert_called_once_with("LIVE")
        app.update_progress.assert_called_once_with(100, "Live Feed")

    def test_finish_detection_preserves_per_camera_availability(self):
        app = self.make_app(False, True)
        app.video_paused = True

        app._finish_detection()

        self.assertFalse(app.video_paused)
        app.detect_btn_L.config.assert_called_once_with(state=tk.DISABLED)
        app.detect_btn_R.config.assert_called_once_with(state=tk.NORMAL)


if __name__ == '__main__':
    unittest.main()

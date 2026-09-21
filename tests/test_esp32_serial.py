import threading
import tkinter as tk
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from main_1366 import IndustrialDashboard


class Esp32SerialTests(unittest.TestCase):
    def make_app(self):
        app = IndustrialDashboard.__new__(IndustrialDashboard)
        app.serial_conn = MagicMock()
        app.serial_conn.is_open = True
        app.serial_lock = threading.Lock()
        app.camera1 = SimpleNamespace(calibration_available=True)
        app.camera2 = SimpleNamespace(calibration_available=True)
        app.inspection_busy = False
        app._serial_pending_side = None
        app.start_detect_thread = MagicMock(return_value=True)
        return app

    def test_left_request_starts_left_then_acknowledges_and_completes(self):
        app = self.make_app()
        app.video_paused = True
        app._refresh_inspection_availability = MagicMock()

        app._handle_serial_button('L')
        app.start_detect_thread.assert_called_once_with('L')
        app.serial_conn.write.assert_called_with(b'L_ACK\n')

        app._finish_detection()
        self.assertEqual(app.serial_conn.write.call_args_list[-1].args, (b'L_DONE\n',))
        self.assertIsNone(app._serial_pending_side)

    def test_right_request_starts_right_not_left(self):
        app = self.make_app()

        app._handle_serial_button('R')

        app.start_detect_thread.assert_called_once_with('R')
        app.serial_conn.write.assert_called_once_with(b'R_ACK\n')

    def test_busy_request_does_not_start_another_inspection(self):
        app = self.make_app()
        app.inspection_busy = True

        app._handle_serial_button('R')

        app.start_detect_thread.assert_not_called()
        app.serial_conn.write.assert_called_once_with(b'R_BUSY\n')

    def test_missing_calibration_rejects_request(self):
        app = self.make_app()
        app.camera1.calibration_available = False

        app._handle_serial_button('L')

        app.start_detect_thread.assert_not_called()
        app.serial_conn.write.assert_called_once_with(b'L_NOT_READY\n')

    def test_failure_sends_error_not_done(self):
        app = self.make_app()
        app._serial_pending_side = 'R'
        app.video_paused = True
        app._refresh_inspection_availability = MagicMock()

        app._finish_detection(completed=False)

        app.serial_conn.write.assert_called_once_with(b'R_ERROR\n')

    @patch('main_1366.threading.Thread')
    def test_busy_guard_does_not_reenable_buttons(self, thread_class):
        app = self.make_app()
        app.detect_btn_L = MagicMock()
        app.detect_btn_R = MagicMock()
        app.active_size = 'M'
        app.video_paused = False
        app._refresh_inspection_availability = MagicMock()
        app.detect_btn_L.__getitem__.return_value = tk.NORMAL

        self.assertTrue(IndustrialDashboard.start_detect_thread(app, 'L'))
        self.assertFalse(IndustrialDashboard.start_detect_thread(app, 'R'))
        self.assertTrue(app.inspection_busy)
        self.assertTrue(app.video_paused)
        thread_class.assert_called_once()
        app._refresh_inspection_availability.assert_not_called()


if __name__ == '__main__':
    unittest.main()

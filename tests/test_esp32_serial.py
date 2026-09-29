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
        app._side_busy = {'L': False, 'R': False}
        app._serial_pending_sides = set()
        app._active_result = None
        app._active_inspection_side = None
        app._result_timer_active = False
        app.start_detect_thread = MagicMock(return_value=True)
        return app

    def test_left_request_starts_acknowledges_and_completes_left(self):
        app = self.make_app()
        app._handle_serial_button('L')

        app.start_detect_thread.assert_called_once_with('L')
        app.serial_conn.write.assert_called_once_with(b'L_ACK\n')
        self.assertEqual(app._serial_pending_sides, {'L'})

        app._report_detection_status('L', True)
        self.assertEqual(app.serial_conn.write.call_args_list[-1].args, (b'L_DONE\n',))
        self.assertEqual(app._serial_pending_sides, set())

    def test_same_busy_side_is_rejected(self):
        app = self.make_app()
        app._side_busy['R'] = True

        app._handle_serial_button('R')

        app.start_detect_thread.assert_not_called()
        app.serial_conn.write.assert_called_once_with(b'R_BUSY\n')

    def test_opposite_side_is_accepted_while_current_side_is_busy(self):
        app = self.make_app()
        app._side_busy['L'] = True
        app._active_inspection_side = 'L'

        app._handle_serial_button('R')

        app.start_detect_thread.assert_called_once_with('R')
        app.serial_conn.write.assert_called_once_with(b'R_ACK\n')
        self.assertEqual(app._serial_pending_sides, {'R'})

    def test_matching_button_toggles_the_result_timer(self):
        app = self.make_app()
        app._side_busy['R'] = True
        app._result_timer_active = True
        app._active_inspection_side = 'R'
        app.toggle_result_timer = MagicMock(
            side_effect=lambda: setattr(app, '_result_timer_paused', True))

        app._handle_serial_button('R')

        app.toggle_result_timer.assert_called_once_with()
        app.serial_conn.write.assert_called_once_with(b'R_PAUSED\n')

    def test_other_button_starts_during_result_timer(self):
        app = self.make_app()
        app._side_busy['L'] = True
        app._result_timer_active = True
        app._active_inspection_side = 'L'
        app.toggle_result_timer = MagicMock()

        app._handle_serial_button('R')

        app.toggle_result_timer.assert_not_called()
        app.start_detect_thread.assert_called_once_with('R')
        app.serial_conn.write.assert_called_once_with(b'R_ACK\n')

    def test_missing_calibration_rejects_request(self):
        app = self.make_app()
        app.camera1.calibration_available = False

        app._handle_serial_button('L')

        app.start_detect_thread.assert_not_called()
        app.serial_conn.write.assert_called_once_with(b'L_NOT_READY\n')

    def test_failure_sends_error_for_that_side(self):
        app = self.make_app()
        app._serial_pending_sides = {'R'}
        app._report_detection_status('R', False)
        app.serial_conn.write.assert_called_once_with(b'R_ERROR\n')

    @patch('main_1366.threading.Thread')
    def test_both_sides_can_start_background_workers(self, thread_class):
        app = self.make_app()
        app.start_detect_thread = IndustrialDashboard.start_detect_thread.__get__(app)
        app.detect_btn_L = MagicMock()
        app.detect_btn_R = MagicMock()
        app.detect_btn_L.__getitem__.return_value = tk.NORMAL
        app.detect_btn_R.__getitem__.return_value = tk.NORMAL
        app.active_size = 'M'
        app.calibration_page = None
        app._inspection_order_counter = 0
        app._inspection_epoch = 0
        app._refresh_inspection_availability = MagicMock()
        app._crop_cameras = MagicMock(return_value={
            '1': [{'saved': True}, {'saved': True}],
            '2': [{'saved': True}, {'saved': True}],
        })

        self.assertTrue(app.start_detect_thread('L'))
        self.assertTrue(app.start_detect_thread('R'))
        self.assertEqual(app._side_busy, {'L': True, 'R': True})
        self.assertEqual(thread_class.call_count, 2)


if __name__ == '__main__':
    unittest.main()

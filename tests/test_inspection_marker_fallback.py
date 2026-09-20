import unittest
from unittest.mock import MagicMock, patch

import numpy as np

import main_1366 as main


class InspectionMarkerFallbackTests(unittest.TestCase):
    @patch.object(main.time, 'sleep')
    @patch.object(main.os.path, 'exists', return_value=False)
    @patch.object(main.os, 'makedirs')
    @patch.object(main.cv2, 'imwrite', return_value=True)
    @patch.object(main.ArucoPlaneEstimator, 'from_calibration_file')
    def test_missing_marker_warns_without_popup(
            self, estimator_factory, _imwrite, _makedirs, _exists, _sleep):
        app = main.IndustrialDashboard.__new__(main.IndustrialDashboard)
        frame = np.zeros((300, 450, 3), np.uint8)
        app.camera1 = MagicMock()
        app.camera1.get_raw_frame_with_ret.return_value = True, frame
        app.camera1.get_undistorted_frame.return_value = frame.copy()
        app.camera2 = MagicMock()
        app.active_size = 'M'
        app.canvas_1 = MagicMock()
        app.canvas_1.winfo_width.return_value = 450
        app.canvas_1.winfo_height.return_value = 300
        app.root = MagicMock()
        app.root.after.side_effect = lambda _delay, callback, *args: callback(*args)
        app._show_error_popup = MagicMock()
        app.update_progress = MagicMock()
        app.set_pass_fail = MagicMock()
        app._display_video_frame = MagicMock()
        app.maximize_camera = MagicMock()
        app.restore_dual_view = MagicMock()
        app._finish_detection = MagicMock()
        estimator_factory.return_value.detect.side_effect = RuntimeError('marker missing')

        app._simulate_detection('L')

        app._show_error_popup.assert_not_called()
        app.set_pass_fail.assert_called_with('WARNING')
        self.assertTrue(any(
            'real measurements were not calculated' in str(call)
            for call in app.update_progress.call_args_list
        ))
        app._finish_detection.assert_called_once()


if __name__ == '__main__':
    unittest.main()

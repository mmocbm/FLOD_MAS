import unittest
from unittest.mock import MagicMock, patch

import numpy as np

import main_1366 as main
from crop_processing import normalized_definition


class InspectionCropPreparationTests(unittest.TestCase):
    @patch.object(main.time, 'sleep')
    @patch.object(main.os, 'makedirs')
    @patch.object(main.cv2, 'imwrite', return_value=True)
    def test_inspection_prepares_two_independent_crops(
            self, imwrite, _makedirs, _sleep):
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
        app.update_progress = MagicMock()
        app._display_video_frame = MagicMock()
        app.maximize_camera = MagicMock()
        app.restore_dual_view = MagicMock()
        app._finish_detection = MagicMock()
        first = normalized_definition(
            (225, 100), 300, 0, [(100, 100), (350, 100)], (450, 300),
        )
        second = normalized_definition(
            (225, 200), 300, 0, [(100, 200), (350, 200)], (450, 300),
        )
        app.crop_definitions = {
            'version': 1, 'cameras': {'1': [first, second], '2': [None, None]},
        }

        app._simulate_detection('L')

        self.assertEqual(imwrite.call_count, 4)
        app._display_video_frame.assert_called_once()
        self.assertTrue(any(
            'two deskewed crops prepared' in str(call)
            for call in app.update_progress.call_args_list
        ))
        app._finish_detection.assert_called_once()


if __name__ == '__main__':
    unittest.main()

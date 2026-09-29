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
        app.root = MagicMock()
        app.root.after.side_effect = lambda _delay, callback, *args: callback(*args)
        app.update_progress = MagicMock()
        app._display_video_frame = MagicMock()
        app.maximize_camera = MagicMock()
        app.restore_dual_view = MagicMock()
        app._finish_detection = MagicMock()
        app.set_pass_fail = MagicMock()
        first = normalized_definition(
            (225, 100), 300, 0, [(100, 100), (350, 100)], (450, 300),
        )
        second = normalized_definition(
            (225, 200), 300, 0, [(100, 200), (350, 200)], (450, 300),
        )
        app.crop_definitions = {
            'version': 1, 'cameras': {'1': [first, second], '2': [None, None]},
        }

        # SAM detection is switched off so this test stays about crop
        # preparation: it must not reach the network, and an overlay write
        # would change the imwrite count asserted below.
        with patch.dict(main.CONFIG['sam_detection'], {'enabled': False}):
            camera_num, shown_crops, labels, warnings = app._simulate_detection('L')

        self.assertEqual(imwrite.call_count, 4)
        self.assertEqual(camera_num, 1)
        self.assertEqual(shown_crops, [])
        self.assertEqual(labels, [])
        self.assertEqual(warnings, [])
        app._display_video_frame.assert_not_called()
        app.maximize_camera.assert_not_called()
        app.restore_dual_view.assert_not_called()
        app._finish_detection.assert_not_called()
        # The 1.2s pause and the red highlight existed only to make the
        # maximized view readable, which the held result replaces.
        _sleep.assert_not_called()
        self.assertTrue(any(
            'no defective segments' in str(call)
            for call in app.update_progress.call_args_list
        ))


if __name__ == '__main__':
    unittest.main()

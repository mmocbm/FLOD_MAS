import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from CalibrateAPP.calibration_ui import CalibrationApp, MIN_CORNERS
from CalibrateAPP.generate_two_boards import generate_two_boards


def detection(board_index, center_x):
    image_points = np.array(
        [[[center_x - 20, 100]], [[center_x + 20, 100]],
         [[center_x + 20, 140]], [[center_x - 20, 140]]], np.float32,
    )
    return {
        'board_index': board_index,
        'corners': image_points.copy(),
        'ids': np.arange(4, dtype=np.int32).reshape(-1, 1),
        'object_points': np.zeros((4, 1, 3), np.float32),
        'image_points': image_points,
    }


class TwoBoardCalibrationTests(unittest.TestCase):
    def test_unique_boards_are_detected_separately_in_one_frame(self):
        app = CalibrationApp.__new__(CalibrationApp)
        app.use_two_boards = True
        app._init_detector()
        canvas = np.full((700, 1900), 255, np.uint8)
        canvas[100:590, 80:850] = app.boards[0].generateImage((770, 490), marginSize=10)
        canvas[100:590, 1050:1820] = app.boards[1].generateImage((770, 490), marginSize=10)

        detected_counts = []
        for detector in app.charuco_detectors:
            ids = detector.detectBoard(canvas)[1]
            detected_counts.append(0 if ids is None else len(ids))

        self.assertTrue(all(count >= MIN_CORNERS for count in detected_counts))
        self.assertTrue(
            set(app.boards[0].getIds()).isdisjoint(set(app.boards[1].getIds()))
        )

    def test_one_board_is_held_until_other_board_is_captured(self):
        app = CalibrationApp.__new__(CalibrationApp)
        app.capture_image_size = None
        app.pending_dual_capture = None
        app.guidance_info = MagicMock()
        app._set_status = MagicMock()
        app.log = MagicMock()
        app._show_frame = MagicMock()
        app._draw_coverage_guide = MagicMock()
        app._draw_capture_history = MagicMock()
        app._finish_capture_review = MagicMock()
        app._accept_dual_capture_set = MagicMock()
        app.root = MagicMock()
        app.root.after.side_effect = lambda _delay, callback, *args: callback(*args)
        first = detection(0, 200)
        second = detection(1, 600)
        frame = np.zeros((400, 800, 3), np.uint8)

        app._dual_capture_detection_complete(
            frame, Path('board_1.png'), [first], (800, 400),
        )
        self.assertEqual(app.pending_dual_capture['detection']['board_index'], 0)
        app._dual_capture_detection_complete(
            frame, Path('board_2.png'), [second], (800, 400),
        )

        app._accept_dual_capture_set.assert_called_once()
        accepted = app._accept_dual_capture_set.call_args.args[2]
        self.assertEqual([item['board_index'] for item in accepted], [0, 1])
        self.assertIsNone(app.pending_dual_capture)

    def test_both_boards_in_one_frame_are_accepted_together(self):
        app = CalibrationApp.__new__(CalibrationApp)
        app.capture_image_size = None
        app.pending_dual_capture = None
        app._accept_dual_capture_set = MagicMock()
        frame = np.zeros((400, 800, 3), np.uint8)

        app._dual_capture_detection_complete(
            frame, Path('both.png'), [detection(1, 600), detection(0, 200)],
            (800, 400),
        )

        accepted = app._accept_dual_capture_set.call_args.args[2]
        self.assertEqual([item['board_index'] for item in accepted], [0, 1])

    def test_printable_board_generator_creates_both_files(self):
        with patch('CalibrateAPP.generate_two_boards.cv2.imwrite', return_value=True) as write:
            paths = generate_two_boards(Path(__file__).parent, pixels_per_square=20)

        self.assertEqual(
            paths, [Path(__file__).parent / 'charuco_board_1.png',
                    Path(__file__).parent / 'charuco_board_2.png'],
        )
        self.assertEqual(write.call_count, 2)
        self.assertFalse(np.array_equal(write.call_args_list[0].args[1],
                                        write.call_args_list[1].args[1]))


if __name__ == '__main__':
    unittest.main()

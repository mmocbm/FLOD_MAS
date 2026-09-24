import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import cv2

from CalibrateAPP.calibration_math import coverage_cell, coverage_percent, next_coverage_cell
from CalibrateAPP import calibration_ui


class GuidanceMathTests(unittest.TestCase):
    def test_coverage_cell_uses_board_center(self):
        points = np.array([[[700.0, 100.0]], [[900.0, 300.0]]], np.float32)
        row, column, center = coverage_cell(points, (1200, 900), 3, 3)
        self.assertEqual((row, column), (0, 2))
        self.assertAlmostEqual(center[0], 2 / 3)

    def test_next_cell_prefers_unseen_area_far_from_last_photo(self):
        counts = np.zeros((3, 3), np.int32)
        counts[1, 1] = 1
        self.assertEqual(next_coverage_cell(counts, (1, 1)), (0, 0))
        counts[0, 0] = 1
        self.assertEqual(coverage_percent(counts), 22)


class PerCaptureProcessingTests(unittest.TestCase):
    @patch.object(calibration_ui.threading, 'Thread')
    def test_capture_button_starts_detection_worker_only(self, thread_class):
        app = calibration_ui.CalibrationApp.__new__(calibration_ui.CalibrationApp)
        app.current_frame = np.zeros((900, 1200, 3), np.uint8)
        app.frame_lock = threading.Lock()
        app.stage = 'capture'
        app.processing_capture = False
        app.camera_index = 0
        app.captured_images = []
        app.object_views = []
        app.image_views = []
        app.capture_image_size = None
        app.capture_centers = []
        app.capture_point_sets = []
        app.coverage_counts = np.zeros((3, 3), np.int32)
        app.capture_info = MagicMock()
        app.progress = MagicMock()
        app._set_status = MagicMock()
        app.log = MagicMock()
        app._refresh_stage_ui = MagicMock()
        app._show_frame = MagicMock()
        app.root = MagicMock()

        app.charuco_detector = MagicMock()

        app.manual_capture()

        app.charuco_detector.detectBoard.assert_not_called()
        self.assertTrue(app.processing_capture)
        thread_class.assert_called_once()
        self.assertEqual(thread_class.call_args.kwargs['target'], app._detect_capture_worker)
        thread_class.return_value.start.assert_called_once()

    def test_detection_worker_returns_points_to_ui(self):
        app = calibration_ui.CalibrationApp.__new__(calibration_ui.CalibrationApp)
        app.root = MagicMock()
        app.root.after.side_effect = lambda _delay, callback, *args: callback(*args)
        app._capture_detection_complete = MagicMock()
        app._capture_rejected = MagicMock()
        point_count = calibration_ui.MIN_CORNERS
        corners = np.array(
            [[[500 + i * 8.0, 350 + i * 3.0]] for i in range(point_count)], np.float32,
        )
        ids = np.arange(point_count, dtype=np.int32).reshape(-1, 1)
        object_points = np.array(
            [[[i * 0.01, 0.0, 0.0]] for i in range(point_count)], np.float32,
        )
        app.charuco_detector = MagicMock()
        app.charuco_detector.detectBoard.return_value = corners, ids, None, None
        app.board = MagicMock()
        app.board.matchImagePoints.return_value = object_points, corners

        saved_frame = np.zeros((900, 1200, 3), np.uint8)
        with (patch.object(calibration_ui.cv2, 'imwrite', return_value=True),
              patch.object(calibration_ui.cv2, 'imread', return_value=saved_frame)):
            app._detect_capture_worker(saved_frame, Path('capture.png'))

        app._capture_detection_complete.assert_called_once()
        app._capture_rejected.assert_not_called()

    def test_rejected_saved_photo_is_deleted(self):
        app = calibration_ui.CalibrationApp.__new__(calibration_ui.CalibrationApp)
        app.root = MagicMock()
        app.root.after.side_effect = lambda _delay, callback, *args: callback(*args)
        app._capture_rejected = MagicMock()
        app.charuco_detector = MagicMock()
        app.charuco_detector.detectBoard.return_value = (
            np.zeros((2, 1, 2), np.float32), np.arange(2, dtype=np.int32).reshape(-1, 1),
            None, None,
        )
        file_path = MagicMock()
        saved_frame = np.zeros((900, 1200, 3), np.uint8)
        with (patch.object(calibration_ui.cv2, 'imwrite', return_value=True),
              patch.object(calibration_ui.cv2, 'imread', return_value=saved_frame)):
            app._detect_capture_worker(saved_frame, file_path)

        file_path.unlink.assert_called_once_with(missing_ok=True)
        app._capture_rejected.assert_called_once()

    @patch.object(calibration_ui.threading, 'Thread')
    def test_accepted_photo_saves_points_without_running_calibration(self, thread_class):
        app = calibration_ui.CalibrationApp.__new__(calibration_ui.CalibrationApp)
        app.camera_index = 0
        app.processing_capture = True
        app.stage = 'capture'
        app.captured_images = []
        app.object_views = []
        app.image_views = []
        app.capture_image_size = None
        app.capture_centers = []
        app.capture_point_sets = []
        app.coverage_counts = np.zeros((3, 3), np.int32)
        app.capture_info = MagicMock()
        app.guidance_info = MagicMock()
        app.progress = MagicMock()
        app._set_status = MagicMock()
        app.log = MagicMock()
        app._refresh_stage_ui = MagicMock()
        app._show_frame = MagicMock()
        app.root = MagicMock()
        point_count = calibration_ui.MIN_CORNERS
        corners = np.array(
            [[[500 + i * 8.0, 350 + i * 3.0]] for i in range(point_count)], np.float32,
        )
        ids = np.arange(point_count, dtype=np.int32).reshape(-1, 1)
        object_points = np.array(
            [[[i * 0.01, 0.0, 0.0]] for i in range(point_count)], np.float32,
        )

        app._capture_detection_complete(
            np.zeros((900, 1200, 3), np.uint8), Path('accepted.png'), corners, ids,
            object_points, corners, (1200, 900),
        )

        self.assertEqual(len(app.object_views), 1)
        self.assertEqual(len(app.capture_point_sets), 1)
        self.assertEqual(int(app.coverage_counts.sum()), 1)
        thread_class.assert_not_called()
        app.root.after.assert_called_once_with(650, app._finish_capture_review)

    def test_live_preview_does_not_detect_board(self):
        app = calibration_ui.CalibrationApp.__new__(calibration_ui.CalibrationApp)
        app.closed = False
        app.processing_capture = False
        app.camera_running = True
        app.frame_lock = threading.Lock()
        app.current_frame = None
        app.capture_centers = []
        app.capture_point_sets = []
        app.coverage_counts = np.zeros((3, 3), np.int32)
        app.charuco_detector = MagicMock()
        app._show_frame = MagicMock()
        app.root = MagicMock()
        app.cap = MagicMock()
        app.cap.isOpened.return_value = True
        app.cap.read.return_value = True, np.zeros((900, 1200, 3), np.uint8)

        app.update_frame()

        app.charuco_detector.detectBoard.assert_not_called()
        app._show_frame.assert_called_once()
        app.root.after.assert_called_once()

    def test_final_calibration_uses_all_saved_views(self):
        app = calibration_ui.CalibrationApp.__new__(calibration_ui.CalibrationApp)
        app.root = MagicMock()
        app.root.after.side_effect = lambda _delay, callback, *args: callback(*args)
        app._intrinsic_complete = MagicMock()
        app._calibration_failed = MagicMock()
        calibration_path = MagicMock()
        app._calibration_path = MagicMock(return_value=calibration_path)
        app.coverage_counts = np.ones((3, 3), np.int32)
        app.capture_centers = [(0.5, 0.5)]
        app.is_calibrating = False

        dictionary = cv2.aruco.getPredefinedDictionary(calibration_ui.DICT_TYPE)
        board = cv2.aruco.CharucoBoard(
            (calibration_ui.SQUARES_X, calibration_ui.SQUARES_Y),
            calibration_ui.SQUARE_LENGTH_MM / 1000.0,
            calibration_ui.MARKER_LENGTH_MM / 1000.0,
            dictionary,
        )
        object_points = np.asarray(board.getChessboardCorners(), np.float32)
        camera_matrix = np.array(
            [[1200.0, 0.0, 600.0], [0.0, 1180.0, 450.0], [0.0, 0.0, 1.0]],
            np.float64,
        )
        object_views, image_views = [], []
        for index in range(10):
            rvec = np.array([[0.04 * index], [-0.03 * index], [0.01 * index]], np.float64)
            tvec = np.array([[0.01 * index], [-0.006 * index], [0.75 + 0.03 * index]], np.float64)
            image_points, _ = cv2.projectPoints(
                object_points, rvec, tvec, camera_matrix, np.zeros((5, 1)),
            )
            object_views.append(object_points.copy())
            image_views.append(image_points.astype(np.float32))

        app._calibrate_intrinsics_worker(object_views, image_views, (1200, 900))

        calibration_path.write_text.assert_called_once()
        app._intrinsic_complete.assert_called_once()
        self.assertTrue(np.isfinite(app._intrinsic_complete.call_args.args[0]).all())
        app._calibration_failed.assert_not_called()

    @patch.object(calibration_ui.messagebox, 'showinfo')
    def test_marker_mode_finishes_after_camera_calibration(self, showinfo):
        app = calibration_ui.CalibrationApp.__new__(calibration_ui.CalibrationApp)
        app.processing_capture = True
        app.camera_running = True
        app.camera_provider = object()
        app.cap = None
        app.camera_index = 0
        app.coverage_counts = np.ones((3, 3), np.int32)
        app.stop_btn = MagicMock()
        app.start_btn = MagicMock()
        app.resolution_label = MagicMock()
        app.capture_info = MagicMock()
        app.guidance_info = MagicMock()
        app._set_status = MagicMock()
        app.log = MagicMock()
        app._refresh_stage_ui = MagicMock()

        with patch.object(calibration_ui, 'SURFACE_SETUP_ENABLED', False):
            app._intrinsic_complete(
                np.eye(3), np.zeros((5, 1)), (1200, 900), 0.5,
                calibration_ui.NUM_CAPTURES,
            )

        self.assertEqual(app.stage, 'complete')
        self.assertFalse(app.camera_running)
        showinfo.assert_called_once()

    def test_saved_surface_mode_remains_available(self):
        app = calibration_ui.CalibrationApp.__new__(calibration_ui.CalibrationApp)
        app.processing_capture = True
        app.camera_index = 0
        app.coverage_counts = np.ones((3, 3), np.int32)
        app.capture_info = MagicMock()
        app.guidance_info = MagicMock()
        app._set_status = MagicMock()
        app.log = MagicMock()
        app._refresh_stage_ui = MagicMock()

        with patch.object(calibration_ui, 'SURFACE_SETUP_ENABLED', True):
            app._intrinsic_complete(
                np.eye(3), np.zeros((5, 1)), (1200, 900), 0.5,
                calibration_ui.NUM_CAPTURES,
            )

        self.assertEqual(app.stage, 'extrinsic_ready')
        self.assertFalse(app.processing_capture)

    def test_coverage_overlay_marks_next_area_and_previous_points(self):
        app = calibration_ui.CalibrationApp.__new__(calibration_ui.CalibrationApp)
        app.coverage_counts = np.zeros((3, 3), np.int32)
        app.coverage_counts[1, 1] = 1
        app.capture_centers = [(0.5, 0.5)]
        app.capture_point_sets = [np.array([[0.45, 0.45], [0.55, 0.55]], np.float32)]
        frame = np.zeros((300, 450, 3), np.uint8)

        app._draw_coverage_guide(frame)
        app._draw_capture_history(frame)

        self.assertGreater(int(np.count_nonzero(frame)), 0)

    def test_annotation_tolerates_corner_id_count_mismatch(self):
        app = calibration_ui.CalibrationApp.__new__(calibration_ui.CalibrationApp)
        corners = np.array(
            [[[100.0, 100.0]], [[150.0, 120.0]], [[200.0, 140.0]]], np.float32,
        )
        ids = np.array([[1], [2]], np.int32)

        annotated = app._annotate_detected_board(
            np.zeros((300, 450, 3), np.uint8), corners, ids,
        )

        self.assertGreater(int(np.count_nonzero(annotated)), 0)

if __name__ == '__main__':
    unittest.main()

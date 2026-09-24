import unittest

import cv2
import numpy as np

from CalibrateAPP.measurement_accuracy import (
    measure_board_accuracy, offset_plane_tvec, summarize_accuracy,
)


class MeasurementAccuracyTests(unittest.TestCase):
    def setUp(self):
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        self.board = cv2.aruco.CharucoBoard((7, 9), 0.028, 0.021, dictionary)
        self.camera_matrix = np.array([
            [1300.0, 0.0, 640.0],
            [0.0, 1290.0, 480.0],
            [0.0, 0.0, 1.0],
        ])
        self.distortion = np.zeros((5, 1), np.float64)
        self.rvec = np.array([[0.08], [-0.12], [0.03]], np.float64)
        self.tvec = np.array([[-0.08], [-0.10], [0.82]], np.float64)
        object_points = self.board.getChessboardCorners()
        image_points, _ = cv2.projectPoints(
            object_points, self.rvec, self.tvec,
            self.camera_matrix, self.distortion,
        )
        self.corners = image_points.astype(np.float32)
        self.ids = np.arange(len(object_points), dtype=np.int32).reshape(-1, 1)

    def test_exact_projection_reports_short_and_long_distances(self):
        rows = measure_board_accuracy(
            self.board, self.corners, self.ids,
            self.camera_matrix, self.distortion,
            self.rvec, self.tvec, 28.0,
        )
        summary = summarize_accuracy(rows)

        self.assertGreater(summary['short']['count'], 0)
        self.assertGreater(summary['long']['count'], 0)
        self.assertLess(summary['overall']['maximum_absolute_error_mm'], 0.001)

    def test_board_thickness_moves_measurement_plane_away_from_camera(self):
        shifted = offset_plane_tvec(
            self.rvec, self.tvec, 2.0, away_from_camera=True,
        )
        rotation, _ = cv2.Rodrigues(self.rvec)
        normal = rotation[:, 2]
        before = abs(float(normal.dot(self.tvec.reshape(3))))
        after = abs(float(normal.dot(shifted.reshape(3))))

        self.assertAlmostEqual(after - before, 0.002, places=9)
        restored = offset_plane_tvec(
            self.rvec, shifted, 2.0, away_from_camera=False,
        )
        np.testing.assert_allclose(restored, self.tvec, atol=1e-12)


if __name__ == '__main__':
    unittest.main()

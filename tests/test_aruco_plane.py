import unittest

import cv2
import numpy as np

from measure.aruco_plane import (
    ArucoPlaneEstimator, MarkerPlaneError, annotate_mask_measurements,
    annotate_mask_pixel_measurements,
)


class ArucoPlaneTests(unittest.TestCase):
    def setUp(self):
        self.camera_matrix = np.array(
            [[800.0, 0.0, 400.0], [0.0, 800.0, 300.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        self.estimator = ArucoPlaneEstimator(
            self.camera_matrix, np.zeros((5, 1)), (800, 600),
            dictionary_name='DICT_4X4_50', marker_id=0,
            marker_length_mm=25.0, minimum_side_px=30.0,
            maximum_reprojection_error_px=2.0,
        )

    @staticmethod
    def marker_image(marker_size=200):
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        marker = cv2.aruco.generateImageMarker(dictionary, 0, marker_size)
        image = np.full((600, 800, 3), 255, np.uint8)
        image[200:200 + marker_size, 300:300 + marker_size] = cv2.cvtColor(
            marker, cv2.COLOR_GRAY2BGR,
        )
        return image

    def test_marker_zero_creates_metric_homography(self):
        image = self.marker_image()
        plane = self.estimator.detect(image, image_is_undistorted=True)

        mapped = plane.pixels_to_mm(plane.corners)
        side_lengths = np.linalg.norm(mapped - np.roll(mapped, -1, axis=0), axis=1)
        self.assertTrue(np.allclose(side_lengths, 25.0, atol=1e-3))
        self.assertLessEqual(plane.reprojection_rms_px, 2.0)
        self.assertGreater(np.count_nonzero(self.estimator.annotate(image, plane)), 0)

    def test_missing_marker_stops_measurement(self):
        with self.assertRaisesRegex(MarkerPlaneError, 'was not found'):
            self.estimator.detect(np.full((600, 800, 3), 255, np.uint8))

    def test_marker_must_be_large_enough(self):
        estimator = ArucoPlaneEstimator(
            self.camera_matrix, np.zeros((5, 1)), (800, 600),
            minimum_side_px=250.0,
        )
        with self.assertRaisesRegex(MarkerPlaneError, 'too small'):
            estimator.detect(self.marker_image())

    def test_mask_component_is_measured_in_marker_millimetres(self):
        image = self.marker_image()
        plane = self.estimator.detect(image)
        mask = np.zeros(image.shape[:2], np.uint8)
        cv2.rectangle(mask, (300, 200), (380, 240), 255, -1)

        _, measurements = annotate_mask_measurements(image, mask, plane)

        self.assertEqual(len(measurements), 1)
        self.assertAlmostEqual(measurements[0]['length_mm'], 10.05, delta=0.2)
        self.assertAlmostEqual(measurements[0]['width_mm'], 5.03, delta=0.2)

    def test_missing_marker_fallback_measures_pixels(self):
        image = np.zeros((300, 450, 3), np.uint8)
        mask = np.zeros(image.shape[:2], np.uint8)
        cv2.rectangle(mask, (100, 100), (180, 140), 255, -1)

        annotated, measurements = annotate_mask_pixel_measurements(image, mask)

        self.assertEqual(len(measurements), 1)
        self.assertAlmostEqual(measurements[0]['length_px'], 80.0, delta=0.1)
        self.assertAlmostEqual(measurements[0]['width_px'], 40.0, delta=0.1)
        self.assertGreater(np.count_nonzero(annotated), 0)


if __name__ == '__main__':
    unittest.main()

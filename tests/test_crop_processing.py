import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from crop_processing import (
    definition_fits_image, extract_rotated_crop, load_crop_store,
    normalized_definition, parallel_line_angle, save_crop_store,
)


class CropProcessingTests(unittest.TestCase):
    def test_rotation_line_is_direction_independent(self):
        forward = parallel_line_angle((10, 10), (20, 20))
        reverse = parallel_line_angle((20, 20), (10, 10))
        self.assertAlmostEqual(forward, 45.0)
        self.assertAlmostEqual(reverse, 45.0)

    def test_rotated_crop_is_deskewed_to_requested_size(self):
        image = np.zeros((600, 1000, 3), np.uint8)
        center = (500.0, 300.0)
        width = 600.0
        angle = 20.0
        radians = np.deg2rad(angle)
        direction = np.array([np.cos(radians), np.sin(radians)])
        line = [np.asarray(center) - direction * 100,
                np.asarray(center) + direction * 100]
        definition = normalized_definition(center, width, angle, line, (1000, 600))
        self.assertTrue(definition_fits_image(definition, (1000, 600)))
        crop = extract_rotated_crop(image, definition, (2208, 552))
        self.assertEqual(crop.shape[:2], (552, 2208))

    def test_store_round_trip_keeps_two_crops_per_camera(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "crops.json"
            data = {"version": 1, "cameras": {"1": [{"x": 1}, None], "2": [None, {"x": 2}]}}
            save_crop_store(path, data)
            self.assertEqual(load_crop_store(path), data)


if __name__ == '__main__':
    unittest.main()

"""Crop-to-millimetre scale, built from a synthetic calibrated plane.

Every fixture here is synthetic: a pinhole camera looking at a plane, with the
crop region and the plane geometry both known exactly. That makes the true
answer to every question independent of the code under test -- a point placed at
a known position in millimetres on the plane must come back at that position, and
a square of a known size must measure as that size.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import plane_scale

FRAME = (3456, 4608)
OUTPUT = (2208, 552)

# A camera roughly a metre from the plane, looking down at it slightly, at the
# sort of focal length and resolution the real rig uses.
CAMERA_MATRIX = np.array([
    [6812.9, 0.0, 1949.6],
    [0.0, 6812.9, 2243.1],
    [0.0, 0.0, 1.0],
], dtype=np.float64)
DISTORTION = np.zeros((5, 1), dtype=np.float64)
RVEC = np.array([[0.62], [0.05], [0.03]], dtype=np.float64)
TVEC = np.array([[0.04], [-0.31], [1.10]], dtype=np.float64)

# A crop region covering a band across the middle of the frame.
DEFINITION = {
    "center_normalized": [0.4931506849, 0.2508561644],
    "width_normalized": 0.7762557078,
    "angle_degrees": 2.3760552180,
    "line_normalized": [[0.2283105023, 0.2859589041], [0.7785388128, 0.3030821918]],
}


def write_json(directory, name, body):
    path = Path(directory) / name
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def calibration(fx=6812.9):
    return {"camera_matrix": [[fx, 0.0, 1949.6], [0.0, fx, 2243.1], [0.0, 0.0, 1.0]],
            "dist_coeffs": [0.0] * 5, "image_size": list(FRAME)}


def extrinsics():
    return {"rvec": RVEC.ravel().tolist(), "tvec": TVEC.ravel().tolist()}


class RoundTripTests(unittest.TestCase):
    """A known position on the plane must survive the trip into the crop."""

    def setUp(self):
        self.scale = plane_scale.load_plane_scale(
            1, DEFINITION, FRAME,
            calibration_path=self._file(calibration()),
            extrinsics_path=self._file(extrinsics()),
        )

    def _file(self, body):
        handle = tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8")
        json.dump(body, handle)
        handle.close()
        self.addCleanup(Path(handle.name).unlink)
        return handle.name

    def _to_crop(self, plane_mm):
        """A known plane position -> source pixel -> crop pixel, the app's path.

        The pose is in metres, so the object points must be too; millimetres go
        in and a thousandth of that is projected.
        """
        points = np.column_stack((plane_mm / 1000.0, np.zeros(len(plane_mm))))
        source, _ = cv2.projectPoints(
            points.astype(np.float64), RVEC, TVEC, CAMERA_MATRIX, DISTORTION)
        source = source.reshape(-1, 2).astype(np.float32)
        from crop_processing import rotated_crop_transform
        to_crop = rotated_crop_transform(DEFINITION, FRAME, OUTPUT)
        return cv2.perspectiveTransform(
            source.reshape(-1, 1, 2), to_crop).reshape(-1, 2)

    def test_known_plane_positions_come_back_unchanged(self):
        # Spread across the region the crop covers, in millimetres.
        plane_mm = np.array([
            [-100.0, -40.0], [0.0, 0.0], [120.0, 30.0],
            [250.0, -60.0], [300.0, 55.0],
        ])
        crop = self._to_crop(plane_mm)
        measured = self.scale.to_mm(crop)

        np.testing.assert_allclose(measured, plane_mm, atol=0.5)

    def test_a_square_measures_its_true_side(self):
        """A 50mm square on the plane reads as 50mm in the crop."""
        side = 50.0
        corners_mm = np.array([
            [0.0, 0.0], [side, 0.0], [side, side], [0.0, side],
        ])
        measured = self.scale.to_mm(self._to_crop(corners_mm))
        sides = np.linalg.norm(measured - np.roll(measured, -1, axis=0), axis=1)

        np.testing.assert_allclose(sides, side, atol=0.5)

    def test_probing_the_plane_intersection_is_what_the_homography_does(self):
        """The map is exactly the ray/plane intersection, sampled four times."""
        from crop_processing import rotated_crop_transform
        from CalibrateAPP.calibration_math import pixel_to_plane

        to_crop = rotated_crop_transform(DEFINITION, FRAME, OUTPUT)
        inverse = np.linalg.inv(to_crop)
        rvec = RVEC.reshape(3, 1)
        tvec = TVEC.reshape(3, 1)

        probes = np.array([[0.0, 0.0], [2207.0, 551.0], [1104.0, 276.0],
                           [500.0, 480.0]], dtype=np.float32)
        source = cv2.perspectiveTransform(
            probes.reshape(-1, 1, 2), inverse).reshape(-1, 2)
        expected = np.array([
            pixel_to_plane(point, CAMERA_MATRIX, rvec, tvec) * 1000.0
            for point in source
        ])
        np.testing.assert_allclose(self.scale.to_mm(probes), expected, atol=0.01)

    def test_mm_per_pixel_summarises_the_crop_centre(self):
        """A one-pixel step at the centre is exactly the reported mm/px."""
        width, height = OUTPUT
        centre = np.array([[width / 2.0, height / 2.0],
                           [width / 2.0 + 1.0, height / 2.0]], dtype=np.float32)
        step = self.scale.to_mm(centre)
        measured = float(np.linalg.norm(step[1] - step[0]))

        # float32 in, float64 out: agree to well under a micron, not exactly.
        self.assertAlmostEqual(measured, self.scale.mm_per_pixel, places=4)

    def test_scale_is_not_a_constant_across_a_projected_crop(self):
        """A pixel is a different number of millimetres at the crop's ends.

        This is the reason distances are recomputed from mapped points rather
        than multiplied by one factor, so it is worth pinning down.
        """
        width = OUTPUT[0]
        ends = np.array([[10.0, 276.0], [width - 10.0, 276.0]], dtype=np.float32)
        mapped = self.scale.to_mm(ends)
        at_left = np.linalg.norm(
            self.scale.to_mm(np.array([[10.0, 276.0], [11.0, 276.0]], np.float32))[1]
            - mapped[0])
        at_right = np.linalg.norm(
            self.scale.to_mm(
                np.array([[width - 11.0, 276.0], [width - 10.0, 276.0]], np.float32))[1]
            - mapped[1])

        self.assertGreater(abs(at_left - at_right), 1e-4)


class PairingTests(unittest.TestCase):
    """Cameras are numbered 1 and 2 by position, as the rest of the app does.

    Matching on the config's ``index`` field instead hands camera 1 the other
    camera's calibration -- a silent few-percent scale error, so it is pinned
    here rather than left to the integration run to notice.
    """

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)

        # Two cameras whose calibrations are told apart by focal length alone.
        for slot, fx in ((0, 5000.0), (1, 9000.0)):
            write_json(root, f"calibration_{slot}.json", calibration(fx))
            write_json(root, f"extrinsics_{slot}.json", extrinsics())

        self.config = {
            "cameras": [
                {"index": 0, "calibration_file": f"calibration_0.json",
                 "extrinsics_file": "extrinsics_0.json"},
                {"index": 1, "calibration_file": f"calibration_1.json",
                 "extrinsics_file": "extrinsics_1.json"},
            ]
        }
        self.root = root

    def _load(self, camera_number):
        with patch("app_config.CONFIG", self.config), \
                patch("app_config.project_path",
                      lambda path: str(self.root / Path(path).name)):
            return plane_scale.load_plane_scale(camera_number, DEFINITION, FRAME)

    def _direct(self, slot):
        """The same camera, with its files named outright -- no lookup involved."""
        return plane_scale.load_plane_scale(
            1, DEFINITION, FRAME,
            calibration_path=self.root / f"calibration_{slot}.json",
            extrinsics_path=self.root / f"extrinsics_{slot}.json",
        )

    def test_camera_one_uses_the_first_configured_camera(self):
        self.assertAlmostEqual(
            self._load(1).mm_per_pixel, self._direct(0).mm_per_pixel, places=9)

    def test_camera_two_uses_the_second_configured_camera(self):
        self.assertAlmostEqual(
            self._load(2).mm_per_pixel, self._direct(1).mm_per_pixel, places=9)

    def test_each_camera_gets_a_different_file(self):
        self.assertNotAlmostEqual(
            self._load(1).mm_per_pixel, self._load(2).mm_per_pixel, places=5)

    def test_an_unknown_camera_number_is_rejected(self):
        for number in (0, 3):
            with self.subTest(camera=number):
                with self.assertRaises(plane_scale.PlaneScaleError):
                    self._load(number)


class FailureTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_a_missing_calibration_file_is_reported(self):
        with self.assertRaisesRegex(plane_scale.PlaneScaleError, "missing"):
            plane_scale.load_plane_scale(
                1, DEFINITION, FRAME,
                calibration_path=self.root / "nope.json",
                extrinsics_path=self.root / "nope.json",
            )

    def test_unreadable_json_is_reported(self):
        broken = self.root / "broken.json"
        broken.write_text("{not json", encoding="utf-8")
        with self.assertRaisesRegex(plane_scale.PlaneScaleError, "could not be read"):
            plane_scale.load_plane_scale(
                1, DEFINITION, FRAME,
                calibration_path=broken, extrinsics_path=broken)

    def test_a_calibration_missing_a_key_is_reported(self):
        body = calibration()
        del body["camera_matrix"]
        path = write_json(self.root, "calibration.json", body)
        extrinsics_path = write_json(self.root, "extrinsics.json", extrinsics())

        with self.assertRaisesRegex(plane_scale.PlaneScaleError, "camera_matrix"):
            plane_scale.load_plane_scale(
                1, DEFINITION, FRAME,
                calibration_path=path, extrinsics_path=extrinsics_path)

    def test_a_degenerate_crop_region_is_reported(self):
        flat = dict(DEFINITION, width_normalized=0.0)
        path = write_json(self.root, "calibration.json", calibration())
        extrinsics_path = write_json(self.root, "extrinsics.json", extrinsics())

        with self.assertRaises(plane_scale.PlaneScaleError):
            plane_scale.load_plane_scale(
                1, flat, FRAME, calibration_path=path,
                extrinsics_path=extrinsics_path)


if __name__ == '__main__':
    unittest.main()

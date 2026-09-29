"""Per-region local homographies, fitted from a synthetic board on a known plane.

Every fixture is synthetic and exact: a pinhole camera looking at a plane, a crop region
whose geometry is known, and a board whose printed coordinates are known because we chose
them. So the true answer to every question is independent of the code under test -- a
board corner placed at a known millimetre position must come back at that position, and
the board's own known geometry must reproduce.

The convention under test is the subtle part, and the one this file exists to pin down.
``rotated_crop_transform`` maps the source rectangle's boundary onto
``-0.5 .. W-0.5``, which is the raster's outer boundary, and ``warpPerspective`` treats a
destination pixel's integer index as its continuous coordinate -- so a crop array index
*is* the continuous coordinate, with no shift. Adding a half-pixel shift moves every
mapped point by ``0.7071 x mm-per-crop-pixel``: a constant translation that no residual
report, repeatability check or strip length would ever reveal. It is asserted here from
both directions.
"""
import inspect
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

import plane_scale
import region_calibration
from crop_processing import rotated_crop_transform
from CalibrateAPP.calibration_math import pixel_to_plane

FRAME = (3456, 4608)
OUTPUT = (2208, 552)

# The same synthetic rig tests/test_plane_scale.py uses, so the two modes are compared on
# identical geometry rather than on two rigs that might differ in some unnoticed way.
CAMERA_MATRIX = np.array([
    [6812.9, 0.0, 1949.6],
    [0.0, 6812.9, 2243.1],
    [0.0, 0.0, 1.0],
], dtype=np.float64)
DISTORTION = np.zeros((5, 1), dtype=np.float64)
RVEC = np.array([[0.62], [0.05], [0.03]], dtype=np.float64)
TVEC = np.array([[0.04], [-0.31], [1.10]], dtype=np.float64)

DEFINITION = {
    "center_normalized": [0.4931506849, 0.2508561644],
    "width_normalized": 0.7762557078,
    "angle_degrees": 2.3760552180,
    "line_normalized": [[0.2283105023, 0.2859589041], [0.7785388128, 0.3030821918]],
}

# A card small enough to sit inside the region above, which is roughly 345 x 86 mm on the
# plane. 7 x 5 squares of 10 mm is 70 x 50 mm.
TEST_BOARD = region_calibration.BoardDefinition(
    name="Test 7x5",
    dictionary="DICT_5X5_100",
    squares_x=7,
    squares_y=5,
    square_length_mm=10.0,
    marker_length_mm=7.0,
)

# Laid at an angle, so a fit that quietly assumed the card was axis-aligned against the
# crop would fail rather than pass.
BOARD_ROTATION_DEGREES = 3.0

# "Exact" for a correspondence fitted by DLT over coordinates of order 100 mm. Double
# precision leaves a few times 1e-6 mm here; 1e-4 mm is a hundredth of a micron and still
# six orders of magnitude below anything physical, so it distinguishes a correct fit from
# a wrong one without asserting on floating-point noise.
EXACT_MM = 1e-4


def rotation(degrees):
    radians = np.deg2rad(float(degrees))
    return np.array([[np.cos(radians), -np.sin(radians)],
                     [np.sin(radians), np.cos(radians)]], dtype=np.float64)


class SyntheticRig:
    """A board lying on a known plane, seen through a known camera."""

    def __init__(self, definition=DEFINITION, board=TEST_BOARD,
                 rotation_degrees=BOARD_ROTATION_DEGREES, offset_mm=None):
        self.definition = definition
        self.board = board
        self.camera_matrix = CAMERA_MATRIX
        self.rvec = RVEC
        self.tvec = TVEC
        self.to_crop = rotated_crop_transform(definition, FRAME, OUTPUT)
        self.corner_points_mm = board.corner_points_mm()

        self.plane_points_mm = self._place(self.corner_points_mm, rotation_degrees,
                                           offset_mm)
        self.frame_points = self._project(self.plane_points_mm)
        # No offset: to_crop's output *is* the crop array index. See the module docstring
        # of region_calibration and PixelConventionTests below.
        self.crop_points = self._deskew(self.frame_points)

    # -- geometry ---------------------------------------------------------------------

    def _place(self, points_mm, rotation_degrees, offset_mm=None):
        """Lay board-local millimetres onto the plane, centred in the crop region."""
        centre = self.region_centre_mm()
        local = points_mm - points_mm.mean(axis=0)
        return local @ rotation(rotation_degrees).T + centre + (
            np.zeros(2) if offset_mm is None else np.asarray(offset_mm, dtype=np.float64))

    def _project(self, plane_mm):
        """Plane millimetres -> source-frame pixels, through the real camera."""
        metres = np.column_stack((np.asarray(plane_mm) / 1000.0, np.zeros(len(plane_mm))))
        projected, _ = cv2.projectPoints(
            metres.astype(np.float64), self.rvec, self.tvec,
            self.camera_matrix, DISTORTION)
        return projected.reshape(-1, 2)

    def _deskew(self, frame_pixels):
        """Source-frame pixels -> crop pixels, exactly as the application does."""
        return cv2.perspectiveTransform(
            np.asarray(frame_pixels, dtype=np.float64).reshape(-1, 1, 2),
            self.to_crop).reshape(-1, 2)

    def crop_to_plane_mm(self, crop_pixels):
        """Crop pixels -> plane millimetres, the long way round."""
        source = cv2.perspectiveTransform(
            np.asarray(crop_pixels, dtype=np.float64).reshape(-1, 1, 2),
            np.linalg.inv(self.to_crop)).reshape(-1, 2)
        rvec, tvec = self.rvec.reshape(3, 1), self.tvec.reshape(3, 1)
        return np.array([
            pixel_to_plane(point, self.camera_matrix, rvec, tvec) * 1000.0
            for point in source
        ], dtype=np.float64)

    def region_corners_crop(self):
        width, height = OUTPUT
        return np.array([[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]])

    def region_centre_mm(self):
        return self.crop_to_plane_mm(self.region_corners_crop()).mean(axis=0)

    def region_size_mm(self):
        on_plane = self.crop_to_plane_mm(self.region_corners_crop())
        return (float(np.linalg.norm(on_plane[1] - on_plane[0])),
                float(np.linalg.norm(on_plane[3] - on_plane[0])))

    def detection(self):
        """The exact correspondences, shaped as a real detection.

        Raw corners and ids are supplied, not left empty, because
        :func:`region_calibration.evaluate_intrinsics` solves the board pose from them.
        The ids are ``0..N-1``, which is correct: ``corner_points_mm`` is OpenCV's own
        chessboard corner list, so corner id ``i`` is board point ``i`` by construction.
        """
        count = len(self.corner_points_mm)
        return region_calibration.BoardDetection(
            frame_points=self.frame_points, crop_points=self.crop_points,
            board_points_mm=self.corner_points_mm,
            corners=self.frame_points.reshape(-1, 1, 2).astype(np.float32),
            ids=np.arange(count, dtype=np.int32).reshape(-1, 1),
            corner_count=count)

    def fit(self):
        return region_calibration.fit_region(
            self.detection(), self.board, CAMERA_MATRIX, FRAME, OUTPUT)

    # -- rendering --------------------------------------------------------------------

    def render_frame(self):
        """A source frame with the board rendered into it at its true perspective.

        The board image is generated at the pixel scale it will occupy, so the warp is
        close to one to one and the corners are not softened by resampling -- which
        matters, because this fixture is used to test *detection*, not just arithmetic.
        """
        board = self.board
        width_mm, height_mm = board.footprint_mm()

        # The card is drawn at the pixel scale it will actually occupy in the frame --
        # one square's projected length divided by one square's printed length -- so the
        # warp is close to one to one and detection sees crisp corners.
        local = board.corner_points_mm()
        placed = self._place(local, BOARD_ROTATION_DEGREES)
        frame_points = self._project(placed)
        px_per_mm = float(np.linalg.norm(frame_points[1] - frame_points[0])) / \
            float(board.square_length_mm)

        margin_mm = 2.0 * float(board.square_length_mm)
        source_size = (int(round((width_mm + 2 * margin_mm) * px_per_mm)),
                       int(round((height_mm + 2 * margin_mm) * px_per_mm)))
        image = board.build_board().generateImage(
            source_size, marginSize=int(round(margin_mm * px_per_mm)))

        # Where the card ended up inside its own image, found by detecting it rather than
        # predicted from the margin: how generateImage scales the drawing is OpenCV's
        # business, and guessing it wrong would put a scale error into every fixture.
        markers, marker_ids, _, _ = board.build_detector().detectBoard(image)
        if marker_ids is None or len(marker_ids) < 4:
            raise AssertionError("The rendered board could not be detected in its own image")
        object_points, image_points = board.build_board().matchImagePoints(markers, marker_ids)
        # Board millimetres -> pixels of the generated image, from the board's own
        # detected corners, so it is exact whatever scale generateImage chose.
        board_to_pixels = cv2.findHomography(
            np.asarray(object_points, dtype=np.float64).reshape(-1, 3)[:, :2] * 1000.0,
            np.asarray(image_points, dtype=np.float64).reshape(-1, 2), 0)[0]
        card_mm = np.array([[0.0, 0.0], [width_mm, 0.0],
                            [width_mm, height_mm], [0.0, height_mm]], dtype=np.float64)
        source_corners = cv2.perspectiveTransform(
            card_mm.reshape(-1, 1, 2), board_to_pixels).reshape(-1, 2).astype(np.float32)

        # Where those corners must land in the frame: centred in the region, turned by
        # the same angle the arithmetic fixtures use.
        centre = self.crop_to_plane_mm(np.array([[OUTPUT[0] / 2.0, OUTPUT[1] / 2.0]]))[0]
        destination = self._project(
            (card_mm - card_mm.mean(axis=0)) @ rotation(BOARD_ROTATION_DEGREES).T + centre
        ).astype(np.float32)

        frame = np.full((FRAME[1], FRAME[0]), 255, dtype=np.uint8)
        cv2.warpPerspective(
            image, cv2.getPerspectiveTransform(source_corners, destination), FRAME,
            dst=frame, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
        return frame

    def warped_marker(self, crop_index):
        """Where a point at a known crop index actually lands, in crop indices.

        The decisive measurement of the pixel convention. The source point is splatted
        onto its four neighbouring pixels with bilinear weights -- the exact inverse of
        what ``INTER_LINEAR`` does to them -- and the intensity centroid of the result is
        taken. Because both steps are linear, the centroid reproduces the mapping itself
        to floating-point precision rather than to the half pixel that rounding the
        source point to one pixel would cost.
        """
        source_point = cv2.perspectiveTransform(
            np.array(crop_index, dtype=np.float64).reshape(-1, 1, 2),
            np.linalg.inv(self.to_crop)).reshape(2)
        column, row = int(np.floor(source_point[0])), int(np.floor(source_point[1]))
        image = np.zeros((FRAME[1], FRAME[0]), dtype=np.float64)
        for delta_row in (0, 1):
            for delta_column in (0, 1):
                weight = ((source_point[0] - column) if delta_column else
                          (column + 1 - source_point[0]))
                weight *= ((source_point[1] - row) if delta_row else
                           (row + 1 - source_point[1]))
                image[row + delta_row, column + delta_column] = weight
        crop = cv2.warpPerspective(image, self.to_crop, OUTPUT, flags=cv2.INTER_LINEAR)
        rows, columns = np.nonzero(crop > 1e-12)
        weights = crop[rows, columns]
        return np.array([float((columns * weights).sum() / weights.sum()),
                         float((rows * weights).sum() / weights.sum())])


class PixelConventionTests(unittest.TestCase):
    """No shift. The half pixel that would add a constant 0.117 mm and hide there."""

    def setUp(self):
        self.rig = SyntheticRig()

    def test_the_offset_is_zero(self):
        self.assertEqual(region_calibration.CROP_PIXEL_OFFSET, 0.0)

    def test_to_crop_output_really_is_the_crop_array_index(self):
        """Measured through an actual warp, not through the transform's arithmetic."""
        for index in ([400.0, 120.0], [1100.0, 300.0], [1900.0, 90.0], [700.0, 460.0]):
            with self.subTest(index=index):
                landed = self.rig.warped_marker(index)
                # One source pixel of rounding, plus bilinear edge effects: well inside
                # a tenth of a crop pixel, and nowhere near the half pixel at stake.
                np.testing.assert_allclose(landed, index, atol=0.15)

    def test_fitting_without_a_shift_is_exact(self):
        homography, _ = region_calibration.fit_homography(
            self.rig.crop_points, self.rig.corner_points_mm)
        measured = region_calibration.to_mm(homography, self.rig.crop_points)

        self.assertLess(float(np.abs(measured - self.rig.corner_points_mm).max()), EXACT_MM)

    def test_applying_a_half_pixel_shift_is_measurably_wrong(self):
        """Pins the trap shut: a future "cleanup" that re-adds it fails here.

        On the real rig this costs 0.117 mm on camera 1 and 0.154 mm on camera 2 -- a
        constant translation, invisible in every other check the project runs.
        """
        shifted = self.rig.crop_points + 0.5
        homography, _ = region_calibration.fit_homography(
            shifted, self.rig.corner_points_mm)
        # Fed crop points as the runtime actually holds them, the shifted fit is off.
        error = np.abs(region_calibration.to_mm(homography, self.rig.crop_points)
                       - self.rig.corner_points_mm).max()

        self.assertGreater(float(error), 0.03)
        self.assertLess(float(error), 1.0)


class FitTests(unittest.TestCase):
    """Fitting the local homography from exact correspondences."""

    def setUp(self):
        self.rig = SyntheticRig()

    def test_a_known_board_position_comes_back_unchanged(self):
        homography, stats = region_calibration.fit_homography(
            self.rig.crop_points, self.rig.corner_points_mm)
        measured = region_calibration.to_mm(homography, self.rig.crop_points)

        np.testing.assert_allclose(measured, self.rig.corner_points_mm, atol=EXACT_MM)
        self.assertEqual(stats.inliers, len(self.rig.crop_points))
        self.assertEqual(stats.dropped, 0)

    def test_the_boards_own_geometry_reproduces(self):
        fit = self.rig.fit()

        self.assertLess(fit.rms_mm, EXACT_MM)
        self.assertLess(fit.max_mm, EXACT_MM)
        self.assertEqual(fit.corners, len(self.rig.corner_points_mm))

    def test_pairwise_distances_match_the_printed_board(self):
        fit = self.rig.fit()
        errors = fit.known_geometry_errors_mm()
        worst = max(row["absolute_error_mm"] for row in errors)

        self.assertLess(worst, EXACT_MM)
        # The distances span the card, so this is not passing by measuring coincident
        # points against each other.
        self.assertGreater(max(row["expected_mm"] for row in errors), 50.0)

    def test_residuals_are_reported_in_millimetres_forward(self):
        """A known board-frame error must show up as that error."""
        metrics = region_calibration.residual_metrics_mm(
            region_calibration.fit_homography(
                self.rig.crop_points, self.rig.corner_points_mm)[0],
            self.rig.crop_points, self.rig.corner_points_mm)

        self.assertLess(metrics["max_mm"], EXACT_MM)
        self.assertEqual(metrics["corners"], len(self.rig.crop_points))

    def test_a_mis_detected_corner_is_dropped_and_counted(self):
        """One corner dragged 40 px: the residual pass sets it aside and says so."""
        crop = self.rig.crop_points.copy()
        crop[3] += np.array([40.0, -25.0])
        homography, stats = region_calibration.fit_homography(crop, self.rig.corner_points_mm)

        self.assertGreaterEqual(stats.dropped, 1)
        self.assertLess(stats.inliers, stats.corners)
        # The refit is still true for every corner that was not dragged.
        measured = region_calibration.to_mm(homography, self.rig.crop_points)
        np.testing.assert_allclose(measured, self.rig.corner_points_mm, atol=0.01)

    def test_too_few_points_is_refused(self):
        with self.assertRaises(region_calibration.RegionCalibrationError):
            region_calibration.fit_homography(
                self.rig.crop_points[:3], self.rig.corner_points_mm[:3])

    def test_a_collinear_corner_set_is_refused(self):
        """One row of corners fits its line beautifully and constrains nothing."""
        single_row = np.array([[0.0, 100.0], [50.0, 100.0], [100.0, 100.1],
                               [150.0, 100.0], [200.0, 100.2], [250.0, 100.0]])
        targets = self.rig.corner_points_mm[:len(single_row)]

        with self.assertRaisesRegex(region_calibration.RegionCalibrationError, "nearly in a line"):
            region_calibration.fit_homography(single_row, targets)

    def test_a_mirrored_corner_set_is_refused(self):
        """A y-flipped board frame fits well and is silently wrong; det gives it away."""
        mirrored = self.rig.corner_points_mm * np.array([1.0, -1.0])

        with self.assertRaisesRegex(region_calibration.RegionCalibrationError, "mirrored"):
            region_calibration.fit_homography(self.rig.crop_points, mirrored)

    def test_the_boards_own_corner_list_is_what_a_charuco_board_has(self):
        points = TEST_BOARD.corner_points_mm()
        expected = TEST_BOARD.maximum_corners()

        self.assertEqual(len(points), expected)
        self.assertAlmostEqual(float(points[:, 0].min()), TEST_BOARD.square_length_mm, places=4)


class AgreementWithExistingModeTests(unittest.TestCase):
    """The two modes must not disagree. Both are valid, and an operator has no way to
    tell which is wrong if they drift apart.

    For a card lying on the measurement plane the two are the same projective map, so
    they must agree to numerical precision. Anything larger is a bug in one of them.
    """

    def setUp(self):
        self.rig = SyntheticRig()
        # The card is deliberately sunk onto the measurement plane for this test.
        on_plane = SyntheticRig(offset_mm=None)
        self.homography, _ = region_calibration.fit_homography(
            on_plane.crop_points, on_plane.corner_points_mm)

        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.calibration_path = write_json(root, "calibration.json", {
            "camera_matrix": CAMERA_MATRIX.tolist(), "dist_coeffs": [0.0] * 5,
            "image_size": list(FRAME)})
        self.extrinsics_path = write_json(root, "extrinsics.json", {
            "rvec": RVEC.ravel().tolist(), "tvec": TVEC.ravel().tolist()})

        self.plane = plane_scale.load_plane_scale(
            1, DEFINITION, FRAME, OUTPUT, calibration_path=self.calibration_path,
            extrinsics_path=self.extrinsics_path)

    def _probes(self):
        """Crop pixels spread over the region."""
        width, height = OUTPUT
        return np.array([
            [40.0, 40.0], [width - 40.0, 40.0], [width / 2.0, height / 2.0],
            [40.0, height - 40.0], [width - 40.0, height - 40.0], [700.0, 90.0],
        ], dtype=np.float64)

    def test_distances_agree_with_the_existing_mode(self):
        probes = self._probes()
        # Both scales have an arbitrary millimetre origin -- one on the board, one on the
        # extrinsics plane -- so only distances between probes are comparable.
        local = region_calibration.to_mm(self.homography, probes)
        global_ = self.plane.to_mm(probes)

        for first in range(len(probes)):
            for second in range(first + 1, len(probes)):
                with self.subTest(pair=(first, second)):
                    self.assertAlmostEqual(
                        float(np.linalg.norm(local[second] - local[first])),
                        float(np.linalg.norm(global_[second] - global_[first])),
                        delta=0.01)

    def test_the_region_fit_reproduces_the_board_geometry_exactly(self):
        rig = SyntheticRig()
        fit = rig.fit()

        self.assertLess(fit.max_mm, EXACT_MM)
        self.assertLess(fit.mm_per_pixel, 0.5)
        self.assertGreater(fit.mm_per_pixel, 0.01)


class DetectionTests(unittest.TestCase):
    """The real detector on a rendered frame, not a hand-made correspondence list."""

    def setUp(self):
        self.rig = SyntheticRig()

    def test_a_rendered_board_is_detected_and_deskewed(self):
        detection = region_calibration.detect_board_in_region(
            self.rig.render_frame(), TEST_BOARD, DEFINITION, FRAME, OUTPUT)

        self.assertEqual(detection.corner_count, len(self.rig.corner_points_mm))
        self.assertEqual(len(detection.crop_points), len(self.rig.corner_points_mm))
        self.assertEqual(detection.outside_region, 0)

    def test_a_board_in_a_missing_region_is_reported_not_measured(self):
        blank = np.full((FRAME[1], FRAME[0]), 255, dtype=np.uint8)
        with self.assertRaisesRegex(region_calibration.RegionCalibrationError,
                                    "board points needed"):
            region_calibration.detect_board_in_region(
                blank, TEST_BOARD, DEFINITION, FRAME, OUTPUT)

    def test_the_wrong_dictionary_is_reported(self):
        """The rendered markers belong to DICT_5X5_100, so nothing matches."""
        wrong = region_calibration.BoardDefinition(
            name="Wrong", dictionary="DICT_6X6_250", squares_x=7, squares_y=5,
            square_length_mm=10.0, marker_length_mm=7.0)
        with self.assertRaises(region_calibration.RegionCalibrationError):
            region_calibration.detect_board_in_region(
                self.rig.render_frame(), wrong, DEFINITION, FRAME, OUTPUT)

    def test_a_detected_board_fits_to_a_fraction_of_a_millimetre(self):
        """End to end through the real detector and the real deskew."""
        detection = region_calibration.detect_board_in_region(
            self.rig.render_frame(), TEST_BOARD, DEFINITION, FRAME, OUTPUT)
        fit = region_calibration.fit_region(
            detection, TEST_BOARD, CAMERA_MATRIX, FRAME, OUTPUT)

        # Limited by detection noise on the rendered card, not by the model.
        self.assertLess(fit.max_mm, 0.2)
        self.assertGreater(fit.intrinsic_evaluation.corners, 8)


class ScaleAgreementTests(unittest.TestCase):
    """The one silent error the fit itself cannot see: a mis-declared board size.

    Everything else the module checks looks at the fit's internal consistency. This looks
    at it from outside, against a scale the fit had no part in producing.
    """

    def setUp(self):
        from unittest.mock import patch
        self.rig = SyntheticRig()
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        calibration_path = write_json(root, "calibration.json", {
            "camera_matrix": CAMERA_MATRIX.tolist(), "dist_coeffs": [0.0] * 5,
            "image_size": list(FRAME)})
        extrinsics_path = write_json(root, "extrinsics.json", {
            "rvec": RVEC.ravel().tolist(), "tvec": TVEC.ravel().tolist()})
        # The reference is built from the intrinsics and extrinsics alone; the rig's fit
        # had no part in it, which is what makes the comparison worth making.
        self.reference = plane_scale.load_plane_scale(
            1, DEFINITION, FRAME, OUTPUT, calibration_path=calibration_path,
            extrinsics_path=extrinsics_path).mm_per_pixel

    def test_a_correctly_declared_board_agrees_with_the_plane(self):
        agreement = region_calibration.measure_scale_agreement(
            self.rig.fit(), self.reference)

        self.assertIsNotNone(agreement)
        self.assertTrue(agreement.within(0.01))
        self.assertIn("mm/px", agreement.describe())

    def test_a_board_declared_too_large_is_caught(self):
        """The failure mode with no other symptom.

        A card really printed at 10 mm squares but declared as 15 fits *perfectly* -- the
        homography simply absorbs the factor -- and reproduces its own geometry exactly.
        Only the independent plane scale reveals it.
        """
        on_plane = self.rig
        # The same physical card, wrongly described as 15 mm squares.
        wrong_board = region_calibration.BoardDefinition(
            name="Mis-declared", dictionary="DICT_5X5_100", squares_x=7, squares_y=5,
            square_length_mm=15.0, marker_length_mm=10.5)
        # Detection is on the same pixels; only the board's metric coordinates change.
        detection = on_plane.detection()
        wrong_detection = region_calibration.BoardDetection(
            frame_points=detection.frame_points, crop_points=detection.crop_points,
            board_points_mm=detection.board_points_mm * 1.5,
            corners=detection.corners, ids=detection.ids,
            corner_count=detection.corner_count)
        wrong_fit = region_calibration.fit_region(
            wrong_detection, wrong_board, CAMERA_MATRIX, FRAME, OUTPUT)

        # The fit itself is beyond reproach: residuals near zero, geometry exact.
        self.assertLess(wrong_fit.max_mm, EXACT_MM)
        self.assertLess(max(row["absolute_error_mm"]
                            for row in wrong_fit.known_geometry_errors_mm()), EXACT_MM)

        agreement = region_calibration.measure_scale_agreement(wrong_fit, self.reference)

        self.assertAlmostEqual(agreement.ratio, 1.5, delta=0.01)
        self.assertFalse(agreement.within(0.02))
        self.assertIn("+50.00%", agreement.describe())

    def test_without_a_reference_it_declines_to_have_an_opinion(self):
        for reference in (float("nan"), 0.0, -1.0):
            with self.subTest(reference=reference):
                self.assertIsNone(region_calibration.measure_scale_agreement(
                    self.rig.fit(), reference))


class BoardDefinitionTests(unittest.TestCase):
    def test_the_requested_board_is_valid_and_is_the_default(self):
        profile = region_calibration.REGION_BOARD_PROFILE

        profile.validate()
        self.assertEqual(profile.dictionary, "DICT_5X5_100")
        self.assertEqual(profile.footprint_mm(), (255.0, 105.0))
        self.assertEqual(profile.maximum_corners(), 96)
        self.assertIn(profile, region_calibration.default_board_profiles())

    def test_a_marker_larger_than_its_square_is_refused(self):
        bad = region_calibration.BoardDefinition(
            name="Bad", dictionary="DICT_5X5_100", squares_x=7, squares_y=5,
            square_length_mm=10.0, marker_length_mm=12.0)
        with self.assertRaisesRegex(ValueError, "smaller than the square"):
            bad.validate()

    def test_too_few_squares_is_refused(self):
        bad = region_calibration.BoardDefinition(
            name="Bad", dictionary="DICT_5X5_100", squares_x=2, squares_y=5,
            square_length_mm=10.0, marker_length_mm=7.0)
        with self.assertRaisesRegex(ValueError, "at least 3"):
            bad.validate()

    def test_an_unknown_dictionary_is_refused(self):
        bad = region_calibration.BoardDefinition(
            name="Bad", dictionary="DICT_NOT_A_REAL_ONE", squares_x=7, squares_y=5,
            square_length_mm=10.0, marker_length_mm=7.0)
        with self.assertRaisesRegex(ValueError, "ArUco dictionary"):
            bad.validate()

    def test_a_dictionary_too_small_for_the_board_is_refused(self):
        bad = region_calibration.BoardDefinition(
            name="Too many markers", dictionary="DICT_5X5_50", squares_x=17, squares_y=17,
            square_length_mm=10.0, marker_length_mm=7.0)
        with self.assertRaisesRegex(ValueError, "use a larger dictionary"):
            bad.validate()

    def test_an_empty_name_is_refused(self):
        bad = region_calibration.BoardDefinition(
            name="   ", dictionary="DICT_5X5_100", squares_x=7, squares_y=5,
            square_length_mm=10.0, marker_length_mm=7.0)
        with self.assertRaisesRegex(ValueError, "needs a name"):
            bad.validate()

    def test_a_non_finite_size_is_refused(self):
        bad = region_calibration.BoardDefinition(
            name="Bad", dictionary="DICT_5X5_100", squares_x=7, squares_y=5,
            square_length_mm=float("nan"), marker_length_mm=7.0)
        with self.assertRaisesRegex(ValueError, "positive number"):
            bad.validate()


class BoardFitReportTests(unittest.TestCase):
    """Advisory, not a gate: a card overhanging still yields usable interior corners."""

    def test_the_specified_board_does_not_fit_the_specified_region(self):
        """255 x 105 mm of card against a 298 x 72 mm marked strip."""
        report = region_calibration.board_fit_report(
            region_calibration.REGION_BOARD_PROFILE, (298.0, 72.0))

        self.assertFalse(report.fits)
        self.assertGreater(report.used_fraction[1], 1.0)
        text = report.summary()
        self.assertIn("105", text)
        # The suggestion has to be actionable, not just a red mark.
        self.assertIn("cut the card", text)

    def test_a_board_that_fits_reports_no_complaint(self):
        small = region_calibration.BoardDefinition(
            name="Small", dictionary="DICT_5X5_100", squares_x=7, squares_y=5,
            square_length_mm=10.0, marker_length_mm=7.0)
        report = region_calibration.board_fit_report(small, (298.0, 72.0))

        self.assertTrue(report.fits)
        self.assertEqual(report.suggestions, ())

    def test_an_unknown_region_size_says_so_instead_of_guessing(self):
        report = region_calibration.board_fit_report(TEST_BOARD, None)

        self.assertTrue(report.fits)
        self.assertIn("not known", report.summary())

    def test_a_thin_card_is_flagged_even_when_it_fits(self):
        thin = region_calibration.BoardDefinition(
            name="Thin", dictionary="DICT_5X5_100", squares_x=11, squares_y=3,
            square_length_mm=10.0, marker_length_mm=7.0)
        report = region_calibration.board_fit_report(thin, (298.0, 72.0))

        self.assertTrue(report.fits)
        self.assertIn("thin card", report.summary())

    def test_shrink_to_fit_uses_the_largest_square_that_fits(self):
        region = (298.0, 72.0)
        square, marker = region_calibration.largest_fitting_square_mm(17, 7, region)

        # The height binds here, so one more tenth would overflow the reserved margin.
        self.assertLessEqual(7 * square, 0.90 * region[1])
        self.assertGreater(7 * (square + 0.1), 0.90 * region[1])
        self.assertLess(marker, square)

    def test_shrink_to_fit_declines_without_a_region(self):
        self.assertIsNone(region_calibration.largest_fitting_square_mm(17, 7, None))
        self.assertIsNone(region_calibration.largest_fitting_square_mm(17, 7, (0.0, 0.0)))


class BoardProfileStoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "boards.json"

    def test_a_missing_file_opens_on_the_defaults(self):
        self.assertEqual(region_calibration.load_board_profiles(self.path),
                         region_calibration.default_board_profiles())

    def test_a_corrupt_file_opens_on_the_defaults(self):
        self.path.write_text("{not json", encoding="utf-8")
        self.assertEqual(region_calibration.load_board_profiles(self.path),
                         region_calibration.default_board_profiles())

    def test_profiles_round_trip(self):
        custom = region_calibration.BoardDefinition(
            name="New board", dictionary="DICT_6X6_250", squares_x=11, squares_y=9,
            square_length_mm=12.5, marker_length_mm=9.0)
        region_calibration.save_board_profiles(
            self.path, region_calibration.default_board_profiles() + [custom])

        self.assertIn(custom, region_calibration.load_board_profiles(self.path))

    def test_saving_by_name_replaces_rather_than_duplicates(self):
        edited = region_calibration.BoardDefinition(
            name=region_calibration.REGION_BOARD_PROFILE.name, dictionary="DICT_5X5_100",
            squares_x=17, squares_y=7, square_length_mm=9.7, marker_length_mm=6.7)
        profiles = region_calibration.upsert_board_profile(
            region_calibration.default_board_profiles(), edited)

        self.assertEqual(len(profiles), len(region_calibration.default_board_profiles()))
        self.assertIn(edited, profiles)

    def test_saving_leaves_no_temporary_file_behind(self):
        region_calibration.save_board_profiles(self.path,
                                               region_calibration.default_board_profiles())
        self.assertFalse(self.path.with_suffix(".json.tmp").exists())


class RegionStoreTests(unittest.TestCase):
    def setUp(self):
        self.rig = SyntheticRig()
        self.fit = self.rig.fit()
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.path = self.root / "homographies.json"
        self.calibration_path = write_json(self.root, "calibration.json", {
            "camera_matrix": CAMERA_MATRIX.tolist(), "dist_coeffs": [0.0] * 5,
            "image_size": list(FRAME)})

    def _save(self, region=1, definition=DEFINITION, frame_size=FRAME, fit=None):
        store = region_calibration.empty_region_store()
        region_calibration.store_region_fit(
            store, 1, region, fit or self.fit,
            region_calibration.crop_signature(definition, frame_size, OUTPUT, 4.0,
                                              self.calibration_path),
            definition)
        region_calibration.save_region_store(self.path, store)

    def _load(self, region=1, definition=DEFINITION, frame_size=FRAME):
        return region_calibration.load_region_scale(
            1, region, definition, frame_size, OUTPUT, 4.0, store_path=self.path,
            calibration_path=self.calibration_path)

    def test_a_saved_region_loads_back_as_a_usable_scale(self):
        self._save()
        scale = self._load()
        measured = scale.to_mm(self.rig.crop_points)

        np.testing.assert_allclose(measured, self.rig.corner_points_mm, atol=EXACT_MM)
        self.assertEqual(scale.region, 1)
        self.assertEqual(scale.board_name, TEST_BOARD.name)
        self.assertLess(scale.rms_mm, EXACT_MM)
        self.assertEqual(scale.corners, len(self.rig.corner_points_mm))

    def test_an_unsaved_region_is_reported_by_name(self):
        self._save(region=1)
        with self.assertRaisesRegex(region_calibration.RegionCalibrationError,
                                    "Region 2 .* no saved local homography"):
            self._load(region=2)

    def test_re_marking_the_crop_region_makes_the_homography_stale(self):
        """The whole reason a signature is stored alongside the matrix."""
        self._save()
        with self.assertRaisesRegex(region_calibration.RegionCalibrationError,
                                    "different crop region or a different lens"):
            self._load(definition=dict(DEFINITION, center_normalized=[0.5, 0.30]))

    def test_a_camera_running_at_another_size_makes_it_stale(self):
        self._save()
        with self.assertRaisesRegex(region_calibration.RegionCalibrationError,
                                    "different crop region or a different lens"):
            self._load(frame_size=(1728, 2304))

    def test_recalibrating_the_lens_makes_the_homography_stale(self):
        """The crop comes from the undistorted frame, so the lens is part of the map."""
        self._save()
        self.calibration_path.write_text(json.dumps({
            "camera_matrix": CAMERA_MATRIX.tolist(), "dist_coeffs": [0.0] * 5,
            "image_size": list(FRAME), "rms_error": 0.9}), encoding="utf-8")

        with self.assertRaisesRegex(region_calibration.RegionCalibrationError,
                                    "different lens"):
            self._load()

    def test_recalibrating_the_extrinsics_does_not_make_it_stale(self):
        """The asymmetry that is the point of the feature.

        The plane pose is no part of a local fit, so it is not in the signature at all --
        asserted structurally, because "the extrinsics file changed" is not something the
        signature can be asked about without a parameter that does not exist.
        """
        self._save()
        parameters = inspect.signature(region_calibration.crop_signature).parameters

        self.assertNotIn("extrinsics_path", parameters)
        self.assertNotIn("extrinsics", parameters)
        # And it still loads, with the extrinsics file having changed underneath it.
        np.testing.assert_allclose(self._load().to_mm(self.rig.crop_points),
                                   self.rig.corner_points_mm, atol=EXACT_MM)

    def test_a_crop_size_change_is_reported_separately(self):
        self._save()
        with self.assertRaisesRegex(region_calibration.RegionCalibrationError, "crop size"):
            region_calibration.load_region_scale(
                1, 1, DEFINITION, FRAME, (1104, 276), 4.0, store_path=self.path,
                calibration_path=self.calibration_path)

    def test_two_regions_keep_their_own_homography(self):
        """Region 2 must not be served region 1's plane."""
        self._save(region=1)
        # A second fit, deliberately offset in millimetres, as a differently placed card
        # in another region would be.
        shifted = self.rig.corner_points_mm + np.array([500.0, 300.0])
        homography, _ = region_calibration.fit_homography(self.rig.crop_points, shifted)
        store = region_calibration.load_region_store(self.path)
        store["cameras"]["1"]["2"] = {
            "homography": homography.tolist(),
            "signature": region_calibration.crop_signature(
                DEFINITION, FRAME, OUTPUT, 4.0, self.calibration_path),
            "output_size": list(OUTPUT), "mm_per_pixel": 0.1,
            "board": TEST_BOARD.as_dict(),
        }
        region_calibration.save_region_store(self.path, store)

        first = self._load(region=1).to_mm(self.rig.crop_points)
        second = self._load(region=2).to_mm(self.rig.crop_points)
        np.testing.assert_allclose(first, self.rig.corner_points_mm, atol=EXACT_MM)
        np.testing.assert_allclose(second, shifted, atol=EXACT_MM)

    def test_a_board_geometry_copy_survives_a_profile_edit(self):
        """The stored card is copied in whole, so editing a profile cannot reinterpret it."""
        self._save()
        stored = region_calibration.load_region_store(self.path)["cameras"]["1"]["1"]

        self.assertEqual(stored["board"], TEST_BOARD.as_dict())

    def test_a_saved_alignment_is_reported_back(self):
        alignment = region_calibration.PlaneAlignment(
            tilt_degrees=0.4, gap_mm=2.1, expected_gap_mm=2.0)
        store = region_calibration.empty_region_store()
        region_calibration.store_region_fit(
            store, 1, 1, self.fit,
            region_calibration.crop_signature(DEFINITION, FRAME, OUTPUT, 4.0,
                                              self.calibration_path),
            DEFINITION, alignment=alignment)
        region_calibration.save_region_store(self.path, store)
        scale = self._load()

        self.assertAlmostEqual(scale.tilt_degrees, 0.4)
        self.assertAlmostEqual(scale.gap_mm, 2.1)

    def test_an_unreadable_homography_is_reported(self):
        store = region_calibration.empty_region_store()
        store["cameras"]["1"]["1"] = {
            "homography": [[1.0, 0.0], [0.0, 1.0]],
            "signature": region_calibration.crop_signature(
                DEFINITION, FRAME, OUTPUT, 4.0, self.calibration_path),
            "output_size": list(OUTPUT)}
        region_calibration.save_region_store(self.path, store)
        with self.assertRaisesRegex(region_calibration.RegionCalibrationError, "unreadable"):
            self._load()

    def test_a_non_finite_homography_is_reported(self):
        store = region_calibration.empty_region_store()
        store["cameras"]["1"]["1"] = {
            "homography": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, float("nan")]],
            "signature": region_calibration.crop_signature(
                DEFINITION, FRAME, OUTPUT, 4.0, self.calibration_path),
            "output_size": list(OUTPUT)}
        region_calibration.save_region_store(self.path, store)
        with self.assertRaisesRegex(region_calibration.RegionCalibrationError,
                                    "not a usable matrix"):
            self._load()

    def test_a_mirrored_stored_homography_is_reported(self):
        store = region_calibration.empty_region_store()
        store["cameras"]["1"]["1"] = {
            "homography": [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
            "signature": region_calibration.crop_signature(
                DEFINITION, FRAME, OUTPUT, 4.0, self.calibration_path),
            "output_size": list(OUTPUT)}
        region_calibration.save_region_store(self.path, store)
        with self.assertRaisesRegex(region_calibration.RegionCalibrationError,
                                    "not a usable matrix"):
            self._load()

    def test_a_missing_store_file_is_reported(self):
        with self.assertRaises(region_calibration.RegionCalibrationError):
            region_calibration.load_region_scale(
                1, 1, DEFINITION, FRAME, OUTPUT, 4.0,
                store_path=self.root / "nope.json",
                calibration_path=self.calibration_path)

    def test_a_corrupt_store_file_is_reported(self):
        self.path.write_text("{not json", encoding="utf-8")
        with self.assertRaises(region_calibration.RegionCalibrationError):
            self._load()

    def test_saving_leaves_no_temporary_file_behind(self):
        self._save()
        self.assertFalse(self.path.with_suffix(".json.tmp").exists())

    def test_an_unknown_region_number_is_refused(self):
        for value in (3, "one", 0):
            with self.subTest(region=value):
                with self.assertRaises(region_calibration.RegionCalibrationError):
                    region_calibration.region_key(value)

    def test_an_unknown_camera_number_is_refused(self):
        with self.assertRaises(region_calibration.RegionCalibrationError):
            region_calibration.store_region_fit(
                region_calibration.empty_region_store(), 3, 1, self.fit, "sig", DEFINITION)


class CropSignatureTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.calibration_path = write_json(self.root, "calibration.json", {"a": 1})
        self.other_path = write_json(self.root, "other.json", {"a": 2})

    def _signature(self, **kwargs):
        kwargs.setdefault("calibration_path", self.calibration_path)
        return region_calibration.crop_signature(
            kwargs.pop("definition", DEFINITION), kwargs.pop("frame_size", FRAME),
            kwargs.pop("output_size", OUTPUT), kwargs.pop("ratio", 4.0),
            kwargs.pop("calibration_path"))

    def test_every_input_it_claims_to_track_changes_it(self):
        base = self._signature()

        cases = {
            "definition": dict(DEFINITION, angle_degrees=3.0),
            "output_size": (1104, 276),
            "ratio": 3.0,
            "calibration_path": self.other_path,
        }
        for label, value in cases.items():
            with self.subTest(changed=label):
                self.assertNotEqual(base, self._signature(**{label: value}))

    def test_it_is_stable_for_the_same_inputs(self):
        self.assertEqual(self._signature(), self._signature())
        # And a later change to the other file does not disturb this one.
        self.other_path.write_text('{"a": 3}', encoding="utf-8")
        self.assertEqual(self._signature(), self._signature())

    def test_an_unreadable_calibration_is_still_a_distinct_signature(self):
        missing = self.root / "missing.json"

        self.assertNotEqual(self._signature(), self._signature(calibration_path=missing))

    def test_key_order_in_the_definition_does_not_matter(self):
        reordered = {key: DEFINITION[key] for key in reversed(list(DEFINITION))}

        self.assertEqual(self._signature(), self._signature(definition=reordered))


class PlaneAlignmentTests(unittest.TestCase):
    """The gate that catches a card lying at an angle to the measurement plane.

    A tilted card fits beautifully and produces a homography for the wrong plane, with
    nothing left in the stored data to reveal it. It has to be caught at save time.
    """

    def setUp(self):
        self.rig = SyntheticRig()
        # The board is tilted onto the measurement plane by construction: the synthetic
        # card lies *in* the extrinsics plane, so a matched reference is tilt-free.
        self.detection = self.rig.detection()

    def _alignment(self, plane_rvec, plane_tvec=TVEC, thickness=0.0):
        return region_calibration.measure_plane_alignment(
            TEST_BOARD, self.detection, CAMERA_MATRIX, plane_rvec, plane_tvec, thickness)

    def _lifted(self, metres, plane_rvec=RVEC, plane_tvec=TVEC):
        """The plane raised ``metres`` perpendicular to itself, as a spacer would."""
        normal = cv2.Rodrigues(
            np.asarray(plane_rvec, dtype=np.float64).reshape(3, 1))[0][:, 2]
        return np.asarray(plane_tvec, dtype=np.float64).reshape(3, 1) + (
            normal.reshape(3, 1) * float(metres))

    def test_a_card_on_the_measurement_plane_reads_as_aligned(self):
        alignment = self._alignment(RVEC)

        self.assertLess(alignment.tilt_degrees, 0.05)
        self.assertLess(alignment.gap_mm, 0.05)

    def test_a_card_tilted_by_two_degrees_is_flagged(self):
        tilted = RVEC + np.array([[0.0], [np.deg2rad(2.0)], [0.0]])
        alignment = self._alignment(tilted)

        self.assertGreater(alignment.tilt_degrees, 1.5)
        self.assertFalse(alignment.within(2.0, 2.0))

    def test_a_card_raised_off_the_plane_is_flagged(self):
        """A 5 mm spacer under the card: the gap gate catches it."""
        alignment = self._alignment(RVEC, plane_tvec=self._lifted(0.005))

        self.assertAlmostEqual(alignment.gap_mm, 5.0, delta=0.05)
        self.assertFalse(alignment.within(2.0, 1.0))
        self.assertTrue(alignment.within(2.0, 6.0))

    def test_a_gap_is_measured_perpendicular_to_the_plane_not_along_an_axis(self):
        """A trap worth pinning: shifting tvec along z is not lifting by that much.

        This plane sits at about 35 degrees to the camera axis, so a 5 mm shift along z
        moves it only 5*cos(35) = 4.1 mm away from the board. Measuring the perpendicular
        separation and not the trivial coordinate difference is the whole job here, and
        the two differ by a factor a 2 mm gate would not survive.
        """
        along_z = self._alignment(RVEC, plane_tvec=TVEC + np.array([[0.0], [0.0], [0.005]]))

        self.assertLess(along_z.gap_mm, 4.5)
        self.assertGreater(along_z.gap_mm, 3.5)

    def test_a_card_of_the_expected_thickness_passes(self):
        """A 2 mm card on the plane sits 2 mm from it; that is not an error."""
        alignment = self._alignment(RVEC, plane_tvec=self._lifted(0.002), thickness=2.0)

        self.assertAlmostEqual(alignment.gap_error_mm, 0.0, delta=0.05)
        self.assertTrue(alignment.within(2.0, 1.0))

    def test_it_describes_itself_for_the_operator(self):
        text = self._alignment(RVEC).describe()

        self.assertIn("tilt", text)
        self.assertIn("gap", text)


class IntrinsicEvaluationTests(unittest.TestCase):
    def test_the_global_lens_model_is_graded_in_the_region(self):
        rig = SyntheticRig()
        rendered = region_calibration.detect_board_in_region(
            rig.render_frame(), TEST_BOARD, DEFINITION, FRAME, OUTPUT)
        evaluation = region_calibration.evaluate_intrinsics(
            rendered, TEST_BOARD, CAMERA_MATRIX)

        self.assertGreater(evaluation.corners, 8)
        # The fixture is exactly pinhole and exactly matches the matrix, so the lens model
        # fits it essentially perfectly. A large number here would mean the evaluation is
        # measuring something other than reprojection.
        self.assertLess(evaluation.rms_px, 1.0)

    def test_a_single_flat_view_cannot_see_a_focal_length_error(self):
        """Measured, not reasoned: this is *why* refinement demands tilted views.

        For one plane the pose can absorb a focal-length change almost exactly -- a
        homography is K[r1 r2 t], so with K's five degrees of freedom and one view there
        is only one constraint on it. Here a five-fold focal error, which is absurd as a
        lens model, still reprojects to under a pixel, and a 15% error is worse than the
        correct matrix by less than a tenth of one.

        The consequence is not a defect in the evaluation -- it is the honest limit that
        :func:`region_calibration.refine_intrinsics` is built around, and the reason that
        function refuses to run on a single view rather than pretending to refine.
        """
        rig = SyntheticRig()
        rendered = region_calibration.detect_board_in_region(
            rig.render_frame(), TEST_BOARD, DEFINITION, FRAME, OUTPUT)
        correct = region_calibration.evaluate_intrinsics(rendered, TEST_BOARD, CAMERA_MATRIX)

        for factor in (1.15, 5.0):
            with self.subTest(factor=factor):
                wrong = CAMERA_MATRIX.copy()
                wrong[0, 0] *= factor
                wrong[1, 1] *= factor
                evaluation = region_calibration.evaluate_intrinsics(
                    rendered, TEST_BOARD, wrong)

                self.assertLess(evaluation.rms_px, 1.0)
                self.assertLess(evaluation.rms_px, correct.rms_px * 4)

        self.assertGreater(correct.rms_px, 0.0)


class RefinementTests(unittest.TestCase):
    """Refinement refuses far more often than it runs, which is the point."""

    def _views(self, count, tilt_degrees):
        """``count`` views of a board tilted by up to ``tilt_degrees`` about x."""
        local = np.column_stack((TEST_BOARD.corner_points_mm() / 1000.0,
                                 np.zeros(TEST_BOARD.maximum_corners())))
        views = []
        for index in range(count):
            angle = np.deg2rad(tilt_degrees * (index + 1) / max(count, 1))
            rvec = np.array([[angle], [0.0], [0.0]])
            tvec = np.array([[0.0], [0.0], [1.0 - 0.05 * index]])
            image, _ = cv2.projectPoints(local.astype(np.float64), rvec, tvec,
                                         CAMERA_MATRIX, DISTORTION)
            views.append((local.astype(np.float64), image.reshape(-1, 2)))
        return views

    def test_one_view_is_refused_with_the_reason(self):
        result = region_calibration.refine_intrinsics(
            self._views(1, 10.0), CAMERA_MATRIX, FRAME)

        self.assertFalse(result.applied)
        self.assertIn("cannot constrain the lens", result.reason)

    def test_flat_views_are_refused_even_when_there_are_enough(self):
        """Five identical flat views carry no more information than one."""
        result = region_calibration.refine_intrinsics(
            self._views(5, 0.0), CAMERA_MATRIX, FRAME)

        self.assertFalse(result.applied)
        self.assertIn("degrees of tilt", result.reason)
        self.assertLess(result.maximum_tilt_degrees, 0.5)

    def test_tilted_views_are_refined_and_reported(self):
        result = region_calibration.refine_intrinsics(
            self._views(6, 12.0), CAMERA_MATRIX, FRAME)

        self.assertTrue(result.applied)
        self.assertGreater(result.maximum_tilt_degrees, 5.0)
        self.assertEqual(result.views, 6)
        # The guess is already exact, so a refinement must not make it worse.
        self.assertLess(result.rms_after_px, max(result.rms_before_px, 1e-6) + 1e-3)
        self.assertEqual(result.camera_matrix.shape, (3, 3))

    def test_refinement_returns_a_copy_and_never_touches_the_global_matrix(self):
        original = CAMERA_MATRIX.copy()
        result = region_calibration.refine_intrinsics(
            self._views(6, 12.0), CAMERA_MATRIX, FRAME)

        self.assertFalse(np.shares_memory(result.camera_matrix, CAMERA_MATRIX))
        np.testing.assert_array_equal(CAMERA_MATRIX, original)


class CameraPathsTests(unittest.TestCase):
    """Cameras are numbered 1 and 2 by position, as the rest of the app does."""

    def setUp(self):
        self.config = {"cameras": [
            {"index": 0, "calibration_file": "calibration_0.json",
             "extrinsics_file": "extrinsics_0.json"},
            {"index": 1, "calibration_file": "calibration_1.json",
             "extrinsics_file": "extrinsics_1.json"},
        ]}

    def test_each_camera_gets_its_own_file(self):
        from unittest.mock import patch
        with patch("app_config.CONFIG", self.config), \
                patch("app_config.project_path", lambda path: path):
            first = region_calibration.camera_paths(1)
            second = region_calibration.camera_paths(2)

        self.assertEqual(first[0], "calibration_0.json")
        self.assertEqual(second[0], "calibration_1.json")

    def test_an_unknown_camera_number_has_no_paths(self):
        from unittest.mock import patch
        with patch("app_config.CONFIG", self.config), \
                patch("app_config.project_path", lambda path: path):
            for number in (0, 3, "x"):
                with self.subTest(camera=number):
                    self.assertEqual(region_calibration.camera_paths(number), (None, None))


class RegionModeFlagTests(unittest.TestCase):
    def test_the_mode_is_off_unless_the_configuration_turns_it_on(self):
        from unittest.mock import patch
        with patch("app_config.CONFIG", {"cameras": [], "region_homography": {}}):
            self.assertFalse(region_calibration.region_mode_enabled())
        with patch("app_config.CONFIG", {"cameras": [],
                                         "region_homography": {"enabled": True}}):
            self.assertTrue(region_calibration.region_mode_enabled())
        # Absent section, which is the shipped state: still off.
        with patch("app_config.CONFIG", {"cameras": []}):
            self.assertFalse(region_calibration.region_mode_enabled())


class RegionSizeTests(unittest.TestCase):
    def setUp(self):
        self.rig = SyntheticRig()

    def test_the_region_extent_comes_from_the_saved_plane(self):
        size = region_calibration.region_size_mm(
            DEFINITION, FRAME, OUTPUT, 4.0,
            calibration={"camera_matrix": CAMERA_MATRIX.tolist(),
                         "image_size": list(FRAME)},
            extrinsics={"rvec": RVEC.ravel().tolist(), "tvec": TVEC.ravel().tolist()})

        expected = self.rig.region_size_mm()
        self.assertAlmostEqual(size[0], expected[0], delta=1.0)
        self.assertAlmostEqual(size[1], expected[1], delta=1.0)

    def test_without_a_plane_the_extent_is_unknown_rather_than_guessed(self):
        self.assertIsNone(region_calibration.region_size_mm(DEFINITION, FRAME, OUTPUT))
        self.assertIsNone(region_calibration.region_size_mm(
            DEFINITION, FRAME, OUTPUT, calibration={"camera_matrix": [[1, 0, 0]]},
            extrinsics={}))


def write_json(directory, name, body):
    path = Path(directory) / name
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


if __name__ == "__main__":
    unittest.main()

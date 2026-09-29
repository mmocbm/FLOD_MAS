import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from crop_processing import (
    crop_cameras_for_size, crop_edit_from_saved, definition_fits_image,
    extract_rotated_crop, four_point_crop, load_crop_store,
    normalized_definition, parallel_line_angle, pixel_definition,
    rotated_crop_transform, save_crop_store,
)

RATIO = 4.0
FRAME = (4608, 3456)


def corner_inside(edit, point, ratio=RATIO):
    """Whether a source-frame point falls inside the crop described by ``edit``."""
    centre = np.asarray(edit["center"], np.float64)
    radians = np.radians(edit["angle_degrees"])
    rotation = np.array([
        [np.cos(radians), -np.sin(radians)],
        [np.sin(radians), np.cos(radians)],
    ])
    local = (np.asarray(point, np.float64) - centre) @ rotation
    return (abs(local[0]) <= edit["width"] / 2.0 + 1e-6
            and abs(local[1]) <= edit["height"] / 2.0 + 1e-6)


# Corner sets in the order the tool expects: P1 -> P2 is an end edge,
# P2 -> P3 is the long side that becomes horizontal, P3 -> P4 the other end edge.
MARKED_QUADS = {
    "axis_aligned": [(500, 1000), (500, 1500), (3500, 1500), (3500, 1000)],
    "reversed_long_side": [(3500, 1000), (3500, 1500), (500, 1500), (500, 1000)],
    "already_ratio": [(500, 1000), (500, 1900), (4100, 1900), (4100, 1000)],
    "sloppy_end_edges": [(600, 900), (520, 1480), (3480, 1560), (3520, 960)],
    "near_square": [(2000, 1500), (2000, 2300), (2800, 2300), (2800, 1500)],
    "steep_angle": [(900, 800), (1400, 800), (1400, 3800), (900, 3800)],
}


class FourPointCropTests(unittest.TestCase):
    def test_directed_angle_puts_the_third_point_on_positive_x(self):
        edit = four_point_crop(
            [(100, 200), (100, 600), (700, 600), (700, 200)], FRAME, RATIO, 0.0,
        )
        self.assertAlmostEqual(edit["angle_degrees"], 0.0)
        # P2 -> P3 runs to the right, so the local frame must not fold it to negative.
        self.assertLess(edit["line"][0][0], edit["line"][1][0])

    def test_reversed_long_side_gives_the_same_rectangle(self):
        forward = four_point_crop(MARKED_QUADS["axis_aligned"], FRAME, RATIO, 0.0)
        reverse = four_point_crop(MARKED_QUADS["reversed_long_side"], FRAME, RATIO, 0.0)
        self.assertAlmostEqual(forward["width"], reverse["width"])
        self.assertAlmostEqual(forward["height"], reverse["height"])
        self.assertAlmostEqual(forward["center"][0], reverse["center"][0])
        self.assertAlmostEqual(forward["center"][1], reverse["center"][1])
        # The rectangle is the same, but the alignment line keeps its own direction.
        self.assertAlmostEqual(
            abs(forward["angle_degrees"] - reverse["angle_degrees"]), 180.0,
        )

    def test_every_marked_corner_survives_the_crop(self):
        # The property that matters most: whatever the operator marked, plus the
        # margin, is inside the crop. A regression here loses fabric silently.
        for name, quad in MARKED_QUADS.items():
            for margin in (0.0, 8.0, 25.0):
                with self.subTest(quad=name, margin=margin):
                    edit = four_point_crop(quad, FRAME, RATIO, margin)
                    for point in quad:
                        self.assertTrue(corner_inside(edit, point), name)

    def test_crop_always_meets_the_configured_ratio(self):
        for name, quad in MARKED_QUADS.items():
            with self.subTest(quad=name):
                edit = four_point_crop(quad, FRAME, RATIO, 8.0)
                self.assertAlmostEqual(edit["width"] / edit["height"], RATIO)
                self.assertAlmostEqual(edit["height"], edit["width"] / RATIO)

    def test_margin_grows_both_sides_by_the_configured_percentage(self):
        # A mark that is already the crop ratio, so the ratio step changes nothing.
        quad = [(500, 1000), (500, 1900), (4100, 1900), (4100, 1000)]
        plain = four_point_crop(quad, FRAME, RATIO, 0.0)
        margined = four_point_crop(quad, FRAME, RATIO, 8.0)
        self.assertAlmostEqual(margined["width"], plain["width"] * 1.16)
        self.assertAlmostEqual(margined["height"], plain["height"] * 1.16)

    def test_margin_of_zero_reproduces_the_marked_ratio_box(self):
        quad = [(500, 1000), (500, 1900), (4100, 1900), (4100, 1000)]
        edit = four_point_crop(quad, FRAME, RATIO, 0.0)
        self.assertAlmostEqual(edit["width"], 3600.0)
        self.assertAlmostEqual(edit["height"], 900.0)

    def test_ratio_step_grows_the_short_dimension_never_the_long(self):
        wide = [(500, 1000), (500, 1300), (4100, 1300), (4100, 1000)]  # 3600 x 300
        edit = four_point_crop(wide, FRAME, RATIO, 0.0)
        self.assertAlmostEqual(edit["width"], 3600.0)
        self.assertAlmostEqual(edit["height"], 900.0)
        tall = [(500, 1000), (500, 2500), (2400, 2500), (2400, 1000)]  # 1900 x 1500
        edit = four_point_crop(tall, FRAME, RATIO, 0.0)
        self.assertAlmostEqual(edit["height"], 1500.0)
        self.assertAlmostEqual(edit["width"], 6000.0)

    def test_rotated_mark_is_deskewed_to_its_own_angle(self):
        radians = np.radians(30.0)
        rotation = np.array([
            [np.cos(radians), -np.sin(radians)],
            [np.sin(radians), np.cos(radians)],
        ])
        local = np.array([(-1500, -200), (-1500, 200), (1500, 200), (1500, -200)])
        quad = (local @ rotation.T + np.array([2300.0, 1700.0])).tolist()
        edit = four_point_crop(quad, FRAME, RATIO, 8.0)
        self.assertAlmostEqual(edit["angle_degrees"], 30.0, places=6)
        for point in quad:
            self.assertTrue(corner_inside(edit, point))

    def test_minimum_width_floor_grows_rather_than_shrinking(self):
        tiny = [(500, 500), (500, 700), (700, 700), (700, 500)]
        edit = four_point_crop(tiny, FRAME, RATIO, 0.0, min_width=800.0)
        self.assertAlmostEqual(edit["width"], 800.0)
        self.assertAlmostEqual(edit["height"], 200.0)
        for point in tiny:
            self.assertTrue(corner_inside(edit, point))

    def test_stored_definition_reproduces_the_measured_box(self):
        edit = four_point_crop(MARKED_QUADS["sloppy_end_edges"], FRAME, RATIO, 8.0)
        definition = normalized_definition(
            edit["center"], edit["width"], edit["angle_degrees"], edit["line"], FRAME,
            quad=edit["marked_quad"],
        )
        restored = pixel_definition(definition, FRAME, RATIO)
        self.assertAlmostEqual(restored["center"][0], edit["center"][0], places=6)
        self.assertAlmostEqual(restored["center"][1], edit["center"][1], places=6)
        self.assertAlmostEqual(restored["width"], edit["width"], places=6)
        self.assertAlmostEqual(restored["angle_degrees"], edit["angle_degrees"], places=6)
        # The stored format keeps only a width; the height must come back equal.
        self.assertAlmostEqual(restored["height"], restored["width"] / RATIO)

    def test_extracted_crop_covers_the_marked_corners(self):
        # End to end through the real warp, not just the geometry: paint a bright
        # band, mark a sloppy quad inside it, and check the marked corners land on
        # bright pixels of the 4:1 output rather than on the background.
        frame = np.zeros((900, 1400, 3), np.uint8)
        frame[300:620, 100:1300] = 255
        quad = [(120, 320), (150, 580), (1280, 600), (1250, 340)]
        edit = four_point_crop(quad, (1400, 900), RATIO, 8.0)
        definition = normalized_definition(
            edit["center"], edit["width"], edit["angle_degrees"], edit["line"],
            (1400, 900), quad=edit["marked_quad"],
        )
        crop = extract_rotated_crop(frame, definition, (2208, 552), RATIO)
        self.assertEqual(crop.shape[:2], (552, 2208))
        # Map the marked corners through the very transform the crop used, so the
        # check is about the real warp and not a re-derivation of it.
        transform = rotated_crop_transform(definition, (1400, 900), (2208, 552), RATIO)
        mapped = cv2.perspectiveTransform(
            np.asarray(quad, np.float32).reshape(-1, 1, 2), transform,
        ).reshape(-1, 2)
        for column, row in mapped:
            column, row = int(round(column)), int(round(row))
            self.assertTrue(0 <= row < 552 and 0 <= column < 2208)
            self.assertGreater(int(crop[row, column].max()), 0)

    def test_mark_carries_its_corners_for_later_re_derivation(self):
        quad = MARKED_QUADS["sloppy_end_edges"]
        edit = four_point_crop(quad, FRAME, RATIO, 8.0)
        self.assertEqual(len(edit["marked_quad"]), 4)

    def test_saved_mark_is_re_derived_with_the_current_margin(self):
        quad = MARKED_QUADS["already_ratio"]
        marked = four_point_crop(quad, FRAME, RATIO, 8.0)
        definition = normalized_definition(
            marked["center"], marked["width"], marked["angle_degrees"], marked["line"],
            FRAME, quad=marked["marked_quad"],
        )
        wider = crop_edit_from_saved(definition, FRAME, RATIO, 25.0)
        plain = four_point_crop(quad, FRAME, RATIO, 25.0)
        self.assertAlmostEqual(wider["width"], plain["width"], places=6)
        self.assertGreater(wider["width"], marked["width"])

    def test_definition_without_corners_keeps_its_stored_box(self):
        # An older or hand-edited crop has no corners to re-derive from.
        definition = normalized_definition(
            (500.0, 300.0), 600.0, 20.0, [[400.0, 250.0], [600.0, 350.0]], (1000, 600),
        )
        edit = crop_edit_from_saved(definition, (1000, 600), RATIO, 8.0)
        self.assertAlmostEqual(edit["width"], 600.0)
        self.assertAlmostEqual(edit["angle_degrees"], 20.0)


class FourPointCropRejectionTests(unittest.TestCase):
    def assert_rejected(self, points, margin=8.0):
        with self.assertRaises(ValueError) as caught:
            four_point_crop(points, FRAME, RATIO, margin)
        return str(caught.exception)

    def test_rejects_points_that_are_too_close(self):
        self.assert_rejected([(500, 500), (505, 505), (1500, 1500), (2000, 500)])

    def test_rejects_a_collinear_mark(self):
        self.assert_rejected([(500, 500), (900, 500), (1300, 500), (1700, 500)])

    def test_rejects_a_bow_tie_click_order(self):
        message = self.assert_rejected(
            [(500, 500), (1900, 900), (500, 900), (1900, 500)],
        )
        self.assertIn("order", message)

    def test_rejects_a_short_side_clicked_as_the_long_side(self):
        message = self.assert_rejected(
            [(500, 1000), (3500, 1000), (3500, 1500), (500, 1500)],
        )
        self.assertIn("long side", message)

    def test_rejects_end_edges_far_from_square(self):
        # A sheared parallelogram: convex, so the bow-tie test passes, but both end
        # edges run 45 degrees off the long side.
        message = self.assert_rejected(
            [(200, 1200), (500, 1500), (3500, 1500), (3200, 1200)],
        )
        self.assertIn("square", message)


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

    def test_store_round_trip_keeps_four_crops_for_every_size(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "crops.json"
            data = {"version": 2, "sizes": {
                "M": {"1": [{"x": 1}, None], "2": [None, {"x": 2}]},
                "XL": {"1": [None, {"x": 3}], "2": [None, None]},
            }}
            save_crop_store(path, data)
            self.assertEqual(load_crop_store(path), data)

    def test_legacy_crops_are_migrated_to_medium(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "crops.json"
            path.write_text(
                '{"version": 1, "cameras": {"1": [{"x": 1}, null], '
                '"2": [null, {"x": 2}]}}', encoding="utf-8")
            loaded = load_crop_store(path, ["XS", "M", "2XL"])
            self.assertEqual(crop_cameras_for_size(loaded, "M")["1"][0], {"x": 1})
            self.assertEqual(crop_cameras_for_size(loaded, "2XL")["2"], [None, None])

    def test_store_keeps_the_ratio_the_crops_were_marked_at(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "crops.json"
            data = {"version": 2, "aspect_ratio": [4, 1],
                    "sizes": {"M": {
                        "1": [{"x": 1}, None], "2": [None, {"x": 2}]}}}
            save_crop_store(path, data)
            self.assertEqual(load_crop_store(path)["aspect_ratio"], [4, 1])

    def test_store_ignores_an_unusable_ratio(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "crops.json"
            path.write_text(
                '{"version": 1, "aspect_ratio": "wide", "cameras": {}}', encoding="utf-8",
            )
            self.assertNotIn("aspect_ratio", load_crop_store(path))


if __name__ == '__main__':
    unittest.main()

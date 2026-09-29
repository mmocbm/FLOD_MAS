"""The result viewer's viewport arithmetic, verified without opening a window.

Every function here is pure, so the zoom-at-cursor invariant, the offset
clamping and the arc timing can all be checked with plain numbers. The widget
itself is exercised by the manual `verify_crop_result_view.py` script.
"""
import unittest

from crop_result_view import (
    ARC_RADIUS, MAX_ARC_STEP_SECONDS, MAX_ZOOM, MIN_ZOOM, SPINNER_PERIOD_SECONDS,
    advance_angle, clamp_offset, fit_view, reanchor, viewport, visible_centre,
    zoom_about,
)

IMAGE = (2208, 552)
CANVAS = (1346, 540)


def scale_of(fit, zoom):
    return fit * zoom


class FitViewTests(unittest.TestCase):
    def test_a_wide_crop_fits_the_full_width_and_centres_vertically(self):
        fit, offset = fit_view(IMAGE, CANVAS)

        # 1346/2208 is the binding ratio for a 4:1 crop in a wider box.
        self.assertAlmostEqual(fit, CANVAS[0] / IMAGE[0])
        self.assertAlmostEqual(offset[0], 0.0)
        self.assertGreater(offset[1], 0.0)
        self.assertAlmostEqual(
            offset[1], (CANVAS[1] - IMAGE[1] * fit) / 2.0)

    def test_a_tall_image_fits_the_height_instead(self):
        fit, offset = fit_view((100, 1000), (800, 400))

        self.assertAlmostEqual(fit, 0.4)
        self.assertGreater(offset[0], 0.0)
        self.assertAlmostEqual(offset[1], 0.0)

    def test_the_whole_image_is_inside_the_canvas_at_fit(self):
        fit, offset = fit_view(IMAGE, CANVAS)

        self.assertGreaterEqual(offset[0], -1e-9)
        self.assertGreaterEqual(offset[1], -1e-9)
        self.assertLessEqual(offset[0] + IMAGE[0] * fit, CANVAS[0] + 1e-9)
        self.assertLessEqual(offset[1] + IMAGE[1] * fit, CANVAS[1] + 1e-9)


class ZoomAboutTests(unittest.TestCase):
    def test_the_image_point_under_the_cursor_does_not_move(self):
        fit, offset = fit_view(IMAGE, CANVAS)
        cursor = (900.0, 300.0)
        before = ((cursor[0] - offset[0]) / scale_of(fit, 1.0),
                  (cursor[1] - offset[1]) / scale_of(fit, 1.0))

        zoom, moved = zoom_about(fit, offset, 1.0, 1.25, cursor)

        after = ((cursor[0] - moved[0]) / scale_of(fit, zoom),
                 (cursor[1] - moved[1]) / scale_of(fit, zoom))
        self.assertAlmostEqual(before[0], after[0])
        self.assertAlmostEqual(before[1], after[1])

    def test_zooming_out_and_back_in_returns_to_the_same_view(self):
        fit, offset = fit_view(IMAGE, CANVAS)

        zoom, moved = zoom_about(fit, offset, 1.0, 1.25, (900.0, 300.0))
        zoom, back = zoom_about(fit, moved, zoom, 0.8, (900.0, 300.0))

        self.assertAlmostEqual(zoom, 1.0)
        self.assertAlmostEqual(back[0], offset[0])
        self.assertAlmostEqual(back[1], offset[1])

    def test_zoom_is_clamped_between_the_fit_view_and_the_maximum(self):
        fit, offset = fit_view(IMAGE, CANVAS)

        zoom, _moved = zoom_about(fit, offset, 1.0, 0.1, (10.0, 10.0))
        self.assertAlmostEqual(zoom, MIN_ZOOM)

        zoom, _moved = zoom_about(fit, offset, MAX_ZOOM, 10.0, (10.0, 10.0))
        self.assertAlmostEqual(zoom, MAX_ZOOM)


class ClampOffsetTests(unittest.TestCase):
    def test_an_image_smaller_than_the_canvas_is_centred(self):
        offset = clamp_offset(IMAGE, CANVAS, 0.1, [-500.0, -500.0])

        self.assertAlmostEqual(offset[0], (CANVAS[0] - IMAGE[0] * 0.1) / 2.0)
        self.assertAlmostEqual(offset[1], (CANVAS[1] - IMAGE[1] * 0.1) / 2.0)

    def test_a_zoomed_in_image_cannot_be_dragged_off_the_canvas(self):
        scale = 1.0
        scaled_width = IMAGE[0] * scale
        scaled_height = IMAGE[1] * scale

        far = clamp_offset(IMAGE, CANVAS, scale, [9999.0, 9999.0])
        self.assertAlmostEqual(far[0], 0.0)
        self.assertAlmostEqual(far[1], 0.0)

        near = clamp_offset(IMAGE, CANVAS, scale, [-9999.0, -9999.0])
        self.assertAlmostEqual(near[0], CANVAS[0] - scaled_width)
        self.assertAlmostEqual(near[1], CANVAS[1] - scaled_height)

    def test_an_offset_inside_the_limits_is_left_alone(self):
        # A 552-tall image in a 540-tall canvas may sit between -12 and 0.
        offset = clamp_offset(IMAGE, CANVAS, 1.0, [-100.0, -5.0])

        self.assertAlmostEqual(offset[0], -100.0)
        self.assertAlmostEqual(offset[1], -5.0)


class ViewportTests(unittest.TestCase):
    def test_a_fitted_image_is_cropped_to_its_own_bounds(self):
        fit, offset = fit_view(IMAGE, CANVAS)

        left, top, right, bottom = viewport(IMAGE, CANVAS, fit, offset)

        self.assertEqual((left, top), (0, 0))
        self.assertEqual((right, bottom), IMAGE)

    def test_a_zoomed_image_is_cropped_to_what_is_on_screen(self):
        scale = 2.0
        left, top, right, bottom = viewport(IMAGE, CANVAS, scale, [-800.0, -100.0])

        self.assertEqual(left, 400)
        self.assertEqual(top, 50)
        self.assertLess(right, IMAGE[0])
        self.assertLess(bottom, IMAGE[1])
        # The crop covers the canvas, not the whole image.
        self.assertLessEqual((right - left) * scale, CANVAS[0] + 2)

    def test_an_image_entirely_off_screen_has_no_viewport(self):
        self.assertIsNone(viewport(IMAGE, CANVAS, 1.0, [-99999.0, 0.0]))

    def test_the_viewport_never_runs_past_the_image(self):
        box = viewport(IMAGE, CANVAS, 1.0, [50.0, 50.0])

        left, top, right, bottom = box
        self.assertGreaterEqual(left, 0)
        self.assertGreaterEqual(top, 0)
        self.assertLessEqual(right, IMAGE[0])
        self.assertLessEqual(bottom, IMAGE[1])


class VisibleCentreTests(unittest.TestCase):
    def test_a_fitted_image_spins_at_the_canvas_centre(self):
        fit, offset = fit_view(IMAGE, CANVAS)

        centre = visible_centre(IMAGE, CANVAS, fit, offset)

        self.assertAlmostEqual(centre[0], CANVAS[0] / 2.0)
        self.assertAlmostEqual(centre[1], CANVAS[1] / 2.0)

    def test_a_corner_panned_view_still_spins_over_the_image(self):
        centre = visible_centre(IMAGE, CANVAS, 1.0, [-600.0, -30.0])

        # Clamped into the visible part of the crop, never into the background.
        self.assertGreater(centre[0], 0.0)
        self.assertLess(centre[0], CANVAS[0])
        self.assertGreater(centre[1], 0.0)
        self.assertLess(centre[1], CANVAS[1])

    def test_an_image_entirely_off_screen_has_no_centre(self):
        self.assertIsNone(visible_centre(IMAGE, CANVAS, 1.0, [-99999.0, 0.0]))

    def test_the_arc_fits_inside_a_short_canvas(self):
        # A 4:1 crop fitted to a 1346x540 canvas is only ~336px tall, so the arc
        # must not be drawn outside it.
        fit, offset = fit_view(IMAGE, CANVAS)
        centre = visible_centre(IMAGE, CANVAS, fit, offset)

        self.assertGreater(centre[1] - ARC_RADIUS, 0.0)
        self.assertLess(centre[1] + ARC_RADIUS, CANVAS[1])


class ReanchorTests(unittest.TestCase):
    def test_the_zoom_factor_survives_a_resize(self):
        fit, offset = fit_view(IMAGE, CANVAS)
        zoom, moved = zoom_about(fit, offset, 1.0, 2.0, (600.0, 200.0))

        new_fit, reoffset = reanchor(
            (1000, 400), CANVAS, IMAGE, zoom, moved)

        self.assertAlmostEqual(new_fit, min(1000 / IMAGE[0], 400 / IMAGE[1]))
        self.assertAlmostEqual(zoom, 2.0)
        self.assertNotEqual(reoffset, moved)

    def test_the_point_at_the_old_centre_stays_at_the_new_centre(self):
        fit, offset = fit_view(IMAGE, CANVAS)
        zoom, moved = zoom_about(fit, offset, 1.0, 3.0, (600.0, 200.0))
        centre_image = ((CANVAS[0] / 2.0 - moved[0]) / (fit * zoom),
                        (CANVAS[1] / 2.0 - moved[1]) / (fit * zoom))

        new_fit, reoffset = reanchor((1000, 400), CANVAS, IMAGE, zoom, moved)

        centre_now = ((1000 / 2.0 - reoffset[0]) / (new_fit * zoom),
                      (400 / 2.0 - reoffset[1]) / (new_fit * zoom))
        self.assertAlmostEqual(centre_image[0], centre_now[0])
        self.assertAlmostEqual(centre_image[1], centre_now[1])


class AdvanceAngleTests(unittest.TestCase):
    def test_a_normal_tick_advances_the_arc_proportionally(self):
        # A quarter of the period is a quarter turn, and is under the clamp.
        self.assertAlmostEqual(advance_angle(0.0, SPINNER_PERIOD_SECONDS / 4), 90.0)

    def test_a_long_stall_does_not_jump_the_arc(self):
        # The paused preview tick re-arms at a flat interval, so a slow frame
        # must not turn into a sudden leap.
        angle = advance_angle(0.0, 30.0)

        self.assertAlmostEqual(
            angle, 360.0 * MAX_ARC_STEP_SECONDS / SPINNER_PERIOD_SECONDS)

    def test_the_angle_stays_within_one_turn(self):
        self.assertLess(advance_angle(359.0, 10.0), 360.0)


if __name__ == '__main__':
    unittest.main()

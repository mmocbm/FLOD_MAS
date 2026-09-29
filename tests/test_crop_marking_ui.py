"""The four-click crop marking sequence in the Two-Crop Setup screen.

The crop editor is a small state machine: MARK 4 POINTS arms it, four canvas
clicks build a crop, a further click starts over, and a click that violates the
marking rules leaves the clicks on screen with an explanation. None of that
needs a camera or a window -- the canvas and the status label are mocks, and the
frozen frame is a plain array -- so it is driven here rather than by hand.
"""
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

import main_1366 as main
from crop_processing import crop_cameras_for_size, empty_crop_store

FRAME = (1400, 900)          # width, height of the frozen undistorted frame
RATIO = main.CROP_RATIO


def make_app():
    """A dashboard sitting in crop setup with one frame frozen and captured."""
    app = main.IndustrialDashboard.__new__(main.IndustrialDashboard)
    app.root = MagicMock()
    app.crop_definitions = empty_crop_store(main.FIXED_SIZES)
    app.active_size = 'M'
    app.crop_selected_size = 'M'
    app.crop_selected_camera = 1
    app.crop_selected_index = 0
    app.crop_frozen = True
    app.crop_frozen_frame = np.zeros((FRAME[1], FRAME[0], 3), np.uint8)
    app.crop_edit = None
    app.crop_edit_mode = "edit"
    app.crop_mark_points = []
    app.crop_drag = None
    app.crop_display_scale = 1.0
    app.crop_display_offset = (0.0, 0.0)
    app.crop_canvas = MagicMock()
    app.crop_canvas.winfo_exists.return_value = True
    app.crop_status = MagicMock()
    app._show_error_popup = MagicMock()
    return app


def press(app, x, y):
    event = MagicMock()
    event.x, event.y = x, y
    app.on_crop_press(event)


def valid_quad():
    """A 400 x 100 marked region well inside the frame."""
    return [(300, 300), (300, 400), (700, 400), (700, 300)]


class CropMarkingTests(unittest.TestCase):
    def test_four_clicks_produce_a_crop_and_return_to_editing(self):
        app = make_app()
        app.start_crop_marking()
        self.assertEqual(app.crop_edit_mode, "mark")
        for point in valid_quad():
            press(app, *point)
        self.assertIsNotNone(app.crop_edit)
        self.assertEqual(app.crop_edit_mode, "edit")
        self.assertAlmostEqual(
            app.crop_edit["width"] / app.crop_edit["height"], RATIO, places=6,
        )

    def test_marking_starts_from_a_clean_slate(self):
        app = make_app()
        app.crop_edit = {"center": [1, 2], "width": 3, "height": 4, "line": []}
        app.start_crop_marking()
        self.assertIsNone(app.crop_edit)

    def test_a_fifth_click_starts_a_fresh_mark(self):
        # The recovery path: after a mark the rules rejected, the clicks stay up so
        # the operator can see them, and the next click begins a new mark.
        app = make_app()
        app.start_crop_marking()
        for point in [(300, 300), (700, 300), (700, 400), (300, 400)]:
            press(app, *point)
        self.assertEqual(len(app.crop_mark_points), 4)
        press(app, 900, 800)
        self.assertEqual(app.crop_mark_points, [[900.0, 800.0]])
        self.assertEqual(app.crop_edit_mode, "mark")

    def test_a_bad_mark_is_explained_and_the_clicks_are_kept(self):
        app = make_app()
        app.start_crop_marking()
        # The short side clicked where the long side belongs.
        for point in [(300, 300), (700, 300), (700, 400), (300, 400)]:
            press(app, *point)
        self.assertIsNone(app.crop_edit)
        self.assertEqual(len(app.crop_mark_points), 4)
        self.assertEqual(app.crop_edit_mode, "mark")
        self.assertIn("long side", app.crop_status.configure.call_args.kwargs["text"])

    def test_press_outside_the_image_is_ignored(self):
        app = make_app()
        app.start_crop_marking()
        press(app, 5000, 5000)
        self.assertEqual(app.crop_mark_points, [])

    def test_marking_needs_a_captured_frame(self):
        app = make_app()
        app.crop_frozen = False
        app.crop_frozen_frame = None
        app.start_crop_marking()
        self.assertEqual(app.crop_edit_mode, "edit")
        app._show_error_popup.assert_called_once()

    def test_a_crop_cannot_be_saved_before_it_is_marked(self):
        app = make_app()
        app.save_crop_region()
        app._show_error_popup.assert_called_once()
        self.assertIsNone(crop_cameras_for_size(app.crop_definitions, 'M')['1'][0])

    def test_dragging_inside_a_marked_crop_moves_it(self):
        app = make_app()
        app.start_crop_marking()
        for point in valid_quad():
            press(app, *point)
        before = list(app.crop_edit["center"])
        press(app, 500, 350)                       # inside the rectangle
        self.assertEqual(app.crop_drag["kind"], "move")
        move = MagicMock()
        move.x, move.y = 520, 360
        app.on_crop_drag(move)
        self.assertGreater(app.crop_edit["center"][0], before[0])
        self.assertGreater(app.crop_edit["center"][1], before[1])
        # The marked corners travel with it, so the region stays re-derivable.
        self.assertEqual(len(app.crop_edit["marked_quad"]), 4)

    def test_resizing_drops_the_marked_corners(self):
        app = make_app()
        app.start_crop_marking()
        for point in valid_quad():
            press(app, *point)
        # 400 x 100 marked, plus 8% margin, is 464 x 116 about (500, 350), so the
        # top-right handle sits at (732, 292).
        press(app, 731, 293)
        self.assertEqual(app.crop_drag["kind"], "resize")
        drag = MagicMock()
        drag.x, drag.y = 800, 250
        app.on_crop_drag(drag)
        # A resized box no longer follows from the mark, so it must not be offered
        # to a later margin or ratio change as if it did.
        self.assertNotIn("marked_quad", app.crop_edit)

    def test_a_press_that_hits_nothing_does_not_start_a_rectangle(self):
        app = make_app()
        app.start_crop_marking()
        for point in valid_quad():
            press(app, *point)
        width = app.crop_edit["width"]
        press(app, 1100, 800)                      # far outside the rectangle
        self.assertIsNone(app.crop_drag)
        self.assertEqual(app.crop_edit["width"], width)


class CropSavingTests(unittest.TestCase):
    def test_saving_stores_the_marked_corners_and_the_ratio(self):
        app = make_app()
        app.start_crop_marking()
        for point in valid_quad():
            press(app, *point)
        with patch.object(main, 'save_crop_store') as store:
            app.save_crop_region()
        self.assertTrue(store.called)
        self.assertEqual(app.crop_definitions["aspect_ratio"],
                         list(main.CONFIG['crop_setup']['aspect_ratio']))
        saved = crop_cameras_for_size(app.crop_definitions, 'M')['1'][0]
        self.assertEqual(len(saved["marked_quad_normalized"]), 4)

    def test_saving_refuses_when_the_margin_leaves_the_image(self):
        app = make_app()
        # A mark against the frame edge: the margin cannot fit around it.
        app.start_crop_marking()
        for point in [(2, 300), (2, 400), (402, 400), (402, 300)]:
            press(app, *point)
        with patch.object(main, 'save_crop_store') as store:
            app.save_crop_region()
        self.assertFalse(store.called)
        self.assertIsNone(crop_cameras_for_size(app.crop_definitions, 'M')['1'][0])
        app._show_error_popup.assert_called_once()
        self.assertIn("margin", app._show_error_popup.call_args.args[0])

    def test_each_size_saves_an_independent_crop(self):
        app = make_app()
        app.start_crop_marking()
        for point in valid_quad():
            press(app, *point)
        with patch.object(main, 'save_crop_store'):
            app.save_crop_region()
        medium = crop_cameras_for_size(app.crop_definitions, 'M')['1'][0]

        app.crop_selected_size = 'XL'
        app.crop_edit = None
        self.assertIsNone(crop_cameras_for_size(app.crop_definitions, 'XL')['1'][0])
        app.start_crop_marking()
        for x, y in valid_quad():
            press(app, x + 100, y)
        with patch.object(main, 'save_crop_store'):
            app.save_crop_region()
        extra_large = crop_cameras_for_size(app.crop_definitions, 'XL')['1'][0]

        self.assertIsNotNone(medium)
        self.assertIsNotNone(extra_large)
        self.assertNotEqual(medium['center_normalized'], extra_large['center_normalized'])


if __name__ == '__main__':
    unittest.main()

"""The crop -> SAM3 -> overlay/record path, with the network stubbed out.

These tests never touch the API: sam_detection.detect_crop is always replaced,
so the suite stays offline and gives the same result whether or not a .env with
a real key is present.
"""
import unittest
from unittest.mock import MagicMock, mock_open, patch

import numpy as np

import main_1366 as main
import sam_detection


def make_app():
    """A dashboard with only the attributes _detect_crops touches."""
    app = main.IndustrialDashboard.__new__(main.IndustrialDashboard)
    app.active_size = 'M'
    app.root = MagicMock()
    # Run marshalled callbacks immediately so the worker path is synchronous.
    app.root.after.side_effect = lambda _delay, callback, *args: callback(*args)
    app.update_progress = MagicMock()
    return app


def crops():
    return [np.zeros((552, 2208, 3), np.uint8) for _ in range(2)]


def ring():
    return np.array([[1, 1], [50, 1], [50, 50], [1, 50]], np.int32)


ONLY_CROP_1 = {'1': [True, False], '2': [False, False]}


class DetectCropsTests(unittest.TestCase):
    def test_disabled_returns_crops_untouched(self):
        app = make_app()
        images = crops()
        with patch.dict(main.CONFIG['sam_detection'], {'enabled': False}), \
                patch.object(sam_detection, 'detect_crop') as detect:
            display, warnings = app._detect_crops(1, 'ts', images)

        detect.assert_not_called()
        self.assertEqual(warnings, [])
        self.assertIs(display[0], images[0])
        self.assertIs(display[1], images[1])

    def test_unselected_crop_is_never_uploaded(self):
        app = make_app()
        images = crops()
        with patch.dict(main.CONFIG['sam_detection'],
                        {'enabled': True, 'send_crops': ONLY_CROP_1}), \
                patch.object(sam_detection, 'detect_crop',
                             return_value=sam_detection.CropDetection([], 0)) as detect, \
                patch.object(main.os, 'makedirs'), \
                patch('builtins.open', mock_open()), \
                patch.object(main.json, 'dump'):
            app._detect_crops(1, 'ts', images)

        self.assertEqual(detect.call_count, 1)  # Crop 1 only, not Crop 2

    def test_polygons_swap_in_overlay_and_write_record(self):
        app = make_app()
        images = crops()
        detection = sam_detection.CropDetection(
            [sam_detection.Polygon('strip', 0.9, ring())], 1234,
        )
        with patch.dict(main.CONFIG['sam_detection'],
                        {'enabled': True, 'send_crops': ONLY_CROP_1}), \
                patch.object(sam_detection, 'detect_crop', return_value=detection), \
                patch.object(main.os, 'makedirs'), \
                patch.object(main.cv2, 'imwrite', return_value=True) as imwrite, \
                patch('builtins.open', mock_open()) as opener, \
                patch.object(main.json, 'dump') as dump:
            display, warnings = app._detect_crops(1, 'ts', images)

        self.assertEqual(warnings, [])
        self.assertEqual(imwrite.call_count, 1)          # the overlay
        self.assertIsNot(display[0], images[0])          # annotated swapped in
        self.assertIs(display[1], images[1])             # other crop untouched
        self.assertEqual(opener.call_count, 1)           # the record
        record = dump.call_args[0][0]
        self.assertIsNone(record['error'])
        self.assertEqual(record['polygons'][0]['class'], 'strip')
        self.assertEqual(record['polygons'][0]['points'][0], {'x': 1, 'y': 1})
        self.assertEqual(record['upload']['bytes'], 1234)
        self.assertTrue(record['overlay_saved'])

    def test_strip_is_measured_and_recorded(self):
        app = make_app()
        images = crops()
        # A long thin ring stands in for the strip; only its shape matters here.
        strip = np.array([[100, 300], [1900, 300], [1900, 330], [100, 330]], np.int32)
        detection = sam_detection.CropDetection(
            [sam_detection.Polygon('strip', 0.9, strip)], 1234,
        )
        with patch.dict(main.CONFIG['sam_detection'],
                        {'enabled': True, 'send_crops': ONLY_CROP_1,
                         'analyze_strip': True, 'strip_segments': 10}), \
                patch.object(sam_detection, 'detect_crop', return_value=detection), \
                patch.object(main.os, 'makedirs'), \
                patch.object(main.cv2, 'imwrite', return_value=True), \
                patch('builtins.open', mock_open()), \
                patch.object(main.json, 'dump') as dump:
            display, warnings = app._detect_crops(1, 'ts', images)

        self.assertEqual(warnings, [])
        record = dump.call_args[0][0]
        self.assertEqual(len(record['strip']['segments']), 10)
        self.assertGreater(record['strip']['total_length_px'], 1000)
        # The measured strip is drawn onto the overlay that gets displayed.
        self.assertIsNot(display[0], images[0])

    def test_strip_is_skipped_when_disabled(self):
        app = make_app()
        images = crops()
        strip = np.array([[100, 300], [1900, 300], [1900, 330], [100, 330]], np.int32)
        detection = sam_detection.CropDetection(
            [sam_detection.Polygon('strip', 0.9, strip)], 1234,
        )
        with patch.dict(main.CONFIG['sam_detection'],
                        {'enabled': True, 'send_crops': ONLY_CROP_1,
                         'analyze_strip': False}), \
                patch.object(sam_detection, 'detect_crop', return_value=detection), \
                patch.object(main.os, 'makedirs'), \
                patch.object(main.cv2, 'imwrite', return_value=True), \
                patch('builtins.open', mock_open()), \
                patch.object(main.json, 'dump') as dump:
            app._detect_crops(1, 'ts', images)

        self.assertIsNone(dump.call_args[0][0]['strip'])

    def test_failure_warns_keeps_raw_crop_and_still_records(self):
        app = make_app()
        images = crops()
        detection = sam_detection.CropDetection([], 500, 'Connection refused')
        with patch.dict(main.CONFIG['sam_detection'],
                        {'enabled': True, 'send_crops': ONLY_CROP_1}), \
                patch.object(sam_detection, 'detect_crop', return_value=detection), \
                patch.object(main.os, 'makedirs'), \
                patch.object(main.cv2, 'imwrite', return_value=True) as imwrite, \
                patch('builtins.open', mock_open()), \
                patch.object(main.json, 'dump') as dump:
            display, warnings = app._detect_crops(1, 'ts', images)

        self.assertEqual(imwrite.call_count, 0)          # no overlay for a negative
        self.assertIs(display[0], images[0])             # raw crop kept
        self.assertEqual(len(warnings), 1)
        self.assertIn('Connection refused', warnings[0])
        # the record is still written, so a dead API is distinguishable
        self.assertEqual(dump.call_args[0][0]['error'], 'Connection refused')


if __name__ == '__main__':
    unittest.main()

"""The crop -> SAM3 -> overlay/record path, with the network stubbed out.

These tests never touch the API: sam_detection.detect_crop is always replaced,
so the suite stays offline and gives the same result whether or not a .env with
a real key is present.
"""
import json
import tempfile
import unittest
from pathlib import Path
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
    # No camera objects and no saved regions: the millimetre scale cannot be
    # built here, which is why the tests that measure a strip turn it off.
    app.strip_scales = {}
    app.crop_definitions = {'cameras': {'1': [], '2': []}}
    app.camera1 = None
    app.camera2 = None
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
                        {'enabled': True, 'send_crops': ONLY_CROP_1,
                         'measure_in_mm': False}), \
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
                         'analyze_strip': True, 'strip_segments': 10,
                         'measure_in_mm': False}), \
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


class StripScaleTests(unittest.TestCase):
    """The millimetre scale is built once per crop, and its absence is reported.

    A missing scale must downgrade the units, never lose the measurement, so
    every way it can fail ends in ``None`` plus a warning the operator can read.
    """

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def definition(self):
        return {
            'center_normalized': [0.4931506849, 0.2508561644],
            'width_normalized': 0.7762557078,
            'angle_degrees': 2.3760552180,
            'line_normalized': [[0.2283105023, 0.2859589041],
                                [0.7785388128, 0.3030821918]],
        }

    def app_with_scale(self):
        """An app whose crop 1 can actually be scaled into millimetres."""
        # Named as the config names them: the lookup resolves by basename.
        matrix = [[6812.9, 0.0, 1949.6], [0.0, 6812.9, 2243.1], [0.0, 0.0, 1.0]]
        (self.root / 'camera_calibration_0.json').write_text(json.dumps(
            {'camera_matrix': matrix, 'dist_coeffs': [0.0] * 5,
             'image_size': [3456, 4608]}), encoding='utf-8')
        (self.root / 'camera_extrinsics_0.json').write_text(json.dumps(
            {'rvec': [0.62, 0.05, 0.03], 'tvec': [0.04, -0.31, 1.10]}),
            encoding='utf-8')

        app = make_app()
        app.crop_definitions = {'cameras': {'1': [self.definition()]}}
        return app

    def test_measuring_in_pixels_needs_no_scale(self):
        app = make_app()
        warnings = []

        with patch.dict(main.CONFIG['sam_detection'], {'measure_in_mm': False}):
            scale = app._strip_scale(1, 1, (3456, 4608), warnings)

        self.assertIsNone(scale)
        self.assertEqual(warnings, [])

    def test_an_unknown_frame_size_is_reported_not_guessed(self):
        app = make_app()
        warnings = []

        with patch.dict(main.CONFIG['sam_detection'], {'measure_in_mm': True}):
            scale = app._strip_scale(1, 1, None, warnings)

        self.assertIsNone(scale)
        self.assertEqual(len(warnings), 1)
        self.assertIn('millimetre', warnings[0])

    def test_missing_calibration_files_fall_back_to_pixels_with_a_reason(self):
        app = make_app()
        app.crop_definitions = {'cameras': {'1': [self.definition()]}}
        warnings = []

        with patch.dict(main.CONFIG['sam_detection'], {'measure_in_mm': True}), \
                patch('app_config.project_path',
                      lambda path: str(self.root / Path(path).name)):
            scale = app._strip_scale(1, 1, (3456, 4608), warnings)

        self.assertIsNone(scale)          # the measurement is not lost
        self.assertEqual(len(warnings), 1)
        self.assertIn('pixels', warnings[0])

    def test_a_usable_calibration_yields_a_scale_that_is_then_cached(self):
        app = self.app_with_scale()
        warnings = []

        with patch.dict(main.CONFIG['sam_detection'], {'measure_in_mm': True}), \
                patch('app_config.project_path',
                      lambda path: str(self.root / Path(path).name)):
            first = app._strip_scale(1, 1, (3456, 4608), warnings)
            second = app._strip_scale(1, 1, (3456, 4608), warnings)

        self.assertIsNotNone(first)
        self.assertGreater(first.mm_per_pixel, 0.0)
        self.assertIs(second, first)      # built once, reused
        self.assertEqual(warnings, [])


if __name__ == '__main__':
    unittest.main()

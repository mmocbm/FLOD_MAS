"""Actual reference geometry with deterministic AI masks; no network or camera."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from concurrent.futures import Future
import cv2
import numpy as np
from inspection.local.engine import LiveInspection
from inspection.line_measurement import measurement_record
from inspection.pipeline import InspectionPipeline
from inspection.storage import CaptureStore
from plane_scale import PlaneScale, PlaneScaleError
from config_validation import validate_config
from auto_trigger.controller import AutoController

CONFIG = json.loads((Path(__file__).resolve().parents[1] / 'config.json').read_text())


def synthetic_inspection(broken=False):
    frame = np.full((800, 1100, 3), 100, np.uint8)
    source = np.zeros(frame.shape[:2], np.uint8)
    y = np.arange(30, 770)
    for x in (350, 850):
        boundary = np.column_stack((x + 30*np.sin((y-30)*2*np.pi/740), y))
        polygon = np.vstack(([x-160, 30], boundary, [x-160, 769]))
        cv2.fillPoly(source, [np.rint(polygon).astype(np.int32)], 255)
    def adhesive(flat):
        h, w = flat.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        mask[35:65, 15:w-15] = 255
        if broken:
            mask[:, w//2-60:w//2+60] = 0
        return mask
    inspector = LiveInspection.__new__(LiveInspection)
    inspector.offset = 100.
    inspector.source = SimpleNamespace(predict=lambda photo: source.copy())
    inspector.glue = SimpleNamespace(width=1024, predict=adhesive)
    return frame, inspector, inspector.run(frame)


class LocalGeometryTests(unittest.TestCase):
    def test_only_rightmost_line_is_refolded_to_full_frame(self):
        frame, _, result = synthetic_inspection()
        self.assertEqual(result['line_count'], 2)
        self.assertEqual(result['selected_line'], 2)
        self.assertEqual(result['mask'].shape, frame.shape[:2])
        self.assertEqual(result['overlay'].shape, frame.shape)
        self.assertFalse(result['mask'][:, :600].any())
        self.assertTrue(result['mask'][:, 700:].any())
        self.assertEqual(len(result['segments']), 10)
        self.assertAlmostEqual(result['mean_width_px'], 30, delta=2)
        # Returned intersections must be in the full undistorted frame, not crop pixels.
        valid = [s for s in result['components'][0]['samples'] if s['valid']]
        self.assertTrue(all(s['center_x'] > 600 and s['left_x'] > 600 for s in valid))

    def test_disconnected_glue_keeps_long_gap_and_sums_real_lengths(self):
        _, _, result = synthetic_inspection(broken=True)
        self.assertEqual(result['component_count'], 2)
        self.assertAlmostEqual(result['length_px'], sum(c['length_px'] for c in result['components']))
        _, _, continuous = synthetic_inspection()
        self.assertLess(result['length_px'], continuous['length_px'])
        main = max(result['components'], key=lambda c: c['length_px'])
        self.assertEqual(result['segments'], main['segments'])

    def test_empty_source_or_empty_adhesive_is_an_error(self):
        frame, inspector, _ = synthetic_inspection()
        inspector.source.predict = lambda f: np.zeros(f.shape[:2], np.uint8)
        with self.assertRaisesRegex(ValueError, 'No accepted source lines'):
            inspector.run(frame)
        _, inspector, _ = synthetic_inspection()
        inspector.glue.predict = lambda f: np.zeros(f.shape[:2], np.uint8)
        with self.assertRaisesRegex(ValueError, 'empty mask'):
            inspector.run(frame)


class CalibrationTests(unittest.TestCase):
    def test_metric_widths_transform_global_intersections_and_grade_all_segments(self):
        _, _, result = synthetic_inspection()
        scale = PlaneScale(np.diag([.1, .1, 1]), .1)
        record = measurement_record(result, scale, 3, 2)
        self.assertEqual(record['status'], 'PASS')
        # Metric length uses transformed sampled centres; the reference stores
        # its pre-resampling arc length, so small chord discretization is expected.
        self.assertAlmostEqual(record['length_mm'], result['length_px']*.1, delta=.1)
        self.assertAlmostEqual(record['mean_width_mm'], result['mean_width_px']*.1, places=5)
        self.assertEqual(len(record['segments']), 10)
        self.assertTrue(all(s['within_tolerance'] for s in record['segments']))
        self.assertEqual(measurement_record(result, scale, 10, 1)['status'], 'FAIL')

    def test_projective_calibration_uses_endpoints_not_summary_scale(self):
        _, _, result = synthetic_inspection()
        scale = PlaneScale(np.array([[.12, .01, 0], [0, .1, 0], [0, .0005, 1]]), 999)
        record = measurement_record(result, scale, 3, 2)
        sample = next(s for s in record['components'][0]['samples'] if s['valid'])
        endpoints = scale.to_mm([[sample['left_x'], sample['left_y']],
                                 [sample['right_x'], sample['right_y']]])
        self.assertAlmostEqual(sample['width_mm'], np.linalg.norm(endpoints[1]-endpoints[0]))
        self.assertLess(sample['width_mm'], 10)

    def test_missing_calibration_or_missing_segment_warns_instead_of_grading(self):
        _, _, result = synthetic_inspection()
        record = measurement_record(result, None, 4, 1)
        # No plane means nothing could be graded, so the fabric warns rather than
        # being called a failure it was never measured against.
        self.assertEqual(record['status'], 'WARNING')
        self.assertFalse(record['metric'])
        self.assertIsNone(record['length_mm'])
        for sample in result['components'][0]['samples']:
            if sample['segment'] == 1:
                sample['valid'] = False
        scale = PlaneScale(np.diag([.1, .1, 1]), .1)
        self.assertEqual(measurement_record(result, scale, 3, 2)['status'], 'WARNING')

    def test_one_ungradeable_segment_warns_even_when_the_rest_are_out_of_tolerance(self):
        _, _, result = synthetic_inspection()
        scale = PlaneScale(np.diag([.1, .1, 1]), .1)
        # Every measured segment is out of tolerance against a 10 mm target, so
        # this would be FAIL -- except segment 1 of the main component loses its
        # samples, and the missing measurement decides the label instead.
        for sample in result['components'][0]['samples']:
            if sample['segment'] == 1:
                sample['valid'] = False
        record = measurement_record(result, scale, 10, .2)
        self.assertEqual(record['status'], 'WARNING')
        # The segments that were measurable still carry their own verdicts.
        graded = {s['segment']: s['within_tolerance']
                  for s in record['components'][0]['segments']}
        self.assertIsNone(graded[1])
        self.assertIn(False, [value for key, value in graded.items() if key != 1])


class LocalPipelineTests(unittest.TestCase):
    def test_corrected_input_lossless_mask_and_serializable_metric_record(self):
        corrected, _, detected = synthetic_inspection()
        original = np.zeros_like(corrected)
        seen = []
        inspector = SimpleNamespace(run=lambda f, progress: (seen.append(f), detected)[1])
        with tempfile.TemporaryDirectory() as directory:
            store = CaptureStore(directory, CONFIG['capture_storage'])
            pipeline = InspectionPipeline(inspector, store, CONFIG)
            with patch('inspection.pipeline.plane_scale.load_frame_scale',
                       return_value=PlaneScale(np.diag([.1,.1,1]), .1)):
                result = pipeline.run(original, corrected, {'size':'M'}, 3, 2)
            self.assertIs(seen[0], corrected)
            path = Path(result['path'])
            for name, expected in [('original.png',original), ('undistorted.png',corrected),
                                    ('mask.png',detected['mask']), ('overlay.png',detected['overlay'])]:
                np.testing.assert_array_equal(cv2.imread(str(path/name), cv2.IMREAD_UNCHANGED), expected)
            record = json.loads((path/'result.json').read_text())
            self.assertEqual(record['measurement']['status'], 'PASS')
            self.assertEqual(record['method'], 'local_onnx_rightmost_line')

    def test_missing_plane_keeps_pixel_result_and_warning(self):
        frame, _, detected = synthetic_inspection()
        with tempfile.TemporaryDirectory() as directory:
            pipeline = InspectionPipeline(SimpleNamespace(run=lambda f, progress: detected),
                                          CaptureStore(directory, CONFIG['capture_storage']), CONFIG)
            with patch('inspection.pipeline.plane_scale.load_frame_scale', side_effect=PlaneScaleError('missing')):
                result = pipeline.run(frame, frame, {}, 4, 1)
            self.assertTrue(result['warnings'])
            self.assertEqual(result['measurement']['status'], 'WARNING')

    def test_invalid_local_settings_are_rejected(self):
        for key, value in [('offset_pixels',0), ('offset_pixels',float('nan')),
                            ('source_width',31), ('source_width',True), ('glue_model','model.h5')]:
            config = copy.deepcopy(CONFIG)
            config['local_inspection'][key] = value
            with self.assertRaises(ValueError):
                validate_config(config)


class ResultLifecycleTests(unittest.TestCase):
    def test_obsolete_inspection_error_does_not_clear_new_results(self):
        controller = AutoController(CONFIG, lambda: None)
        controller.generation = 2
        controller.job_generation = 1
        controller.future = Future()
        controller.future.set_exception(ValueError('old empty adhesive mask'))
        controller._complete()
        self.assertEqual(list(controller.poll()), [])

    def test_current_error_is_reported_and_stopped_result_is_suppressed(self):
        controller = AutoController(CONFIG, lambda: None)
        controller.job_generation = controller.generation
        controller.future = Future()
        controller.future.set_exception(ValueError('empty adhesive mask'))
        controller._complete()
        self.assertEqual(list(controller.poll()), [('error', 0, 'empty adhesive mask')])
        controller.future = Future()
        controller.future.set_result({'old': True})
        controller.stop()
        controller._complete()
        self.assertNotIn('result', [event[0] for event in controller.poll()])


if __name__ == '__main__':
    unittest.main()

"""Exclude tip artefacts without shortening the recorded detected glue line."""
import copy
import unittest
import numpy as np
from inspection.line_measurement import measurement_record
from inspection.overlay import annotated_overlay
from plane_scale import PlaneScale
from config_validation import validate_config
from test_local_inspection import synthetic_inspection, CONFIG


class EndExclusionTests(unittest.TestCase):
    def test_five_percent_removes_tip_failures_and_keeps_full_length(self):
        frame, _, result = synthetic_inspection()
        for component in result['components']:
            length = component['length_px']
            for sample in component['samples']:
                if sample['arc_px'] < length*.05 or sample['arc_px'] > length*.95:
                    if sample['valid']:
                        left = np.array([sample['left_x'], sample['left_y']])
                        right = np.array([sample['right_x'], sample['right_y']])
                        centre = (left+right)/2
                        direction = (right-left)/np.linalg.norm(right-left)
                        sample['left_x'], sample['left_y'] = centre-direction*.5
                        sample['right_x'], sample['right_y'] = centre+direction*.5
                        sample['width_px'] = 1.0
        original = copy.deepcopy(result['components'])
        scale = PlaneScale(np.diag([.1, .1, 1]), .1)
        untrimmed = measurement_record(result, scale, 3, .2, 0)
        trimmed = measurement_record(result, scale, 3, .2, 5)
        self.assertEqual(untrimmed['status'], 'FAIL')
        self.assertEqual(trimmed['status'], 'PASS')
        self.assertEqual(trimmed['length_px'], untrimmed['length_px'])
        self.assertEqual(trimmed['length_mm'], untrimmed['length_mm'])
        self.assertEqual(result['components'], original)
        for component in trimmed['components']:
            self.assertAlmostEqual(component['inspected_length_px'], component['length_px']*.9)
            for sample in component['samples']:
                if sample['excluded_end']:
                    self.assertEqual(sample['segment'], 0)
                    self.assertFalse(sample['valid'])
                    self.assertIsNone(sample['width_mm'])
            self.assertEqual(len(component['segments']), 10)
            self.assertTrue(all(s['within_tolerance'] for s in component['segments']))
        self.assertEqual(annotated_overlay(frame, trimmed).shape, frame.shape)

    def test_excluded_samples_are_not_drawn(self):
        image = np.zeros((250, 250, 3), np.uint8)
        m = {'components': [{'samples': [{'segment': 0, 'center_x': 30, 'center_y': 30}],
            'segments': [{'segment': 1, 'within_tolerance': False, 'average_width_mm': 3}]}]}
        np.testing.assert_array_equal(annotated_overlay(image, m), image)

    def test_invalid_trim_settings_rejected(self):
        _, _, result = synthetic_inspection()
        for value in (-1, 50, 90, float('nan'), float('inf'), True):
            cfg = copy.deepcopy(CONFIG)
            cfg['inspection']['end_exclusion_percent'] = value
            with self.assertRaises(ValueError):
                validate_config(cfg)
            with self.assertRaises(ValueError):
                measurement_record(result, None, 3, .2, value)


if __name__ == '__main__':
    unittest.main()

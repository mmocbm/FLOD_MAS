"""Exclude tip artefacts without shortening the recorded detected glue line."""
import copy
import unittest
import numpy as np
from inspection.line_measurement import measurement_record
from inspection.overlay import annotated_overlay, segment_joins
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

    def test_the_detected_mask_is_tinted_under_the_grades(self):
        image = np.zeros((120, 160, 3), np.uint8)
        mask = np.zeros((120, 160), np.uint8)
        mask[30:90, 40:120] = 255
        plain = annotated_overlay(image, {'components': []})
        tinted = annotated_overlay(image, {'components': []}, mask)
        # Grades alone leave a bare frame; the mask is the only thing added here.
        np.testing.assert_array_equal(plain, image)
        self.assertFalse(np.array_equal(tinted, plain))
        self.assertEqual(tinted.shape, image.shape)
        # The fill covers the middle of the strip, not just its boundary.
        self.assertTrue(tinted[60, 80].any())
        np.testing.assert_array_equal(tinted[5, 5], image[5, 5])

    def test_an_empty_mask_draws_nothing(self):
        image = np.zeros((40, 40, 3), np.uint8)
        np.testing.assert_array_equal(
            annotated_overlay(image, {'components': []}, np.zeros((40, 40), np.uint8)), image)

    def test_a_mask_of_the_wrong_size_is_rejected(self):
        with self.assertRaises(ValueError):
            annotated_overlay(np.zeros((40, 40, 3), np.uint8), {'components': []},
                              np.zeros((20, 20), np.uint8))

    def test_invalid_trim_settings_rejected(self):
        _, _, result = synthetic_inspection()
        for value in (-1, 50, 90, float('nan'), float('inf'), True):
            cfg = copy.deepcopy(CONFIG)
            cfg['inspection']['end_exclusion_percent'] = value
            with self.assertRaises(ValueError):
                validate_config(cfg)
            with self.assertRaises(ValueError):
                measurement_record(result, None, 3, .2, value)


class SegmentSeparationTests(unittest.TestCase):
    """The segments meet exactly, so the overlay has to mark the join itself."""

    def test_every_join_is_drawn_across_the_adhesive(self):
        frame, _, result = synthetic_inspection()
        measurement = measurement_record(result, PlaneScale(np.diag([.1, .1, 1]), .1), 4, 1)
        component = max(measurement['components'], key=lambda c: len(c['segments']))
        joins = segment_joins(component)
        # Ten segments have nine joins between them, and the first segment has
        # nothing ahead of it to mark.
        self.assertEqual(len(joins), len(component['segments']) - 1)
        drawn = annotated_overlay(frame, measurement)
        for start, end in joins:
            # The bar spans the adhesive rather than sitting on the centreline.
            self.assertGreater(np.hypot(end[0]-start[0], end[1]-start[1]), 10)
            middle = (int(round((start[0]+end[0])/2)), int(round((start[1]+end[1])/2)))
            patch = drawn[middle[1]-2:middle[1]+3, middle[0]-2:middle[0]+3]
            self.assertTrue((patch.max(axis=(0, 1)) > 200).all(), middle)

    def test_a_rejected_boundary_sample_still_marks_the_join(self):
        # The bar is the last resort of the boundary geometry, never a guess:
        # a segment whose only located sample failed still shows its join.
        component = {'segments': [{'segment': 1}, {'segment': 2}],
                     'samples': [{'segment': 2, 'valid': False, 'left_x': 1, 'left_y': 2,
                                  'right_x': 3, 'right_y': 4}]}
        self.assertEqual(segment_joins(component), [((1, 2), (3, 4))])

    def test_the_accepted_sample_is_preferred_over_a_rejected_one(self):
        component = {'segments': [{'segment': 1}, {'segment': 2}],
                     'samples': [{'segment': 2, 'valid': False, 'left_x': 0, 'left_y': 0,
                                  'right_x': 0, 'right_y': 0},
                                 {'segment': 2, 'valid': True, 'left_x': 5, 'left_y': 6,
                                  'right_x': 7, 'right_y': 8}]}
        self.assertEqual(segment_joins(component), [((5, 6), (7, 8))])

    def test_a_segment_without_boundary_geometry_adds_no_bar(self):
        component = {'segments': [{'segment': 1}, {'segment': 2}],
                     'samples': [{'segment': 2, 'valid': True, 'center_x': 5, 'center_y': 5}]}
        self.assertEqual(segment_joins(component), [])


if __name__ == '__main__':
    unittest.main()

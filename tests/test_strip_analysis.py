"""Strip centreline, segmentation and width measurement.

The measurements are checked against ribbons built from a known curve, so the
true length and width are known independently of the code under test. The
tolerances are wider than the algorithm's own error because rasterising a
ribbon rounds its edges: a nominal 25px ribbon occupies 27 pixel rows, and that
is a property of the test fixture, not of the measurement.
"""
import unittest
from unittest.mock import patch

import numpy as np

import sam_detection
import strip_analysis

SHAPE = (552, 2208, 3)
CAMERA_FRAME = (552, 2208)


def make_ribbon(curve_x, curve_y, half_width):
    """A closed ring tracing both edges of a ribbon of the given half-width."""
    dx, dy = np.gradient(curve_x), np.gradient(curve_y)
    length = np.hypot(dx, dy)
    nx, ny = -dy / length, dx / length
    top = np.column_stack([curve_x + nx * half_width, curve_y + ny * half_width])
    bottom = np.column_stack([curve_x - nx * half_width, curve_y - ny * half_width])

    true_length = float(np.sum(np.hypot(np.diff(curve_x), np.diff(curve_y))))
    return np.round(np.vstack([top, bottom[::-1]])).astype(np.int32), true_length


def curved_curve(points=600):
    t = np.linspace(0, 1, points)
    return (120 + 1950 * t,
            150 + 110 * np.sin(2 * np.pi * 1.5 * t) + 90 * t)


def straight_curve(points=600):
    t = np.linspace(0, 1, points)
    return 150 + 1900 * t, np.full(points, 275.0)


class MeasurementTests(unittest.TestCase):
    def test_curved_ribbon_length_and_width(self):
        ring, true_length = make_ribbon(*curved_curve(), half_width=9.0)
        result = strip_analysis.analyze_ring(ring, SHAPE, segment_count=10)

        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.total_length_px / true_length, 1.0, delta=0.05)
        self.assertAlmostEqual(result.average_width_px, 18.0, delta=1.5)

    def test_uniform_ribbon_measures_uniformly(self):
        """Every segment of a constant-width strip must agree with the others.

        This is the property the per-segment labels rely on, so it is worth
        asserting on its own rather than only through the overall average.
        """
        ring, _ = make_ribbon(*curved_curve(), half_width=9.0)
        result = strip_analysis.analyze_ring(ring, SHAPE, segment_count=10)

        widths = [segment.average_width_px for segment in result.segments]
        self.assertEqual(len(widths), 10)
        self.assertLess(max(widths) - min(widths), 1.0)

    def test_segments_are_indexed_from_one_and_are_equal_length(self):
        ring, _ = make_ribbon(*curved_curve(), half_width=9.0)
        result = strip_analysis.analyze_ring(ring, SHAPE, segment_count=10)

        self.assertEqual([s.index for s in result.segments], list(range(1, 11)))
        # Boundaries land on whole centreline samples, so segment lengths can
        # differ by one sample's worth of arc length. That is the floor for any
        # index-based split, so it is the bound asserted here.
        spacing = result.total_length_px / len(result.centerline)
        lengths = [s.length_px for s in result.segments]
        self.assertLessEqual(max(lengths) - min(lengths), spacing + 1e-9)

    def test_straight_ribbon_width_matches_its_rasterised_extent(self):
        ring, true_length = make_ribbon(*straight_curve(), half_width=12.5)
        result = strip_analysis.analyze_ring(ring, SHAPE, segment_count=10)

        self.assertAlmostEqual(result.total_length_px / true_length, 1.0, delta=0.05)
        # The fixture's own extent is what a correct measurement must report.
        rows = np.unique(ring[:, 1])
        self.assertAlmostEqual(result.average_width_px, float(np.ptp(rows)), delta=1.0)

    def test_thin_ribbon_is_still_measured(self):
        ring, _ = make_ribbon(*curved_curve(), half_width=4.0)
        result = strip_analysis.analyze_ring(ring, SHAPE, segment_count=10)

        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.average_width_px, 8.0, delta=1.0)

    def test_segment_count_is_respected(self):
        ring, _ = make_ribbon(*curved_curve(), half_width=9.0)
        result = strip_analysis.analyze_ring(ring, SHAPE, segment_count=4)

        self.assertEqual(len(result.segments), 4)

    def test_centerline_stays_inside_the_image(self):
        ring, _ = make_ribbon(*curved_curve(), half_width=9.0)
        result = strip_analysis.analyze_ring(ring, SHAPE, segment_count=10)

        self.assertGreaterEqual(result.centerline[:, 0].min(), 0)
        self.assertLess(result.centerline[:, 0].max(), SHAPE[1])
        self.assertGreaterEqual(result.centerline[:, 1].min(), 0)
        self.assertLess(result.centerline[:, 1].max(), SHAPE[0])


class DegenerateInputTests(unittest.TestCase):
    """A stray prediction must never raise -- it returns nothing to report."""

    def test_degenerate_rings_return_none(self):
        cases = {
            'two points': np.array([[10, 10], [20, 20]], np.int32),
            'repeated point': np.array([[10, 10], [10, 10], [10, 10]], np.int32),
            'speck': np.array([[5, 5], [8, 5], [8, 8], [5, 8]], np.int32),
            'off image': np.array([[-99, -99], [-90, -99], [-90, -90]], np.int32),
        }
        for name, ring in cases.items():
            with self.subTest(name):
                self.assertIsNone(strip_analysis.analyze_ring(ring, SHAPE, 10))

    def test_zero_segments_returns_none(self):
        ring, _ = make_ribbon(*curved_curve(), half_width=9.0)
        self.assertIsNone(strip_analysis.analyze_ring(ring, SHAPE, segment_count=0))


class DrawingTests(unittest.TestCase):
    def test_drawing_marks_the_canvas_and_leaves_the_source_alone(self):
        ring, _ = make_ribbon(*curved_curve(), half_width=9.0)
        result = strip_analysis.analyze_ring(ring, SHAPE, segment_count=10)
        canvas = np.full(SHAPE, 210, np.uint8)
        before = canvas.copy()

        returned = strip_analysis.draw_strip_analysis(canvas, result)

        self.assertIs(returned, canvas)
        self.assertGreater(int(np.count_nonzero(canvas != before)), 0)

    def test_labels_stay_inside_the_image(self):
        ring, _ = make_ribbon(*curved_curve(), half_width=9.0)
        result = strip_analysis.analyze_ring(ring, SHAPE, segment_count=10)

        for segment in result.segments:
            left, top, width, height = strip_analysis._place_label(segment, SHAPE)
            self.assertGreaterEqual(left, 0)
            self.assertGreaterEqual(top, 0)
            self.assertLessEqual(left + width, SHAPE[1])
            self.assertLessEqual(top + height, SHAPE[0])


class AnalyzeDetectionTests(unittest.TestCase):
    def test_largest_polygon_is_the_one_measured(self):
        small = sam_detection.Polygon('strip', 0.5, np.array(
            [[10, 10], [30, 10], [30, 30], [10, 30]], np.int32))
        big_ring, true_length = make_ribbon(*curved_curve(), half_width=9.0)
        big = sam_detection.Polygon('strip', 0.9, big_ring)

        measurement = sam_detection.analyze_detection([small, big], CAMERA_FRAME, 10)

        self.assertIsNotNone(measurement)
        self.assertEqual(measurement.polygon_index, 1)
        self.assertAlmostEqual(measurement.analysis.total_length_px / true_length,
                               1.0, delta=0.05)

    def test_no_polygons_means_no_measurement(self):
        self.assertIsNone(sam_detection.analyze_detection([], CAMERA_FRAME, 10))

    def test_unmeasurable_polygon_means_no_measurement(self):
        speck = sam_detection.Polygon('strip', 0.9, np.array(
            [[5, 5], [8, 5], [8, 8], [5, 8]], np.int32))
        self.assertIsNone(sam_detection.analyze_detection([speck], CAMERA_FRAME, 10))

    def test_a_broken_ring_is_reported_as_no_measurement(self):
        """A measurement must never be able to fail an inspection."""
        broken = sam_detection.Polygon('strip', 0.9, np.zeros((4, 2), np.int32))
        with patch.object(strip_analysis, 'analyze_ring',
                          side_effect=RuntimeError('boom')):
            self.assertIsNone(
                sam_detection.analyze_detection([broken], CAMERA_FRAME, 10))


class StripRecordTests(unittest.TestCase):
    def measurement(self):
        ring, _ = make_ribbon(*curved_curve(), half_width=9.0)
        return sam_detection.analyze_detection(
            [sam_detection.Polygon('strip', 0.9, ring)], CAMERA_FRAME, 10)

    def test_record_carries_every_segment(self):
        record = sam_detection.strip_record(self.measurement())

        self.assertEqual(record['polygon_index'], 0)
        self.assertEqual(len(record['segments']), 10)
        self.assertEqual([s['index'] for s in record['segments']], list(range(1, 11)))
        for segment in record['segments']:
            for key in ('length_px', 'average_width_px', 'minimum_width_px',
                        'maximum_width_px', 'samples', 'midpoint'):
                self.assertIn(key, segment)
            self.assertIn('x', segment['midpoint'])
            self.assertIn('y', segment['midpoint'])

    def test_missing_measurement_serialises_as_null(self):
        self.assertIsNone(sam_detection.strip_record(None))

    def test_detection_record_includes_the_strip(self):
        record = sam_detection.detection_record(
            1, 2, 'ts', 'M', 'prompt', (2208, 552),
            sam_detection.CropDetection([], 0), 75, False, self.measurement(),
        )

        self.assertEqual(len(record['strip']['segments']), 10)
        self.assertIn('total_length_px', record['strip'])


class _DoublingScale:
    """A stand-in millimetre scale: doubles every coordinate.

    Linear, so the true answer is arithmetic rather than another projection, and
    deliberately *not* uniform in effect -- a doubling of coordinates doubles
    every distance, whereas a real projective scale would not, which is what the
    separate plane-scale tests cover.
    """

    def to_mm(self, points):
        return np.asarray(points, dtype=np.float64) * 2.0


class MetricTests(unittest.TestCase):
    def ring(self):
        return np.int32(make_ribbon(*curved_curve(), half_width=9.0)[0])

    def analysis(self):
        return sam_detection.analyze_detection(
            [sam_detection.Polygon('strip', 0.9, self.ring())], CAMERA_FRAME, 10)

    def metric_analysis(self):
        return sam_detection.analyze_detection(
            [sam_detection.Polygon('strip', 0.9, self.ring())],
            CAMERA_FRAME, 10, _DoublingScale()).analysis

    def test_a_scale_turns_the_measurement_metric(self):
        pixels = self.analysis()
        self.assertFalse(pixels.analysis.metric)

        metric = sam_detection.analyze_detection(
            [sam_detection.Polygon('strip', 0.9, self.ring())],
            CAMERA_FRAME, 10, _DoublingScale())

        self.assertTrue(metric.analysis.metric)
        # The pixel total is the arc of the smoothed path; the millimetre one is
        # the arc of the path that was actually mapped, so the two differ by the
        # resampling's own chord shortening -- a thousandth of a percent here.
        self.assertAlmostEqual(
            metric.analysis.total_length_mm, 2.0 * metric.analysis.total_length_px,
            delta=0.001 * metric.analysis.total_length_px)
        self.assertAlmostEqual(
            metric.analysis.average_width_mm,
            2.0 * metric.analysis.average_width_px, places=6)

    def test_each_segment_keeps_its_pixel_figures(self):
        pixels = self.analysis().analysis
        metric = sam_detection.analyze_detection(
            [sam_detection.Polygon('strip', 0.9, self.ring())],
            CAMERA_FRAME, 10, _DoublingScale()).analysis

        self.assertEqual(len(metric.segments), len(pixels.segments))
        for before, after in zip(pixels.segments, metric.segments):
            self.assertAlmostEqual(
                after.average_width_mm, 2.0 * before.average_width_px, places=6)
            # Proportional to the same total the pixel figure uses, so the two
            # agree to the total's own rounding rather than exactly.
            self.assertAlmostEqual(
                after.length_mm, 2.0 * before.length_px, delta=0.001 * before.length_px)
            # The pixel figures survive the conversion unchanged.
            self.assertAlmostEqual(
                after.average_width_px, before.average_width_px, places=9)
            self.assertAlmostEqual(after.length_px, before.length_px, places=9)

    def test_summary_widths_span_the_segments(self):
        metric = sam_detection.analyze_detection(
            [sam_detection.Polygon('strip', 0.9, self.ring())],
            CAMERA_FRAME, 10, _DoublingScale()).analysis

        # The summary spans the per-sample extremes the segments saw, so it
        # brackets the per-segment averages rather than equalling their min/max.
        averages = [s.average_width_mm for s in metric.segments]
        self.assertAlmostEqual(
            metric.minimum_width_mm,
            min(s.minimum_width_mm for s in metric.segments), places=6)
        self.assertAlmostEqual(
            metric.maximum_width_mm,
            max(s.maximum_width_mm for s in metric.segments), places=6)
        self.assertLessEqual(metric.minimum_width_mm, min(averages))
        self.assertGreaterEqual(metric.maximum_width_mm, max(averages))

    def test_a_missing_scale_leaves_the_measurement_in_pixels(self):
        analysis = self.analysis().analysis

        self.assertFalse(analysis.metric)
        self.assertIsNone(analysis.total_length_mm)
        self.assertIsNone(analysis.average_width_mm)
        self.assertIsNone(analysis.minimum_width_mm)
        self.assertIsNone(analysis.maximum_width_mm)
        for segment in analysis.segments:
            self.assertIsNone(segment.length_mm)
            self.assertIsNone(segment.average_width_mm)

    def test_metric_labels_read_in_millimetres_to_two_decimals(self):
        metric = sam_detection.analyze_detection(
            [sam_detection.Polygon('strip', 0.9, self.ring())],
            CAMERA_FRAME, 10, _DoublingScale()).analysis

        for segment in metric.segments:
            text = strip_analysis._label_text(segment)
            self.assertRegex(text, r'^\d+: \d+\.\d\dmm$')

    def test_pixel_labels_are_still_used_without_a_scale(self):
        for segment in self.analysis().analysis.segments:
            self.assertRegex(strip_analysis._label_text(segment), r'^\d+: [\d.]+px$')

    def test_metric_record_carries_both_units(self):
        metric = sam_detection.analyze_detection(
            [sam_detection.Polygon('strip', 0.9, self.ring())],
            CAMERA_FRAME, 10, _DoublingScale())
        record = sam_detection.strip_record(metric)

        self.assertTrue(record['metric'])
        for key in ('total_length_mm', 'average_width_mm',
                    'minimum_width_mm', 'maximum_width_mm'):
            self.assertIsNotNone(record[key])
            self.assertIsNotNone(record[key.replace('_mm', '_px')])
        for segment in record['segments']:
            self.assertIsNotNone(segment['average_width_mm'])
            self.assertIsNotNone(segment['average_width_px'])

    def test_pixel_record_leaves_the_millimetre_fields_null(self):
        record = sam_detection.strip_record(self.analysis())

        self.assertFalse(record['metric'])
        self.assertIsNone(record['total_length_mm'])
        self.assertIsNone(record['average_width_mm'])
        for segment in record['segments']:
            self.assertIsNone(segment['average_width_mm'])
            self.assertIsNone(segment['length_mm'])
            self.assertIsNotNone(segment['average_width_px'])


class DrawAnalysisTests(unittest.TestCase):
    def test_draw_analysis_survives_a_missing_measurement(self):
        crop = np.zeros((552, 2208, 3), np.uint8)
        ring = np.array([[10, 10], [400, 10], [400, 60], [10, 60]], np.int32)
        polygon = sam_detection.Polygon('strip', 0.9, ring)

        with_measurement = sam_detection.draw_analysis(crop, [polygon], None)
        self.assertEqual(with_measurement.shape, crop.shape)
        self.assertGreater(int(np.count_nonzero(with_measurement)), 0)


if __name__ == '__main__':
    unittest.main()

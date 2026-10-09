"""Pipeline Lab: pre-processing steps, their order, recipes and the Run job."""
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import pipeline_lab  # noqa: E402
import pipeline_steps as steps  # noqa: E402


def scene():
    """Pale fabric with one wavy glue bead: bright core, a dark line each side."""
    rng = np.random.default_rng(4)
    fabric = np.full((700, 500), 160.0)
    y = np.arange(60, 640, dtype=np.float64)
    centre = 250 + 50 * np.sin(2 * np.pi * (y - 60) / 580)
    for offset, value, thickness in ((0, 172.0, 8), (-8, 148.0, 3), (8, 148.0, 3)):
        points = np.column_stack([centre + offset, y]).round().astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(fabric, [points], False, value, thickness, cv2.LINE_AA)
    fabric += rng.normal(0, 0.8, fabric.shape)
    return cv2.cvtColor(np.clip(fabric, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


class StepTests(unittest.TestCase):
    def test_every_step_runs_and_keeps_the_picture_usable(self):
        image = scene()
        roi = np.zeros(image.shape[:2], np.uint8)
        roi[50:650, 100:400] = 1
        for name, step in steps.STEPS.items():
            entry = steps.new_entry(name)
            result, mask, _ = steps.run_pipeline(image, roi, [entry])
            self.assertEqual(result.dtype, np.uint8, name)
            self.assertEqual(result.ndim, 3, name)
            if step.resizes:
                self.assertEqual(result.shape[1], image.shape[1] * 2, name)
            else:
                self.assertEqual(result.shape, image.shape, name)
            self.assertEqual(mask.shape, result.shape[:2], name)

    def test_order_matters_and_disabled_steps_are_skipped(self):
        image = scene()
        quantise, sharpen = steps.new_entry('quantise'), steps.new_entry('sharpen')
        first, _, _ = steps.run_pipeline(image, None, [quantise, sharpen])
        second, _, _ = steps.run_pipeline(image, None, [sharpen, quantise])
        self.assertFalse(np.array_equal(first, second))
        off = dict(sharpen, enabled=False)
        only, _, stages = steps.run_pipeline(image, None, [quantise, off], keep_stages=True)
        alone, _, _ = steps.run_pipeline(image, None, [quantise])
        np.testing.assert_array_equal(only, alone)
        self.assertEqual(len(stages), 1)

    def test_quantise_leaves_the_requested_number_of_levels(self):
        entry = steps.new_entry('quantise')
        entry['params']['levels'] = 3
        result, _, _ = steps.run_pipeline(scene(), None, [entry])
        self.assertLessEqual(len(np.unique(result)), 3)

    def test_recipe_round_trips_and_unknown_steps_are_refused(self):
        recipe = {'roi': 'box', 'segmenter': 'sam3',
                  'pipeline': [steps.new_entry('denoise'), steps.new_entry('resize')]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'recipe.json'
            steps.save_recipe(path, recipe)
            self.assertEqual(steps.load_recipe(path), recipe)
            steps.save_recipe(path, {'pipeline': [{'step': 'nonsense'}]})
            with self.assertRaises(ValueError):
                steps.load_recipe(path)

    def test_ring_stats_report_a_ribbons_width(self):
        ring = np.array([[10, 10], [410, 10], [410, 30], [10, 30]], float)
        stats = steps.ring_stats(ring)
        self.assertAlmostEqual(stats['width_px'], 20, delta=3)
        self.assertGreater(stats['length_px'], 350)


class RunJobTests(unittest.TestCase):
    def recipe(self, **changes):
        base = {'roi': 'whole', 'pipeline': [], 'segmenter': 'local', 'sam2_size': 'hiera_large',
                'point_source': 'lines', 'prompt': 'thin glossy line'}
        base.update(changes)
        return base

    def test_local_detector_finds_the_bead_in_a_hand_drawn_box(self):
        image = scene()
        result = pipeline_lab.run_job(image, self.recipe(roi='box'), (120, 30, 400, 680), [])
        self.assertEqual(len(result['lines']), 1)
        self.assertAlmostEqual(result['lines'][0]['width_px'], 16, delta=5)
        self.assertEqual(result['overlay'].shape, image.shape)
        self.assertIn('1 line', pipeline_lab.describe(result))

    def test_a_resize_step_scales_the_result_and_the_clicked_points(self):
        image = scene()
        seen = {}

        class Client:
            def sam3_visual_segment(self, tile, prompts=None, multimask_output=False):
                seen['shape'], seen['points'] = tile.shape, prompts[0]['points']
                return {'predictions': [{'confidence': 0.9, 'masks': [
                    [[480, 200], [520, 200], [520, 1100], [480, 1100]]]}]}

        recipe = self.recipe(segmenter='sam3', point_source='clicks',
                             pipeline=[steps.new_entry('resize')])
        result = pipeline_lab.run_job(image, recipe, None, [(250, 300, True), (300, 300, False)],
                                      client=Client())
        self.assertEqual(result['scale'], 2.0)
        self.assertEqual(seen['shape'][:2], (1024, 1000))
        # A click at x=250 on the original is x=500 on the doubled picture.
        self.assertAlmostEqual(seen['points'][0]['x'], 500, delta=1)
        self.assertEqual([p['positive'] for p in seen['points']], [True, False])
        self.assertEqual(result['lines'][0]['source'], 'sam3')

    def test_sam2_is_called_with_the_chosen_size(self):
        calls = []

        class Client:
            def sam2_segment_image(self, tile, prompts=None, sam2_version_id=None,
                                   multimask_output=False):
                calls.append(sam2_version_id)
                return {'predictions': []}

        recipe = self.recipe(segmenter='sam2', sam2_size='hiera_small', point_source='clicks')
        result = pipeline_lab.run_job(scene(), recipe, None, [(250, 300, True)], client=Client())
        self.assertEqual(calls, ['hiera_small'])
        self.assertEqual(result['lines'], [])

    def test_clicking_no_glue_point_is_reported(self):
        recipe = self.recipe(segmenter='sam3', point_source='clicks')
        with self.assertRaisesRegex(ValueError, 'at least one point'):
            pipeline_lab.run_job(scene(), recipe, None, [(10, 10, False)], client=object())


class LeakRepairTests(unittest.TestCase):
    """SAM's mask as an even ribbon, with the stretches where it spilled repaired."""

    def setUp(self):
        from segmentation import sam_refine
        self.sam_refine = sam_refine
        y = np.arange(60, 900, dtype=np.float64)
        self.centre = np.column_stack([300 + 40 * np.sin(2 * np.pi * (y - 60) / 840), y])
        self.mask = np.zeros((960, 600), np.uint8)
        cv2.polylines(self.mask, [self.centre.round().astype(np.int32).reshape(-1, 1, 2)], False, 1, 30)

    def test_a_clean_ribbon_is_kept_as_it_is(self):
        ring, seen, widths = self.sam_refine.trim_ring(self.mask, self.centre)
        self.assertGreater(seen, 0.95)
        self.assertAlmostEqual(float(np.median(widths)), 30.0, delta=3.0)
        self.assertAlmostEqual(steps.ring_stats(ring)['width_px'], 30.0, delta=4.0)

    def test_a_spill_to_one_side_is_cut_back_to_the_ribbon(self):
        self.mask[300:480, 300:420] = 1                       # a blob beside the bead
        ring, seen, widths = self.sam_refine.trim_ring(self.mask, self.centre)
        self.assertLess(seen, 0.9)
        self.assertGreater(seen, 0.6)
        self.assertLess(float(widths.max()), 40.0)

    def test_a_line_taken_too_wide_is_corrected_with_the_usual_width(self):
        wide = np.zeros_like(self.mask)
        cv2.polylines(wide, [(self.centre - (15, 0)).round().astype(np.int32).reshape(-1, 1, 2)],
                      False, 1, 60)
        wide[200:500] = self.mask[200:500]                    # only this stretch is the bead
        _, _, own = self.sam_refine.trim_ring(wide, self.centre)
        self.assertGreater(float(np.median(own)), 50.0)
        _, seen, widths = self.sam_refine.trim_ring(wide, self.centre, expected_width=30.0)
        self.assertAlmostEqual(float(np.median(widths)), 30.0, delta=4.0)
        self.assertLess(seen, 0.75)          # one edge is shared with the wide outline

    def test_a_second_answer_fills_the_stretch_the_first_one_spilled_in(self):
        spilled = self.mask.copy()
        spilled[300:480, 300:420] = 1
        _, alone, _ = self.sam_refine.trim_ring(spilled, self.centre)
        other = self.mask.copy()
        other[600:760, 180:300] = 1                           # spills somewhere else
        _, both, widths = self.sam_refine.trim_ring([spilled, other], self.centre)
        self.assertGreater(both, alone)
        self.assertGreater(both, 0.95)
        self.assertLess(float(widths.max()), 40.0)

    def test_no_mask_near_the_line_gives_nothing(self):
        self.assertIsNone(self.sam_refine.trim_ring(np.zeros_like(self.mask), self.centre))


def pale_bead(shift=0.0, start=200, stop=1300, core=True):
    """Pale fabric with a bead as the camera shows it: a band 30 px wide and 6
    levels darker, a thin shiny core with a black line each side, and noise."""
    rng = np.random.default_rng(7)
    fabric = np.full((1500, 600), 200.0, np.float32)
    y = np.arange(start, stop, dtype=np.float64)
    middle = 300 + 40 * np.sin(2 * np.pi * (y - start) / (stop - start)) + shift
    points = np.column_stack([middle, y]).round().astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(fabric, [points], False, 194.0, 30)
    if core:
        cv2.polylines(fabric, [points[300:800]], False, 186.0, 12)
        cv2.polylines(fabric, [points[300:800]], False, 240.0, 6)
    fabric += rng.normal(0, 3.0, fabric.shape)
    image = cv2.cvtColor(np.clip(fabric, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    return image, np.column_stack([middle - shift, y])


class BeadEdgeTests(unittest.TestCase):
    """The bead's own edges on pale fabric, where it is only a few levels darker."""

    def setUp(self):
        from segmentation import bead_edges
        self.find = bead_edges.bead_edges

    def test_the_band_is_measured_through_noise_and_the_shiny_core(self):
        image, centre = pale_bead()
        ring, seen, widths = self.find(image, centre)
        self.assertGreater(seen, 0.9)
        self.assertAlmostEqual(float(np.median(widths)), 30.0, delta=3.0)
        self.assertLess(float(np.percentile(widths, 90) - np.percentile(widths, 10)), 5.0)

    def test_a_rough_line_lying_beside_the_bead_is_corrected(self):
        image, centre = pale_bead(shift=14.0)                 # the bead is 14 px off the line
        ring, _, widths = self.find(image, centre)
        half = len(ring) // 2
        found = (ring[:half] + ring[half:][::-1]) / 2.0
        rows = np.round(found[:, 1]).astype(int)
        truth = np.interp(rows, centre[:, 1], centre[:, 0] + 14.0)
        self.assertLess(float(np.median(np.abs(found[:, 0] - truth))), 3.0)
        self.assertAlmostEqual(float(np.median(widths)), 30.0, delta=3.0)

    def test_the_outline_stops_at_the_beads_ends(self):
        image, centre = pale_bead()
        longer = np.vstack([centre[0] - [[0, 80]], centre, centre[-1] + [[0, 80]]])   # overshoots
        shorter = centre[60:-60]                                                    # stops short
        for line in (longer, shorter):
            ring, _, _ = self.find(image, line)
            self.assertAlmostEqual(float(ring[:, 1].min()), 200.0, delta=30.0)
            self.assertAlmostEqual(float(ring[:, 1].max()), 1300.0, delta=30.0)

    def test_plain_fabric_gives_nothing(self):
        image, centre = pale_bead()
        image[:] = 200
        self.assertIsNone(self.find(image, centre))


class PlaceByPlaceTests(unittest.TestCase):
    """What the picture is like along the line, and the oriented filter."""

    def test_shiny_and_matte_stretches_and_the_ends_are_told_apart(self):
        from segmentation import bead_edges
        image, centre = pale_bead()                           # core on samples 300-800 of 1100
        longer = np.vstack([centre[0] - [[0, 80]], centre, centre[-1] + [[0, 80]]])
        found = bead_edges.measure(image, longer)
        rows = found['points'][:, 1]
        state = found['state']
        self.assertGreater(np.mean(state[(rows > 560) & (rows < 940)] == 'shiny'), 0.9)
        self.assertGreater(np.mean(state[(rows > 260) & (rows < 440)] == 'matte'), 0.9)
        self.assertGreater(np.mean(state[rows < 150] == 'none'), 0.9)

    def test_a_core_along_one_edge_is_called_one_sided(self):
        from segmentation import bead_edges
        rng = np.random.default_rng(3)
        fabric = np.full((1500, 600), 200.0, np.float32)
        fabric[200:1300, 285:315] = 194.0                     # the band
        fabric[500:1000, 309:315] = 240.0                     # the core, at the band's right edge
        fabric += rng.normal(0, 3.0, fabric.shape)
        image = cv2.cvtColor(np.clip(fabric, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        centre = np.column_stack([np.full(1100, 300.0), np.arange(200.0, 1300.0)])
        found = bead_edges.measure(image, centre)
        rows = found['points'][:, 1]
        self.assertGreater(np.mean(found['state'][(rows > 560) & (rows < 940)] == 'one-sided'), 0.8)

    def test_smoothing_along_the_line_removes_weave_and_keeps_the_beads_edges(self):
        image, centre = pale_bead(core=False)
        entry = steps.new_entry('smooth_along')
        result, _, _ = steps.run_pipeline(image, None, [entry])
        self.assertEqual(result.shape, image.shape)
        self.assertLess(float(result[100:180, 50:200].std()), 0.6 * float(image[100:180, 50:200].std()))
        row = result[750, :, 0].astype(float)                 # across the bead
        middle = int(round(np.interp(750, centre[:, 1], centre[:, 0])))
        self.assertGreater(row[middle - 40:middle - 25].mean() - row[middle - 8:middle + 8].mean(), 3.0)


class RibbonTests(unittest.TestCase):
    """One smooth, even ribbon from jagged edge evidence."""

    def setUp(self):
        from segmentation import ribbon
        self.ribbon = ribbon
        y = np.arange(0.0, 1600.0, 2.0)
        self.base = np.column_stack([300 + 60 * np.sin(2 * np.pi * y / 1600), y])
        self.normals = ribbon._frame(self.base)

    def evidence(self, half_width, wobble=0.0, seed=0):
        rng = np.random.default_rng(seed)
        rows = []
        for sign in (-1, 1):
            distance = sign * half_width + rng.normal(0, wobble, len(self.base))
            rows.append(np.column_stack([self.base + self.normals * np.reshape(distance, (-1, 1)),
                                         np.ones(len(self.base))]))
        return rows

    def test_wobbly_evidence_gives_a_smooth_ribbon_of_the_right_width(self):
        low, high = self.evidence(np.full(len(self.base), 15.0), wobble=3.0)
        fitted = self.ribbon.fit_ribbon(self.base, low, high)
        self.assertAlmostEqual(float(np.median(fitted['widths'])), 30.0, delta=1.0)
        self.assertLess(float(np.ptp(fitted['widths'][50:-50])), 2.0)
        self.assertLess(float(np.abs(fitted['centre'] - self.base).max()), 2.0)
        self.assertEqual(fitted['deviations'], [])

    def test_a_notch_and_a_spill_do_not_pull_the_ribbon(self):
        low, high = self.evidence(np.full(len(self.base), 15.0), wobble=1.0)
        high[300:330, :2] -= self.normals[300:330] * 12.0     # a notch inwards, 60 px long
        low[500:520, :2] -= self.normals[500:520] * 30.0      # a spill outwards, 40 px long
        fitted = self.ribbon.fit_ribbon(self.base, low, high)
        self.assertLess(float(np.abs(fitted['widths'] - 30.0)[50:-50].max()), 2.0)

    def test_a_long_real_bulge_is_reported_not_smoothed_away(self):
        half = np.full(len(self.base), 15.0)
        half[350:450] = 21.0                                  # 12 px wider over 200 px
        low, high = self.evidence(half, wobble=1.0)
        fitted = self.ribbon.fit_ribbon(self.base, low, high)
        self.assertEqual(len(fitted['deviations']), 1)
        first, last, pixels = fitted['deviations'][0]
        self.assertTrue(330 < first < 380 and 420 < last < 470)
        self.assertGreater(pixels, 5.0)

    def test_too_little_evidence_gives_nothing(self):
        low, high = self.evidence(np.full(len(self.base), 15.0))
        self.assertIsNone(self.ribbon.fit_ribbon(self.base, low[:10], high[:10]))


class LocalSamTests(unittest.TestCase):
    """SAM on this computer: chosen by name, a tile encoded once, no cloud call."""

    class Model:
        """Stands in for a local SAM model: paints a 30 px ribbon through the points."""

        def __init__(self):
            self.encodes = 0
            self.asks = 0

        def open(self, tile):
            self.encodes += 1
            return {'size': tile.shape[:2]}

        def ask(self, opened, points, multimask=False):
            self.asks += 1
            mask = np.zeros(opened['size'], np.uint8)
            chosen = [(int(point['x']), int(point['y'])) for point in points if point.get('positive', True)]
            cv2.polylines(mask, [np.array(chosen, np.int32).reshape(-1, 1, 2)], False, 1, 30)
            return [(mask, 0.9)]

    def setUp(self):
        from segmentation import sam_local, sam_refine
        self.sam_local, self.sam_refine = sam_local, sam_refine
        self.model = self.Model()
        self.kept = dict(sam_local._loaded)
        sam_local._loaded['stub'] = self.model
        sam_local._last.update(tile=None, opened=None, model=None)

    def tearDown(self):
        self.sam_local._loaded.clear()
        self.sam_local._loaded.update(self.kept)

    def test_a_local_name_goes_to_the_local_model_and_never_to_the_cloud(self):
        class Cloud:
            def __getattr__(self, name):
                raise AssertionError('the cloud client was used')
        tile = np.zeros((400, 400, 3), np.uint8)
        prompt = [{'points': [{'x': 100.0, 'y': 50.0, 'positive': True},
                              {'x': 100.0, 'y': 350.0, 'positive': True}]}]
        reply = self.sam_refine.segment_points(Cloud(), tile, prompt, 'local:stub')
        mask, confidence = self.sam_refine.reply_mask(reply, tile.shape)
        self.assertAlmostEqual(confidence, 0.9)
        self.assertGreater(int(mask[200, 100]), 0)
        self.assertEqual(int(mask[200, 300]), 0)

    def test_a_tile_is_encoded_once_however_often_it_is_asked(self):
        tile = np.zeros((400, 400, 3), np.uint8)
        prompt = [{'points': [{'x': 100.0, 'y': 50.0, 'positive': True},
                              {'x': 100.0, 'y': 350.0, 'positive': True}]}]
        for _ in range(3):
            self.sam_refine.segment_points(None, tile, prompt, 'local:stub')
        self.assertEqual((self.model.encodes, self.model.asks), (1, 3))
        self.sam_refine.segment_points(None, tile.copy(), prompt, 'local:stub')
        self.assertEqual(self.model.encodes, 2)

    def test_the_run_job_uses_the_local_model_and_says_nothing_was_uploaded(self):
        image, centre = pale_bead()
        box = (150, 100, 450, 1400)
        recipe = {'roi': 'box', 'pipeline': [], 'segmenter': 'sam3', 'point_source': 'lines',
                  'sam_runs': 'local', 'local_model': 'stub', 'ask_once': True, 'ribbon': True}
        rough = [{'ring': np.vstack([centre - (6, 0), (centre + (6, 0))[::-1]]), 'source': 'local',
                  'confidence': 1.0}]
        original = steps.segment_local
        steps.segment_local = lambda *args, **kwargs: [dict(line) for line in rough]
        try:
            result = pipeline_lab.run_job(image, recipe, box, [])
        finally:
            steps.segment_local = original
        self.assertEqual(len(result['lines']), 1)
        self.assertTrue(result['lines'][0]['source'].startswith('local:stub'))
        self.assertAlmostEqual(result['lines'][0]['width_px'], 30.0, delta=4.0)
        self.assertTrue(any('nothing was uploaded' in note for note in result['notes']))
        self.assertGreater(self.model.encodes, 0)

    def test_an_unknown_local_model_is_refused(self):
        with self.assertRaises(ValueError):
            self.sam_local.get('no-such-model')


if __name__ == '__main__':
    unittest.main()

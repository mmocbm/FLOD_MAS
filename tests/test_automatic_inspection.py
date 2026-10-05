"""Automatic workflow contracts, real geometry, storage and asynchronous lifecycle."""
import copy
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import cv2
import numpy as np
import plane_scale
from auto_trigger.checkpoint import Checkpoint, Debouncer
from auto_trigger.controller import AutoController
from auto_trigger.models import FabricGate
from inspection.result_view import FabricResultView
from segmentation import Instance, SegmentationResult, validate_result, mask_to_polygons
from segmentation.provider import WorkflowSegmenter
from inspection.storage import CaptureStore
from inspection.measurement import measure_instances
from inspection.pipeline import InspectionPipeline, keep_largest_fragments
from config_validation import validate_config

CONFIG = json.loads((Path(__file__).resolve().parents[1] / 'config.json').read_text())


class TriggerTests(unittest.TestCase):
    def test_confirmation_is_configurable_and_reentry_cancels(self):
        checkpoint = Checkpoint(2.5, .1)
        self.assertFalse(checkpoint.update(False, 0))
        checkpoint.update(True, 1)
        checkpoint.update(False, 2)
        self.assertFalse(checkpoint.update(False, 4.4))
        checkpoint.update(True, 4.45)
        checkpoint.update(False, 5)
        self.assertFalse(checkpoint.update(False, 7.4))
        self.assertTrue(checkpoint.update(False, 7.5))
        self.assertFalse(checkpoint.update(False, 8))

    def test_half_and_full_accepted_but_low_confidence_empty_does_not_clear(self):
        gate = FabricGate.__new__(FabricGate)
        gate.settings = CONFIG['auto_trigger']
        gate.labels = ['full_fabric', 'half_fabric', 'no_fabric']
        frame = np.zeros((32, 32, 3), np.uint8)
        for scores, accepted, empty in [([.9, .05, .05], True, False),
                                        ([.05, .9, .05], True, False),
                                        ([.05, .05, .9], False, True),
                                        ([.3, .3, .4], False, False)]:
            gate.model = lambda *args, s=scores, **kwargs: np.array([s])
            result = gate.classify(frame)
            self.assertEqual((result['accepted'], result['empty']), (accepted, empty))


class SegmentationTests(unittest.TestCase):
    def test_nonuniform_resize_maps_back_to_original_frame(self):
        cfg = dict(CONFIG['segmentation'], input_width=100, input_height=80)
        provider = WorkflowSegmenter(cfg, client=object())
        response = SimpleNamespace(image_width=100, image_height=80,
            panel_boxes=[(0, 0, 99, 79)], strips=[SimpleNamespace(
                points=[(10, 10), (30, 10), (30, 20), (10, 20)], confidence=.95, class_name='strip')])
        frame = np.zeros((240, 500, 3), np.uint8)
        with patch('segmentation.provider.workflow.run_workflow', return_value=response) as run:
            result = provider.segment(frame)
        self.assertEqual(run.call_args.args[0].shape, (80, 100, 3))
        np.testing.assert_allclose(result.instances[0].polygon,
                                   [[50, 30], [150, 30], [150, 60], [50, 60]])
        self.assertIs(validate_result(result, frame), result)

    def test_local_mask_adapter_and_coordinate_validation(self):
        mask = np.zeros((80, 100), np.uint8)
        mask[10:30, 20:60] = 1
        rings = mask_to_polygons(mask)
        self.assertEqual(len(rings), 1)
        frame = np.zeros((80, 100, 3), np.uint8)
        with self.assertRaises(ValueError):
            validate_result(SegmentationResult((Instance(rings[0]),), (50, 40)), frame)


class MeasurementTests(unittest.TestCase):
    def test_four_strips_use_existing_metric_algorithm(self):
        frame = np.zeros((300, 1000, 3), np.uint8)
        instances = tuple(Instance(np.array([[40+i*240, 40], [80+i*240, 40],
                                             [80+i*240, 240], [40+i*240, 240]], float))
                          for i in range(4))
        scale = plane_scale.PlaneScale(np.diag([.1, .1, 1]), .1)
        result = measure_instances(frame, SegmentationResult(instances, (1000, 300)),
                                   scale, CONFIG['sam_detection'], 4, 1)
        self.assertEqual(len(result), 4)
        for fabric in result:
            self.assertEqual(fabric.record['status'], 'PASS')
            metric = fabric.record['measurement']
            self.assertEqual(len(metric['segments']), 10)
            self.assertAlmostEqual(metric['average_width_mm'], 4, delta=.2)

    def test_full_frame_plane_recovers_known_millimetres(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            camera = root / 'camera.json'
            surface = root / 'surface.json'
            camera.write_text(json.dumps({'camera_matrix': [[1000,0,500],[0,1000,400],[0,0,1]],
                                          'dist_coeffs': [0]*5, 'image_size': [1000,800]}))
            surface.write_text(json.dumps({'rvec': [0,0,0], 'tvec': [0,0,1]}))
            scale = plane_scale.load_frame_scale(1, (2000,1600), camera, surface)
            np.testing.assert_allclose(scale.to_mm(np.array([[1000,800],[1200,1000]], float)),
                                       [[0,0],[100,100]], atol=.001)


class StorageTests(unittest.TestCase):
    def test_lossless_sources_jpeg_results_and_whole_set_retention(self):
        with tempfile.TemporaryDirectory() as directory:
            store = CaptureStore(directory, dict(CONFIG['capture_storage'], max_sets=2))
            frame = np.random.default_rng(7).integers(0, 256, (40, 60, 3), dtype=np.uint8)
            paths = []
            unrelated = Path(directory) / 'keep_me'
            unrelated.mkdir()
            for index in range(3):
                path = store.begin(frame, frame, {'capture': index})
                store.save_preview(path, 'fabric_01', frame)
                store.finish(path, {'status': 'complete'})
                paths.append(path)
            self.assertFalse(paths[0].exists())
            self.assertTrue(unrelated.exists())
            self.assertEqual(len(store._sets()), 2)
            np.testing.assert_array_equal(cv2.imread(str(paths[-1]/'original.png')), frame)
            self.assertTrue((paths[-1]/'undistorted.png').is_file())
            self.assertTrue((paths[-1]/'fabric_01.jpg').is_file())

    def test_inflight_set_is_never_pruned(self):
        with tempfile.TemporaryDirectory() as directory:
            store = CaptureStore(directory, dict(CONFIG['capture_storage'], max_sets=1))
            frame = np.zeros((12, 12, 3), np.uint8)
            path = store.begin(frame, frame, {})
            with self.assertRaises(OSError):
                store.begin(frame, frame, {})
            self.assertTrue(path.is_dir())
            self.assertEqual(len(store._sets()), 1)

    def test_restart_finalizes_interrupted_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = dict(CONFIG['capture_storage'], max_sets=2)
            store = CaptureStore(directory, settings)
            path = store.begin(np.zeros((12,12,3), np.uint8), None, {})
            CaptureStore(directory, settings)
            self.assertEqual(json.loads((path/'result.json').read_text())['status'], 'interrupted')


class PipelineTests(unittest.TestCase):
    def test_workflow_error_keeps_lossless_capture_and_error_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            store = CaptureStore(directory, CONFIG['capture_storage'])
            class Broken:
                def segment(self, frame):
                    raise RuntimeError('workflow offline')
            pipeline = InspectionPipeline(Broken(), store, CONFIG)
            frame = np.zeros((40, 60, 3), np.uint8)
            with self.assertRaisesRegex(RuntimeError, 'workflow offline'):
                pipeline.run(frame, frame, {}, 4, 1)
            path = store._sets()[0]
            self.assertTrue((path/'original.png').exists())
            record = json.loads((path/'result.json').read_text())
            self.assertEqual(record['status'], 'error')
            self.assertEqual(record['error'], 'workflow offline')


class ConfigTests(unittest.TestCase):
    def test_defaults_valid_and_invalid_retention_delay_rejected(self):
        validate_config(CONFIG)
        for section, key, value in [('capture_storage','max_sets',0),
                                    ('result_view','show_live_preview','false'),
                                    ('auto_trigger','hand_absence_seconds',float('nan')),
                                    ('capture_storage','directory','../other')]:
            config = copy.deepcopy(CONFIG)
            config[section][key] = value
            with self.assertRaises(ValueError):
                validate_config(config)


def _ribbon(x, top=40, bottom=240, width=40):
    return np.array([[x, top], [x+width, top], [x+width, bottom], [x, bottom]], float)


class LowContrastOptionTests(unittest.TestCase):
    """Opt-in handling for pale fabric. Every option is off in the shipped config."""

    def response(self, strips, boxes):
        return SimpleNamespace(image_width=100, image_height=80, panel_boxes=boxes, strips=[
            SimpleNamespace(points=points, confidence=confidence, class_name='strip')
            for points, confidence in strips])

    def segment(self, response, frame, **options):
        cfg = dict(CONFIG['segmentation'], input_width=100, input_height=80, **options)
        provider = WorkflowSegmenter(cfg, client=object())
        with patch('segmentation.provider.workflow.run_workflow', return_value=response) as run:
            return provider.segment(frame), run, provider

    def test_shipped_config_keeps_every_new_option_off(self):
        segmentation, strip = CONFIG['segmentation'], CONFIG['sam_detection']
        self.assertEqual(segmentation['min_confidence'], 0.0)
        self.assertFalse(segmentation['report_missing_strips'])
        self.assertEqual(segmentation['fragment_policy'], 'error')
        self.assertEqual(segmentation['roi'], [0.0, 0.0, 1.0, 1.0])
        self.assertFalse(segmentation['enhance']['enabled'])
        self.assertFalse(segmentation['save_input'])
        self.assertFalse(strip['refine_edges'])
        self.assertFalse(CONFIG['capture']['controls']['enabled'])

    def test_settings_without_the_new_keys_behave_as_before(self):
        legacy = {key: CONFIG['segmentation'][key] for key in (
            'provider', 'convert_to_rgb', 'timeout_seconds', 'max_retries', 'max_fabrics')}
        response = self.response([([(10, 10), (30, 10), (30, 20), (10, 20)], .95)],
                                 [(0, 0, 99, 79), (50, 0, 99, 79)])
        provider = WorkflowSegmenter(dict(legacy, input_width=100, input_height=80),
                                     client=object())
        frame = np.zeros((240, 500, 3), np.uint8)
        with patch('segmentation.provider.workflow.run_workflow', return_value=response) as run:
            result = provider.segment(frame)
        self.assertEqual(len(result.instances), 1)
        np.testing.assert_array_equal(run.call_args.args[0], np.zeros((80, 100, 3), np.uint8))
        np.testing.assert_allclose(result.instances[0].polygon,
                                   [[50, 30], [150, 30], [150, 60], [50, 60]])

    def test_roi_crop_maps_polygons_and_boxes_back_to_the_full_frame(self):
        frame = np.zeros((400, 1000, 3), np.uint8)
        frame[100:300, 200:700] = 200
        response = self.response([([(10, 10), (30, 10), (30, 20), (10, 20)], .9)],
                                 [(0, 0, 100, 80)])
        result, run, _ = self.segment(response, frame, roi=[0.2, 0.25, 0.7, 0.75])
        # Only the region of interest was sent, at the configured input size.
        self.assertEqual(run.call_args.args[0].shape, (80, 100, 3))
        self.assertTrue((run.call_args.args[0] == 200).all())
        np.testing.assert_allclose(result.instances[0].polygon,
                                   [[250, 125], [350, 125], [350, 150], [250, 150]])
        np.testing.assert_allclose(result.instances[0].panel_box, (200, 100, 700, 300))
        self.assertIs(validate_result(result, frame), result)

    def test_enhancement_changes_only_the_image_sent(self):
        rng = np.random.default_rng(3)
        frame = rng.integers(150, 165, (240, 500, 3), dtype=np.uint8)
        untouched = frame.copy()
        response = self.response([], [])
        _, plain, _ = self.segment(response, frame)
        _, enhanced, provider = self.segment(
            response, frame, enhance={'enabled': True, 'clip_limit': 3.0, 'tile_grid': 4})
        self.assertGreater(float(enhanced.call_args.args[0].std()),
                           float(plain.call_args.args[0].std()))
        np.testing.assert_array_equal(frame, untouched)
        np.testing.assert_array_equal(provider.last_input, enhanced.call_args.args[0])

    def test_panel_without_a_strip_is_reported_only_when_asked(self):
        frame = np.zeros((240, 500, 3), np.uint8)
        response = self.response([([(10, 10), (30, 10), (30, 20), (10, 20)], .9)],
                                 [(0, 0, 49, 79), (50, 0, 99, 79)])
        hidden, _, _ = self.segment(response, frame)
        self.assertEqual(len(hidden.instances), 1)
        shown, _, _ = self.segment(response, frame, report_missing_strips=True)
        self.assertEqual([i.polygon is None for i in shown.instances], [False, True])
        self.assertIs(validate_result(shown, frame), shown)
        fabrics = measure_instances(frame, shown, None, CONFIG['sam_detection'], 4, 1)
        self.assertEqual(fabrics[1].record['status'], 'NO STRIP')
        self.assertIsNone(fabrics[1].record['measurement'])
        self.assertGreater(fabrics[1].image.size, 0)
        json.dumps(fabrics[1].record, allow_nan=False)

    def test_a_missing_strip_instance_needs_its_panel_box(self):
        frame = np.zeros((80, 100, 3), np.uint8)
        with self.assertRaises(ValueError):
            validate_result(SegmentationResult((Instance(None),), (100, 80)), frame)

    def pipeline_run(self, instances, **options):
        frame = np.zeros((300, 1000, 3), np.uint8)
        class Fixed:
            last_input = np.full((8, 8, 3), 90, np.uint8)
            def segment(self, image):
                return SegmentationResult(tuple(instances), (1000, 300))
        config = copy.deepcopy(CONFIG)
        config['segmentation'].update(options)
        scale = plane_scale.PlaneScale(np.diag([.1, .1, 1]), .1)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        store = CaptureStore(self.directory.name, config['capture_storage'])
        with patch('inspection.pipeline.plane_scale.load_frame_scale', return_value=scale):
            outcome = InspectionPipeline(Fixed(), store, config).run(frame, frame, {}, 4, 1)
        return outcome, store._sets()[0]

    def fragmented(self):
        box_a, box_b = (20, 20, 100, 260), (260, 20, 340, 260)
        return [Instance(_ribbon(40), .9, 'strip', box_a),
                Instance(_ribbon(40, 250, 258, 10), .9, 'strip', box_a),
                Instance(_ribbon(280), .9, 'strip', box_b)]

    def test_fragments_still_stop_the_capture_by_default(self):
        with self.assertRaisesRegex(ValueError, 'Multiple strip fragments'):
            self.pipeline_run(self.fragmented())

    def test_largest_fragment_policy_keeps_the_other_fabrics_graded(self):
        outcome, _ = self.pipeline_run(self.fragmented(), fragment_policy='largest')
        states = [fabric.record['status'] for fabric in outcome['fabrics']]
        self.assertEqual(states, ['FRAGMENTED', 'PASS'])
        self.assertEqual(outcome['fabrics'][0].record['note'], 'fragmented')
        self.assertTrue(any('pieces' in warning for warning in outcome['warnings']))
        merged = keep_largest_fragments(SegmentationResult(tuple(self.fragmented()), (1000, 300)))
        np.testing.assert_array_equal(merged.instances[0].polygon, _ribbon(40))

    def test_low_confidence_strip_is_flagged_instead_of_graded(self):
        strips = [Instance(_ribbon(40), .30), Instance(_ribbon(280), .90)]
        graded, _ = self.pipeline_run(strips)
        self.assertEqual([f.record['status'] for f in graded['fabrics']], ['PASS', 'PASS'])
        self.assertEqual(graded['warnings'], [])
        flagged, _ = self.pipeline_run(strips, min_confidence=.5)
        self.assertEqual([f.record['status'] for f in flagged['fabrics']],
                         ['LOW CONFIDENCE', 'PASS'])
        self.assertTrue(flagged['warnings'])

    def test_model_input_is_saved_only_when_asked(self):
        strips = [Instance(_ribbon(40), .9)]
        _, default_set = self.pipeline_run(strips)
        self.assertFalse((default_set / 'segmentation_input.jpg').exists())
        _, audited_set = self.pipeline_run(strips, save_input=True)
        self.assertTrue((audited_set / 'segmentation_input.jpg').is_file())

    def test_default_record_carries_no_new_keys(self):
        outcome, _ = self.pipeline_run([Instance(_ribbon(40), .9)])
        self.assertEqual(set(outcome['fabrics'][0].record), {
            'fabric', 'status', 'polygon', 'confidence', 'label', 'display_box',
            'measurement_coordinates', 'measurement'})

    def test_new_options_are_validated(self):
        for section, key, value in [('segmentation', 'min_confidence', 1.5),
                                    ('segmentation', 'fragment_policy', 'merge'),
                                    ('segmentation', 'roi', [0.5, 0.0, 0.4, 1.0]),
                                    ('segmentation', 'roi', None),
                                    ('segmentation', 'report_missing_strips', 'yes'),
                                    ('segmentation', 'enhance', {'enabled': True, 'tile_grid': 0}),
                                    ('sam_detection', 'refine_edges', 1),
                                    ('sam_detection', 'refine_search_px', 0),
                                    ('capture', 'controls', {'enabled': 'no'}),
                                    ('capture', 'controls', {'enabled': True, 'exposure': 'low'})]:
            config = copy.deepcopy(CONFIG)
            config[section][key] = value
            with self.assertRaises(ValueError, msg=f'{section}.{key}={value!r}'):
                validate_config(config)
        legacy = copy.deepcopy(CONFIG)
        for key in ('min_confidence', 'report_missing_strips', 'fragment_policy', 'roi',
                    'enhance', 'save_input'):
            del legacy['segmentation'][key]
        del legacy['sam_detection']['refine_edges'], legacy['sam_detection']['refine_search_px']
        del legacy['capture']['controls']
        validate_config(legacy)


class GlueLineDetectorTests(unittest.TestCase):
    """The local detector for pale fabric: a glue bead is a thin bright line
    with a thin dark line along each edge."""

    PROVIDER = 'segmentation.glue_line:GlueLineSegmenter'

    def scene(self, strip=True, bead=True, stray_line=True, gap=16, fade=None):
        rng = np.random.default_rng(5)
        frame = np.full((1400, 1000, 3), 235, np.uint8)             # white table
        frame[80:1320, 60:940] = (110, 150, 190)                     # cardboard
        fabric = np.full((1100, 700), 160.0)
        y = np.arange(60, 1040, dtype=np.float64)
        centre = 300 + 70 * np.sin(2 * np.pi * (y - 60) / 980)
        def line(x, value, thickness=3):
            points = np.column_stack([x, y]).round().astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(fabric, [points], False, value, thickness, cv2.LINE_AA)
        if strip:
            if bead:
                line(centre, 172.0, max(3, gap - 8))                 # the glossy bead
            line(centre - gap / 2, 148.0)                            # its two edges
            line(centre + gap / 2, 148.0)
        if stray_line:
            line(centre + 260, 148.0)                                # a fold: one line only
        if fade is not None:
            # A stretch where the bead and both edges fade towards the fabric, so
            # the detector loses it there and has to join across it.
            flat = slice(560, 660)
            fabric[flat] = 160.0 + (fabric[flat] - 160.0) * fade
        fabric += rng.normal(0, 0.8, fabric.shape)
        frame[150:1250, 150:850] = np.clip(fabric, 0, 255)[:, :, None]
        return frame

    def segment(self, frame, **options):
        settings = dict(CONFIG['segmentation'], provider=self.PROVIDER)
        settings['glue_line'] = dict(settings['glue_line'], min_length_px=400.0, **options)
        from segmentation import create_segmenter
        return create_segmenter(settings).segment(frame)

    def width_of(self, frame):
        fabrics = measure_instances(frame, self.segment(frame), None,
                                    CONFIG['sam_detection'], 4, 1)
        return fabrics[0].record['measurement']

    def test_a_bead_between_two_edge_lines_is_one_strip_of_the_right_width(self):
        frame = self.scene()
        result = self.segment(frame)
        self.assertIs(validate_result(result, frame), result)
        self.assertEqual(len(result.instances), 1)
        self.assertGreater(result.instances[0].confidence, 0.9)
        measurement = self.width_of(frame)
        self.assertAlmostEqual(measurement['average_width_px'], 16, delta=3)
        self.assertGreater(measurement['total_length_px'], 800)

    def test_width_follows_the_distance_between_the_edges(self):
        for gap in (12, 24):
            width = self.width_of(self.scene(gap=gap, stray_line=False))['average_width_px']
            self.assertAlmostEqual(width, gap, delta=3, msg=f'gap {gap}')

    def test_a_single_line_is_not_a_strip(self):
        self.assertEqual(len(self.segment(self.scene(strip=False)).instances), 0)

    def test_two_lines_without_a_bead_between_them_are_not_a_strip(self):
        self.assertEqual(len(self.segment(self.scene(bead=False)).instances), 0)

    def test_dark_fabric_and_empty_table_give_nothing(self):
        dark = self.scene()
        dark[150:1250, 150:850] //= 4
        self.assertEqual(len(self.segment(dark).instances), 0)
        self.assertEqual(len(self.segment(np.full((600, 800, 3), 235, np.uint8)).instances), 0)

    def test_edges_farther_apart_than_the_widest_strip_are_not_paired(self):
        frame = self.scene(gap=24, stray_line=False)
        self.assertEqual(len(self.segment(frame, max_width_px=16.0).instances), 0)

    def test_a_faded_stretch_is_measured_rather_than_guessed(self):
        # The bead fades far enough for the detector to lose it and break the run
        # into two pieces, but its edges are still there: the gap between them is
        # measured, so it counts as seen and the width survives it.
        frame = self.scene(fade=0.10)
        tracked, guessed = self.segment(frame), self.segment(frame, track_gaps=False)
        self.assertGreater(tracked.instances[0].confidence,
                           guessed.instances[0].confidence)
        self.assertAlmostEqual(self.width_of(frame)['average_width_px'], 16, delta=2)

    def test_a_stretch_with_no_edges_left_is_still_only_a_guess(self):
        # Nothing of the bead is left at all: the two pieces are joined by a
        # straight line and confidence has to fall for it, exactly as it always did.
        frame = self.scene(fade=0.0)
        self.assertLess(self.segment(frame).instances[0].confidence, 0.95)
        self.assertAlmostEqual(self.segment(frame).instances[0].confidence,
                               self.segment(frame, track_gaps=False)
                               .instances[0].confidence, places=6)

    def test_detector_runs_through_the_inspection_pipeline(self):
        frame = self.scene()
        config = copy.deepcopy(CONFIG)
        config['segmentation']['provider'] = self.PROVIDER
        config['segmentation']['save_input'] = True
        config['segmentation']['glue_line']['min_length_px'] = 400.0
        validate_config(config)
        from segmentation import create_segmenter
        scale = plane_scale.PlaneScale(np.diag([.25, .25, 1]), .25)
        with tempfile.TemporaryDirectory() as directory:
            store = CaptureStore(directory, config['capture_storage'])
            pipeline = InspectionPipeline(create_segmenter(config['segmentation']), store, config)
            with patch('inspection.pipeline.plane_scale.load_frame_scale', return_value=scale):
                outcome = pipeline.run(frame, frame, {}, 4, 1)
            self.assertEqual([f.record['status'] for f in outcome['fabrics']], ['PASS'])
            self.assertAlmostEqual(
                outcome['fabrics'][0].record['measurement']['average_width_mm'], 4.0, delta=0.8)
            self.assertTrue((store._sets()[0] / 'segmentation_input.jpg').is_file())

    def test_shipped_provider_is_still_the_workflow(self):
        self.assertEqual(CONFIG['segmentation']['provider'], 'workflow')
        config = copy.deepcopy(CONFIG)
        config['segmentation']['glue_line']['min_width_px'] = 200.0
        with self.assertRaises(ValueError):
            validate_config(config)


class PreviewOptionTests(unittest.TestCase):
    def test_disabled_preview_does_not_resize_or_access_camera_widgets(self):
        view = FabricResultView.__new__(FabricResultView)
        view.show_live_preview = False
        with patch('inspection.result_view.cv2.resize') as resize:
            view.update_live_preview(np.zeros((40,60,3), np.uint8))
        resize.assert_not_called()


class ControllerTests(unittest.TestCase):
    def test_next_arrangement_is_processed_after_busy_request_without_another_trigger(self):
        cfg = copy.deepcopy(CONFIG)
        cfg['auto_trigger'].update(debounce_frames=1, hand_absence_seconds=.05)
        started, release = threading.Event(), threading.Event()
        processed = []
        class Hand:
            def __init__(self, settings): pass
            def present(self, frame): return bool(frame[0,0,0])
            def close(self): pass
        class Gate:
            def __init__(self, settings): pass
            def classify(self, frame):
                return {'accepted': True, 'empty': False, 'label': 'half_fabric', 'confidence': .99}
        class Pipeline:
            def run(self, original, *args):
                count = int(original[0,0,1])
                processed.append(count)
                if len(processed) == 1:
                    started.set()
                    release.wait(4)
                return {'fabrics': list(range(count))}
        controller = AutoController(cfg, Pipeline, Hand, Gate)
        controller.start()
        sequence = 0
        def feed(hand, count):
            nonlocal sequence
            sequence += 1
            frame = np.zeros((20,20,3), np.uint8)
            frame[:,:,0], frame[:,:,1] = int(hand), count
            controller.submit(frame, sequence, SimpleNamespace(undistort=lambda f: f.copy()),
                              {'size':'M','width':4,'tolerance':1})
            time.sleep(.08)
        try:
            for present in (True, False, False): feed(present, 1)
            self.assertTrue(started.wait(2))
            time.sleep(.12)
            for present in (True, True, False, False): feed(present, 2)
            self.assertIsNotNone(controller.pending_job)
            self.assertEqual(processed, [1])
            release.set()
            results = []
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline and not results:
                results += [payload for event, _, payload in controller.poll() if event == 'result']
                time.sleep(.02)
            self.assertEqual(processed, [1, 2])
            self.assertEqual([len(result['fabrics']) for result in results], [2])
        finally:
            release.set()
            controller.stop()
            controller.join(2)

    def test_save_all_keeps_sources_when_classifier_fails(self):
        cfg = copy.deepcopy(CONFIG)
        cfg['auto_trigger'].update(debounce_frames=1, hand_absence_seconds=.05)
        cfg['capture_storage']['save_rejected_triggers'] = True
        class Hand:
            def __init__(self, settings): pass
            def present(self, frame): return bool(frame[0,0,0])
            def close(self): pass
        class Gate:
            def __init__(self, settings): pass
            def classify(self, frame): raise RuntimeError('classifier failed')
        with tempfile.TemporaryDirectory() as directory:
            store = CaptureStore(directory, cfg['capture_storage'])
            controller = AutoController(cfg, lambda: SimpleNamespace(store=store), Hand, Gate)
            controller.start()
            try:
                for seq, value in enumerate((True, False, False)):
                    controller.submit(np.full((20,20,3), value, np.uint8), seq,
                        SimpleNamespace(undistort=lambda f: f.copy()), {'size':'M','width':4,'tolerance':1})
                    time.sleep(.1)
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline and not store._sets():
                    time.sleep(.02)
                controller.stop()
                controller.join(2)
                path = store._sets()[0]
                self.assertTrue((path/'original.png').is_file())
                self.assertTrue((path/'undistorted.png').is_file())
                record = json.loads((path/'result.json').read_text())
                self.assertEqual(record['error'], 'classifier failed')
            finally:
                controller.stop()
                controller.join(2)

    def test_empty_after_hand_cycle_clears_and_old_work_cannot_restore_tabs(self):
        cfg = copy.deepcopy(CONFIG)
        cfg['auto_trigger'].update(debounce_frames=1, hand_absence_seconds=.05)
        started, release = threading.Event(), threading.Event()
        class Hand:
            def __init__(self, settings): pass
            def present(self, frame): return bool(frame[0,0,0])
            def close(self): pass
        class Gate:
            def __init__(self, settings): self.count = 0
            def classify(self, frame):
                self.count += 1
                return {'accepted': self.count <= 2, 'empty': self.count > 2,
                        'label': 'full_fabric' if self.count <= 2 else 'no_fabric', 'confidence': .99}
        class Pipeline:
            def run(self, *args):
                started.set()
                release.wait(3)
                return {'fabrics': ['old result']}
        controller = AutoController(cfg, Pipeline, Hand, Gate)
        controller.start()
        seq = 0
        def feed(present):
            nonlocal seq
            seq += 1
            frame = np.full((20,20,3), int(present), np.uint8)
            controller.submit(frame, seq, SimpleNamespace(undistort=lambda f: f.copy()),
                              {'size':'M','width':4,'tolerance':1})
            time.sleep(.08)
        try:
            for value in (True, False, False): feed(value)
            self.assertTrue(started.wait(2))
            # Let the flash end before a new hand cycle.
            time.sleep(.12)
            for value in (True, True, False, False): feed(value)
            self.assertIsNotNone(controller.pending_job)
            time.sleep(.12)
            for value in (True, True, False, False): feed(value)
            events = list(controller.poll())
            self.assertIn('clear', [e[0] for e in events])
            self.assertIsNone(controller.pending_job)
            release.set()
            time.sleep(.15)
            self.assertNotIn('result', [e[0] for e in controller.poll()])
        finally:
            release.set()
            controller.stop()
            controller.join(2)


if __name__ == '__main__':
    unittest.main()

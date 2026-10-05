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
from inspection.pipeline import InspectionPipeline
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

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

    def test_two_class_gate_uses_strict_comparison_including_ties(self):
        gate = FabricGate.__new__(FabricGate)
        gate.settings = CONFIG['auto_trigger']
        gate.labels = ['full_fabric', 'no_fabric']
        frame = np.zeros((32, 32, 3), np.uint8)
        for scores, accepted, empty in [([.9, .1], True, False),
                                        ([.5001, .4999], True, False),
                                        ([.1, .9], False, True),
                                        ([.5, .5], False, False),
                                        ([.4, .3], True, False)]:
            gate.model = lambda *args, s=scores, **kwargs: np.array([s])
            result = gate.classify(frame)
            self.assertEqual((result['accepted'], result['empty']), (accepted, empty))
        for scores in ([.5], [.2, .3, .5], [float('nan'), .5]):
            gate.model = lambda *args, s=scores, **kwargs: np.array([s])
            with self.assertRaises(ValueError):
                gate.classify(frame)


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

    def test_shipped_retention_default_is_two_hundred_sets(self):
        self.assertEqual(CONFIG['capture_storage']['max_sets'], 200)

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
                def run(self, frame, progress):
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
        owed = []
        class Hand:
            def __init__(self, settings): pass
            def present(self, frame): return bool(frame[0,0,0])
            def close(self): pass
        class Gate:
            def __init__(self, settings): pass
            def classify(self, frame):
                return {'accepted': True, 'empty': False, 'label': 'full_fabric', 'confidence': .99}
        class Pipeline:
            def run(self, original, *args):
                count = int(original[0,0,1])
                processed.append(count)
                owed.append((args[1]['inspected_lines'], args[1]['queued_captures']))
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
            controller.set_paused(True)
            time.sleep(.12)
            for present in (True, True, False, False): feed(present, 9)
            self.assertFalse(controller.pending_jobs)
            controller.set_paused(False)
            time.sleep(.12)
            for present in (True, True, False, False): feed(present, 2)
            self.assertTrue(controller.pending_jobs)
            self.assertEqual(processed, [1])
            release.set()
            results = []
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline and len(results) < 2:
                results += [payload for event, _, payload in controller.poll() if event == 'result']
                time.sleep(.02)
            self.assertEqual(processed, [1, 2])
            # This pipeline measures no lines, so nothing is ever recorded as
            # measured and neither capture has anything queued behind it.
            self.assertEqual(owed, [([], 0), ([], 0)])
            self.assertEqual([len(result['fabrics']) for result in results], [1, 2])
        finally:
            release.set()
            controller.stop()
            controller.join(2)

    def test_a_capture_owes_the_lines_nothing_has_measured(self):
        seen = {}
        class Pipeline:
            def run_batch(self, original, corrected, metadata, width, tolerance, path,
                          on_result, on_error):
                seen.update(metadata)
                on_result({'measurement': {'selected_line': 2}, 'warnings': []})
        pipeline = Pipeline()
        controller = AutoController(CONFIG, lambda: pipeline)
        controller.pipeline = pipeline
        controller.inspected_lines = {1}
        controller.preparing_count = 1
        controller.pending_jobs.append({'generation': 0, 'epoch': 0})
        controller._start_job({'original': None, 'corrected': None, 'metadata': {},
                               'profile': {'width': 4, 'tolerance': 1},
                               'path': None, 'generation': 0, 'epoch': 0})
        controller.future.result(timeout=2)
        # Line 1 is already measured, and the capture queued in front - the one
        # checking plus the one waiting - takes the next, so this capture is told
        # it owes two lines fewer.
        self.assertEqual(seen['inspected_lines'], [1])
        self.assertEqual(seen['queued_captures'], 2)
        # The line it did measure is recorded, so no later capture repeats it.
        self.assertEqual(controller.inspected_lines, {1, 2})

    def test_slow_fabric_check_keeps_hand_monitor_running_and_preserves_fifo(self):
        cfg = copy.deepcopy(CONFIG)
        cfg['auto_trigger'].update(debounce_frames=1, hand_absence_seconds=.05)
        cfg['capture_storage']['save_rejected_triggers'] = False
        checking, release = threading.Event(), threading.Event()
        processed, detected = [], []
        class Hand:
            def __init__(self, settings): pass
            def present(self, frame):
                detected.append(int(frame[0, 0, 1]))
                return bool(frame[0, 0, 0])
            def close(self): pass
        class Gate:
            def __init__(self, settings): pass
            def classify(self, frame):
                checking.set()
                release.wait(4)
                return {'accepted': True, 'label': 'full_fabric', 'confidence': .99}
        class Pipeline:
            def run(self, original, *args):
                processed.append(int(original[0, 0, 1]))
                return {'fabrics': []}
        controller = AutoController(cfg, Pipeline, Hand, Gate)
        controller.start()
        sequence = 0
        def feed(present, identity):
            nonlocal sequence
            sequence += 1
            frame = np.zeros((20, 20, 3), np.uint8)
            frame[:, :, 0], frame[:, :, 1] = present, identity
            controller.submit(frame, sequence, SimpleNamespace(undistort=lambda f: f.copy()),
                              {'size': 'M', 'width': 4, 'tolerance': 1})
            time.sleep(.09)
        try:
            for value in (True, False, False): feed(value, 1)
            self.assertTrue(checking.wait(2))
            for value in (True, True, False, False): feed(value, 2)
            self.assertIn(2, detected)
            self.assertEqual(controller.preparing_count, 2)
            self.assertIn('2 checking', controller.queue_text)
            controller.set_paused(True)
            release.set()
            deadline = time.monotonic() + 2
            while len(processed) < 2 and time.monotonic() < deadline:
                time.sleep(.02)
            self.assertEqual(processed, [1, 2])
            deadline = time.monotonic() + 2
            while controller.busy and time.monotonic() < deadline:
                time.sleep(.02)
            self.assertEqual(controller.queue_text, 'Queue: 0 checking | 0 waiting | 0 inspecting')
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

    def test_empty_fabric_does_not_clear_or_discard_accepted_work(self):
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
            self.assertTrue(controller.pending_jobs)
            time.sleep(.12)
            for value in (True, True, False, False): feed(value)
            events = list(controller.poll())
            self.assertNotIn('clear', [e[0] for e in events])
            self.assertTrue(controller.pending_jobs)
            release.set()
            time.sleep(.15)
            self.assertEqual(2, sum(e[0] == 'result' for e in controller.poll()))
        finally:
            release.set()
            controller.stop()
            controller.join(2)


class DetectionPublishingTests(unittest.TestCase):
    """The controller publishes preview geometry without changing the trigger."""

    def build(self, hand_class):
        cfg = copy.deepcopy(CONFIG)
        cfg['auto_trigger'].update(debounce_frames=1, hand_absence_seconds=.05)

        class Gate:
            def __init__(self, settings): pass

            def classify(self, frame):
                return {'accepted': True, 'empty': False, 'label': 'full_fabric', 'confidence': .99}

        class Pipeline:
            def run(self, original, *args):
                return {'fabrics': []}

        controller = AutoController(cfg, Pipeline, hand_class, Gate)
        controller.start()
        self.addCleanup(self.stop, controller)
        return controller

    def stop(self, controller):
        controller.stop()
        controller.join(2)

    def feed(self, controller, sequence, value=0):
        frame = np.full((20, 20, 3), value, np.uint8)
        controller.submit(frame, sequence, SimpleNamespace(undistort=lambda f: f.copy()),
                          {'size': 'M', 'width': 4, 'tolerance': 1})
        # The absence timer is wall-clock, so frames must be spaced out for it.
        time.sleep(.08)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if getattr(controller, 'detections', None) is not None:
                snapshot = controller.detections
                if snapshot[0] == sequence:
                    return snapshot
            time.sleep(.01)
        self.fail(f'no detections published for sequence {sequence}')

    def test_landmarks_are_published_when_the_hand_can_report_them(self):
        landmarks = tuple((float(index), float(index)) for index in range(21))

        class Hand:
            def __init__(self, settings): pass

            def detect(self, frame): return True, (landmarks,)

            def present(self, frame): return True

            def close(self): pass

        controller = self.build(Hand)
        _, _, _, size, hands, markers = self.feed(controller, 1)
        self.assertEqual(size, (20, 20))
        self.assertEqual(hands, (landmarks,))
        self.assertEqual(markers, ())

    def test_a_hand_without_detection_still_triggers_and_publishes_no_landmarks(self):
        # The injected stand-in used across these tests implements only
        # present(); it must keep working exactly as before.
        triggered = threading.Event()

        class Hand:
            def __init__(self, settings): pass

            def present(self, frame): return bool(frame[0, 0, 0])

            def close(self): pass

        cfg = copy.deepcopy(CONFIG)
        cfg['auto_trigger'].update(debounce_frames=1, hand_absence_seconds=.05)

        class Gate:
            def __init__(self, settings): pass

            def classify(self, frame): return {'accepted': True, 'empty': False,
                                               'label': 'full_fabric', 'confidence': .99}

        class Pipeline:
            def run(self, original, *args):
                triggered.set()
                return {'fabrics': []}

        controller = AutoController(cfg, Pipeline, Hand, Gate)
        controller.start()
        self.addCleanup(self.stop, controller)
        sequence = 0
        for value in (1, 0, 0):
            sequence += 1
            self.feed(controller, sequence, value=255 if value else 0)
        self.assertTrue(triggered.wait(2))
        self.assertEqual(controller.detections[4], ())

    def test_markers_are_published_even_while_paused_with_no_hands(self):
        class Hand:
            def __init__(self, settings): pass

            def detect(self, frame): return True, (((1.0, 2.0),),)

            def present(self, frame): return True

            def close(self): pass

        controller = self.build(Hand)
        controller.set_paused(True)
        _, _, _, _, hands, markers = self.feed(controller, 1, value=255)
        # Hand inference is skipped on the paused path, so no skeleton is left
        # drawn over a frame that was never checked.
        self.assertEqual(hands, ())


if __name__ == '__main__':
    unittest.main()

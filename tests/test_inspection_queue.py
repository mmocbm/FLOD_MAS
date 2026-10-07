"""Queue, bed-reset, line selection and streamed overlay regression checks."""
import copy
from concurrent.futures import Future
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from auto_trigger.bed_markers import BedMarkers
from auto_trigger.controller import AutoController
from inspection.pipeline import InspectionPipeline
from inspection.storage import CaptureStore
from inspection.dashboard import AutomaticDashboard
from plane_scale import PlaneScale
from test_local_inspection import synthetic_inspection, CONFIG


class MarkerTests(unittest.TestCase):
    def test_two_physical_markers_with_same_id_and_continuous_confirmation(self):
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        marker = cv2.aruco.generateImageMarker(dictionary, 0, 100)
        frame = np.full((260, 400), 255, np.uint8)
        frame[60:160, 30:130] = marker
        one = frame.copy()
        frame[60:160, 250:350] = marker
        monitor = BedMarkers(.5)
        self.assertFalse(monitor.update(one, 0))
        self.assertFalse(monitor.update(frame, 1))
        self.assertFalse(monitor.update(frame, 1.49))
        self.assertTrue(monitor.update(frame, 1.51))
        self.assertFalse(monitor.update(one, 2))
        self.assertFalse(monitor.update(frame, 3))


class QueueTests(unittest.TestCase):
    def test_fifo_preserves_every_result_and_reset_waits_for_drain_even_paused(self):
        controller = AutoController(CONFIG, lambda: None)
        controller.job_generation = 0
        controller.paused = True
        controller.reset_requested = True
        controller.pending_jobs.extend([{'number': 2, 'generation': 0, 'epoch': 0},
                                        {'number': 3, 'generation': 0, 'epoch': 0}])
        started = []
        def start(job):
            started.append(job['number'])
            controller.future = Future()
        controller._start_job = start
        controller.future = Future()
        for number in (1, 2, 3):
            controller.future.set_result({'number': number})
            controller._complete()
            events = list(controller.poll())
            self.assertEqual([p['number'] for e, _, p in events if e == 'result'], [number])
            self.assertEqual(any(e == 'reset_ready' for e, _, _ in events), number == 3)
        self.assertEqual(started, [2, 3])
        self.assertTrue(controller.has_results)
        controller.finish_reset()
        self.assertFalse(controller.has_results)
        self.assertFalse(controller.reset_requested)
        self.assertTrue(controller.paused)

    def test_ui_waits_two_seconds_after_final_result_before_clearing(self):
        controller = AutoController(CONFIG, lambda: None)
        controller.has_results = True
        controller.reset_requested = True
        controller.events.put(('result', 0, {'line': 1}))
        controller.events.put(('reset_ready', 0, None))
        calls = []
        app = SimpleNamespace(auto_controller=controller, video_paused=False, video_streaming=True,
            _reset_due=None, root=SimpleNamespace(after=lambda *args: None),
            _show_automatic_results=lambda p: calls.append('result'),
            _dismiss_result_view=lambda: calls.append('clear'),
            update_progress=lambda *args: None, set_pass_fail=lambda *args: None,
            _poll_automatic=lambda: None)
        with patch('inspection.dashboard.time.monotonic', return_value=10):
            AutomaticDashboard._poll_automatic(app)
        self.assertEqual(calls, ['result'])
        with patch('inspection.dashboard.time.monotonic', return_value=11.99):
            AutomaticDashboard._poll_automatic(app)
        self.assertEqual(calls, ['result'])
        with patch('inspection.dashboard.time.monotonic', return_value=12.01):
            AutomaticDashboard._poll_automatic(app)
        self.assertEqual(calls, ['result', 'clear'])
        self.assertFalse(controller.reset_requested)


class BatchTests(unittest.TestCase):
    def test_all_and_leftmost_selection(self):
        frame, inspector, _ = synthetic_inspection()
        results = list(inspector.run_lines(frame, selection='all'))
        self.assertEqual([r['selected_line'] for r in results], [1, 2])
        self.assertFalse(results[0]['mask'][:, 600:].any())
        result = next(inspector.run_lines(frame, selection='leftmost'))
        self.assertEqual(result['selected_line'], 1)

    def test_failed_line_does_not_skip_other_lines(self):
        frame, inspector, _ = synthetic_inspection()
        inspect = inspector._inspect_line
        failures = []
        def fail_first(photo, source, line, index, *args):
            if index == 1:
                raise ValueError('empty adhesive')
            return inspect(photo, source, line, index, *args)
        inspector._inspect_line = fail_first
        results = list(inspector.run_lines(frame, selection='all',
                       on_error=lambda index, error: failures.append(index)))
        self.assertEqual(failures, [1])
        self.assertEqual([r['selected_line'] for r in results], [2])

    def test_batch_streams_individual_results_and_retains_line_artifacts(self):
        frame, inspector, _ = synthetic_inspection()
        config = copy.deepcopy(CONFIG)
        config['capture_storage']['max_sets'] = 1
        with tempfile.TemporaryDirectory() as directory:
            store = CaptureStore(directory, config['capture_storage'])
            pipeline = InspectionPipeline(inspector, store, config)
            results = []
            with patch('inspection.pipeline.plane_scale.load_frame_scale',
                       return_value=PlaneScale(np.diag([.1,.1,1]), .1)):
                pipeline.run_batch(frame, frame, {'line_selection': 'all'}, 10, .1,
                                   on_result=results.append)
            self.assertEqual(len(results), 2)
            self.assertTrue(all(r['measurement']['status'] == 'FAIL' for r in results))
            for result in results:
                path = Path(result['path'])
                self.assertTrue((path/'overlay.png').is_file())
                self.assertTrue((path/'result.json').is_file())
                self.assertFalse(np.array_equal(frame, result['overlay']))
            old = store._sets()[0]
            store.begin(frame, frame, {})
            self.assertFalse(old.exists())


if __name__ == '__main__':
    unittest.main()

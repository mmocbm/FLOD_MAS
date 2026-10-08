"""Queue, bed-reset, line selection and streamed overlay regression checks."""
import copy
from concurrent.futures import Future
import json
from pathlib import Path
import tempfile
import time
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
from inspection.local.engine import LiveInspection
from test_local_inspection import synthetic_inspection, CONFIG


def four_line_inspection():
    """Four glue lines on one frame, so the line-count rule can be exercised.

    The shipped machine holds at most four, and the rule distinguishes four
    lines from one, which a two-line frame cannot.
    """
    frame = np.full((800, 1500, 3), 100, np.uint8)
    source = np.zeros(frame.shape[:2], np.uint8)
    y = np.arange(30, 770)
    for x in (250, 500, 750, 1000):
        boundary = np.column_stack((x + 30*np.sin((y-30)*2*np.pi/740), y))
        cv2.fillPoly(source, [np.rint(np.vstack(([x-150, 30], boundary,
                                                 [x-150, 769]))).astype(np.int32)], 255)
    def adhesive(flat):
        h, w = flat.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        mask[35:65, 15:w-15] = 255
        return mask
    inspector = LiveInspection.__new__(LiveInspection)
    inspector.offset = 100.
    inspector.source = SimpleNamespace(predict=lambda photo: source.copy())
    inspector.glue = SimpleNamespace(width=1024, predict=adhesive)
    return frame, inspector


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

    def test_display_scan_reports_full_frame_corners_and_leaves_the_timer_alone(self):
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        marker = cv2.aruco.generateImageMarker(dictionary, 0, 100)
        frame = np.full((260, 400), 255, np.uint8)
        frame[60:160, 30:130] = marker
        frame[60:160, 250:350] = marker
        monitor = BedMarkers(.5)
        # A downscaled scan must scale its corners back, or the overlay would be
        # drawn at half size in the corner of the preview.
        for width in (0, 200, 100):
            corners = monitor.detect_for_display(frame, width)
            self.assertEqual(len(corners), 2, width)
            xs = [x for corner in corners for x, _ in corner]
            self.assertAlmostEqual(min(xs), 30, delta=3, msg=width)
            self.assertAlmostEqual(max(xs), 350, delta=3, msg=width)
        # It exists only to be drawn, so the reset decision cannot depend on it.
        self.assertIsNone(monitor.since)

    def test_display_scan_returns_nothing_for_a_clear_frame(self):
        monitor = BedMarkers(.5)
        self.assertEqual(monitor.detect_for_display(np.full((260, 400), 255, np.uint8), 200), ())


class OverlayGeometryTests(unittest.TestCase):
    """The preview reads one snapshot; these pin every reason it draws nothing."""

    def app(self, controller_epoch=3, snapshot_epoch=None, age=0.0,
            hands=(((0, 0),),), markers=(), show=True):
        if snapshot_epoch is None:
            snapshot_epoch = controller_epoch
        controller = SimpleNamespace(epoch=controller_epoch)
        controller.detections = (1, snapshot_epoch, time.monotonic() - age,
                                 (10, 10), hands, markers)
        app = SimpleNamespace(show_overlays=show, auto_controller=controller)
        return app

    def test_geometry_is_returned_while_fresh(self):
        geometry = AutomaticDashboard._overlay_geometry(self.app())
        self.assertEqual(geometry['hands'], (((0, 0),),))
        self.assertEqual(geometry['size'], (10, 10))

    def test_nothing_is_drawn_when_the_switch_is_off(self):
        self.assertIsNone(AutomaticDashboard._overlay_geometry(self.app(show=False)))

    def test_stale_geometry_is_dropped_when_the_worker_stops(self):
        self.assertIsNone(AutomaticDashboard._overlay_geometry(self.app(age=1.0)))

    def test_geometry_from_a_previous_session_is_dropped(self):
        # The controller bumped its epoch, so the snapshot is from the camera
        # that was showing before the switch.
        self.assertIsNone(AutomaticDashboard._overlay_geometry(
            self.app(controller_epoch=4, snapshot_epoch=3)))

    def test_an_empty_snapshot_draws_nothing(self):
        self.assertIsNone(
            AutomaticDashboard._overlay_geometry(self.app(hands=(), markers=())))

    def test_a_controller_that_never_ran_draws_nothing(self):
        app = SimpleNamespace(show_overlays=True, auto_controller=SimpleNamespace(epoch=0))
        self.assertIsNone(AutomaticDashboard._overlay_geometry(app))


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
    def test_only_the_lines_still_owed_are_measured_from_the_left(self):
        frame, inspector, _ = synthetic_inspection()
        results = list(inspector.run_lines(frame))
        self.assertEqual([r['selected_line'] for r in results], [1, 2])
        self.assertFalse(results[0]['mask'][:, 600:].any())
        # Line 1 is already measured, so only line 2 is still owed.
        rest = list(inspector.run_lines(frame, inspected=[1]))
        self.assertEqual([r['selected_line'] for r in rest], [2])
        # Every detected line is accounted for: nothing is loaded at all.
        self.assertEqual(list(inspector.run_lines(frame, inspected=[1, 2])), [])

    def test_a_queued_capture_takes_the_leftmost_of_what_is_left(self):
        frame, inspector, _ = synthetic_inspection()
        # One capture is already queued behind this one, so this capture measures
        # the rightmost line and leaves the leftmost for the queued one.
        results = list(inspector.run_lines(frame, reserved=1))
        self.assertEqual([r['selected_line'] for r in results], [2])
        self.assertEqual(list(inspector.run_lines(frame, reserved=2)), [])

    def test_the_line_count_rule_matches_the_operator_examples(self):
        frame, inspector = four_line_inspection()
        self.assertEqual([r['selected_line'] for r in inspector.run_lines(frame)], [1, 2, 3, 4])
        # Four lines, two already measured, nothing queued: the rightmost two,
        # loaded left to right so the third is inspected before the fourth.
        taken = list(inspector.run_lines(frame, inspected=[1, 2]))
        self.assertEqual([r['selected_line'] for r in taken], [3, 4])
        centres = [np.nonzero(r['mask'])[1].mean() for r in taken]
        self.assertEqual(centres, sorted(centres))
        # A capture already queued takes the next line, so only the fourth is
        # left; two queued take both, and then nothing is loaded at all.
        self.assertEqual([r['selected_line'] for r in
                          inspector.run_lines(frame, inspected=[1, 2], reserved=1)], [4])
        self.assertEqual(list(inspector.run_lines(frame, inspected=[1, 2], reserved=2)), [])
        # Every detected line accounted for: no inspection starts.
        self.assertEqual(list(inspector.run_lines(frame, inspected=[1, 2, 3, 4])), [])

    def test_failed_line_does_not_skip_other_lines(self):
        frame, inspector, _ = synthetic_inspection()
        inspect = inspector._inspect_line
        failures = []
        def fail_first(photo, source, line, index, *args):
            if index == 1:
                raise ValueError('empty adhesive')
            return inspect(photo, source, line, index, *args)
        inspector._inspect_line = fail_first
        results = list(inspector.run_lines(frame,
                       on_error=lambda index, error: failures.append(index)))
        self.assertEqual(failures, [1])
        self.assertEqual([r['selected_line'] for r in results], [2])
        # Line 1 never produced a result, so it is the only one offered again.
        inspector._inspect_line = inspect
        retry = list(inspector.run_lines(frame, inspected=[2]))
        self.assertEqual([r['selected_line'] for r in retry], [1])

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
                pipeline.run_batch(frame, frame, {}, 10, .1,
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

    def test_batch_overlay_shows_the_detected_adhesive_region(self):
        frame, inspector, _ = synthetic_inspection()
        with tempfile.TemporaryDirectory() as directory:
            pipeline = InspectionPipeline(inspector, CaptureStore(directory, CONFIG['capture_storage']),
                                          CONFIG)
            results = []
            with patch('inspection.pipeline.plane_scale.load_frame_scale',
                       return_value=PlaneScale(np.diag([.1,.1,1]), .1)):
                pipeline.run_batch(frame, frame, {}, 10, .1,
                                   on_result=results.append)
            self.assertEqual(len(results), 2)
            for result in results:
                mask = np.squeeze(result['mask'])
                # Eroded so the assertion is about the fill, not the outline.
                interior = cv2.erode(mask, np.ones((9, 9), np.uint8)).astype(bool)
                self.assertTrue(interior.any())
                self.assertFalse(np.array_equal(result['overlay'][interior], frame[interior]))


if __name__ == '__main__':
    unittest.main()

"""Regression checks for preview scheduling and bounded canvas item count."""
import os
import sys
import tkinter as tk
import time
import unittest
import threading
from concurrent.futures import Future
from unittest.mock import MagicMock, patch

import numpy as np
import main_1366
from main_1366 import IndustrialDashboard
from app_config import CONFIG
from CalibrateAPP.calibration_ui import CalibrationApp
from ui_theme import HAND_EDGES, draw_tracking_overlay, preview_transform


class PreviewUITests(unittest.TestCase):
    @patch('main_1366.ImageTk.PhotoImage')
    def test_duplicate_frame_skips_scaling_but_resize_repaints(self, photo):
        app = IndustrialDashboard.__new__(IndustrialDashboard)
        app.canvas_1 = canvas = MagicMock()
        canvas.winfo_width.return_value = 2000
        canvas.winfo_height.return_value = 1500
        canvas.find_withtag.return_value = (12,)
        original = np.zeros((1800, 2400, 3), np.uint8)
        app._display_video_frame(original, 1)
        rendered = photo.call_args.args[0]
        self.assertLessEqual(rendered.width, CONFIG['preview']['max_width'])
        self.assertLessEqual(rendered.height, CONFIG['preview']['max_height'])
        self.assertEqual(original.shape, (1800, 2400, 3))
        app._display_video_frame(original, 1)
        self.assertEqual(photo.call_count, 1)
        canvas.winfo_width.return_value = 500
        app._display_video_frame(original, 1)
        self.assertEqual(photo.call_count, 2)
        # A cleared canvas must be repainted even if the device has stalled.
        canvas.find_withtag.return_value = ()
        app._display_video_frame(original, 1)
        self.assertEqual(photo.call_count, 3)

    def test_paused_feed_does_not_animate_or_read_cameras(self):
        app = IndustrialDashboard.__new__(IndustrialDashboard)
        app.root = MagicMock()
        app.video_streaming = app.video_paused = True
        app.camera1 = MagicMock()
        app.result_view = MagicMock()
        app.update_video_feed()
        app.camera1.get_raw_frame_with_ret.assert_not_called()
        app.result_view.tick.assert_not_called()
        app.root.after.assert_called_once_with(CONFIG['preview']['interval_ms'], app.update_video_feed)

    @patch('main_1366.time.perf_counter', side_effect=[0.0, 0.5])
    def test_slow_feed_yields_and_does_not_read_hidden_camera(self, clock):
        app = IndustrialDashboard.__new__(IndustrialDashboard)
        app.root = MagicMock()
        app.video_streaming = True
        app.video_paused = False
        app.maximized_camera = 1
        app.camera1 = MagicMock()
        app.camera2 = MagicMock()
        app.camera1.stream.read_snapshot.return_value = (np.zeros((4, 4, 3)), 1)
        app.result_view = None
        app._submit_automatic_frame = MagicMock()
        app._display_video_frame = MagicMock()
        app.update_video_feed()
        app.camera2.get_raw_frame_with_ret.assert_not_called()
        app._display_video_frame.assert_called_once()
        self.assertGreaterEqual(app.root.after.call_args.args[0], CONFIG['preview']['interval_ms'])

    def calibration_app(self):
        app = CalibrationApp.__new__(CalibrationApp)
        app.closed = app.processing_capture = app.check_panel_visible = app.check_preview_frozen = False
        app.camera_running = True
        app.cap = MagicMock()
        app.root = MagicMock()
        app.video_label = MagicMock()
        app.video_label.winfo_width.return_value = 800
        app.video_label.winfo_height.return_value = 600
        app.frame_lock = threading.Lock()
        app._preview_executor = MagicMock()
        app._preview_future = None
        app._show_frame = MagicMock()
        app._draw_coverage_guide = MagicMock()
        app._draw_capture_history = MagicMock()
        return app

    def test_preview_worker_keeps_full_resolution_capture(self):
        original = np.zeros((2000, 3000, 3), np.uint8)
        source = MagicMock()
        source.read.return_value = True, original
        camera, frame, display = CalibrationApp._prepare_live_preview(source, (600, 400))
        self.assertIs(camera, source)
        self.assertIs(frame, original)
        self.assertEqual(display.shape, (400, 600, 3))
        self.assertFalse(np.shares_memory(display, original))

    def test_slow_worker_does_not_block_or_queue_more_frames(self):
        app = self.calibration_app()
        app._preview_future = Future()  # Deliberately not complete.
        app.update_frame()
        app.cap.read.assert_not_called()
        app._preview_executor.submit.assert_not_called()
        app.root.after.assert_called_once()

    def test_switched_camera_discards_old_result(self):
        app = self.calibration_app()
        app._preview_future = Future()
        app._preview_future.set_result((object(), np.zeros((10, 10, 3)), np.zeros((10, 10, 3))))
        app.update_frame()
        app._show_frame.assert_not_called()
        app._preview_executor.submit.assert_called_once()

    def test_frozen_review_is_not_overwritten_by_pending_preview(self):
        app = self.calibration_app()
        app.check_panel_visible = app.check_preview_frozen = True
        app._preview_future = Future()
        original = np.zeros((10, 10, 3))
        app._preview_future.set_result((app.cap, original, original.copy()))
        app.update_frame()
        app._show_frame.assert_not_called()
        self.assertIs(app.current_frame, original)
        self.assertIsNone(app._preview_executor.submit.call_args.args[-1])

    def auto_lock_app(self):
        app = self.calibration_app()
        app.auto_lock_state = 'idle'
        app.auto_lock_job = None
        app.auto_lock_deadline = 0.0
        app.auto_btn = MagicMock()
        app.auto_instruction = MagicMock()
        app.auto_status = MagicMock()
        app._set_status = MagicMock()
        app.log = MagicMock()
        return app

    def test_auto_lock_refuses_to_start_without_a_camera(self):
        app = self.auto_lock_app()
        app.camera_running = False
        app.start_auto_lock()
        self.assertEqual(app.auto_lock_state, 'idle')
        app.root.after.assert_not_called()
        app.auto_btn.configure.assert_not_called()

    def test_auto_lock_first_press_starts_the_settle_countdown(self):
        app = self.auto_lock_app()
        app.start_auto_lock()
        self.assertEqual(app.auto_lock_state, 'settling')
        app.auto_btn.configure.assert_called_once_with(state=tk.DISABLED, text="SETTLING…")
        self.assertGreater(app.auto_lock_deadline, time.monotonic())

    def test_auto_lock_countdown_reschedules_until_it_reaches_ready(self):
        app = self.auto_lock_app()
        app.auto_lock_state = 'settling'
        app.auto_lock_deadline = time.monotonic() + 5
        app._tick_auto_lock()
        self.assertEqual(app.auto_lock_state, 'settling')
        app.root.after.assert_called_once_with(100, app._tick_auto_lock)
        self.assertEqual(app.auto_lock_job, app.root.after.return_value)
        self.assertIn('Settling', app.auto_status.configure.call_args.kwargs['text'])

    def test_auto_lock_offers_the_lock_once_settled(self):
        app = self.auto_lock_app()
        app.auto_lock_state = 'settling'
        app.auto_lock_deadline = -1.0  # Already past, so the timer is due.
        app._tick_auto_lock()
        self.assertEqual(app.auto_lock_state, 'ready')
        self.assertIsNone(app.auto_lock_job)
        app.auto_btn.configure.assert_called_once_with(state=tk.NORMAL, text="TURN AUTO OFF")
        app.root.after.assert_not_called()

    @patch('CalibrateAPP.calibration_ui.threading.Thread')
    def test_second_press_runs_the_lock_off_the_tk_thread(self, thread):
        app = self.auto_lock_app()
        app.auto_lock_state = 'ready'
        app.start_auto_lock()
        self.assertEqual(app.auto_lock_state, 'locking')
        self.assertTrue(thread.call_args.kwargs['daemon'])
        self.assertEqual(thread.call_args.kwargs['args'], (app.cap,))
        self.assertEqual(thread.call_args.kwargs['target'], app._auto_lock_worker)
        thread.return_value.start.assert_called_once_with()

    def test_auto_lock_worker_returns_the_driver_result_to_tk(self):
        app = self.auto_lock_app()
        result = {'white_balance': {'ok': True, 'actual': 0.0},
                  'exposure': {'ok': True, 'actual': 0.25}}
        app.cap.lock_automatic.return_value = result
        app._auto_lock_worker(app.cap)
        app.cap.lock_automatic.assert_called_once_with()
        app.root.after.assert_called_once_with(0, app._auto_lock_finished, app.cap, result)

    def test_auto_lock_worker_survives_a_driver_that_raises(self):
        app = self.auto_lock_app()
        app.cap.lock_automatic.side_effect = RuntimeError('device busy')
        app._auto_lock_worker(app.cap)
        cap, result = app.root.after.call_args.args[2:]
        self.assertIs(cap, app.cap)
        self.assertEqual(result, {'error': 'device busy'})

    def test_confirmed_lock_reports_both_controls_off(self):
        app = self.auto_lock_app()
        app._auto_lock_finished(app.cap, {'white_balance': {'ok': True, 'actual': 0.0},
                                          'exposure': {'ok': True, 'actual': 0.25}})
        self.assertEqual(app.auto_lock_state, 'done')
        self.assertEqual(app.auto_status.configure.call_args.kwargs['text'],
                         "White balance OFF and exposure OFF.")

    @patch('CalibrateAPP.calibration_ui.messagebox')
    def test_unconfirmed_control_is_reported_not_hidden(self, box):
        app = self.auto_lock_app()
        app._auto_lock_finished(app.cap, {'white_balance': {'ok': True, 'actual': 0.0},
                                          'exposure': {'ok': False, 'actual': 0.75}})
        self.assertEqual(app.auto_lock_state, 'failed')
        text = app.auto_status.configure.call_args.kwargs['text']
        self.assertIn('exposure NOT locked (driver reported 0.75)', text)
        self.assertIn('white balance OFF', text)
        box.showwarning.assert_called_once()

    @patch('CalibrateAPP.calibration_ui.messagebox')
    def test_result_for_a_previous_camera_is_dropped(self, box):
        app = self.auto_lock_app()
        app._auto_lock_finished(MagicMock(), {'white_balance': {'ok': True, 'actual': 0.0},
                                              'exposure': {'ok': True, 'actual': 0.25}})
        self.assertEqual(app.auto_lock_state, 'idle')
        app.auto_status.configure.assert_not_called()
        app.auto_btn.configure.assert_not_called()
        box.showwarning.assert_not_called()

    def test_result_arriving_after_close_is_dropped(self):
        app = self.auto_lock_app()
        app.closed = True
        app._auto_lock_finished(app.cap, {'error': 'device busy'})
        app.auto_status.configure.assert_not_called()

    def test_cancel_auto_lock_stops_the_countdown(self):
        app = self.auto_lock_app()
        app.auto_lock_state = 'settling'
        app.auto_lock_job = 'after#7'
        app._cancel_auto_lock()
        app.root.after_cancel.assert_called_once_with('after#7')
        self.assertIsNone(app.auto_lock_job)
        self.assertEqual(app.auto_lock_state, 'idle')

    def test_refresh_ui_disables_the_button_without_a_camera(self):
        app = self.auto_lock_app()
        app.camera_running = False
        app._refresh_auto_lock_ui()
        app.auto_btn.configure.assert_called_once_with(state=tk.DISABLED, text="RUN AUTO LOCK")

    def test_refresh_ui_leaves_a_settle_countdown_alone(self):
        app = self.auto_lock_app()
        app.auto_lock_state = 'settling'
        app._refresh_auto_lock_ui()
        self.assertEqual(app.auto_lock_state, 'settling')
        app.auto_btn.configure.assert_not_called()

    def test_transform_matches_the_main_preview_resize(self):
        # The overlay must land on the pixels the image was actually scaled to.
        scale, offset_x, offset_y = preview_transform((4608, 3456), 640, 400, 880, 540)
        self.assertAlmostEqual(scale, 400 / 3456)
        width, height = int(4608 * scale), int(3456 * scale)
        self.assertAlmostEqual(offset_x, 640 / 2 - width / 2)
        self.assertAlmostEqual(offset_y, 400 / 2 - height / 2)

    def test_transform_never_upscales_the_main_preview(self):
        # A small frame in a large canvas keeps its own size, as the image does.
        scale, _, _ = preview_transform((100, 50), 2000, 1500, 880, 540)
        self.assertEqual(scale, 1.0)

    def test_transform_lets_the_side_panel_grow_the_image(self):
        scale, _, _ = preview_transform((100, 50), 300, 150)
        self.assertEqual(scale, 3.0)

    def test_overlay_maps_landmarks_through_the_shared_transform(self):
        canvas = MagicMock()
        hand = tuple((x, y) for x, y in ((0, 0), (10, 10)) + ((0, 0),) * 19)
        draw_tracking_overlay(canvas, {'hands': (hand,), 'markers': ()},
                              scale=2.0, offset_x=10, offset_y=20)
        canvas.delete.assert_called_once_with('overlay')
        self.assertEqual(canvas.create_line.call_count, len(HAND_EDGES))
        # Edge 0 joins landmarks 0 and 1: (0,0) and (10,10), both offset by
        # (10,20) and doubled.
        self.assertEqual(canvas.create_line.call_args_list[0].args[:4], (10, 20, 30, 40))

    def test_overlay_clears_only_its_own_tag(self):
        canvas = MagicMock()
        draw_tracking_overlay(canvas, None, 1.0, 0, 0)
        canvas.delete.assert_called_once_with('overlay')
        canvas.create_line.assert_not_called()
        canvas.create_oval.assert_not_called()
        canvas.create_polygon.assert_not_called()

    def test_overlay_never_uses_the_image_tag(self):
        # find_withtag("img")[0] must stay the frame image on every canvas.
        canvas = MagicMock()
        hand = tuple((0, 0) for _ in range(21))
        draw_tracking_overlay(canvas, {'hands': (hand,), 'markers': (((0, 0), (1, 0), (1, 1), (0, 1)),)},
                              1.0, 0, 0)
        for call in (canvas.create_line.call_args_list + canvas.create_oval.call_args_list
                     + canvas.create_polygon.call_args_list):
            self.assertEqual(call.kwargs.get('tags'), 'overlay')

    def test_overlay_handles_a_partial_landmark_list(self):
        # Fewer than 21 points cannot be joined into a skeleton, so only dots.
        canvas = MagicMock()
        draw_tracking_overlay(canvas, {'hands': (((1, 2), (3, 4)),), 'markers': ()}, 1.0, 0, 0)
        canvas.create_line.assert_not_called()
        self.assertEqual(canvas.create_oval.call_count, 2)

    def test_resume_cancels_previous_feed_timer(self):
        app = IndustrialDashboard.__new__(IndustrialDashboard)
        app.root = MagicMock()
        app.camera1 = app.camera2 = object()
        app._video_job = 'after#42'
        app.update_video_feed = MagicMock()
        app._refresh_inspection_availability = MagicMock()
        app.start_video_stream()
        app.root.after_cancel.assert_called_once_with('after#42')
        app.update_video_feed.assert_called_once_with()
        self.assertFalse(app.video_paused)

    @patch('main_1366.subprocess.Popen')
    def test_reset_relaunches_the_app_without_the_startup_countdown(self, popen):
        app = IndustrialDashboard.__new__(IndustrialDashboard)
        app.root = MagicMock()
        app.auto_controller = MagicMock()
        app.camera1 = MagicMock()
        app.camera2 = None
        app._camera_futures = []
        app._dismiss_result_view = MagicMock()
        app.restart_application()
        # The devices must be free before the replacement process opens them.
        app.auto_controller.stop.assert_called_once_with()
        app.camera1.release.assert_called_once_with()
        self.assertEqual(popen.call_args.args[0][0], sys.executable)
        self.assertEqual(popen.call_args.args[0][2], main_1366.NO_STARTUP_FLAG)
        self.assertEqual(popen.call_args.kwargs['cwd'],
                         os.path.dirname(os.path.abspath(main_1366.__file__)))
        app.root.destroy.assert_called_once_with()

    @patch('main_1366.ImageTk.PhotoImage')
    def test_live_frame_reuses_existing_image_below_overlays(self, photo):
        app = IndustrialDashboard.__new__(IndustrialDashboard)
        app.canvas_1 = canvas = MagicMock()
        canvas.winfo_width.return_value = 640
        canvas.winfo_height.return_value = 480
        canvas.find_withtag.return_value = (12,)
        app._display_video_frame(np.zeros((800, 1200, 3), np.uint8), 1)
        canvas.create_image.assert_not_called()
        canvas.delete.assert_not_called()
        canvas.itemconfigure.assert_called_once_with(12, image=photo.return_value)
        canvas.coords.assert_called_once_with(12, 320, 240)
        canvas.tag_lower.assert_called_once_with('img')


if __name__ == '__main__':
    unittest.main()

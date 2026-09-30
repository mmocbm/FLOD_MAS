"""Regression checks for preview scheduling and bounded canvas item count."""
import tkinter as tk
import unittest
import threading
from concurrent.futures import Future
from unittest.mock import MagicMock, patch

import numpy as np
from main_1366 import IndustrialDashboard
from app_config import CONFIG
from CalibrateAPP.calibration_ui import CalibrationApp


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
        app.root.after.assert_called_once_with(250, app.update_video_feed)

    @patch('main_1366.time.perf_counter', side_effect=[0.0, 0.5])
    def test_slow_feed_yields_and_does_not_read_hidden_camera(self, clock):
        app = IndustrialDashboard.__new__(IndustrialDashboard)
        app.root = MagicMock()
        app.video_streaming = True
        app.video_paused = False
        app.maximized_camera = 1
        app.camera1 = MagicMock()
        app.camera2 = MagicMock()
        app.camera1.get_raw_frame_with_ret.return_value = (True, np.zeros((4, 4, 3)))
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

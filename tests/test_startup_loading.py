"""Startup readiness is explicit; a dashboard camera cannot start early."""
import json
from pathlib import Path
import threading
import unittest
from unittest.mock import Mock, patch

from auto_trigger.controller import AutoController
from main_1366 import IndustrialDashboard

CONFIG = json.loads((Path(__file__).parents[1] / 'config.json').read_text())


class StartupLoadingTests(unittest.TestCase):
    def test_activity_animation_starts_once_and_stops_when_work_finishes(self):
        app = IndustrialDashboard.__new__(IndustrialDashboard)
        app.progress_bar = Mock()
        app._set_activity_animation(True)
        app._set_activity_animation(True)
        app.progress_bar.start.assert_called_once_with(35)
        app.progress_bar.reset_mock()
        app._set_activity_animation(False)
        app.progress_bar.stop.assert_called_once()
        app.progress_bar.configure.assert_called_with(mode='determinate', value=0)

    def test_camera_is_not_constructed_while_models_load(self):
        app = IndustrialDashboard.__new__(IndustrialDashboard)
        app._models_loading = True
        with patch('main_1366.CameraHandler') as camera:
            app.start_video_stream()
        camera.assert_not_called()
        self.assertTrue(app._camera_start_requested)

    def test_ready_is_emitted_only_after_all_factories_finish(self):
        entered, release = threading.Event(), threading.Event()
        hand = Mock()
        def pipeline():
            entered.set()
            release.wait(3)
            return Mock()
        controller = AutoController(CONFIG, pipeline, lambda _: hand, lambda _: Mock())
        controller.start()
        try:
            self.assertTrue(entered.wait(3))
            self.assertFalse(controller.ready)
            self.assertNotIn('ready', [e[0] for e in controller.poll()])
            release.set()
            # Wait using the worker's event queue, not guessed startup timing.
            ready = False
            import time
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline and not ready:
                ready = any(e[0] == 'ready' for e in controller.poll())
                if not ready:
                    threading.Event().wait(.01)
            self.assertTrue(ready)
            self.assertTrue(controller.ready)
        finally:
            release.set()
            controller.stop()
            controller.join(3)

    def test_failure_has_startup_error_and_never_ready(self):
        def broken(_):
            raise RuntimeError('Missing model')
        controller = AutoController(CONFIG, Mock(), broken, Mock())
        controller.start()
        controller.join(3)
        events = list(controller.poll())
        self.assertFalse(controller.ready)
        self.assertNotIn('ready', [e[0] for e in events])
        self.assertTrue(any(e[0] == 'startup_error' and 'Missing model' in e[2] for e in events))

    def test_settings_return_starts_deferred_camera_after_ready(self):
        app = IndustrialDashboard.__new__(IndustrialDashboard)
        app._models_loading = False
        app._camera_start_requested = True
        app.camera1 = None
        app.start_video_stream = Mock()
        app.resume_video()
        app.start_video_stream.assert_called_once()

"""Hardware-free Tk navigation regression check; run from the project root."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import time
import tkinter as tk
from unittest.mock import patch
import numpy as np
import main_1366 as main


class FakeCamera:
    opens = 0
    def __init__(self, index, path):
        time.sleep(0.15)  # Driver initialization must not block the Tk thread.
        type(self).opens += 1
        self.camera_index = index
        self.stream = self
        self.size = (2560, 1440)
        self.resolution_warning = 'Camera fallback test'
        self.released = False
    def read(self): return True, np.zeros((1440, 2560, 3), np.uint8)
    def isOpened(self): return not self.released
    def get_raw_frame_with_ret(self): return False, None
    def reload_calibration(self): pass
    def release(self): self.released = True


def run():
    root = tk.Tk()
    errors = []
    root.report_callback_exception = lambda *args: errors.append(args)
    with patch.object(main, 'CameraHandler', FakeCamera), patch.object(main.IndustrialDashboard, '_init_serial'), patch.object(main.messagebox, 'showwarning') as warning:
        app = main.IndustrialDashboard(root, lambda: None)
        start = time.perf_counter()
        app.start_video_stream()
        assert time.perf_counter() - start < 0.1, 'Camera startup blocked the UI'
        deadline = time.monotonic() + 5
        while app.camera1 is None and time.monotonic() < deadline:
            root.update()
            time.sleep(0.01)
        assert app.camera1 is not None
        cameras = (app.camera1, app.camera2)
        for _ in range(3):
            app.open_settings_selector()
            app.open_camera_setup()
            setup = app.calibration_app
            setup.select_camera(cameras[0].camera_index)
            setup.cap = setup.camera_provider(cameras[0].camera_index)
            setup.camera_running = True
            setup.stage = 'capture'
            root.update()
            setup.on_closing()
            root.update()
            assert app.main_frame.winfo_ismapped()
            assert (app.camera1, app.camera2) == cameras
            assert not any(c.released for c in cameras)
        assert FakeCamera.opens == 2, 'Switching screens reopened a camera'
        assert warning.call_count == 0, 'Camera startup should not show a resolution popup'
        assert len(root.tk.call('after', 'info')) == 1, 'Duplicate preview timers'
        assert not errors, errors
        app.close_application()
        assert all(c.released for c in cameras)
    print('PASS: asynchronous startup; three page switches; two device opens total; one preview timer')


if __name__ == '__main__':
    run()

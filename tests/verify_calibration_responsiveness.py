"""Hardware-free Tk smoke check: live frames, scrolling and page teardown."""
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tkinter as tk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from CalibrateAPP.calibration_ui import CalibrationApp, CalibrationCheckApp


class SimulatedCamera:
    def __init__(self):
        self.frame = np.zeros((2160, 3840, 3), np.uint8)

    def isOpened(self):
        return True

    def read(self):
        time.sleep(0.015)
        return True, self.frame

    def release(self):
        pass


def run():
    root = tk.Tk()
    root.geometry('1366x768')
    errors = []
    root.report_callback_exception = lambda *args: errors.append(args)
    try:
        for app_type in (CalibrationApp, CalibrationCheckApp):
            host = tk.Frame(root)
            host.pack(fill='both', expand=True)
            app = app_type(root, host=host)
            root.update()
            root.lift()
            root.attributes('-topmost', True)
            root.update()
            app.cap = SimulatedCamera()
            app.camera_running = True
            gaps = []
            previous = time.perf_counter()
            deadline = previous + 1.5
            next_scroll = previous
            start_scroll = app.controls_canvas.yview()[0]
            while time.perf_counter() < deadline:
                now = time.perf_counter()
                if now >= next_scroll:
                    app._scroll_controls_with_mouse(SimpleNamespace(
                        delta=-120,
                        x_root=app.controls_canvas.winfo_rootx() + 5,
                        y_root=app.controls_canvas.winfo_rooty() + 5))
                    next_scroll = now + 0.15
                root.update()
                current = time.perf_counter()
                gaps.append(current - previous)
                previous = current
                time.sleep(0.002)
            assert app.current_frame is app.cap.frame
            assert app.video_label.cget('image')
            assert app.controls_canvas.yview()[0] > start_scroll, (app.controls_canvas.yview(), app.controls_canvas.bbox('all'), app.controls_canvas.winfo_height())
            assert not errors, errors
            assert app.shutdown()
            assert app._scroll_job is None
            host.destroy()
            root.update()
            print(f'{app_type.__name__}: PASS; max observed event-loop gap {max(gaps)*1000:.1f} ms')
        assert not errors, errors
    finally:
        root.destroy()


if __name__ == '__main__':
    run()

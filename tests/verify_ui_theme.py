"""Hardware-free layout check for the shared application theme."""

import sys
import tkinter as tk
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main_1366 as main
from ui_theme import COLORS, button, set_button_role


def visible_inside(widget, window):
    window.update_idletasks()
    assert widget.winfo_ismapped(), f"{widget} is not visible"
    right = widget.winfo_rootx() - window.winfo_rootx() + widget.winfo_width()
    bottom = widget.winfo_rooty() - window.winfo_rooty() + widget.winfo_height()
    assert right <= window.winfo_width(), f"{widget} extends past the right edge"
    assert bottom <= window.winfo_height(), f"{widget} extends past the bottom edge"


def run():
    root = tk.Tk()
    errors = []
    root.report_callback_exception = lambda *args: errors.append(args)
    with patch.object(main.IndustrialDashboard, '_init_serial'):
        app = main.IndustrialDashboard(root, lambda: None)
        root.update()
        assert root.cget('bg') == COLORS['bg']
        assert not root.tk.call('after', 'info'), 'Theme must not add background animation timers'
        visible_inside(app.detect_btn_L, root)
        visible_inside(app.detect_btn_R, root)
        calls = []
        probe = button(root, 'Action', lambda: calls.append(True))
        probe.invoke()
        probe.configure(state=tk.DISABLED)
        probe.invoke()
        assert calls == [True], 'Disabled rounded buttons must not invoke actions'
        probe.configure(state=tk.NORMAL, text='Updated action')
        set_button_role(probe, 'selected')
        probe.invoke()
        assert calls == [True, True]
        probe.destroy()

        app.open_settings_selector()
        root.update()
        assert app.sel_win.cget('bg') == COLORS['bg']

        app.open_size_window()
        root.update()
        for size_button in app._size_buttons.values():
            visible_inside(size_button, app.size_win)
        visible_inside(app.strip_width_entry, app.size_win)
        visible_inside(app.strip_width_tolerance_entry, app.size_win)

        app.open_crop_setup_window()
        root.update()
        for size_button in app.crop_size_buttons.values():
            visible_inside(size_button, app.crop_win)
        assert app.crop_canvas.winfo_width() > 800
        assert app.crop_canvas.winfo_height() > 400
        app.crop_setup_active = False
        app.crop_win.destroy()

        app.open_calibration_checks_page()
        root.update()
        visible_inside(app.calibration_app.controls_canvas, root)
        app.open_camera_setup()
        root.update()
        visible_inside(app.calibration_app.controls_canvas, root)
        with patch.object(app, 'start_video_stream'):
            app.calibration_app.on_closing()
        root.update()

        assert not errors, errors
        app.close_application()
    print('PASS: dashboard, settings, profile, and crop layouts fit 1366x768')


if __name__ == '__main__':
    run()

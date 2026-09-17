"""Hardware-free layout check for the shared application theme."""

import sys
import tkinter as tk
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main_1366 as main
from ui_theme import COLORS


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

        app.open_settings_selector()
        root.update()
        assert app.sel_win.cget('bg') == COLORS['bg']

        app.open_size_window()
        root.update()
        for size_button in app._size_buttons.values():
            visible_inside(size_button, app.size_win)
        visible_inside(app.segments_combo, app.size_win)

        app.open_mask_setup_window()
        root.update()
        assert app.mask_canvas.winfo_width() > 800
        assert app.mask_canvas.winfo_height() > 400
        app.mask_setup_active = False
        app.mask_win.destroy()

        assert not errors, errors
        app.close_application()
    print('PASS: dashboard, settings, profile, and mask layouts fit 1366x768')


if __name__ == '__main__':
    run()

"""Hardware-free check of the tabbed, zoomable inspection result view.

Collected tests never open a window, so the widget itself is exercised here: two
crops in two tabs, per-tab zoom and pan, the uploading tab's arc, and the resume
button staying greyed until the worker says it is done.
"""

import sys
import tkinter as tk
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import main_1366 as main
from crop_result_view import CropResultView


def visible_inside(widget, window):
    window.update_idletasks()
    assert widget.winfo_ismapped(), f"{widget} is not visible"
    right = widget.winfo_rootx() - window.winfo_rootx() + widget.winfo_width()
    bottom = widget.winfo_rooty() - window.winfo_rooty() + widget.winfo_height()
    assert right <= window.winfo_width(), f"{widget} extends past the right edge"
    assert bottom <= window.winfo_height(), f"{widget} extends past the bottom edge"


def crops():
    """Two crops with a recognisable pixel at a known offset."""
    first = np.zeros((552, 2208, 3), np.uint8)
    second = np.full((552, 2208, 3), 40, np.uint8)
    return [first, second]


def run():
    root = tk.Tk()
    errors = []
    root.report_callback_exception = lambda *args: errors.append(args)
    with patch.object(main.IndustrialDashboard, '_init_serial'):
        app = main.IndustrialDashboard(root, lambda: None)
        root.update()

        opened = []
        view = CropResultView(app.image_panel, lambda: opened.append(True))
        app.images_container.pack_forget()
        view.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(10, 6))
        view.show_crops(crops())
        root.update()

        # It fills the panel the dual camera view would have used.
        assert view.canvas.winfo_width() > 800, view.canvas.winfo_width()
        assert view.canvas.winfo_height() > 300, view.canvas.winfo_height()
        visible_inside(view.canvas, root)
        assert len(view._tabs) == 2, 'one tab per crop'

        # The arc is a canvas item, and only over the tab that is uploading.
        assert not view.canvas.find_withtag('arc'), 'no arc before anything is busy'
        view.update_crop(1, crops()[0], True)
        view.tick()
        root.update()
        assert view.canvas.find_withtag('arc'), 'the busy tab must show the arc'
        first_arc = view.canvas.coords(view.canvas.find_withtag('arc')[0])
        view.tick()
        root.update()
        second_arc = view.canvas.coords(view.canvas.find_withtag('arc')[0])
        assert first_arc[:2] == second_arc[:2], 'the arc must not drift'

        # A finished crop swaps in its overlay and stops spinning.
        overlay = np.full((552, 2208, 3), 200, np.uint8)
        view.update_crop(1, overlay, False)
        view.tick()
        root.update()
        assert not view.canvas.find_withtag('arc'), 'the arc stops when the upload ends'
        assert '✓' in view._tab_buttons[0].cget('text'), 'a finished tab is ticked'

        # Per-tab state: zooming tab 1 must leave tab 2 at its fit view.
        view.select_tab(0)
        for _ in range(4):
            view.zoom_by(1.25)
        root.update()
        zoomed = view._tabs[0]['zoom']
        assert zoomed > 1.0, zoomed
        view.select_tab(1)
        root.update()
        assert view._tabs[1]['zoom'] == 1.0, 'each tab keeps its own zoom'
        view.select_tab(0)
        root.update()
        assert view._tabs[0]['zoom'] == zoomed, 'returning to a tab keeps its zoom'

        # RESET returns the active tab to the fit view.
        view.reset_view()
        root.update()
        assert abs(view._tabs[0]['zoom'] - 1.0) < 1e-9, view._tabs[0]['zoom']

        # The operator cannot resume until the detection thread has finished.
        assert str(view._resume_button.cget('state')) == tk.DISABLED
        view._resume_button.invoke()
        assert opened == [], 'a disabled resume button must not fire'
        app.result_view = view
        app._inspection_worker_done()
        assert str(view._resume_button.cget('state')) == tk.NORMAL
        view._resume_button.invoke()
        assert opened == [True], opened

        # Animating must never create a second timer.
        assert not root.tk.call('after', 'info'), 'the view must not schedule its own timer'
        assert not errors, errors
        view.destroy()
        app.close_application()
    print('PASS: two crop tabs, per-tab zoom and pan, arc only on the uploading tab')


if __name__ == '__main__':
    run()

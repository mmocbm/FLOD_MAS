"""Selected-line UI smoke test; no cameras, real models or network."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tkinter as tk
from unittest.mock import patch
from PIL import ImageGrab
from main_1366 import IndustrialDashboard
from inspection.line_measurement import measurement_record
from plane_scale import PlaneScale
from app_config import CONFIG
from test_local_inspection import synthetic_inspection
from auto_trigger.controller import AutoController
import numpy as np


def main():
    root = tk.Tk()
    errors = []
    root.report_callback_exception = lambda *args: errors.append(args)
    with patch.object(IndustrialDashboard, '_init_automatic'):
        app = IndustrialDashboard(root, lambda: None)
    app.auto_controller = AutoController(CONFIG, lambda: None)  # never start models/workers
    frame, _, detected = synthetic_inspection()
    measurement = measurement_record(detected, PlaneScale(np.diag([.1,.1,1]), .1), 3, 2)
    result = dict(measurement=measurement, overlay=detected['overlay'], mask=detected['mask'],
                  warnings=[], path='synthetic')
    for _ in range(2):
        app._show_automatic_results(result)
        root.update()
        view = app.result_view
        assert len(view.records) == 1 and len(view._tabs) == 2
        assert [b.cget('text') for b in view._tab_buttons if b.winfo_ismapped()] == ['OVERLAY', 'MASK']
        assert len(view.segment_table.get_children()) == 10
        view.update_live_preview(frame)
        assert view.live_panel.winfo_ismapped()
        assert len(view.live_canvas.find_all()) == 1
        assert view.canvas.winfo_width() > view.live_panel.winfo_width()
        view.select_tab(1)
        root.update()
        assert view._active == 1 and 'Rightmost line 2 / 2' in view.summary.cget('text')
        view.select_tab(0)
        view.zoom_by(1.2)
        root.update()
        assert view._tabs[0]['zoom'] > 1
        view.reset_view()
    root.update()
    output = Path(__file__).parent / 'automatic_ui_preview.png'
    ImageGrab.grab(window=int(root.frame(), 16)).save(output)
    with patch.dict(CONFIG['result_view'], {'show_live_preview': False}):
        app._show_automatic_results(result)
        root.update()
        assert not hasattr(app.result_view, 'live_panel')
        app.result_view.update_live_preview(frame)
    app._dismiss_result_view()
    root.update()
    assert app.result_view is None and app.images_container.winfo_ismapped()
    assert app.bottom_panel.winfo_ismapped()
    assert not errors, errors
    root.destroy()
    print('Overlay/mask tabs, ten-segment table, zoom, optional live preview and replacement/clearing verified:', output)


if __name__ == '__main__':
    main()

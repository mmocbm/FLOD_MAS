"""Selected-line UI smoke test; no cameras, real models or network."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tkinter as tk
from types import SimpleNamespace
from unittest.mock import patch
from PIL import ImageGrab
from main_1366 import IndustrialDashboard
from inspection.line_measurement import measurement_record
from inspection.overlay import annotated_overlay
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
    measurement = measurement_record(detected, PlaneScale(np.diag([.1,.1,1]), .1), 3, .05)
    result = dict(measurement=measurement, overlay=annotated_overlay(frame, measurement), mask=detected['mask'],
                  warnings=[], path='synthetic')
    for index in range(6):
        app._show_automatic_results(result)
        root.update()
        assert app.inspection_queue_label.winfo_ismapped()
        app.auto_controller.preparing_count = 2
        app._poll_automatic()
        assert '2 checking' in app.inspection_queue_label.cget('text')
        app.auto_controller.preparing_count = 0
        view = app.result_view
        assert len(view.records) == index+1 and len(view._tabs) == index+1
        assert view._active == index
        assert len(view._tab_buttons) == index+1
        view.update_live_preview(frame)
        assert view.live_panel.winfo_ismapped()
        assert len(view.live_canvas.find_all()) == 1
        assert view.canvas.winfo_width() > view.live_panel.winfo_width()
        assert f'Fabric {index+1}' in view.summary.cget('text')
        app._toggle_automatic_pause()
        assert app.auto_controller.paused
        assert app.auto_pause_button.cget('text') == 'PLAY'
        view.select_tab(0)
        view.zoom_by(2.5)
        root.update()
        assert view._tabs[0]['zoom'] > 1
        previous_offset = list(view._tabs[0]['offset'])
        view._on_pan_start(SimpleNamespace(x=300, y=300))
        view._on_pan_drag(SimpleNamespace(x=330, y=350))
        assert view._tabs[0]['offset'] != previous_offset
        view.reset_view()
        view.select_tab(index)
        app._toggle_automatic_pause()
    root.update()
    output = Path(__file__).parent / 'automatic_ui_preview.png'
    ImageGrab.grab(window=int(root.frame(), 16)).save(output)
    app._dismiss_result_view()
    with patch.dict(CONFIG['result_view'], {'show_live_preview': False}):
        app._show_automatic_results(result)
        root.update()
        assert not hasattr(app.result_view, 'live_panel')
        app.result_view.update_live_preview(frame)
    app._dismiss_result_view()
    root.update()
    assert app.result_view is None and app.images_container.winfo_ismapped()
    assert app.bottom_panel.winfo_ismapped()
    app.open_settings_selector()
    app.open_size_window()
    root.update()
    assert app.end_exclusion_entry.winfo_ismapped()
    assert app.end_exclusion_entry.winfo_rooty()+app.end_exclusion_entry.winfo_height() <= app.size_win.winfo_rooty()+app.size_win.winfo_height()
    app.end_exclusion_var.set('50')
    with patch('main_1366.messagebox.showerror') as error, patch('startup_settings.save_config') as save:
        app.save_settings()
        error.assert_called_once()
        save.assert_not_called()
    app.end_exclusion_var.set('7.5')
    with patch('startup_settings.save_config') as save, patch.object(app, '_show_success_popup'):
        app.save_settings()
        assert save.call_args.args[0]['inspection']['end_exclusion_percent'] == 7.5
        assert app.active_end_exclusion_percent == 7.5
    ImageGrab.grab(window=int(app.size_win.frame(), 16)).save(Path(__file__).parent/'end_exclusion_profile.png')
    app.size_win.destroy()
    assert not errors, errors
    root.destroy()
    print('Persistent line tabs, newest selection, pause/zoom, live preview and clearing verified:', output)


if __name__ == '__main__':
    main()

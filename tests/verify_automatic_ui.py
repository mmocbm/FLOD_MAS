"""Synthetic UI smoke test; never opens cameras, models or network connections."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tkinter as tk
from unittest.mock import patch
import numpy as np
from PIL import ImageGrab
from main_1366 import IndustrialDashboard
from inspection.measurement import measure_instances
from segmentation import Instance, SegmentationResult
from plane_scale import PlaneScale
from app_config import CONFIG


def main():
    root = tk.Tk()
    with patch.object(IndustrialDashboard, '_init_automatic'):
        app = IndustrialDashboard(root, lambda: None)
    frame = np.full((350, 1200, 3), 55, np.uint8)
    instances = []
    for index in range(4):
        x = 60 + index*280
        frame[30:320, x-20:x+80] = (85, 105, 125)
        frame[50:300, x:x+40] = (190, 195, 200)
        instances.append(Instance(np.array([[x,50],[x+40,50],[x+40,300],[x,300]], float)))
    fabrics = measure_instances(frame, SegmentationResult(tuple(instances), (1200,350)),
                                PlaneScale(np.diag([.1,.1,1]), .1), CONFIG['sam_detection'], 4, 1)
    for count in (1, 2, 3, 4, 2, 4):
        app._show_automatic_results({'fabrics':fabrics[:count], 'warnings':[], 'path':'synthetic'})
        root.update()
        assert len(app.result_view.records) == count
        assert len([b for b in app.result_view._tab_buttons if b.winfo_ismapped()]) == count
        app.result_view.update_live_preview(frame)
        assert app.result_view.live_panel.winfo_ismapped()
        assert app.result_view.canvas.winfo_ismapped()
        assert app.result_view.canvas.winfo_width() > app.result_view.live_panel.winfo_width()
        assert len(app.result_view.live_canvas.find_all()) == 1
    app.result_view.select_tab(3)
    root.update()
    assert 'Fabric 4' in app.result_view.summary.cget('text')
    output = Path(__file__).parent / 'automatic_ui_preview.png'
    # Capture only our own window, including when another app is foreground.
    ImageGrab.grab(window=int(root.frame(), 16)).save(output)
    with patch.dict(CONFIG['result_view'], {'show_live_preview': False}):
        app._show_automatic_results({'fabrics':fabrics[:1], 'warnings':[], 'path':'synthetic'})
        root.update()
        assert not hasattr(app.result_view, 'live_panel')
        app.result_view.update_live_preview(frame)
    app._dismiss_result_view()
    root.update()
    assert app.result_view is None
    assert app.images_container.winfo_ismapped()
    assert app.bottom_panel.winfo_ismapped()
    root.destroy()
    print('1–4 replacement tabs, optional side preview, per-tab summaries, and clearing verified:', output)


if __name__ == '__main__':
    main()

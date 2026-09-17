"""Open a hardware-free dashboard preview for visual review."""

import sys
import tkinter as tk
from pathlib import Path
from unittest.mock import patch
from PIL import ImageGrab

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main_1366 as main


root = tk.Tk()
root.title("Vision Inspection UI Preview")
with patch.object(main.IndustrialDashboard, '_init_serial'):
    app = main.IndustrialDashboard(root, lambda: None)
    app.update_progress(100, "Cameras ready")
    if "--capture" in sys.argv:
        def capture():
            root.update_idletasks()
            bounds = (
                root.winfo_rootx(), root.winfo_rooty(),
                root.winfo_rootx() + root.winfo_width(),
                root.winfo_rooty() + root.winfo_height(),
            )
            ImageGrab.grab(bbox=bounds).save(Path(__file__).parent / "ui_preview.png")
            root.destroy()
        root.attributes("-topmost", True)
        root.lift()
        root.focus_force()
        root.after(900, capture)
    root.mainloop()

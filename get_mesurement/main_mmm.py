import tkinter as tk
from tkinter import ttk
import threading
import os
import time
import subprocess
import json
from PIL import Image, ImageTk
from unet.predictor import UNetPredictor
import cv2
import numpy as np
from measure.measure import CylinderWidthMeasurer
from camera_handler import CameraHandler
import datetime
import sys
from app_config import CONFIG, project_path
from Image_Processing.get_mask import SkeletonSegmentationPredictor
from get_mesurement.measure_real import LinearFeatureInspectorOptimized


# ---------------- CONFIG ----------------
WINDOW_WIDTH = 1024
WINDOW_HEIGHT = 600
BOTTOM_PANEL_HEIGHT = 120
TITLE_BAR_HEIGHT = 30

# Camera configuration
CAMERA_INDEX_1, CAMERA_INDEX_2 = [c['index'] for c in CONFIG['cameras']]
CALIB_FILE_1, CALIB_FILE_2 = [project_path(c['calibration_file']) for c in CONFIG['cameras']]
Extrinsics_FILE_1, Extrinsics_FILE_2 = [project_path(c['extrinsics_file']) for c in CONFIG['cameras']]

# Model path
MODEL_PATH = r"unet\models\unet_AllAptern_CoveerdMask.h5"

# JSON file with pattern lengths
PATTERN_JSON_PATH = r"Files\pattern_lengths.json"
# ---------------------------------------

# Fixed pattern and size lists (replaces CSV dropdown)
FIXED_PATTERNS = ["10373", "10376", "10383", "10381", "10384"]
FIXED_SIZES = ["XXS", "XS", "S", "M", "L", "XL", "2X", "3X", "4X"]


def get_initial_fit_scale(pil_image, frame_width, frame_height):
    img_ratio = pil_image.width / pil_image.height
    frame_ratio = frame_width / frame_height
    return frame_width / pil_image.width if img_ratio > frame_ratio else frame_height / pil_image.height


def open_windows_keyboard():
    try:
        subprocess.Popen('osk.exe')
    except Exception as e:
        print(f"Could not open on-screen keyboard: {e}")


# ======================================================================
class IndustrialDashboard:
    def __init__(self, root, on_detect_thread_func, on_reset_callback):
        self.root = root
        self.on_detect_thread_func = on_detect_thread_func
        self.on_reset_callback = on_reset_callback

        # Camera handlers
        self.camera1 = None
        self.camera2 = None
        
        # --- zoom / pan state (per canvas) ---
        self.original_full_res_1 = None
        self.base_scale_1 = 1.0
        self.zoom_level_1 = 1.0
        self.pan_x_1, self.pan_y_1 = 0, 0

        self.original_full_res_2 = None
        self.base_scale_2 = 1.0
        self.zoom_level_2 = 1.0
        self.pan_x_2, self.pan_y_2 = 0, 0

        # --- video / frame state ---
        self.video_streaming = False
        self.video_paused = False

        self.result_image_1 = None
        self.result_image_2 = None

        # --- mode flags ---
        self.inspection_mode = False
        self.overlay_mode = False

        self.overlay_result_1 = None
        self.overlay_result_2 = None

        # --- calibration preview ---
        self.cap1 = None
        self.cap2 = None
        self.camera_streaming = False

        # --- settings variables ---
        self.n_segments_var = tk.IntVar(value=3)          # default 3 segments
        self.main_pattern_var = tk.StringVar()
        self.size_var = tk.StringVar()
        self.len_threshold_var = tk.StringVar(value="± 1mm")
        self.wid_threshold_var = tk.StringVar(value="± 1mm")
        self.strip_width_var = tk.StringVar(value="2")
        self.enable_check_var = tk.BooleanVar(value=True)

        # Set defaults for pattern and size
        self.main_pattern_var.set("10373")
        self.size_var.set("XXS")
        self.available_sizes = FIXED_SIZES          # keep for compatibility

        # Lengths loaded from JSON
        self.length_camera1 = 0.0
        self.length_camera2 = 0.0
        self.expected_lengths_list = []
        self.current_pattern_sizes = []              # available sizes for current pattern
        self._load_lengths_from_json()                # initial load

        # --- active (saved) values used during detection ---
        self.active_main_pattern = self.main_pattern_var.get()
        self.active_size = self.size_var.get()
        self.active_len_threshold = self.len_threshold_var.get()
        self.active_wid_threshold = self.wid_threshold_var.get()
        self.active_segments = self.n_segments_var.get()
        self.active_strip_width = self._validate_strip_width(self.strip_width_var.get())
        self.active_enable_check = self.enable_check_var.get()
        self.active_length_camera1 = self.length_camera1
        self.active_length_camera2 = self.length_camera2
        self.active_expected_lengths = self.expected_lengths_list.copy()
        self.active_pattern_sizes = self.current_pattern_sizes.copy()

        # --- window chrome ---
        self.root.overrideredirect(True)

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width // 2) - (WINDOW_WIDTH // 2)
        y = (screen_height // 2) - (WINDOW_HEIGHT // 2)
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}+{x}+{y}")
        self.root.configure(bg="#1e1e1e")

        self._create_custom_title_bar()
        self._create_layout()

    # ==============================================================
    # helpers
    # ==============================================================
    def _validate_strip_width(self, value):
        try:
            w = float(value)
            return w if w > 0 else 2.0
        except (ValueError, TypeError):
            return 2.0

    def _parse_tolerance(self, tol_string, default=1.0):
        """Parse tolerance string like '± 1mm' to float value"""
        try:
            return float(tol_string.replace("±", "").replace("mm", "").strip())
        except (ValueError, AttributeError):
            return default

    def _load_lengths_from_json(self):
        """Load the length list for current pattern/size from JSON."""
        pattern = self.main_pattern_var.get()
        size = self.size_var.get()

        self.length_camera1 = 0.0
        self.length_camera2 = 0.0
        self.expected_lengths_list = []
        self.current_pattern_sizes = []

        if not os.path.exists(PATTERN_JSON_PATH):
            print(f"JSON file not found: {PATTERN_JSON_PATH}")
            return

        try:
            with open(PATTERN_JSON_PATH, 'r') as f:
                data = json.load(f)

            pattern_data = data.get(pattern)
            if pattern_data is None:
                print(f"Pattern '{pattern}' not found in JSON")
                return

            # Store available sizes for this pattern
            self.current_pattern_sizes = list(pattern_data.keys())
            print(f"Available sizes for {pattern}: {self.current_pattern_sizes}")

            size_list = pattern_data.get(size)
            if size_list is None:
                print(f"Size '{size}' not found for pattern '{pattern}'")
                return

            self.expected_lengths_list = [float(val) for val in size_list]
            print(f"Loaded expected lengths for {pattern} {size}: {self.expected_lengths_list}")

            if len(size_list) >= 2:
                self.length_camera1 = float(size_list[0])
                self.length_camera2 = float(size_list[1])
                print(f"Camera lengths – Cam1: {self.length_camera1}mm, Cam2: {self.length_camera2}mm")

        except Exception as e:
            print(f"Error reading JSON: {e}")

    # ==============================================================
    # title-bar / window management
    # ==============================================================
    def _create_custom_title_bar(self):
        self.title_bar = tk.Frame(self.root, bg="#111", height=TITLE_BAR_HEIGHT)
        self.title_bar.pack(side=tk.TOP, fill=tk.X)
        tk.Label(self.title_bar, text="Industrial Vision Dashboard",
                 fg="#888", bg="#111", font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=10)
        tk.Button(self.title_bar, text="✕", command=self.close_application,
                  bg="#111", fg="white", bd=0, padx=12, pady=5,
                  activebackground="#e81123").pack(side=tk.RIGHT, fill=tk.Y)
        tk.Button(self.title_bar, text="—", command=self.minimize_window,
                  bg="#111", fg="white", bd=0, padx=12, pady=5,
                  activebackground="#333").pack(side=tk.RIGHT, fill=tk.Y)
        self.title_bar.bind("<ButtonPress-1>", self._start_move)
        self.title_bar.bind("<B1-Motion>", self._do_move)

    def close_application(self):
        self.video_streaming = False
        if self.camera1: self.camera1.release()
        if self.camera2: self.camera2.release()
        self.root.destroy()

    def minimize_window(self, target_win=None):
        win = target_win or self.root
        win.overrideredirect(False)
        win.iconify()
        win.bind("<FocusIn>", lambda e: self._restore_frameless(e, win))

    def _restore_frameless(self, event, win):
        if win.state() == 'normal':
            win.overrideredirect(True)
            win.unbind("<FocusIn>")

    def _start_move(self, event):
        self.x, self.y = event.x, event.y

    def _do_move(self, event):
        win = event.widget.winfo_toplevel()
        dx, dy = event.x - self.x, event.y - self.y
        win.geometry(f"+{win.winfo_x() + dx}+{win.winfo_y() + dy}")

    # ==============================================================
    # layout
    # ==============================================================
    def _create_layout(self):
        self.main_frame = tk.Frame(self.root, bg="#1e1e1e")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        self._create_image_panel()
        self._create_bottom_panel()

    # --------------------------------------------------------------
    # bottom panel (simplified)
    # --------------------------------------------------------------
    def _create_bottom_panel(self):
        self.bottom_panel = tk.Frame(self.main_frame, height=BOTTOM_PANEL_HEIGHT, bg="#2a2a2a")
        self.bottom_panel.pack(side=tk.BOTTOM, fill=tk.X)
        self.bottom_panel.pack_propagate(False)

        # ---- left: pattern & size (large) + length list ----
        left_section = tk.Frame(self.bottom_panel, bg="#2a2a2a")
        left_section.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Large pattern and size display
        self.lbl_large_pattern_size = tk.Label(
            left_section,
            text=f"{self.active_main_pattern} - {self.active_size}",
            fg="#00c8ff", bg="#2a2a2a",
            font=("Segoe UI", 20, "bold")
        )
        self.lbl_large_pattern_size.pack(anchor="w")

        # Label showing the actual length list (in mm) for the selected pattern/size
        self.lbl_length_list = tk.Label(
            left_section,
            text=f"Lengths: {self._format_length_list(self.active_expected_lengths)}",
            fg="#aaa", bg="#2a2a2a",
            font=("Segoe UI", 9),
            wraplength=400,
            justify=tk.LEFT
        )
        self.lbl_length_list.pack(anchor="w", pady=(0, 5))

        # ---- center: PASS / FAIL ----
        center_section = tk.Frame(self.bottom_panel, bg="#2a2a2a")
        center_section.pack(side=tk.LEFT, padx=20)
        self.status_result_label = tk.Label(center_section, text="LIVE", fg="#00c8ff", bg="#2a2a2a",
                                            font=("Segoe UI", 20, "bold"))
        self.status_result_label.pack(pady=10)

        # ---- right: buttons (unchanged) ----
        right_section = tk.Frame(self.bottom_panel, bg="#2a2a2a")
        right_section.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=5)

        btn_font = ("Segoe UI", 9, "bold")
        btn_font_large = ("Segoe UI", 12, "bold")

        self.btn_container = tk.Frame(right_section, bg="#2a2a2a")
        self.btn_container.pack(expand=True)

        tk.Button(self.btn_container, text="Settings", font=btn_font, bg="#444", fg="white",
                  relief=tk.FLAT, width=10, height=2,
                  command=self.open_settings_selector).pack(side=tk.LEFT, padx=3)

        tk.Button(self.btn_container, text="RESET", font=btn_font_large, bg="#d32f2f", fg="white",
                  relief=tk.FLAT, width=15, height=3,
                  command=self.on_reset_callback).pack(side=tk.LEFT, padx=3)

        self.detect_btn = tk.Button(self.btn_container, text="START DETECTION", font=btn_font_large,
                                    bg="#1e5fd6", fg="white", relief=tk.FLAT,
                                    width=18, height=3, command=self.start_detect_thread)
        self.detect_btn.pack(side=tk.LEFT, padx=3)

        # hidden until needed
        self.inspection_btn = tk.Button(self.btn_container, text="INSPECTION", font=btn_font_large,
                                        bg="#4CAF50", fg="white", relief=tk.FLAT,
                                        width=18, height=3, command=self.switch_to_inspection_mode)
        self.overlay_btn = tk.Button(self.btn_container, text="OVERLAY", font=btn_font_large,
                                     bg="#ff9800", fg="white", relief=tk.FLAT,
                                     width=18, height=3, command=self.switch_to_overlay_mode)

    def _format_length_list(self, lengths):
        """Format the list of lengths for display."""
        if not lengths:
            return "No data"
        # Show up to 10 values, then "..."
        if len(lengths) > 10:
            return ", ".join(f"{l:.1f}" for l in lengths[:10]) + " ..."
        else:
            return ", ".join(f"{l:.1f}" for l in lengths)

    # ==============================================================
    # settings windows
    # ==============================================================
    def open_settings_selector(self):
        self.video_paused = True
        self.root.withdraw()

        self.sel_win = tk.Toplevel(self.root)
        self.sel_win.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}+{self.root.winfo_x()}+{self.root.winfo_y()}")
        self.sel_win.configure(bg="#1e1e1e")
        self.sel_win.overrideredirect(True)
        self.sel_win.attributes("-topmost", True)

        header = tk.Frame(self.sel_win, bg="#111", height=TITLE_BAR_HEIGHT)
        header.pack(fill=tk.X)
        tk.Button(header, text="🏠 Home",
                  command=lambda: [self.sel_win.destroy(), self.root.deiconify(), self.resume_video()],
                  bg="#111", fg="#00c8ff", bd=0, padx=15, font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, fill=tk.Y)
        tk.Label(header, text="Configuration Selector", fg="#888", bg="#111", font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=10)
        tk.Button(header, text="✕", command=self.close_application, bg="#111", fg="white", bd=0, padx=12).pack(side=tk.RIGHT, fill=tk.Y)
        header.bind("<ButtonPress-1>", self._start_move)
        header.bind("<B1-Motion>", self._do_move)

        container = tk.Frame(self.sel_win, bg="#1e1e1e")
        container.pack(expand=True)
        tk.Label(container, text="System Settings", fg="white", bg="#1e1e1e",
                 font=("Segoe UI", 24, "bold")).pack(pady=30)

        btn_style = {"font": ("Segoe UI", 14, "bold"), "bg": "#333", "fg": "white",
                     "relief": tk.FLAT, "width": 25, "height": 3}

        # Existing buttons
        tk.Button(container, text="1. Pattern Setting",      command=self.open_pattern_window,      **btn_style).pack(pady=10)
        tk.Button(container, text="2. Camera Calibration",   command=self.run_external_calibration, **btn_style).pack(pady=10)

        # New buttons for checking individual cameras
        tk.Button(container, text="Check Camera 1",          command=self.check_camera1,            **btn_style).pack(pady=10)
        tk.Button(container, text="Check Camera 2",          command=self.check_camera2,            **btn_style).pack(pady=10)

    def run_external_calibration(self):
        """Launches external calibration app and relaunches main after it closes."""
        self.video_streaming = False
        if self.camera1: self.camera1.release()
        if self.camera2: self.camera2.release()
        
        calib_script = os.path.join(os.getcwd(), "CalibrateAPP", "calibration_ui.py")
        
        def monitor_and_restart():
            try:
                subprocess.run([sys.executable, calib_script], check=False)
                print("Calibration finished. Relaunching main app...")
                subprocess.Popen([sys.executable] + sys.argv)
                os._exit(0)
            except Exception as e:
                print(f"Error in monitor_and_restart: {e}")
                self.root.after(0, self.root.deiconify)

        try:
            threading.Thread(target=monitor_and_restart, daemon=True).start()
            self.root.withdraw()
            if hasattr(self, 'sel_win'): 
                self.sel_win.destroy()
        except Exception as e:
            print(f"Error launching calibration app: {e}")
            self._show_error_popup(f"Could not open calibration app:\n{e}")
            self.root.deiconify()

    def check_camera1(self):
        """Launch check script for camera 1 and return to main window after it closes."""
        self._run_check_script("CalibrateAPP/check_Camera_1.py")

    def check_camera2(self):
        """Launch check script for camera 2 and return to main window after it closes."""
        self._run_check_script("CalibrateAPP/check_Camera_2.py")

    def _run_check_script(self, script_path):
        """Generic method to run a check script, pause video, hide window, then resume."""
        # Stop video streaming
        self.video_streaming = False
        if self.camera1:
            self.camera1.release()
        if self.camera2:
            self.camera2.release()
        self.camera1 = None
        self.camera2 = None

        # Hide main window and settings window
        self.root.withdraw()
        if hasattr(self, 'sel_win'):
            self.sel_win.destroy()

        # Run the script in a separate thread to avoid blocking UI (but we want to wait)
        # Use subprocess.run to block until script finishes
        def run_and_resume():
            try:
                full_path = os.path.join(os.getcwd(), script_path)
                print(f"Running {full_path}...")
                subprocess.run([sys.executable, full_path], check=False)
                print(f"{script_path} finished.")
            except Exception as e:
                print(f"Error running {script_path}: {e}")
            finally:
                # After script finishes, restart cameras and show main window
                self.root.after(0, self._restart_cameras_and_show)

        threading.Thread(target=run_and_resume, daemon=True).start()

    def _restart_cameras_and_show(self):
        """Re-initialize cameras and show the main window."""
        try:
            self.camera1 = CameraHandler(CAMERA_INDEX_1, CALIB_FILE_1)
            self.camera2 = CameraHandler(CAMERA_INDEX_2, CALIB_FILE_2)
            self.video_streaming = True
            self.video_paused = False
            self.update_video_feed()
        except Exception as e:
            print(f"Error restarting cameras: {e}")
        self.root.deiconify()

    # --------------------------------------------------------------
    # Pattern window (button-based selector) with dropdown for segments
    # --------------------------------------------------------------
    def open_pattern_window(self):
        self.sel_win.destroy()

        self.pat_win = tk.Toplevel(self.root)
        self.pat_win.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}+{self.root.winfo_x()}+{self.root.winfo_y()}")
        self.pat_win.configure(bg="#1e1e1e")
        self.pat_win.overrideredirect(True)
        self.pat_win.attributes("-topmost", True)

        # ================= HEADER =================
        header = tk.Frame(self.pat_win, bg="#111", height=TITLE_BAR_HEIGHT)
        header.pack(fill=tk.X)

        tk.Button(
            header, text="← Back",
            command=lambda: [self.pat_win.destroy(), self.open_settings_selector()],
            bg="#111", fg="#00c8ff", bd=0, padx=15,
            font=("Segoe UI", 9, "bold")
        ).pack(side=tk.LEFT)

        tk.Label(
            header, text="Pattern Setting",
            fg="#888", bg="#111",
            font=("Segoe UI", 9)
        ).pack(side=tk.LEFT, padx=10)

        tk.Button(
            header, text="✕",
            command=self.close_application,
            bg="#111", fg="white", bd=0, padx=12
        ).pack(side=tk.RIGHT)

        # ================= CONTENT =================
        content = tk.Frame(self.pat_win, bg="#1e1e1e")
        content.pack(expand=True, fill=tk.BOTH, padx=40, pady=30)

        # ---------- PATTERN ----------
        tk.Label(
            content, text="Pattern",
            fg="#00c8ff", bg="#1e1e1e",
            font=("Segoe UI", 13, "bold")
        ).pack(anchor="w", pady=(10, 6))

        pat_frame = tk.Frame(content, bg="#1e1e1e")
        pat_frame.pack(anchor="w")

        self._pattern_buttons = {}
        for pat in FIXED_PATTERNS:
            btn = tk.Button(
                pat_frame,
                text=pat,
                width=10,
                height=2,
                bg="#3a3a3a",
                fg="white",
                relief=tk.FLAT,
                font=("Segoe UI", 10, "bold"),
                command=lambda p=pat: self._select_pattern(p)
            )
            btn.pack(side=tk.LEFT, padx=8, pady=6)
            self._pattern_buttons[pat] = btn

        # ---------- SIZE ----------
        tk.Label(
            content, text="Size",
            fg="#00c8ff", bg="#1e1e1e",
            font=("Segoe UI", 13, "bold")
        ).pack(anchor="w", pady=(25, 6))

        size_frame = tk.Frame(content, bg="#1e1e1e")
        size_frame.pack(anchor="w")

        self._size_buttons = {}
        for size in FIXED_SIZES:
            btn = tk.Button(
                size_frame,
                text=size,
                width=8,
                height=2,
                bg="#2e7d32",
                fg="white",
                relief=tk.FLAT,
                font=("Segoe UI", 10, "bold"),
                command=lambda s=size: self._select_size(s)
            )
            btn.pack(side=tk.LEFT, padx=6, pady=6)
            self._size_buttons[size] = btn

        # ---------- SETTINGS (with dropdowns) ----------
        settings = tk.Frame(content, bg="#1e1e1e")
        settings.pack(fill=tk.X, pady=30)

        # Length tolerance (expanded options)
        tk.Label(settings, text="Length Tolerance", fg="#888", bg="#1e1e1e").grid(row=0, column=0, sticky="w")
        self.len_tolerance_menu = tk.OptionMenu(
            settings, self.len_threshold_var,
            "± 0.5mm", "± 1mm", "± 2mm", "± 3mm", "± 5mm", "± 10mm"
        )
        self.len_tolerance_menu.config(bg="#3a3a3a", fg="white", font=("Segoe UI", 10), relief=tk.FLAT)
        self.len_tolerance_menu["menu"].config(bg="#3a3a3a", fg="white")
        self.len_tolerance_menu.grid(row=1, column=0, padx=10, sticky="w")

        # Width tolerance (expanded options)
        tk.Label(settings, text="Width Tolerance", fg="#888", bg="#1e1e1e").grid(row=0, column=1, sticky="w")
        self.wid_tolerance_menu = tk.OptionMenu(
            settings, self.wid_threshold_var,
            "± 0.5mm", "± 1mm", "± 2mm", "± 3mm", "± 5mm", "± 10mm"
        )
        self.wid_tolerance_menu.config(bg="#3a3a3a", fg="white", font=("Segoe UI", 10), relief=tk.FLAT)
        self.wid_tolerance_menu["menu"].config(bg="#3a3a3a", fg="white")
        self.wid_tolerance_menu.grid(row=1, column=1, padx=10, sticky="w")

        # Segments dropdown (1-6)
        tk.Label(settings, text="Segments", fg="#888", bg="#1e1e1e").grid(row=0, column=2, sticky="w")
        self.segments_combo = ttk.Combobox(
            settings,
            values=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"],
            state="readonly",
            width=8,
            font=("Segoe UI", 10)
        )
        self.segments_combo.set(str(self.n_segments_var.get()))
        self.segments_combo.grid(row=1, column=2, padx=10, sticky="w")
        # Bind to update the IntVar
        self.segments_combo.bind("<<ComboboxSelected>>", lambda e: self.n_segments_var.set(int(self.segments_combo.get())))

        # ---------- ENABLE INSPECT ----------
        tk.Checkbutton(
            content,
            text="Enable Inspect (PASS / FAIL)",
            variable=self.enable_check_var,
            bg="#1e1e1e",
            fg="white",
            selectcolor="#333",
            font=("Segoe UI", 10, "bold")
        ).pack(anchor="w", pady=15)

        # ---------- SAVE ----------
        tk.Button(
            content,
            text="SAVE",
            bg="#1e5fd6",
            fg="white",
            font=("Segoe UI", 14, "bold"),
            width=14,
            height=2,
            relief=tk.FLAT,
            command=self.save_settings
        ).pack(pady=25)

        # ---------- DEFAULT HIGHLIGHT ----------
        self._select_pattern(self.main_pattern_var.get())
        self._select_size(self.size_var.get())

    # ---------- Helper methods for pattern/size buttons ----------
    def _select_pattern(self, pattern):
        self.main_pattern_var.set(pattern)
        self._load_lengths_from_json()
        for p, btn in self._pattern_buttons.items():
            btn.config(
                bg="#1e5fd6" if p == pattern else "#3a3a3a",
                fg="white"
            )
        # Update the size list display in bottom panel (will be applied on save)

    def _select_size(self, size):
        self.size_var.set(size)
        self._load_lengths_from_json()
        for s, btn in self._size_buttons.items():
            btn.config(
                bg="#00c853" if s == size else "#2e7d32",
                fg="white"
            )

    def save_settings(self):
        strip_value = self.strip_width_var.get()
        validated_width = self._validate_strip_width(strip_value)

        # Update segments from combobox (if changed)
        if hasattr(self, 'segments_combo'):
            self.n_segments_var.set(int(self.segments_combo.get()))

        # Commit to active values
        self.active_main_pattern      = self.main_pattern_var.get()
        self.active_size              = self.size_var.get()
        self.active_len_threshold     = self.len_threshold_var.get()
        self.active_wid_threshold     = self.wid_threshold_var.get()
        self.active_segments          = self.n_segments_var.get()
        self.active_strip_width       = validated_width
        self.active_enable_check      = self.enable_check_var.get()
        self.active_length_camera1    = self.length_camera1
        self.active_length_camera2    = self.length_camera2
        self.active_expected_lengths  = self.expected_lengths_list.copy()
        self.active_pattern_sizes     = self.current_pattern_sizes.copy()

        # Refresh bottom-panel displays
        self.lbl_large_pattern_size.config(text=f"{self.active_main_pattern} - {self.active_size}")
        self.lbl_length_list.config(text=f"Lengths: {self._format_length_list(self.active_expected_lengths)}")

        self._show_success_popup()

    def _show_success_popup(self):
        pop = tk.Toplevel(self.pat_win)
        pop.overrideredirect(True)
        pop.attributes("-topmost", True)
        pop.configure(bg="#2a2a2a")

        px = self.pat_win.winfo_x() + (WINDOW_WIDTH // 2) - 140
        py = self.pat_win.winfo_y() + (WINDOW_HEIGHT // 2) - 70
        pop.geometry(f"280x140+{px}+{py}")

        tk.Label(pop, text="✓", fg="#4CAF50", bg="#2a2a2a", font=("Segoe UI", 30)).pack(pady=(15, 0))
        tk.Label(pop, text="Settings Saved Successfully", fg="white", bg="#2a2a2a",
                 font=("Segoe UI", 10, "bold")).pack()

        def close_pop():
            pop.destroy()
            self.pat_win.destroy()
            self.open_settings_selector()

        tk.Button(pop, text="OK", bg="#444", fg="white", relief=tk.FLAT, width=10, command=close_pop).pack(pady=15)

    def _show_error_popup(self, message):
        pop = tk.Toplevel(self.root)
        pop.overrideredirect(True)
        pop.attributes("-topmost", True)
        pop.configure(bg="#2a2a2a")

        px = self.root.winfo_x() + (WINDOW_WIDTH // 2) - 190
        py = self.root.winfo_y() + (WINDOW_HEIGHT // 2) - 120
        pop.geometry(f"380x240+{px}+{py}")

        title_bar = tk.Frame(pop, bg="#1a1a1a", height=30)
        title_bar.pack(fill=tk.X)
        title_bar.pack_propagate(False)
        tk.Label(title_bar, text="Error", fg="#888", bg="#1a1a1a", font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=10)
        tk.Button(title_bar, text="✕", command=pop.destroy,
                  bg="#1a1a1a", fg="white", bd=0, padx=10, pady=5,
                  font=("Segoe UI", 10), activebackground="#e81123").pack(side=tk.RIGHT, fill=tk.Y)

        content_frame = tk.Frame(pop, bg="#2a2a2a")
        content_frame.pack(fill=tk.BOTH, expand=True, pady=15)
        tk.Label(content_frame, text="⚠",      fg="#F44336", bg="#2a2a2a", font=("Segoe UI", 48)).pack(pady=(15, 10))
        tk.Label(content_frame, text=message,  fg="white",   bg="#2a2a2a", font=("Segoe UI", 16, "bold")).pack(pady=(5, 20))

        btn_frame = tk.Frame(content_frame, bg="#2a2a2a")
        btn_frame.pack(pady=(0, 15))

        tk.Button(btn_frame, text="Go to Settings", bg="#1e5fd6", fg="white",
                  relief=tk.FLAT, width=14, height=2, font=("Segoe UI", 10, "bold"),
                  command=lambda: [pop.destroy(), self.open_settings_selector()]).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="OK", bg="#444", fg="white",
                  relief=tk.FLAT, width=12, height=2, font=("Segoe UI", 10),
                  command=pop.destroy).pack(side=tk.LEFT, padx=5)

    def resume_video(self):
        self.video_paused = False

    # ==============================================================
    # image panel (dual canvases + zoom toolbars) – unchanged
    # ==============================================================
    def _create_image_panel(self):
        self.image_panel = tk.Frame(self.main_frame, bg="#111111")
        self.image_panel.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        images_container = tk.Frame(self.image_panel, bg="#111111")
        images_container.pack(fill=tk.BOTH, expand=True)

        half_w = WINDOW_WIDTH // 2

        # ---- left ----
        left_frame = tk.Frame(images_container, bg="#0a0a0a")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 1))
        tk.Label(left_frame, text="Camera 1", bg="#0a0a0a", fg="#888", font=("Segoe UI", 9)).pack(pady=(5, 0))

        self.canvas_1 = tk.Canvas(left_frame, bg="#000000", highlightthickness=0)
        self.canvas_1.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.toolbar_1 = tk.Frame(self.canvas_1, bg="#2a2a2a", bd=1, relief=tk.RAISED)
        self.canvas_1.create_window(half_w - 25, 20, anchor="ne", window=self.toolbar_1)

        btn_s = {"bg": "#333", "fg": "#00c8ff", "font": ("Arial", 9, "bold"), "width": 2, "relief": tk.FLAT}
        tk.Button(self.toolbar_1, text="+", command=self.zoom_in_1,    **btn_s).pack(side=tk.TOP, pady=1, padx=1)
        tk.Button(self.toolbar_1, text="-", command=self.zoom_out_1,   **btn_s).pack(side=tk.TOP, pady=1, padx=1)
        tk.Button(self.toolbar_1, text="⟲", command=self.reset_view_1, **btn_s).pack(side=tk.TOP, pady=1, padx=1)

        # ---- right ----
        right_frame = tk.Frame(images_container, bg="#0a0a0a")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(1, 0))
        tk.Label(right_frame, text="Camera 2", bg="#0a0a0a", fg="#888", font=("Segoe UI", 9)).pack(pady=(5, 0))

        self.canvas_2 = tk.Canvas(right_frame, bg="#000000", highlightthickness=0)
        self.canvas_2.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.toolbar_2 = tk.Frame(self.canvas_2, bg="#2a2a2a", bd=1, relief=tk.RAISED)
        self.canvas_2.create_window(half_w - 25, 20, anchor="ne", window=self.toolbar_2)

        tk.Button(self.toolbar_2, text="+", command=self.zoom_in_2,    **btn_s).pack(side=tk.TOP, pady=1, padx=1)
        tk.Button(self.toolbar_2, text="-", command=self.zoom_out_2,   **btn_s).pack(side=tk.TOP, pady=1, padx=1)
        tk.Button(self.toolbar_2, text="⟲", command=self.reset_view_2, **btn_s).pack(side=tk.TOP, pady=1, padx=1)

        # ---- progress bar ----
        self.loading_container = tk.Frame(self.image_panel, bg="#1e1e1e", height=20)
        self.loading_container.pack(side=tk.BOTTOM, fill=tk.X)

        self.progress_label = tk.Label(self.loading_container, text="Live Feed",
                                       fg="#888", bg="#1e1e1e", font=("Segoe UI", 8))
        self.progress_label.pack(side=tk.LEFT, padx=10)

        self.style = ttk.Style()
        self.style.theme_use('default')
        self.style.configure("Sleek.Horizontal.TProgressbar", thickness=3,
                             troughcolor='#111', background='#a855f7', bordercolor="#1e1e1e")
        self.progress_bar = ttk.Progressbar(self.loading_container, orient=tk.HORIZONTAL,
                                            mode='determinate', style="Sleek.Horizontal.TProgressbar")
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        # ---- bind pan/zoom ----
        self.canvas_1.bind("<ButtonPress-1>",  self._on_pan_start_1)
        self.canvas_1.bind("<B1-Motion>",      self._on_pan_drag_1)
        self.canvas_1.bind("<MouseWheel>",     self._on_mouse_wheel_1)

        self.canvas_2.bind("<ButtonPress-1>",  self._on_pan_start_2)
        self.canvas_2.bind("<B1-Motion>",      self._on_pan_drag_2)
        self.canvas_2.bind("<MouseWheel>",     self._on_mouse_wheel_2)

    # ==============================================================
    # video streaming – unchanged
    # ==============================================================
    def start_video_stream(self):
        try:
            self.camera1 = CameraHandler(CAMERA_INDEX_1, CALIB_FILE_1)
            self.camera2 = CameraHandler(CAMERA_INDEX_2, CALIB_FILE_2)

            self.video_streaming = True
            self.video_paused = False
            self.update_video_feed()

        except Exception as e:
            print(f"Video init error: {e}")

    def update_video_feed(self):
        if not self.video_streaming:
            return
        if self.video_paused:
            self.root.after(30, self.update_video_feed)
            return

        ret1 = self.camera1.read_frame()
        ret2 = self.camera2.read_frame()

        if ret1 and ret2:
            undist1 = self.camera1.get_undistorted_frame()
            undist2 = self.camera2.get_undistorted_frame()
            
            self._display_video_frame(undist1, 1)
            self._display_video_frame(undist2, 2)

        self.root.after(30, self.update_video_feed)

    def _display_video_frame(self, frame, canvas_num):
        cw = (WINDOW_WIDTH // 2) - 10
        ch = WINDOW_HEIGHT - TITLE_BAR_HEIGHT - BOTTOM_PANEL_HEIGHT - 40

        frame_resized = cv2.resize(frame, (cw, ch))

        if self.overlay_mode:
            overlay = self.overlay_result_1 if canvas_num == 1 else self.overlay_result_2
            if overlay is not None:
                overlay_resized = cv2.resize(overlay, (cw, ch))
                frame_resized = cv2.addWeighted(frame_resized, 0.6, overlay_resized, 0.4, 0)

        pil_img = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))

        if canvas_num == 1:
            self.tk_image_1 = ImageTk.PhotoImage(pil_img)
            self.canvas_1.delete("img")
            self.canvas_1.create_image(cw // 2, ch // 2, image=self.tk_image_1, anchor=tk.CENTER, tags="img")
        else:
            self.tk_image_2 = ImageTk.PhotoImage(pil_img)
            self.canvas_2.delete("img")
            self.canvas_2.create_image(cw // 2, ch // 2, image=self.tk_image_2, anchor=tk.CENTER, tags="img")

    # ==============================================================
    # inspection / overlay toggle – unchanged
    # ==============================================================
    def switch_to_inspection_mode(self):
        self.overlay_mode = False
        self.video_paused = True

        self.inspection_btn.pack_forget()
        self.overlay_btn.pack(side=tk.LEFT, padx=3)

        if self.result_image_1 and self.result_image_2:
            self._update_canvas_1()
            self._update_canvas_2()

    def switch_to_overlay_mode(self):
        self.overlay_mode = True

        self.overlay_btn.pack_forget()
        self.inspection_btn.pack(side=tk.LEFT, padx=3)

        self.zoom_level_1 = self.zoom_level_2 = 1.0
        self.pan_x_1 = self.pan_y_1 = 0
        self.pan_x_2 = self.pan_y_2 = 0

        self.video_paused = False

    # ==============================================================
    # status / progress helpers
    # ==============================================================
    def set_pass_fail(self, status):
        colours = {"PASS": "#4CAF50", "FAIL": "#F44336", "READY": "#888", "LIVE": "#00c8ff", "OVERLAY": "#00c853"}
        self.status_result_label.config(text=status, fg=colours.get(status, "#888"))

    def update_progress(self, value, text):
        self.progress_bar['value'] = value
        self.progress_label.config(text=text)
        self.root.update_idletasks()

    # ==============================================================
    # detection trigger
    # ==============================================================
    def start_detect_thread(self):
        if not self.active_main_pattern:
            self._show_error_popup("Set Pattern First")
            return
        if not self.active_size:
            self._show_error_popup("Set Size First")
            return
        if not self.active_expected_lengths:
            self._show_error_popup("No length data available for this pattern/size")
            return

        self.video_paused = True
        self.detect_btn.config(state=tk.DISABLED)
        threading.Thread(target=self.on_detect_thread_func).start()

    # ==============================================================
    # show result images after detection – unchanged
    # ==============================================================
    def show_image(self, pil_image, canvas_num=1, switch_buttons=True):
        if canvas_num == 1:
            self.result_image_1 = pil_image
            self.original_full_res_1 = pil_image
            cw = (WINDOW_WIDTH // 2) - 10
            ch = WINDOW_HEIGHT - TITLE_BAR_HEIGHT - BOTTOM_PANEL_HEIGHT - 40
            self.base_scale_1 = get_initial_fit_scale(pil_image, cw, ch)
            self.zoom_level_1, self.pan_x_1, self.pan_y_1 = 1.0, 0, 0
            self._update_canvas_1()
        else:
            self.result_image_2 = pil_image
            self.original_full_res_2 = pil_image
            cw = (WINDOW_WIDTH // 2) - 10
            ch = WINDOW_HEIGHT - TITLE_BAR_HEIGHT - BOTTOM_PANEL_HEIGHT - 40
            self.base_scale_2 = get_initial_fit_scale(pil_image, cw, ch)
            self.zoom_level_2, self.pan_x_2, self.pan_y_2 = 1.0, 0, 0
            self._update_canvas_2()

        if canvas_num == 2 and switch_buttons:
            self.inspection_mode = True
            self.overlay_mode = False
            self.detect_btn.pack_forget()
            self.inspection_btn.pack_forget()
            self.overlay_btn.pack(side=tk.LEFT, padx=3)
            self.detect_btn.config(state=tk.NORMAL)

    # --------------------------------------------------------------
    # canvas update – unchanged
    # --------------------------------------------------------------
    def _update_canvas_1(self):
        if self.original_full_res_1 is None: return
        scale = self.base_scale_1 * self.zoom_level_1
        nw = int(self.original_full_res_1.width * scale)
        nh = int(self.original_full_res_1.height * scale)
        self.tk_image_1 = ImageTk.PhotoImage(self.original_full_res_1.resize((nw, nh), Image.LANCZOS))
        self.canvas_1.delete("img")
        self.canvas_1.delete("overlay")
        cw = (WINDOW_WIDTH // 2) - 10
        ch = WINDOW_HEIGHT - TITLE_BAR_HEIGHT - BOTTOM_PANEL_HEIGHT - 40
        self.canvas_1.create_image((cw // 2) + self.pan_x_1, (ch // 2) + self.pan_y_1,
                                   image=self.tk_image_1, anchor=tk.CENTER, tags="img")

    def _update_canvas_2(self):
        if self.original_full_res_2 is None: return
        scale = self.base_scale_2 * self.zoom_level_2
        nw = int(self.original_full_res_2.width * scale)
        nh = int(self.original_full_res_2.height * scale)
        self.tk_image_2 = ImageTk.PhotoImage(self.original_full_res_2.resize((nw, nh), Image.LANCZOS))
        self.canvas_2.delete("img")
        self.canvas_2.delete("overlay")
        cw = (WINDOW_WIDTH // 2) - 10
        ch = WINDOW_HEIGHT - TITLE_BAR_HEIGHT - BOTTOM_PANEL_HEIGHT - 40
        self.canvas_2.create_image((cw // 2) + self.pan_x_2, (ch // 2) + self.pan_y_2,
                                   image=self.tk_image_2, anchor=tk.CENTER, tags="img")

    # ==============================================================
    # zoom / pan – unchanged
    # ==============================================================
    def zoom_in_1(self):
        if not self.overlay_mode:
            self.zoom_level_1 *= 1.25; self._update_canvas_1()
    def zoom_out_1(self):
        if not self.overlay_mode:
            self.zoom_level_1 /= 1.25; self._update_canvas_1()
    def reset_view_1(self):
        if not self.overlay_mode:
            self.zoom_level_1, self.pan_x_1, self.pan_y_1 = 1.0, 0, 0; self._update_canvas_1()
    def _on_pan_start_1(self, event):
        if not self.overlay_mode:
            self.last_x_1, self.last_y_1 = event.x, event.y
    def _on_pan_drag_1(self, event):
        if not self.overlay_mode:
            self.pan_x_1 += event.x - self.last_x_1
            self.pan_y_1 += event.y - self.last_y_1
            self.last_x_1, self.last_y_1 = event.x, event.y
            self._update_canvas_1()
    def _on_mouse_wheel_1(self, event):
        if not self.overlay_mode:
            (self.zoom_in_1 if event.delta > 0 else self.zoom_out_1)()

    def zoom_in_2(self):
        if not self.overlay_mode:
            self.zoom_level_2 *= 1.25; self._update_canvas_2()
    def zoom_out_2(self):
        if not self.overlay_mode:
            self.zoom_level_2 /= 1.25; self._update_canvas_2()
    def reset_view_2(self):
        if not self.overlay_mode:
            self.zoom_level_2, self.pan_x_2, self.pan_y_2 = 1.0, 0, 0; self._update_canvas_2()
    def _on_pan_start_2(self, event):
        if not self.overlay_mode:
            self.last_x_2, self.last_y_2 = event.x, event.y
    def _on_pan_drag_2(self, event):
        if not self.overlay_mode:
            self.pan_x_2 += event.x - self.last_x_2
            self.pan_y_2 += event.y - self.last_y_2
            self.last_x_2, self.last_y_2 = event.x, event.y
            self._update_canvas_2()
    def _on_mouse_wheel_2(self, event):
        if not self.overlay_mode:
            (self.zoom_in_2 if event.delta > 0 else self.zoom_out_2)()


# ======================================================================
# MAIN
# ======================================================================
def main():
    root = tk.Tk()

    def load_initial():
        app.update_progress(0, "Initializing cameras…")
        app.start_video_stream()
        app.update_progress(100, "Live Feed")
        app.set_pass_fail("LIVE")

    # ------------------------------------------------------------------
    def on_detect_logic():
        try:
            app.update_progress(5, "Capturing frames…")
            raw_frame_1 = app.camera1.get_raw_frame()
            raw_frame_2 = app.camera2.get_raw_frame()

            undist_frame_1 = app.camera1.get_undistorted_frame()
            undist_frame_2 = app.camera2.get_undistorted_frame()

            if raw_frame_1 is None or raw_frame_2 is None:
                print("Error: No frames captured")
                root.after(0, lambda: [app.detect_btn.config(state=tk.NORMAL), app.resume_video()])
                return

            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            for subdir, frame in [("Camera1", raw_frame_1), ("Camera2", raw_frame_2)]:
                folder = os.path.join("Dataset_Capture", subdir)
                os.makedirs(folder, exist_ok=True)
                cv2.imwrite(os.path.join(folder, f"raw_{ts}.jpg"), frame)

            app.update_progress(15, "Loading AI model…")
            predictor = SkeletonSegmentationPredictor()

            app.update_progress(20, "AI Predicting Camera 1…")
            mask1 = predictor.predict(undist_frame_1)
            mask_uint8_1 = (mask1 * 255).astype(np.uint8)

            app.update_progress(40, "AI Predicting Camera 2…")
            mask2 = predictor.predict(undist_frame_2)
            mask_uint8_2 = (mask2 * 255).astype(np.uint8)

            for subdir, undist, mask in [("Camera1", undist_frame_1, mask_uint8_1), 
                                         ("Camera2", undist_frame_2, mask_uint8_2)]:
                folder = os.path.join("Dataset_Capture", subdir)
                cv2.imwrite(os.path.join(folder, f"undist_{ts}.jpg"), undist)
                cv2.imwrite(os.path.join(folder, f"mask_{ts}.jpg"), mask)
            print(f"Saved dataset for timestamp: {ts}")

            length_tolerance = app._parse_tolerance(app.active_len_threshold)
            width_tolerance = app._parse_tolerance(app.active_wid_threshold)

            app.update_progress(70, "Measuring…")

            measurer_1 = LinearFeatureInspectorOptimized(
                expected_lengths=app.active_expected_lengths,
                length_tolerance=length_tolerance,
                expected_width=3.5,
                width_tolerance=width_tolerance,
                calibration_path=CALIB_FILE_1,
                extrinsics_path=Extrinsics_FILE_1,
                num_segments=app.active_segments,
                debug=True,
                profile=True,
                enable_inspection=app.active_enable_check,
                me_length=1.0,
                me_width=0.5,
            )

            measurer_2 = LinearFeatureInspectorOptimized(
                expected_lengths=app.active_expected_lengths,
                length_tolerance=length_tolerance,
                expected_width=3.5,
                width_tolerance=width_tolerance,
                calibration_path=CALIB_FILE_2,
                extrinsics_path=Extrinsics_FILE_2,
                num_segments=app.active_segments,
                debug=True,
                profile=True,
                enable_inspection=app.active_enable_check,
                me_length=1.0,
                me_width=0.5,
            )

            app.update_progress(85, "Generating results…")
            result_img_1 = measurer_1.inspect(undist_frame_1, mask_uint8_1)
            result_img_2 = measurer_2.inspect(undist_frame_2, mask_uint8_2)

            pil_1 = Image.fromarray(cv2.cvtColor(result_img_1, cv2.COLOR_BGR2RGB))
            pil_2 = Image.fromarray(cv2.cvtColor(result_img_2, cv2.COLOR_BGR2RGB))

            app.overlay_result_1 = result_img_1.copy()
            app.overlay_result_2 = result_img_2.copy()

            app.update_progress(100, "Done")

            def after_detection():
                app.show_image(pil_1, canvas_num=1)
                app.show_image(pil_2, canvas_num=2)

            root.after(0, after_detection)

        except Exception as e:
            print("Detection Error:", e)
            import traceback
            traceback.print_exc()
            root.after(0, lambda: [app.detect_btn.config(state=tk.NORMAL), app.resume_video()])

    # ------------------------------------------------------------------
    def reset_system():
        app.update_progress(0, "Resetting…")

        app.inspection_mode = False
        app.overlay_mode = False

        app.result_image_1 = app.result_image_2 = None
        app.overlay_result_1 = app.overlay_result_2 = None

        app.canvas_1.delete("img"); app.canvas_1.delete("overlay")
        app.canvas_2.delete("img"); app.canvas_2.delete("overlay")

        app.zoom_level_1 = app.zoom_level_2 = 1.0
        app.pan_x_1 = app.pan_y_1 = app.pan_x_2 = app.pan_y_2 = 0

        app.inspection_btn.pack_forget()
        app.overlay_btn.pack_forget()
        app.detect_btn.pack(side=tk.LEFT, padx=3)
        app.detect_btn.config(state=tk.NORMAL)

        app.video_paused = False
        app.set_pass_fail("LIVE")
        app.update_progress(100, "Live Feed")

    # ------------------------------------------------------------------
    app = IndustrialDashboard(root, on_detect_logic, reset_system)
    load_initial()
    root.mainloop()


if __name__ == "__main__":
    main()

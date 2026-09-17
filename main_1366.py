import tkinter as tk
from tkinter import ttk, messagebox
import threading
import os
import time
import subprocess
import json
from PIL import Image, ImageTk
import cv2
import numpy as np
from camera_handler import CameraHandler
import datetime
import sys
from concurrent.futures import ThreadPoolExecutor
from app_config import CONFIG, project_path
from Image_Processing.color_mask_generator import ColorMaskGenerator
from CalibrateAPP.calibration_ui import CalibrationApp

try:
    import serial
except ImportError:
    serial = None

# ---------------- CONFIG ----------------
WINDOW_WIDTH = 1366
WINDOW_HEIGHT = 768
BOTTOM_PANEL_HEIGHT = 100
TITLE_BAR_HEIGHT = 38

# Serial Configuration
SERIAL_PORT = CONFIG['serial']['port']
BAUD_RATE = CONFIG['serial']['baud_rate']

# Camera configuration
CAMERA_INDEX_1, CAMERA_INDEX_2 = [c['index'] for c in CONFIG['cameras']]
CALIB_FILE_1, CALIB_FILE_2 = [project_path(c['calibration_file']) for c in CONFIG['cameras']]
Extrinsics_FILE_1, Extrinsics_FILE_2 = [project_path(c['extrinsics_file']) for c in CONFIG['cameras']]

# Fixed sizes only (no patterns)
FIXED_SIZES = CONFIG['inspection']['sizes']

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
    def __init__(self, root, on_reset_callback):
        self.root = root
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

        # --- maximized camera mode ---
        self.maximized_camera = None  # None, 1, or 2

        # --- settings variables ---
        self.n_segments_var = tk.IntVar(value=CONFIG['inspection']['segments'])
        self.size_var = tk.StringVar()
        self.len_threshold_var = tk.StringVar(value=CONFIG['inspection']['length_tolerance'])
        self.wid_threshold_var = tk.StringVar(value=CONFIG['inspection']['width_tolerance'])
        self.strip_width_var = tk.StringVar(value="2")
        self.enable_check_var = tk.BooleanVar(value=True)

        # Set default size
        self.size_var.set(CONFIG['inspection']['default_size'])
        self.available_sizes = FIXED_SIZES

        # --- active (saved) values used during detection ---
        self.active_size = self.size_var.get()
        self.active_len_threshold = self.len_threshold_var.get()
        self.active_wid_threshold = self.wid_threshold_var.get()
        self.active_segments = self.n_segments_var.get()
        self.active_strip_width = self._validate_strip_width(self.strip_width_var.get())
        self.active_enable_check = self.enable_check_var.get()
        self.session_start_time = ""

        # --- serial communication ---
        self.serial_conn = None
        self._init_serial()

        # --- window chrome ---
        self.root.overrideredirect(True)

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width // 2) - (WINDOW_WIDTH // 2)
        y = (screen_height // 2) - (WINDOW_HEIGHT // 2)
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}+{x}+{y}")
        self.root.configure(bg="#1e293b")

        self._create_custom_title_bar()
        self._create_layout()

    # ==============================================================
    # helpers
    # ==============================================================
    def _init_serial(self):
        if not CONFIG['serial']['enabled']:
            return
        if serial is None:
            print("pyserial is not installed. Serial features disabled.")
            return

        try:
            self.serial_conn = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
            threading.Thread(target=self._serial_monitor, daemon=True).start()
            print(f"Serial connected on {SERIAL_PORT}")
        except Exception as e:
            print(f"Could not connect to serial port {SERIAL_PORT}: {e}")

    def _serial_monitor(self):
        while True:
            if self.serial_conn and self.serial_conn.is_open:
                try:
                    if self.serial_conn.in_waiting > 0:
                        line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                        if line == "LeftCheck":
                            print("Arduino -> LeftCheck")
                            try:
                                self.serial_conn.write(b"L2\n")
                            except Exception as e:
                                print(f"Serial write error L2: {e}")
                            self.root.after(0, lambda: self.start_detect_thread("R"))
                        elif line == "RightCheck":
                            print("Arduino -> RightCheck")
                            try:
                                self.serial_conn.write(b"R2\n")
                            except Exception as e:
                                print(f"Serial write error R2: {e}")
                            self.root.after(0, lambda: self.start_detect_thread("L"))
                except Exception as e:
                    print(f"Serial read error: {e}")
                    time.sleep(1)
            time.sleep(0.01)

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

    # ==============================================================
    # title-bar / window management
    # ==============================================================
    def _create_custom_title_bar(self):
        self.title_bar = tk.Frame(self.root, bg="#0f172a", height=TITLE_BAR_HEIGHT)
        self.title_bar.pack(side=tk.TOP, fill=tk.X)
        tk.Label(self.title_bar, text="Industrial Vision Dashboard",
                 fg="#94a3b8", bg="#0f172a", font=("Segoe UI", 11)).pack(side=tk.LEFT, padx=10)
        tk.Button(self.title_bar, text="✕", command=self.close_application,
                  bg="#0f172a", fg="white", bd=0, padx=12, pady=5, font=("Segoe UI", 12),
                  activebackground="#e11d48").pack(side=tk.RIGHT, fill=tk.Y)
        tk.Button(self.title_bar, text="—", command=self.minimize_window,
                  bg="#0f172a", fg="white", bd=0, padx=12, pady=5, font=("Segoe UI", 12),
                  activebackground="#475569").pack(side=tk.RIGHT, fill=tk.Y)
        self.title_bar.bind("<ButtonPress-1>", self._start_move)
        self.title_bar.bind("<B1-Motion>", self._do_move)

    def close_application(self):
        self._closing = True
        for future in getattr(self, '_camera_futures', []):
            def release_when_ready(done):
                if not done.cancelled() and done.exception() is None:
                    done.result().release()
            future.add_done_callback(release_when_ready)
        self.video_streaming = False
        if getattr(self, "serial_conn", None) and self.serial_conn.is_open:
            self.serial_conn.close()
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
        self.main_frame = tk.Frame(self.root, bg="#1e293b")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        self._create_image_panel()
        self._create_bottom_panel()

    # --------------------------------------------------------------
    # bottom panel
    # --------------------------------------------------------------
    def _create_bottom_panel(self):
        self.bottom_panel = tk.Frame(self.main_frame, height=BOTTOM_PANEL_HEIGHT, bg="#334155")
        self.bottom_panel.pack(side=tk.BOTTOM, fill=tk.X)
        self.bottom_panel.pack_propagate(False)

        # ---- left: size display only ----
        left_section = tk.Frame(self.bottom_panel, bg="#334155")
        left_section.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Large size display
        self.lbl_large_size = tk.Label(
            left_section,
            text=f"Size: {self.active_size}",
            fg="#22d3ee", bg="#334155",
            font=("Segoe UI", 26, "bold")
        )
        self.lbl_large_size.pack(anchor="w")

        self.session_time_label = tk.Label(
            left_section,
            text="",
            fg="#f8fafc", bg="#334155",
            font=("Segoe UI", 18, "bold")
        )
        self.session_time_label.pack(anchor="w", pady=(2, 0))

        # ---- center: status label (LIVE) ----
        center_section = tk.Frame(self.bottom_panel, bg="#334155")
        center_section.pack(side=tk.LEFT, padx=20)
        self.status_result_label = tk.Label(center_section, text="LIVE", fg="#22d3ee", bg="#334155",
                                            font=("Segoe UI", 26, "bold"))
        self.status_result_label.pack(pady=10)

        # ---- right: buttons ----
        right_section = tk.Frame(self.bottom_panel, bg="#334155")
        right_section.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=5)

        btn_font = ("Segoe UI", 11, "bold")
        btn_font_large = ("Segoe UI", 13, "bold")

        self.btn_container = tk.Frame(right_section, bg="#334155")
        self.btn_container.pack(expand=True)

        # Settings button
        tk.Button(self.btn_container, text="Settings", font=btn_font_large, bg="#64748b", fg="white",
                  relief=tk.FLAT, width=10, height=2,
                  command=self.open_settings_selector).pack(side=tk.LEFT, padx=3)

        # Reset button
        tk.Button(self.btn_container, text="RESET", font=btn_font_large, bg="#f43f5e", fg="white",
                  relief=tk.FLAT, width=10, height=2,
                  command=self.on_reset_callback).pack(side=tk.LEFT, padx=3)

        # Camera L button (left)
        self.detect_btn_L = tk.Button(self.btn_container, text="Camera L", font=btn_font_large,
                                      bg="#3b82f6", fg="white", relief=tk.FLAT,
                                      width=10, height=2, command=lambda: self.start_detect_thread("L"))
        self.detect_btn_L.pack(side=tk.LEFT, padx=3)

        # Camera R button (right)
        self.detect_btn_R = tk.Button(self.btn_container, text="Camera R", font=btn_font_large,
                                      bg="#3b82f6", fg="white", relief=tk.FLAT,
                                      width=10, height=2, command=lambda: self.start_detect_thread("R"))
        self.detect_btn_R.pack(side=tk.LEFT, padx=3)

    # ==============================================================
    # settings windows
    # ==============================================================
    def open_settings_selector(self):
        self.video_paused = True
        self.root.withdraw()

        self.sel_win = tk.Toplevel(self.root)
        self.sel_win.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}+{self.root.winfo_x()}+{self.root.winfo_y()}")
        self.sel_win.configure(bg="#1e293b")
        self.sel_win.overrideredirect(True)
        self.sel_win.attributes("-topmost", True)

        header = tk.Frame(self.sel_win, bg="#0f172a", height=TITLE_BAR_HEIGHT)
        header.pack(fill=tk.X)
        tk.Button(header, text="🏠 Home",
                  command=lambda: [self.sel_win.destroy(), self.root.deiconify(), self.resume_video()],
                  bg="#0f172a", fg="#22d3ee", bd=0, padx=15, font=("Segoe UI", 11, "bold")).pack(side=tk.LEFT, fill=tk.Y)
        tk.Label(header, text="Configuration Selector", fg="#94a3b8", bg="#0f172a", font=("Segoe UI", 11)).pack(side=tk.LEFT, padx=10)
        tk.Button(header, text="✕", command=self.close_application, bg="#0f172a", fg="white", bd=0, padx=12, font=("Segoe UI", 12)).pack(side=tk.RIGHT, fill=tk.Y)
        header.bind("<ButtonPress-1>", self._start_move)
        header.bind("<B1-Motion>", self._do_move)

        container = tk.Frame(self.sel_win, bg="#1e293b")
        container.pack(expand=True)
        tk.Label(container, text="System Settings", fg="white", bg="#1e293b",
                 font=("Segoe UI", 32, "bold")).pack(pady=40)

        btn_style = {"font": ("Segoe UI", 18, "bold"), "bg": "#475569", "fg": "white",
                     "relief": tk.FLAT, "width": 25, "height": 3}

        tk.Button(container, text="Size Setting",      command=self.open_size_window,      **btn_style).pack(pady=10)
        tk.Button(container, text="Camera Setup", command=self.open_camera_setup, **btn_style).pack(pady=10)

    def open_camera_setup(self):
        """Show setup inside the existing dashboard window and event loop."""
        if getattr(self, 'calibration_page', None) is not None:
            return
        self.video_streaming = False
        if getattr(self, '_video_job', None) is not None:
            self.root.after_cancel(self._video_job)
            self._video_job = None
        # Keep the streams open; setup borrows the same full-resolution devices.
        if hasattr(self, 'sel_win') and self.sel_win.winfo_exists():
            self.sel_win.destroy()
        self.title_bar.pack_forget()
        self.main_frame.pack_forget()
        self.calibration_page = tk.Frame(self.root, bg="#1e293b")
        self.calibration_page.pack(fill=tk.BOTH, expand=True)
        self.root.deiconify()
        try:
            self.calibration_app = CalibrationApp(
                self.root, host=self.calibration_page, on_close=self.close_camera_setup,
                camera_provider=self._setup_camera_stream,
            )
        except Exception as e:
            self.close_camera_setup()
            self._show_error_popup(f"Could not open camera setup:\n{e}")

    def close_camera_setup(self):
        """Return immediately; keep devices open and reload saved calibration."""
        if getattr(self, 'calibration_page', None) is not None:
            self.calibration_page.destroy()
        self.calibration_page = None
        self.calibration_app = None
        self.title_bar.pack(side=tk.TOP, fill=tk.X)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        if self.camera1: self.camera1.reload_calibration()
        if self.camera2: self.camera2.reload_calibration()
        self.start_video_stream()

    def _setup_camera_stream(self, index):
        for camera in (self.camera1, self.camera2):
            if camera is not None and camera.camera_index == index:
                return camera.stream
        raise RuntimeError("Camera is still starting or unavailable. Return to the dashboard and try again.")

    # --------------------------------------------------------------
    # Size window (replaces pattern window)
    # --------------------------------------------------------------
    def open_size_window(self):
        self.sel_win.destroy()

        self.size_win = tk.Toplevel(self.root)
        self.size_win.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}+{self.root.winfo_x()}+{self.root.winfo_y()}")
        self.size_win.configure(bg="#1e293b")
        self.size_win.overrideredirect(True)
        self.size_win.attributes("-topmost", True)

        # ================= HEADER =================
        header = tk.Frame(self.size_win, bg="#0f172a", height=TITLE_BAR_HEIGHT)
        header.pack(fill=tk.X)

        tk.Button(
            header, text="← Back",
            command=lambda: [self.size_win.destroy(), self.open_settings_selector()],
            bg="#0f172a", fg="#22d3ee", bd=0, padx=15,
            font=("Segoe UI", 11, "bold")
        ).pack(side=tk.LEFT)

        tk.Label(
            header, text="Size Setting",
            fg="#94a3b8", bg="#0f172a",
            font=("Segoe UI", 11)
        ).pack(side=tk.LEFT, padx=10)

        tk.Button(
            header, text="✕",
            command=self.close_application,
            bg="#0f172a", fg="white", bd=0, padx=12, font=("Segoe UI", 12)
        ).pack(side=tk.RIGHT)

        # ================= CONTENT =================
        content = tk.Frame(self.size_win, bg="#1e293b")
        content.pack(expand=True, fill=tk.BOTH, padx=40, pady=30)

        # ---------- SIZE ----------
        tk.Label(
            content, text="Select Size",
            fg="#22d3ee", bg="#1e293b",
            font=("Segoe UI", 18, "bold")
        ).pack(anchor="w", pady=(15, 10))

        size_frame = tk.Frame(content, bg="#1e293b")
        size_frame.pack(anchor="w")

        self._size_buttons = {}
        for size in FIXED_SIZES:
            btn = tk.Button(
                size_frame,
                text=size,
                width=10,
                height=2,
                bg="#047857",
                fg="white",
                relief=tk.FLAT,
                font=("Segoe UI", 13, "bold"),
                command=lambda s=size: self._select_size(s)
            )
            btn.pack(side=tk.LEFT, padx=10, pady=10)
            self._size_buttons[size] = btn

        # ---------- SETTINGS (keep for future use) ----------
        settings = tk.Frame(content, bg="#1e293b")
        settings.pack(fill=tk.X, pady=30)

        tk.Label(settings, text="Length Tolerance", fg="#94a3b8", bg="#1e293b", font=("Segoe UI", 12)).grid(row=0, column=0, sticky="w")
        self.len_tolerance_menu = tk.OptionMenu(
            settings, self.len_threshold_var,
            "± 0.5mm", "± 1mm", "± 1.5mm", "± 2mm", "± 2.5mm", "± 3mm", "± 3.5mm", "± 4mm", "± 4.5mm", "± 5mm", "± 10mm"
        )
        self.len_tolerance_menu.config(bg="#475569", fg="white", font=("Segoe UI", 12), relief=tk.FLAT)
        self.len_tolerance_menu["menu"].config(bg="#475569", fg="white", font=("Segoe UI", 12))
        self.len_tolerance_menu.grid(row=1, column=0, padx=10, sticky="w")

        tk.Label(settings, text="Width Tolerance", fg="#94a3b8", bg="#1e293b", font=("Segoe UI", 12)).grid(row=0, column=1, sticky="w")
        self.wid_tolerance_menu = tk.OptionMenu(
            settings, self.wid_threshold_var,
            "± 0.5mm", "± 1mm", "± 1.5mm", "± 2mm", "± 2.5mm", "± 3mm", "± 3.5mm", "± 4mm", "± 4.5mm", "± 5mm", "± 10mm"
        )
        self.wid_tolerance_menu.config(bg="#475569", fg="white", font=("Segoe UI", 12), relief=tk.FLAT)
        self.wid_tolerance_menu["menu"].config(bg="#475569", fg="white", font=("Segoe UI", 12))
        self.wid_tolerance_menu.grid(row=1, column=1, padx=10, sticky="w")

        tk.Label(settings, text="Segments", fg="#94a3b8", bg="#1e293b", font=("Segoe UI", 12)).grid(row=0, column=2, sticky="w")
        self.segments_combo = ttk.Combobox(
            settings,
            values=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"],
            state="readonly",
            width=10,
            font=("Segoe UI", 12)
        )
        self.segments_combo.set(str(self.n_segments_var.get()))
        self.segments_combo.grid(row=1, column=2, padx=10, sticky="w")
        self.segments_combo.bind("<<ComboboxSelected>>", lambda e: self.n_segments_var.set(int(self.segments_combo.get())))

        tk.Checkbutton(
            content,
            text="Enable Inspect (PASS / FAIL)",
            variable=self.enable_check_var,
            bg="#1e293b",
            fg="white",
            selectcolor="#475569",
            font=("Segoe UI", 14, "bold")
        ).pack(anchor="w", pady=20)

        # ---------- ACTION BUTTONS ----------
        btn_frame = tk.Frame(content, bg="#1e293b")
        btn_frame.pack(pady=25)

        tk.Button(
            btn_frame,
            text="Mask Setup",
            bg="#8b5cf6",
            fg="white",
            font=("Segoe UI", 14, "bold"),
            width=14,
            height=2,
            relief=tk.FLAT,
            command=self.open_mask_setup_window
        ).pack(side=tk.LEFT, padx=10)

        tk.Button(
            btn_frame,
            text="SAVE",
            bg="#3b82f6",
            fg="white",
            font=("Segoe UI", 14, "bold"),
            width=14,
            height=2,
            relief=tk.FLAT,
            command=self.save_settings
        ).pack(side=tk.LEFT, padx=10)

        # ---------- DEFAULT HIGHLIGHT ----------
        self._select_size(self.size_var.get())

    # ---------- Helper methods for size buttons ----------
    def _select_size(self, size):
        self.size_var.set(size)
        for s, btn in self._size_buttons.items():
            btn.config(
                bg="#10b981" if s == size else "#047857",
                fg="white"
            )

    def save_settings(self):
        strip_value = self.strip_width_var.get()
        validated_width = self._validate_strip_width(strip_value)

        if hasattr(self, 'segments_combo'):
            self.n_segments_var.set(int(self.segments_combo.get()))

        # Commit to active values
        self.active_size              = self.size_var.get()
        self.active_len_threshold     = self.len_threshold_var.get()
        self.active_wid_threshold     = self.wid_threshold_var.get()
        self.active_segments          = self.n_segments_var.get()
        self.active_strip_width       = validated_width
        self.active_enable_check      = self.enable_check_var.get()

        # Refresh bottom-panel display
        self.lbl_large_size.config(text=f"Size: {self.active_size}")
        
        self.session_start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.session_time_label.config(text=f"Session Start: {self.session_start_time}")

        self._show_success_popup()

    def _show_success_popup(self):
        pop = tk.Toplevel(self.size_win)
        pop.overrideredirect(True)
        pop.attributes("-topmost", True)
        pop.configure(bg="#334155")

        px = self.size_win.winfo_x() + (WINDOW_WIDTH // 2) - 160
        py = self.size_win.winfo_y() + (WINDOW_HEIGHT // 2) - 80
        pop.geometry(f"320x160+{px}+{py}")

        tk.Label(pop, text="✓", fg="#10b981", bg="#334155", font=("Segoe UI", 36)).pack(pady=(15, 0))
        tk.Label(pop, text="Settings Saved Successfully", fg="white", bg="#334155",
                 font=("Segoe UI", 13, "bold")).pack()

        def close_pop():
            pop.destroy()
            self.size_win.destroy()
            self.open_settings_selector()

        tk.Button(pop, text="OK", bg="#64748b", fg="white", relief=tk.FLAT, width=10, command=close_pop).pack(pady=15)

    def _show_error_popup(self, message):
        pop = tk.Toplevel(self.root)
        pop.overrideredirect(True)
        pop.attributes("-topmost", True)
        pop.configure(bg="#334155")

        px = self.root.winfo_x() + (WINDOW_WIDTH // 2) - 225
        py = self.root.winfo_y() + (WINDOW_HEIGHT // 2) - 140
        pop.geometry(f"450x280+{px}+{py}")

        title_bar = tk.Frame(pop, bg="#0f172a", height=38)
        title_bar.pack(fill=tk.X)
        title_bar.pack_propagate(False)
        tk.Label(title_bar, text="Error", fg="#94a3b8", bg="#0f172a", font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=10)
        tk.Button(title_bar, text="✕", command=pop.destroy,
                  bg="#0f172a", fg="white", bd=0, padx=10, pady=5,
                  font=("Segoe UI", 10), activebackground="#e11d48").pack(side=tk.RIGHT, fill=tk.Y)

        content_frame = tk.Frame(pop, bg="#334155")
        content_frame.pack(fill=tk.BOTH, expand=True, pady=15)
        tk.Label(content_frame, text="⚠",      fg="#ef4444", bg="#334155", font=("Segoe UI", 56)).pack(pady=(15, 10))
        tk.Label(content_frame, text=message,  fg="white",   bg="#334155", font=("Segoe UI", 18, "bold")).pack(pady=(5, 20))

        btn_frame = tk.Frame(content_frame, bg="#334155")
        btn_frame.pack(pady=(0, 15))

        tk.Button(btn_frame, text="Go to Settings", bg="#3b82f6", fg="white",
                  relief=tk.FLAT, width=16, height=2, font=("Segoe UI", 12, "bold"),
                  command=lambda: [pop.destroy(), self.open_settings_selector()]).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="OK", bg="#64748b", fg="white",
                  relief=tk.FLAT, width=14, height=2, font=("Segoe UI", 12),
                  command=pop.destroy).pack(side=tk.LEFT, padx=5)

    def resume_video(self):
        self.video_paused = False

    def open_mask_setup_window(self):
        self.size_win.destroy()

        self.mask_win = tk.Toplevel(self.root)
        self.mask_win.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}+{self.root.winfo_x()}+{self.root.winfo_y()}")
        self.mask_win.configure(bg="#1e293b")
        self.mask_win.overrideredirect(True)
        self.mask_win.attributes("-topmost", True)

        self.mask_setup_active = True
        self.mask_selected_camera = 1
        self.video_paused = False  # Resume video for mask setup
        self.mask_frozen = False
        self.mask_frozen_frame = None
        self.mask_start_x = None
        self.mask_start_y = None
        self.mask_rect = None
        self.mask_rect_coords = None
        self.mask_pick_point = None
        self.mask_pick_circle = None

        # ================= HEADER =================
        header = tk.Frame(self.mask_win, bg="#0f172a", height=TITLE_BAR_HEIGHT)
        header.pack(fill=tk.X)

        def back_to_size():
            self.mask_setup_active = False
            self.mask_frozen = False
            self.video_paused = True
            self.mask_win.destroy()
            self.open_size_window()

        tk.Button(
            header, text="← Back",
            command=back_to_size,
            bg="#0f172a", fg="#22d3ee", bd=0, padx=15,
            font=("Segoe UI", 11, "bold")
        ).pack(side=tk.LEFT)

        tk.Label(
            header, text="Mask Setup",
            fg="#94a3b8", bg="#0f172a",
            font=("Segoe UI", 11)
        ).pack(side=tk.LEFT, padx=10)

        tk.Button(
            header, text="✕",
            command=self.close_application,
            bg="#0f172a", fg="white", bd=0, padx=12, font=("Segoe UI", 12)
        ).pack(side=tk.RIGHT)

        # ================= CONTENT =================
        content = tk.Frame(self.mask_win, bg="#1e293b")
        content.pack(expand=True, fill=tk.BOTH, padx=20, pady=10)

        # Camera selection frame
        cam_sel_frame = tk.Frame(content, bg="#1e293b")
        cam_sel_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(cam_sel_frame, text="Select Camera:", fg="white", bg="#1e293b", font=("Segoe UI", 12)).pack(side=tk.LEFT, padx=(0, 10))

        def select_cam(c):
            if getattr(self, "mask_frozen", False): return
            self.mask_selected_camera = c
            btn_cam1.config(bg="#10b981" if c == 1 else "#047857")
            btn_cam2.config(bg="#10b981" if c == 2 else "#047857")

        btn_cam1 = tk.Button(cam_sel_frame, text="Camera 1", bg="#10b981", fg="white", font=("Segoe UI", 12, "bold"), relief=tk.FLAT, width=12, command=lambda: select_cam(1))
        btn_cam1.pack(side=tk.LEFT, padx=8)

        btn_cam2 = tk.Button(cam_sel_frame, text="Camera 2", bg="#047857", fg="white", font=("Segoe UI", 12, "bold"), relief=tk.FLAT, width=12, command=lambda: select_cam(2))
        btn_cam2.pack(side=tk.LEFT, padx=8)

        # Actions
        tk.Frame(cam_sel_frame, width=30, bg="#1e293b").pack(side=tk.LEFT) # spacer
        btn_capture = tk.Button(cam_sel_frame, text="Capture Frame", bg="#f59e0b", fg="white", font=("Segoe UI", 12, "bold"), relief=tk.FLAT, width=16, command=self.capture_mask_frame)
        btn_capture.pack(side=tk.LEFT, padx=5)

        btn_clear = tk.Button(cam_sel_frame, text="Clear", bg="#f43f5e", fg="white", font=("Segoe UI", 12, "bold"), relief=tk.FLAT, width=10, command=self.clear_mask_frame)
        btn_clear.pack(side=tk.LEFT, padx=5)

        btn_save = tk.Button(cam_sel_frame, text="Save Mask", bg="#3b82f6", fg="white", font=("Segoe UI", 12, "bold"), relief=tk.FLAT, width=12, command=self.save_mask_frame)
        btn_save.pack(side=tk.LEFT, padx=5)

        # Canvas for live video
        self.mask_canvas = tk.Canvas(content, bg="#020617", highlightthickness=0)
        self.mask_canvas.pack(fill=tk.BOTH, expand=True)

        self.mask_canvas.bind("<ButtonPress-1>", self.on_mask_press)
        self.mask_canvas.bind("<B1-Motion>", self.on_mask_drag)
        self.mask_canvas.bind("<ButtonRelease-1>", self.on_mask_release)

    def _display_mask_video_frame(self, frame):
        if not hasattr(self, 'mask_canvas') or not self.mask_canvas.winfo_exists():
            return
        cw = self.mask_canvas.winfo_width()
        ch = self.mask_canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            cw = WINDOW_WIDTH - 40
            ch = WINDOW_HEIGHT - TITLE_BAR_HEIGHT - 60

        if cw > 1 and ch > 1:
            frame_resized = cv2.resize(frame, (cw, ch))
            pil_img = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
            self.tk_image_mask = ImageTk.PhotoImage(pil_img)
            self.mask_canvas.delete("img")
            self.mask_canvas.create_image(cw // 2, ch // 2, image=self.tk_image_mask, anchor=tk.CENTER, tags="img")
            self.mask_canvas.tag_lower("img")

    def capture_mask_frame(self):
        self.mask_frozen = True

    def clear_mask_frame(self):
        self.mask_frozen = False
        self.mask_frozen_frame = None
        if getattr(self, "mask_rect", None):
            self.mask_canvas.delete(self.mask_rect)
            self.mask_rect = None
        self.mask_rect_coords = None
        if getattr(self, "mask_pick_circle", None):
            self.mask_canvas.delete(self.mask_pick_circle)
            self.mask_pick_circle = None
        self.mask_pick_point = None

    def save_mask_frame(self):
        if not getattr(self, "mask_frozen", False) or getattr(self, "mask_frozen_frame", None) is None:
            self._show_error_popup("Capture a frame first!")
            return
        if not getattr(self, "mask_pick_point", None):
            self._show_error_popup("Click to pick a color point first!")
            return
        if not getattr(self, "mask_rect_coords", None):
            self._show_error_popup("Draw a region first!")
            return

        cw = self.mask_canvas.winfo_width()
        ch = self.mask_canvas.winfo_height()
        
        orig_h, orig_w = self.mask_frozen_frame.shape[:2]
        
        # Convert pick point
        px, py = self.mask_pick_point
        px_orig = int(px * orig_w / cw)
        py_orig = int(py * orig_h / ch)
        px_orig = max(0, min(orig_w - 1, px_orig))
        py_orig = max(0, min(orig_h - 1, py_orig))

        # Convert region rect
        x1, y1, x2, y2 = self.mask_rect_coords
        x1_orig = int(x1 * orig_w / cw)
        x2_orig = int(x2 * orig_w / cw)
        y1_orig = int(y1 * orig_h / ch)
        y2_orig = int(y2 * orig_h / ch)

        x1_orig, x2_orig = sorted([x1_orig, x2_orig])
        y1_orig, y2_orig = sorted([y1_orig, y2_orig])

        x1_orig = max(0, min(orig_w - 1, x1_orig))
        x2_orig = max(0, min(orig_w - 1, x2_orig))
        y1_orig = max(0, min(orig_h - 1, y1_orig))
        y2_orig = max(0, min(orig_h - 1, y2_orig))
        
        region_orig = (x1_orig, y1_orig, x2_orig - x1_orig, y2_orig - y1_orig)

        generator = ColorMaskGenerator()
        mask = generator.create_mask(self.mask_frozen_frame, region_orig, (px_orig, py_orig))

        filename = f"mask_{self.mask_selected_camera}.png"
        cv2.imwrite(filename, mask)
        
        self.clear_mask_frame() # unfreeze
        self._show_mask_success(filename)

    def _show_mask_success(self, filename):
        pop = tk.Toplevel(self.mask_win)
        pop.overrideredirect(True)
        pop.attributes("-topmost", True)
        pop.configure(bg="#334155")
        px = self.mask_win.winfo_x() + (WINDOW_WIDTH // 2) - 160
        py = self.mask_win.winfo_y() + (WINDOW_HEIGHT // 2) - 80
        pop.geometry(f"320x160+{px}+{py}")
        tk.Label(pop, text="✓", fg="#10b981", bg="#334155", font=("Segoe UI", 36)).pack(pady=(15, 0))
        tk.Label(pop, text=f"Saved {filename}", fg="white", bg="#334155", font=("Segoe UI", 13, "bold")).pack()
        tk.Button(pop, text="OK", bg="#64748b", fg="white", relief=tk.FLAT, width=10, command=pop.destroy).pack(pady=15)

    def on_mask_press(self, event):
        if not getattr(self, "mask_frozen", False): return
        self.mask_start_x = event.x
        self.mask_start_y = event.y

    def on_mask_drag(self, event):
        if not getattr(self, "mask_frozen", False): return
        dx = abs(event.x - self.mask_start_x)
        dy = abs(event.y - self.mask_start_y)
        if dx > 3 or dy > 3:
            if not getattr(self, "mask_rect", None):
                self.mask_rect = self.mask_canvas.create_rectangle(self.mask_start_x, self.mask_start_y, event.x, event.y, outline="red", width=2)
            else:
                self.mask_canvas.coords(self.mask_rect, self.mask_start_x, self.mask_start_y, event.x, event.y)

    def on_mask_release(self, event):
        if not getattr(self, "mask_frozen", False): return
        dx = abs(event.x - self.mask_start_x)
        dy = abs(event.y - self.mask_start_y)
        if dx <= 3 and dy <= 3:
            self.mask_pick_point = (event.x, event.y)
            if getattr(self, "mask_pick_circle", None):
                self.mask_canvas.delete(self.mask_pick_circle)
            self.mask_pick_circle = self.mask_canvas.create_oval(
                event.x-4, event.y-4, event.x+4, event.y+4, outline="#10b981", width=2
            )
        else:
            if getattr(self, "mask_rect", None):
                self.mask_canvas.coords(self.mask_rect, self.mask_start_x, self.mask_start_y, event.x, event.y)
                self.mask_rect_coords = (self.mask_start_x, self.mask_start_y, event.x, event.y)

    # ==============================================================
    # image panel (dual canvases + zoom toolbars)
    # ==============================================================
    def _create_image_panel(self):
        self.image_panel = tk.Frame(self.main_frame, bg="#0f172a")
        self.image_panel.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.images_container = tk.Frame(self.image_panel, bg="#0f172a")
        self.images_container.pack(fill=tk.BOTH, expand=True)

        # ---- left ----
        self.left_frame = tk.Frame(self.images_container, bg="#0f172a")
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 1))
        tk.Label(self.left_frame, text="Camera 1", bg="#0f172a", fg="#94a3b8", font=("Segoe UI", 9)).pack(pady=(5, 0))

        self.canvas_1 = tk.Canvas(self.left_frame, bg="#020617", highlightthickness=0)
        self.canvas_1.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.toolbar_1 = tk.Frame(self.canvas_1, bg="#334155", bd=1, relief=tk.RAISED)
        btn_s = {"bg": "#475569", "fg": "#22d3ee", "font": ("Arial", 11, "bold"), "width": 3, "relief": tk.FLAT}
        tk.Button(self.toolbar_1, text="+", command=self.zoom_in_1,    **btn_s).pack(side=tk.TOP, pady=2, padx=2)
        tk.Button(self.toolbar_1, text="-", command=self.zoom_out_1,   **btn_s).pack(side=tk.TOP, pady=2, padx=2)
        tk.Button(self.toolbar_1, text="⟲", command=self.reset_view_1, **btn_s).pack(side=tk.TOP, pady=2, padx=2)
        # Store the canvas window ID for later repositioning
        self.toolbar_win_id_1 = self.canvas_1.create_window(0, 0, anchor="ne", window=self.toolbar_1)

        # ---- right ----
        self.right_frame = tk.Frame(self.images_container, bg="#0f172a")
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(1, 0))
        tk.Label(self.right_frame, text="Camera 2", bg="#0f172a", fg="#94a3b8", font=("Segoe UI", 9)).pack(pady=(5, 0))

        self.canvas_2 = tk.Canvas(self.right_frame, bg="#020617", highlightthickness=0)
        self.canvas_2.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.toolbar_2 = tk.Frame(self.canvas_2, bg="#334155", bd=1, relief=tk.RAISED)
        tk.Button(self.toolbar_2, text="+", command=self.zoom_in_2,    **btn_s).pack(side=tk.TOP, pady=2, padx=2)
        tk.Button(self.toolbar_2, text="-", command=self.zoom_out_2,   **btn_s).pack(side=tk.TOP, pady=2, padx=2)
        tk.Button(self.toolbar_2, text="⟲", command=self.reset_view_2, **btn_s).pack(side=tk.TOP, pady=2, padx=2)
        self.toolbar_win_id_2 = self.canvas_2.create_window(0, 0, anchor="ne", window=self.toolbar_2)

        # Bind configure events to reposition toolbars when canvas resizes
        self.canvas_1.bind("<Configure>", self._reposition_toolbar_1)
        self.canvas_2.bind("<Configure>", self._reposition_toolbar_2)

        # ---- progress bar ----
        self.loading_container = tk.Frame(self.image_panel, bg="#1e293b", height=20)
        self.loading_container.pack(side=tk.BOTTOM, fill=tk.X)

        self.progress_label = tk.Label(self.loading_container, text="Live Feed",
                                       fg="#94a3b8", bg="#1e293b", font=("Segoe UI", 10))
        self.progress_label.pack(side=tk.LEFT, padx=10)

        self.style = ttk.Style()
        self.style.theme_use('default')
        self.style.configure("Sleek.Horizontal.TProgressbar", thickness=5,
                             troughcolor='#0f172a', background='#8b5cf6', bordercolor="#1e293b")
        self.progress_bar = ttk.Progressbar(self.loading_container, orient=tk.HORIZONTAL,
                                            mode='determinate', style="Sleek.Horizontal.TProgressbar")
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        # Store original pack options for restoration
        self.left_frame_pack_opts = {'side': tk.LEFT, 'fill': tk.BOTH, 'expand': True, 'padx': (0, 1)}
        self.right_frame_pack_opts = {'side': tk.RIGHT, 'fill': tk.BOTH, 'expand': True, 'padx': (1, 0)}

        # ---- bind pan/zoom ----
        self.canvas_1.bind("<ButtonPress-1>",  self._on_pan_start_1)
        self.canvas_1.bind("<B1-Motion>",      self._on_pan_drag_1)
        self.canvas_1.bind("<MouseWheel>",     self._on_mouse_wheel_1)

        self.canvas_2.bind("<ButtonPress-1>",  self._on_pan_start_2)
        self.canvas_2.bind("<B1-Motion>",      self._on_pan_drag_2)
        self.canvas_2.bind("<MouseWheel>",     self._on_mouse_wheel_2)

    # --------------------------------------------------------------
    # Toolbar repositioning (called on canvas resize)
    # --------------------------------------------------------------
    def _reposition_toolbar_1(self, event):
        """Place toolbar at top-right corner of canvas 1."""
        # margin from top and right edges
        margin_x = 5
        margin_y = 5
        # get toolbar width and height
        self.toolbar_1.update_idletasks()
        toolbar_w = self.toolbar_1.winfo_reqwidth()
        toolbar_h = self.toolbar_1.winfo_reqheight()
        x = event.width - toolbar_w - margin_x
        y = margin_y
        self.canvas_1.coords(self.toolbar_win_id_1, x, y)

    def _reposition_toolbar_2(self, event):
        """Place toolbar at top-right corner of canvas 2."""
        margin_x = 5
        margin_y = 5
        self.toolbar_2.update_idletasks()
        toolbar_w = self.toolbar_2.winfo_reqwidth()
        toolbar_h = self.toolbar_2.winfo_reqheight()
        x = event.width - toolbar_w - margin_x
        y = margin_y
        self.canvas_2.coords(self.toolbar_win_id_2, x, y)

    # --------------------------------------------------------------
    # Maximize / restore functions
    # --------------------------------------------------------------
    def maximize_camera(self, camera_num):
        """Show the selected camera in the center of the container."""
        if self.maximized_camera is not None:
            return  # already maximized
        self.maximized_camera = camera_num
        
        # Hide both frames from pack layout
        self.left_frame.pack_forget()
        self.right_frame.pack_forget()
        
        # Place the selected frame in the center
        if camera_num == 1:
            self.left_frame.place(in_=self.images_container, relx=0.5, rely=0.5, anchor=tk.CENTER, relwidth=0.5, relheight=1.0)
        else:
            self.right_frame.place(in_=self.images_container, relx=0.5, rely=0.5, anchor=tk.CENTER, relwidth=0.5, relheight=1.0)
        self.root.update_idletasks()

    def restore_dual_view(self):
        """Restore the original dual‑camera layout."""
        if self.maximized_camera is None:
            return
        # Forget the currently placed frame
        if self.maximized_camera == 1:
            self.left_frame.place_forget()
        else:
            self.right_frame.place_forget()
        # Re-pack both frames with original options
        self.left_frame.pack(**self.left_frame_pack_opts)
        self.right_frame.pack(**self.right_frame_pack_opts)
        self.maximized_camera = None
        self.root.update_idletasks()

    # ==============================================================
    # video streaming – using raw frames
    # ==============================================================
    def start_video_stream(self):
        if self.camera1 is not None and self.camera2 is not None:
            self.video_streaming = True
            self.video_paused = False
            self.set_pass_fail("LIVE")
            self.update_progress(100, "Live Feed")
            self.update_video_feed()
            return
        if getattr(self, '_camera_starting', False):
            return
        self._camera_starting = True
        self.update_progress(0, "Starting cameras…")
        self.set_pass_fail("STARTING")
        executor = ThreadPoolExecutor(max_workers=2)
        futures = [executor.submit(CameraHandler, index, path) for index, path in
                   ((CAMERA_INDEX_1, CALIB_FILE_1), (CAMERA_INDEX_2, CALIB_FILE_2))]
        self._camera_futures = futures
        executor.shutdown(wait=False)

        def finish():
            if not all(f.done() for f in futures):
                self.root.after(50, finish)
                return
            self._camera_starting = False
            cameras, errors = [], []
            for future in futures:
                try:
                    cameras.append(future.result())
                except Exception as error:
                    errors.append(str(error))
            if errors or getattr(self, '_closing', False):
                for camera in cameras: camera.release()
                if not getattr(self, '_closing', False):
                    self.set_pass_fail("CAMERA ERROR")
                    self.update_progress(0, " | ".join(errors))
                return
            self.camera1, self.camera2 = cameras
            if getattr(self, 'calibration_page', None) is None:
                self.start_video_stream()
        self.root.after(50, finish)

    def update_video_feed(self):
        if not self.video_streaming:
            return
        if self.video_paused:
            self._video_job = self.root.after(CONFIG['preview']['interval_ms'], self.update_video_feed)
            return

        # Read frames from both cameras (we need both to keep them alive)
        ret1, raw1 = self.camera1.get_raw_frame_with_ret()
        ret2, raw2 = self.camera2.get_raw_frame_with_ret()

        if ret1 and ret2:
            # Force layout update so canvas sizes are correct
            self.root.update_idletasks()
            
            if getattr(self, "mask_setup_active", False):
                if getattr(self, "mask_frozen", False):
                    if not hasattr(self, "mask_frozen_frame") or self.mask_frozen_frame is None:
                        cam_sel = getattr(self, "mask_selected_camera", 1)
                        self.mask_frozen_frame = (raw1 if cam_sel == 1 else raw2).copy()
                        self._display_mask_video_frame(self.mask_frozen_frame)
                else:
                    cam_sel = getattr(self, "mask_selected_camera", 1)
                    frame = raw1 if cam_sel == 1 else raw2
                    self._display_mask_video_frame(frame)
            else:
                if self.maximized_camera is None:
                    # Show both
                    self._display_video_frame(raw1, 1)
                    self._display_video_frame(raw2, 2)
                else:
                    # Show only the selected camera
                    if self.maximized_camera == 1:
                        self._display_video_frame(raw1, 1)
                    else:
                        self._display_video_frame(raw2, 2)

        self._video_job = self.root.after(CONFIG['preview']['interval_ms'], self.update_video_feed)

    def _display_video_frame(self, frame, canvas_num):
        """Display a frame on the specified canvas, automatically resizing to canvas size."""
        if canvas_num == 1:
            canvas = self.canvas_1
        else:
            canvas = self.canvas_2

        # Get current canvas dimensions (may be 0 if not yet visible)
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            # Fallback: use container dimensions
            if canvas_num == 1:
                parent = self.left_frame
            else:
                parent = self.right_frame
            cw = parent.winfo_width()
            ch = parent.winfo_height()
            if cw <= 1 or ch <= 1:
                # Still 0? Use default based on window size
                cw = (WINDOW_WIDTH // 2) - 10
                ch = WINDOW_HEIGHT - TITLE_BAR_HEIGHT - BOTTOM_PANEL_HEIGHT - 40

        preview_scale = min(1.0, CONFIG['preview']['max_width'] / cw,
                            CONFIG['preview']['max_height'] / ch)
        frame_resized = cv2.resize(frame, (max(1, int(cw * preview_scale)),
                                          max(1, int(ch * preview_scale))))
        pil_img = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))

        if canvas_num == 1:
            self.tk_image_1 = ImageTk.PhotoImage(pil_img)
            canvas.delete("img")
            canvas.create_image(cw // 2, ch // 2, image=self.tk_image_1, anchor=tk.CENTER, tags="img")
        else:
            self.tk_image_2 = ImageTk.PhotoImage(pil_img)
            canvas.delete("img")
            canvas.create_image(cw // 2, ch // 2, image=self.tk_image_2, anchor=tk.CENTER, tags="img")

    # ==============================================================
    # status / progress helpers
    # ==============================================================
    def set_pass_fail(self, status):
        colours = {"PASS": "#10b981", "FAIL": "#ef4444", "READY": "#94a3b8", "LIVE": "#22d3ee"}
        self.status_result_label.config(text=status, fg=colours.get(status, "#94a3b8"))

    def update_progress(self, value, text):
        self.progress_bar['value'] = value
        self.progress_label.config(text=text)
        self.root.update_idletasks()

    # ==============================================================
    # detection trigger – with maximize & highlight
    # ==============================================================
    def start_detect_thread(self, side):
        if getattr(self, 'calibration_page', None) is not None:
            return
        if self.camera1 is None or self.camera2 is None:
            return
        if self.detect_btn_L['state'] == tk.DISABLED:
            return  # Prevent overlapping detections
            
        if not self.active_size:
            self._show_error_popup("Set Size First")
            return

        # Disable both detection buttons while processing
        self.detect_btn_L.config(state=tk.DISABLED)
        self.detect_btn_R.config(state=tk.DISABLED)
        self.video_paused = True

        # Start a thread that does the detection
        threading.Thread(target=self._simulate_detection, args=(side,)).start()

    def _simulate_detection(self, side):
        """
        Placeholder for actual detection.
        Maximizes the selected camera, highlights it with a red border,
        simulates processing, then restores dual view.
        """
        import os
        import datetime
        
        # Determine camera number and canvas
        if side == "L":
            camera_num = 1
            canvas = self.canvas_1
            cam_name = "Camera 1"
            _, frame = self.camera1.get_raw_frame_with_ret()
            frame_undist = self.camera1.get_undistorted_frame()
            mask_path = "mask_1.png"
        else:
            camera_num = 2
            canvas = self.canvas_2
            cam_name = "Camera 2"
            _, frame = self.camera2.get_raw_frame_with_ret()
            frame_undist = self.camera2.get_undistorted_frame()
            mask_path = "mask_2.png"

        if frame is not None:
            ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            orig_dir = os.path.join("Dataset_capture", f"Camera{camera_num}", "Original")
            undist_dir = os.path.join("Dataset_capture", f"Camera{camera_num}", "Undistorted")
            os.makedirs(orig_dir, exist_ok=True)
            os.makedirs(undist_dir, exist_ok=True)
            
            cv2.imwrite(os.path.join(orig_dir, f"frame_{ts}.jpg"), frame)
            if frame_undist is not None:
                cv2.imwrite(os.path.join(undist_dir, f"frame_{ts}.jpg"), frame_undist)

        display_frame = frame
        if frame is not None and os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                if mask.shape[:2] != frame.shape[:2]:
                    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                display_frame = cv2.bitwise_and(frame, frame, mask=mask)
                
                v_coords = cv2.findNonZero(mask)
                if v_coords is not None:
                    x, y, w, h = cv2.boundingRect(v_coords)
                    pad = 30
                    orig_h, orig_w = display_frame.shape[:2]
                    
                    x1 = max(0, x - pad)
                    y1 = max(0, y - pad)
                    x2 = min(orig_w, x + w + pad)
                    y2 = min(orig_h, y + h + pad)
                    
                    display_frame = display_frame[y1:y2, x1:x2]
                
        if display_frame is not None:
            df_copy = display_frame.copy()
            self.root.after(0, lambda f=df_copy, c=camera_num: self._display_video_frame(f, c))

        # Maximize the selected camera on the main thread
        self.root.after(0, lambda: self.maximize_camera(camera_num))

        # Short delay to allow UI to update
        time.sleep(1.2)

        # Draw highlight border on the selected canvas
        def highlight():
            canvas.update_idletasks()
            w = canvas.winfo_width()
            h = canvas.winfo_height()
            # Remove any existing highlight
            canvas.delete("highlight")
            canvas.create_rectangle(2, 2, w-2, h-2, outline="red", width=5, tags="highlight")
            # Remove after 2 seconds
            canvas.after(1000, lambda: canvas.delete("highlight"))

        self.root.after(0, highlight)

        # Simulate some processing time
        time.sleep(1)

        # Print a message (replace with actual detection)
        print(f"Detection triggered for {cam_name} (size: {self.active_size})")

        # Restore dual view on main thread
        self.root.after(0, self.restore_dual_view)

        # Re-enable buttons and resume video on main thread
        self.root.after(0, self._finish_detection)

    def _finish_detection(self):
        self.detect_btn_L.config(state=tk.NORMAL)
        self.detect_btn_R.config(state=tk.NORMAL)
        self.video_paused = False

    # ==============================================================
    # zoom / pan (only used when images are shown, not in live feed)
    # ==============================================================
    def zoom_in_1(self):
        if self.original_full_res_1 is not None:
            self.zoom_level_1 *= 1.25
            self._update_canvas_1()

    def zoom_out_1(self):
        if self.original_full_res_1 is not None:
            self.zoom_level_1 /= 1.25
            self._update_canvas_1()

    def reset_view_1(self):
        if self.original_full_res_1 is not None:
            self.zoom_level_1, self.pan_x_1, self.pan_y_1 = 1.0, 0, 0
            self._update_canvas_1()

    def _on_pan_start_1(self, event):
        if self.original_full_res_1 is not None:
            self.last_x_1, self.last_y_1 = event.x, event.y

    def _on_pan_drag_1(self, event):
        if self.original_full_res_1 is not None:
            self.pan_x_1 += event.x - self.last_x_1
            self.pan_y_1 += event.y - self.last_y_1
            self.last_x_1, self.last_y_1 = event.x, event.y
            self._update_canvas_1()

    def _on_mouse_wheel_1(self, event):
        if self.original_full_res_1 is not None:
            if event.delta > 0:
                self.zoom_in_1()
            else:
                self.zoom_out_1()

    def zoom_in_2(self):
        if self.original_full_res_2 is not None:
            self.zoom_level_2 *= 1.25
            self._update_canvas_2()

    def zoom_out_2(self):
        if self.original_full_res_2 is not None:
            self.zoom_level_2 /= 1.25
            self._update_canvas_2()

    def reset_view_2(self):
        if self.original_full_res_2 is not None:
            self.zoom_level_2, self.pan_x_2, self.pan_y_2 = 1.0, 0, 0
            self._update_canvas_2()

    def _on_pan_start_2(self, event):
        if self.original_full_res_2 is not None:
            self.last_x_2, self.last_y_2 = event.x, event.y

    def _on_pan_drag_2(self, event):
        if self.original_full_res_2 is not None:
            self.pan_x_2 += event.x - self.last_x_2
            self.pan_y_2 += event.y - self.last_y_2
            self.last_x_2, self.last_y_2 = event.x, event.y
            self._update_canvas_2()

    def _on_mouse_wheel_2(self, event):
        if self.original_full_res_2 is not None:
            if event.delta > 0:
                self.zoom_in_2()
            else:
                self.zoom_out_2()

    def _update_canvas_1(self):
        if self.original_full_res_1 is None:
            return
        scale = self.base_scale_1 * self.zoom_level_1
        nw = int(self.original_full_res_1.width * scale)
        nh = int(self.original_full_res_1.height * scale)
        self.tk_image_1 = ImageTk.PhotoImage(self.original_full_res_1.resize((nw, nh), Image.LANCZOS))
        self.canvas_1.delete("img")
        cw = self.canvas_1.winfo_width()
        ch = self.canvas_1.winfo_height()
        if cw <= 1 or ch <= 1:
            cw = (WINDOW_WIDTH // 2) - 10
            ch = WINDOW_HEIGHT - TITLE_BAR_HEIGHT - BOTTOM_PANEL_HEIGHT - 40
        self.canvas_1.create_image((cw // 2) + self.pan_x_1, (ch // 2) + self.pan_y_1,
                                   image=self.tk_image_1, anchor=tk.CENTER, tags="img")

    def _update_canvas_2(self):
        if self.original_full_res_2 is None:
            return
        scale = self.base_scale_2 * self.zoom_level_2
        nw = int(self.original_full_res_2.width * scale)
        nh = int(self.original_full_res_2.height * scale)
        self.tk_image_2 = ImageTk.PhotoImage(self.original_full_res_2.resize((nw, nh), Image.LANCZOS))
        self.canvas_2.delete("img")
        cw = self.canvas_2.winfo_width()
        ch = self.canvas_2.winfo_height()
        if cw <= 1 or ch <= 1:
            cw = (WINDOW_WIDTH // 2) - 10
            ch = WINDOW_HEIGHT - TITLE_BAR_HEIGHT - BOTTOM_PANEL_HEIGHT - 40
        self.canvas_2.create_image((cw // 2) + self.pan_x_2, (ch // 2) + self.pan_y_2,
                                   image=self.tk_image_2, anchor=tk.CENTER, tags="img")

    def show_image(self, pil_image, canvas_num=1):
        """Keep for compatibility – not used in current live feed, but can be used later."""
        if canvas_num == 1:
            self.original_full_res_1 = pil_image
            cw = (WINDOW_WIDTH // 2) - 10
            ch = WINDOW_HEIGHT - TITLE_BAR_HEIGHT - BOTTOM_PANEL_HEIGHT - 40
            self.base_scale_1 = get_initial_fit_scale(pil_image, cw, ch)
            self.zoom_level_1, self.pan_x_1, self.pan_y_1 = 1.0, 0, 0
            self._update_canvas_1()
        else:
            self.original_full_res_2 = pil_image
            cw = (WINDOW_WIDTH // 2) - 10
            ch = WINDOW_HEIGHT - TITLE_BAR_HEIGHT - BOTTOM_PANEL_HEIGHT - 40
            self.base_scale_2 = get_initial_fit_scale(pil_image, cw, ch)
            self.zoom_level_2, self.pan_x_2, self.pan_y_2 = 1.0, 0, 0
            self._update_canvas_2()

# ======================================================================
# MAIN
# ======================================================================
def main():
    root = tk.Tk()

    def load_initial():
        app.update_progress(0, "Initializing cameras…")
        app.start_video_stream()

    # ------------------------------------------------------------------
    def reset_system():
        app.update_progress(0, "Resetting…")

        app.result_image_1 = app.result_image_2 = None
        app.original_full_res_1 = app.original_full_res_2 = None

        # Ensure we are in dual view
        app.restore_dual_view()

        # Clear any leftover drawings on canvases
        app.canvas_1.delete("all")
        app.canvas_2.delete("all")

        # Reset zoom/pan values
        app.zoom_level_1 = app.zoom_level_2 = 1.0
        app.pan_x_1 = app.pan_y_1 = app.pan_x_2 = app.pan_y_2 = 0

        # Re-enable detection buttons
        app.detect_btn_L.config(state=tk.NORMAL)
        app.detect_btn_R.config(state=tk.NORMAL)

        app.video_paused = False
        app.set_pass_fail("LIVE")
        app.update_progress(100, "Live Feed")

    # ------------------------------------------------------------------
    app = IndustrialDashboard(root, reset_system)
    load_initial()
    root.mainloop()

if __name__ == "__main__":
    main()

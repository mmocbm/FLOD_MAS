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
import math
from concurrent.futures import ThreadPoolExecutor
from app_config import CONFIG, project_path
import plane_scale
import sam_detection
from CalibrateAPP.calibration_ui import CalibrationApp, CalibrationCheckApp
from crop_processing import (
    definition_fits_image, extract_rotated_crop,
    load_crop_store, normalized_definition, pixel_definition,
    parallel_line_angle, rotated_crop_corners, save_crop_store, stack_crop_results,
)
from ui_theme import (
    COLORS as C, FONT, button as themed_button, card as themed_card,
    configure_ttk, section_label, set_button_role, status_dot,
)

try:
    import serial
except ImportError:
    serial = None

# ---------------- CONFIG ----------------
WINDOW_WIDTH = 1366
WINDOW_HEIGHT = 768
BOTTOM_PANEL_HEIGHT = 118
TITLE_BAR_HEIGHT = 44

# Serial Configuration
SERIAL_PORT = CONFIG['serial']['port']
BAUD_RATE = CONFIG['serial']['baud_rate']

# Camera configuration
CAMERA_INDEX_1, CAMERA_INDEX_2 = [c['index'] for c in CONFIG['cameras']]
CALIB_FILE_1, CALIB_FILE_2 = [project_path(c['calibration_file']) for c in CONFIG['cameras']]
CROP_DEFINITIONS_FILE = project_path(CONFIG['crop_setup']['definitions_file'])
CROP_RATIO = CONFIG['crop_setup']['aspect_ratio'][0] / CONFIG['crop_setup']['aspect_ratio'][1]
CROP_OUTPUT_SIZE = tuple(CONFIG['crop_setup']['output_size'])

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

        # Two persistent, independently deskewed crop regions per camera.
        self.crop_definitions = load_crop_store(CROP_DEFINITIONS_FILE)
        self.crop_setup_active = False
        # Millimetre scale per (camera, crop). It depends only on the saved
        # region and the calibration files, so it is built once and reused.
        self.strip_scales = {}

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
        self.serial_lock = threading.Lock()
        self.inspection_busy = False
        self._serial_pending_side = None
        self._init_serial()

        # --- window chrome ---
        self.root.overrideredirect(True)

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width // 2) - (WINDOW_WIDTH // 2)
        y = (screen_height // 2) - (WINDOW_HEIGHT // 2)
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}+{x}+{y}")
        self.root.configure(bg=C["bg"])
        self.ui_style = configure_ttk(self.root)

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
            self.serial_conn = serial.Serial(
                SERIAL_PORT, BAUD_RATE, timeout=0.1, write_timeout=0.2,
            )
            threading.Thread(target=self._serial_monitor, daemon=True).start()
            print(f"Serial connected on {SERIAL_PORT}")
        except Exception as e:
            print(f"Could not connect to serial port {SERIAL_PORT}: {e}")

    def _serial_monitor(self):
        while not getattr(self, '_closing', False):
            if self.serial_conn and self.serial_conn.is_open:
                try:
                    if self.serial_conn.in_waiting > 0:
                        line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                        if line == "LeftCheck":
                            print("ESP32 -> LeftCheck")
                            self.root.after(0, self._handle_serial_button, "L")
                        elif line == "RightCheck":
                            print("ESP32 -> RightCheck")
                            self.root.after(0, self._handle_serial_button, "R")
                except Exception as e:
                    print(f"Serial read error: {e}")
                    time.sleep(1)
            time.sleep(0.01)

    def _send_serial_status(self, message):
        """Reply only after the dashboard has accepted or rejected a request."""
        connection = getattr(self, 'serial_conn', None)
        if connection is None or not connection.is_open:
            return
        try:
            with self.serial_lock:
                connection.write((message + "\n").encode('ascii'))
        except Exception as error:
            print(f"Serial write error ({message}): {error}")

    def _handle_serial_button(self, side):
        if getattr(self, 'inspection_busy', False):
            self._send_serial_status(f"{side}_BUSY")
            return
        camera = self.camera1 if side == "L" else self.camera2
        if (camera is None or not getattr(camera, 'calibration_available', False)
                or self.camera1 is None or self.camera2 is None
                or getattr(self, 'calibration_page', None) is not None):
            self._send_serial_status(f"{side}_NOT_READY")
            return
        # Set this before starting the worker, so even a very fast failure
        # still sends its completion status to the correct button.
        self._serial_pending_side = side
        if self.start_detect_thread(side):
            self._send_serial_status(f"{side}_ACK")
        else:
            self._serial_pending_side = None
            self._send_serial_status(f"{side}_NOT_READY")

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
        self.title_bar = tk.Frame(self.root, bg=C["surface"], height=TITLE_BAR_HEIGHT,
                                  highlightbackground=C["border"], highlightthickness=1)
        self.title_bar.pack(side=tk.TOP, fill=tk.X)
        self.title_bar.pack_propagate(False)
        brand = tk.Frame(self.title_bar, bg=C["surface"])
        brand.pack(side=tk.LEFT, fill=tk.Y, padx=(16, 0))
        tk.Frame(brand, bg=C["accent"], width=4, height=22).pack(side=tk.LEFT, pady=10)
        tk.Label(brand, text="VISION INSPECTION", fg=C["text"], bg=C["surface"],
                 font=(FONT, 11, "bold")).pack(side=tk.LEFT, padx=(10, 8))
        tk.Label(brand, text="Production console", fg=C["muted"], bg=C["surface"],
                 font=(FONT, 9)).pack(side=tk.LEFT)
        themed_button(self.title_bar, "✕", self.close_application, role="quiet",
                      padx=16, pady=6, font_size=12).pack(side=tk.RIGHT, fill=tk.Y)
        themed_button(self.title_bar, "—", self.minimize_window, role="quiet",
                      padx=16, pady=6, font_size=12).pack(side=tk.RIGHT, fill=tk.Y)
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
        self.main_frame = tk.Frame(self.root, bg=C["bg"])
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        self._create_image_panel()
        self._create_bottom_panel()

    # --------------------------------------------------------------
    # bottom panel
    # --------------------------------------------------------------
    def _create_bottom_panel(self):
        self.bottom_panel = tk.Frame(
            self.main_frame, height=BOTTOM_PANEL_HEIGHT, bg=C["surface"],
            highlightbackground=C["border"], highlightthickness=1,
        )
        self.bottom_panel.pack(side=tk.BOTTOM, fill=tk.X)
        self.bottom_panel.pack_propagate(False)

        # ---- left: size display only ----
        left_section = tk.Frame(self.bottom_panel, bg=C["surface"])
        left_section.pack(side=tk.LEFT, fill=tk.Y, padx=(18, 12), pady=13)

        # Large size display
        self.lbl_large_size = tk.Label(
            left_section,
            text=self.active_size,
            fg=C["text"], bg=C["surface"],
            font=(FONT, 28, "bold")
        )
        tk.Label(left_section, text="ACTIVE SIZE", fg=C["muted"], bg=C["surface"],
                 font=(FONT, 8, "bold")).pack(anchor="w")
        self.lbl_large_size.pack(anchor="w", pady=(0, 1))

        self.session_time_label = tk.Label(
            left_section,
            text="",
            fg=C["muted"], bg=C["surface"],
            font=(FONT, 9)
        )
        self.session_time_label.pack(anchor="w", pady=(2, 0))

        # ---- center: status label (LIVE) ----
        center_section = themed_card(self.bottom_panel, bg=C["surface_2"])
        center_section.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=13)
        status_head = tk.Frame(center_section, bg=C["surface_2"])
        status_head.pack(fill=tk.X, padx=16, pady=(10, 0))
        status_dot(status_head).pack(side=tk.LEFT, padx=(0, 7))
        tk.Label(status_head, text="SYSTEM STATUS", fg=C["muted"], bg=C["surface_2"],
                 font=(FONT, 8, "bold")).pack(side=tk.LEFT)
        self.status_result_label = tk.Label(
            center_section, text="LIVE", fg=C["accent"], bg=C["surface_2"],
            font=(FONT, 20, "bold"), width=9, anchor="w",
        )
        self.status_result_label.pack(fill=tk.X, padx=16, pady=(2, 8))

        # ---- right: buttons ----
        right_section = tk.Frame(self.bottom_panel, bg=C["surface"])
        right_section.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 16), pady=13)
        self.btn_container = tk.Frame(right_section, bg=C["surface"])
        self.btn_container.pack(expand=True)
        themed_button(self.btn_container, "SETTINGS", self.open_settings_selector,
                      role="secondary", width=11, pady=15).pack(side=tk.LEFT, padx=4)
        themed_button(self.btn_container, "RESET", self.on_reset_callback,
                      role="danger", width=9, pady=15).pack(side=tk.LEFT, padx=4)
        self.detect_btn_L = themed_button(
            self.btn_container, "INSPECT LEFT", lambda: self.start_detect_thread("L"),
            role="blue", width=13, pady=15,
        )
        self.detect_btn_L.pack(side=tk.LEFT, padx=4)
        self.detect_btn_R = themed_button(
            self.btn_container, "INSPECT RIGHT", lambda: self.start_detect_thread("R"),
            role="blue", width=13, pady=15,
        )
        self.detect_btn_R.pack(side=tk.LEFT, padx=4)

    # ==============================================================
    # settings windows
    # ==============================================================
    def open_settings_selector(self):
        self.video_paused = True
        self.root.withdraw()

        self.sel_win = tk.Toplevel(self.root)
        self.sel_win.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}+{self.root.winfo_x()}+{self.root.winfo_y()}")
        self.sel_win.configure(bg=C["bg"])
        self.sel_win.overrideredirect(True)
        self.sel_win.attributes("-topmost", True)

        header = tk.Frame(self.sel_win, bg=C["surface"], height=TITLE_BAR_HEIGHT,
                          highlightbackground=C["border"], highlightthickness=1)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        themed_button(
            header, "←  DASHBOARD",
            lambda: [self.sel_win.destroy(), self.root.deiconify(), self.resume_video()],
            role="quiet", padx=16, pady=6,
        ).pack(side=tk.LEFT, fill=tk.Y)
        tk.Label(header, text="SYSTEM SETTINGS", fg=C["muted"], bg=C["surface"],
                 font=(FONT, 10, "bold")).pack(side=tk.LEFT, padx=12)
        themed_button(header, "✕", self.close_application, role="quiet",
                      padx=16, pady=6, font_size=12).pack(side=tk.RIGHT, fill=tk.Y)
        header.bind("<ButtonPress-1>", self._start_move)
        header.bind("<B1-Motion>", self._do_move)

        container = tk.Frame(self.sel_win, bg=C["bg"])
        container.pack(fill=tk.BOTH, expand=True, padx=72, pady=54)
        tk.Label(container, text="System settings", fg=C["text"], bg=C["bg"],
                 font=(FONT, 30, "bold")).pack(anchor="w")
        tk.Label(
            container, text="Choose what you want to prepare before inspection.",
            fg=C["muted"], bg=C["bg"], font=(FONT, 12),
        ).pack(anchor="w", pady=(5, 30))

        choices = tk.Frame(container, bg=C["bg"])
        choices.pack(fill=tk.X)
        for column in range(2):
            choices.columnconfigure(column, weight=1, uniform="settings")

        size_card = themed_card(choices)
        size_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        section_label(size_card, "Inspection profile").pack(anchor="w", padx=24, pady=(24, 8))
        tk.Label(size_card, text="Size & limits", fg=C["text"], bg=C["card"],
                 font=(FONT, 22, "bold")).pack(anchor="w", padx=24)
        tk.Label(
            size_card, text="Select the product size, allowed variation, segments, and color mask.",
            fg=C["muted"], bg=C["card"], font=(FONT, 11), justify=tk.LEFT,
            wraplength=430,
        ).pack(anchor="w", padx=24, pady=(8, 30))
        themed_button(size_card, "OPEN SIZE SETTINGS  →", self.open_size_window,
                      role="primary").pack(anchor="w", padx=24, pady=(0, 24))

        camera_card = themed_card(choices)
        camera_card.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        section_label(camera_card, "Measurement quality").pack(anchor="w", padx=24, pady=(24, 8))
        tk.Label(camera_card, text="Camera setup", fg=C["text"], bg=C["card"],
                 font=(FONT, 22, "bold")).pack(anchor="w", padx=24)
        tk.Label(
            camera_card, text="Prepare each camera and save the flat measurement surface.",
            fg=C["muted"], bg=C["card"], font=(FONT, 11), justify=tk.LEFT,
            wraplength=430,
        ).pack(anchor="w", padx=24, pady=(8, 30))
        themed_button(camera_card, "OPEN CAMERA SETUP  →", self.open_camera_setup,
                      role="blue").pack(anchor="w", padx=24, pady=(0, 24))
        themed_button(
            camera_card, "OPEN CALIBRATION CHECK  →", self.open_calibration_checks_page,
            role="purple",
        ).pack(anchor="w", padx=24, pady=(0, 24))

    def open_camera_setup(self):
        """Show setup inside the existing dashboard window and event loop."""
        self._open_camera_tool("setup")

    def open_calibration_checks_page(self):
        """Show the dedicated live calibration-verification page."""
        self._open_camera_tool("checks")

    def _open_camera_tool(self, page):
        existing_app = getattr(self, 'calibration_app', None)
        if existing_app is not None:
            if getattr(self, '_camera_tool_page', None) == page:
                return
            if not existing_app.shutdown():
                return
            if getattr(self, 'calibration_page', None) is not None:
                self.calibration_page.destroy()
            self.calibration_page = None
            self.calibration_app = None
        else:
            self.video_streaming = False
            if getattr(self, '_video_job', None) is not None:
                self.root.after_cancel(self._video_job)
                self._video_job = None
            # Keep the streams open; both pages borrow the same full-resolution devices.
            if hasattr(self, 'sel_win') and self.sel_win.winfo_exists():
                self.sel_win.destroy()
            self.title_bar.pack_forget()
            self.main_frame.pack_forget()

        self._camera_tool_page = page
        self.calibration_page = tk.Frame(self.root, bg=C["bg"])
        self.calibration_page.pack(fill=tk.BOTH, expand=True)
        self.root.deiconify()
        try:
            if page == "checks":
                self.calibration_app = CalibrationCheckApp(
                    self.root, host=self.calibration_page,
                    on_close=self.close_camera_setup,
                    on_open_setup=self.open_camera_setup,
                    camera_provider=self._setup_camera_stream,
                )
            else:
                self.calibration_app = CalibrationApp(
                    self.root, host=self.calibration_page,
                    on_close=self.close_camera_setup,
                    on_open_checks=self.open_calibration_checks_page,
                    camera_provider=self._setup_camera_stream,
                )
        except Exception as e:
            self.close_camera_setup()
            self._show_error_popup(f"Could not open camera tool:\n{e}")

    def close_camera_setup(self):
        """Return immediately; keep devices open and reload saved calibration."""
        if getattr(self, 'calibration_page', None) is not None:
            self.calibration_page.destroy()
        self.calibration_page = None
        self.calibration_app = None
        self._camera_tool_page = None
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
        self.size_win.configure(bg=C["bg"])
        self.size_win.overrideredirect(True)
        self.size_win.attributes("-topmost", True)

        header = tk.Frame(self.size_win, bg=C["surface"], height=TITLE_BAR_HEIGHT,
                          highlightbackground=C["border"], highlightthickness=1)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        themed_button(
            header, "←  SETTINGS",
            command=lambda: [self.size_win.destroy(), self.open_settings_selector()],
            role="quiet", padx=16, pady=6,
        ).pack(side=tk.LEFT, fill=tk.Y)
        tk.Label(
            header, text="INSPECTION PROFILE",
            fg=C["muted"], bg=C["surface"], font=(FONT, 10, "bold"),
        ).pack(side=tk.LEFT, padx=10)
        themed_button(header, "✕", self.close_application, role="quiet",
                      padx=16, pady=6, font_size=12).pack(side=tk.RIGHT, fill=tk.Y)

        content = tk.Frame(self.size_win, bg=C["bg"])
        content.pack(expand=True, fill=tk.BOTH, padx=54, pady=30)
        tk.Label(
            content, text="Inspection profile", fg=C["text"], bg=C["bg"],
            font=(FONT, 28, "bold"),
        ).pack(anchor="w")
        tk.Label(
            content, text="Set the product size and allowed measurement variation.",
            fg=C["muted"], bg=C["bg"], font=(FONT, 11),
        ).pack(anchor="w", pady=(4, 20))

        size_card = themed_card(content)
        size_card.pack(fill=tk.X)
        section_label(size_card, "Product size").pack(anchor="w", padx=20, pady=(17, 4))
        tk.Label(size_card, text="Choose the size being inspected", fg=C["text_soft"],
                 bg=C["card"], font=(FONT, 10)).pack(anchor="w", padx=20)
        size_frame = tk.Frame(size_card, bg=C["card"])
        size_frame.pack(fill=tk.X, padx=16, pady=(10, 17))

        self._size_buttons = {}
        for size in FIXED_SIZES:
            btn = themed_button(
                size_frame, size, lambda s=size: self._select_size(s),
                role="secondary", width=9, font_size=13, pady=12,
            )
            btn.pack(side=tk.LEFT, padx=4)
            self._size_buttons[size] = btn

        settings = themed_card(content)
        settings.pack(fill=tk.X, pady=14)
        for column in range(3):
            settings.columnconfigure(column, weight=1, uniform="limits")
        section_label(settings, "Measurement limits").grid(
            row=0, column=0, columnspan=3, sticky="w", padx=20, pady=(17, 12))

        label_options = {"fg": C["text_soft"], "bg": C["card"], "font": (FONT, 10, "bold")}
        tk.Label(settings, text="Length variation", **label_options).grid(row=1, column=0, sticky="w", padx=20)
        self.len_tolerance_menu = tk.OptionMenu(
            settings, self.len_threshold_var,
            "± 0.5mm", "± 1mm", "± 1.5mm", "± 2mm", "± 2.5mm", "± 3mm", "± 3.5mm", "± 4mm", "± 4.5mm", "± 5mm", "± 10mm"
        )
        self.len_tolerance_menu.config(bg=C["surface_2"], fg=C["text"], activebackground=C["card_hover"],
                                       activeforeground=C["text"], font=(FONT, 11), relief=tk.FLAT,
                                       highlightthickness=1, highlightbackground=C["border"], width=18)
        self.len_tolerance_menu["menu"].config(bg=C["surface_2"], fg=C["text"], font=(FONT, 11))
        self.len_tolerance_menu.grid(row=2, column=0, padx=20, pady=(7, 20), sticky="ew")

        tk.Label(settings, text="Width variation", **label_options).grid(row=1, column=1, sticky="w", padx=20)
        self.wid_tolerance_menu = tk.OptionMenu(
            settings, self.wid_threshold_var,
            "± 0.5mm", "± 1mm", "± 1.5mm", "± 2mm", "± 2.5mm", "± 3mm", "± 3.5mm", "± 4mm", "± 4.5mm", "± 5mm", "± 10mm"
        )
        self.wid_tolerance_menu.config(bg=C["surface_2"], fg=C["text"], activebackground=C["card_hover"],
                                       activeforeground=C["text"], font=(FONT, 11), relief=tk.FLAT,
                                       highlightthickness=1, highlightbackground=C["border"], width=18)
        self.wid_tolerance_menu["menu"].config(bg=C["surface_2"], fg=C["text"], font=(FONT, 11))
        self.wid_tolerance_menu.grid(row=2, column=1, padx=20, pady=(7, 20), sticky="ew")

        tk.Label(settings, text="Measurement segments", **label_options).grid(row=1, column=2, sticky="w", padx=20)
        self.segments_combo = ttk.Combobox(
            settings,
            values=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"],
            state="readonly",
            font=(FONT, 11), style="App.TCombobox",
        )
        self.segments_combo.set(str(self.n_segments_var.get()))
        self.segments_combo.grid(row=2, column=2, padx=20, pady=(7, 20), sticky="ew")
        self.segments_combo.bind("<<ComboboxSelected>>", lambda e: self.n_segments_var.set(int(self.segments_combo.get())))

        footer = tk.Frame(content, bg=C["bg"])
        footer.pack(fill=tk.X, pady=(4, 0))
        check = tk.Checkbutton(
            footer, text="Use PASS / FAIL checking",
            variable=self.enable_check_var,
            bg=C["bg"], fg=C["text_soft"], activebackground=C["bg"],
            activeforeground=C["text"], selectcolor=C["surface_2"],
            font=(FONT, 11, "bold"), bd=0, highlightthickness=0,
        )
        check.pack(side=tk.LEFT)
        themed_button(footer, "CROP SETUP", self.open_crop_setup_window,
                      role="purple", width=13).pack(side=tk.RIGHT, padx=(8, 0))
        themed_button(footer, "SAVE PROFILE", self.save_settings,
                      role="primary", width=14).pack(side=tk.RIGHT)

        self._select_size(self.size_var.get())

    # ---------- Helper methods for size buttons ----------
    def _select_size(self, size):
        self.size_var.set(size)
        for s, btn in self._size_buttons.items():
            set_button_role(btn, "selected" if s == size else "secondary")

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
        self.lbl_large_size.config(text=self.active_size)
        
        self.session_start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.session_time_label.config(text=f"Session Start: {self.session_start_time}")

        self._show_success_popup()

    def _show_success_popup(self):
        pop = tk.Toplevel(self.size_win)
        pop.overrideredirect(True)
        pop.attributes("-topmost", True)
        pop.configure(bg=C["card"], highlightbackground=C["border_strong"], highlightthickness=1)

        px = self.size_win.winfo_x() + (WINDOW_WIDTH // 2) - 160
        py = self.size_win.winfo_y() + (WINDOW_HEIGHT // 2) - 80
        pop.geometry(f"320x160+{px}+{py}")

        tk.Label(pop, text="✓", fg=C["success"], bg=C["card"], font=(FONT, 32, "bold")).pack(pady=(14, 0))
        tk.Label(pop, text="Profile saved", fg=C["text"], bg=C["card"],
                 font=(FONT, 13, "bold")).pack()

        def close_pop():
            pop.destroy()
            self.size_win.destroy()
            self.open_settings_selector()

        themed_button(pop, "DONE", close_pop, role="primary", width=10, pady=7).pack(pady=14)

    def _show_error_popup(self, message):
        pop = tk.Toplevel(self.root)
        pop.overrideredirect(True)
        pop.attributes("-topmost", True)
        pop.configure(bg=C["card"], highlightbackground=C["border_strong"], highlightthickness=1)

        px = self.root.winfo_x() + (WINDOW_WIDTH // 2) - 225
        py = self.root.winfo_y() + (WINDOW_HEIGHT // 2) - 140
        pop.geometry(f"450x280+{px}+{py}")

        title_bar = tk.Frame(pop, bg=C["surface"], height=38)
        title_bar.pack(fill=tk.X)
        title_bar.pack_propagate(False)
        tk.Label(title_bar, text="ACTION NEEDED", fg=C["muted"], bg=C["surface"],
                 font=(FONT, 9, "bold")).pack(side=tk.LEFT, padx=12)
        themed_button(title_bar, "✕", pop.destroy, role="quiet", padx=12,
                      pady=4).pack(side=tk.RIGHT, fill=tk.Y)

        content_frame = tk.Frame(pop, bg=C["card"])
        content_frame.pack(fill=tk.BOTH, expand=True, pady=15)
        tk.Label(content_frame, text="!", fg=C["danger"], bg=C["card"],
                 font=(FONT, 36, "bold")).pack(pady=(10, 4))
        tk.Label(content_frame, text=message, fg=C["text"], bg=C["card"],
                 font=(FONT, 14, "bold"), wraplength=390).pack(pady=(2, 18))

        btn_frame = tk.Frame(content_frame, bg=C["card"])
        btn_frame.pack(pady=(0, 15))

        themed_button(btn_frame, "OPEN SETTINGS", lambda: [pop.destroy(), self.open_settings_selector()],
                      role="blue", width=15).pack(side=tk.LEFT, padx=5)
        themed_button(btn_frame, "CLOSE", pop.destroy, role="secondary",
                      width=11).pack(side=tk.LEFT, padx=5)

    def resume_video(self):
        self.video_paused = False

    def open_mask_setup_window(self):
        """Backward-compatible name for the crop setup that replaced masks."""
        return self.open_crop_setup_window()
    # ==============================================================
    # two-crop setup (undistorted image, 4:1 crop, two-point deskew)
    # ==============================================================
    def open_crop_setup_window(self):
        self.size_win.destroy()
        self.crop_win = tk.Toplevel(self.root)
        self.crop_win.geometry(
            f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}+{self.root.winfo_x()}+{self.root.winfo_y()}"
        )
        self.crop_win.configure(bg=C["bg"])
        self.crop_win.overrideredirect(True)
        self.crop_win.attributes("-topmost", True)

        self.crop_setup_active = True
        self.crop_selected_camera = 1
        self.crop_selected_index = 0
        self.crop_frozen = False
        self.crop_live_frame = None
        self.crop_frozen_frame = None
        self.crop_edit = None
        self.crop_edit_mode = "draw"
        self.crop_line_points = []
        self.crop_drag = None
        self.video_paused = False

        header = tk.Frame(
            self.crop_win, bg=C["surface"], height=TITLE_BAR_HEIGHT,
            highlightbackground=C["border"], highlightthickness=1,
        )
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        def back_to_profile():
            self.crop_setup_active = False
            self.crop_frozen = False
            self.video_paused = True
            self.crop_win.destroy()
            self.open_size_window()

        themed_button(header, "←  PROFILE", back_to_profile, role="quiet",
                      padx=16, pady=6).pack(side=tk.LEFT, fill=tk.Y)
        tk.Label(header, text="TWO-CROP SETUP", fg=C["muted"], bg=C["surface"],
                 font=(FONT, 10, "bold")).pack(side=tk.LEFT, padx=10)
        themed_button(header, "✕", self.close_application, role="quiet",
                      padx=16, pady=6, font_size=12).pack(side=tk.RIGHT, fill=tk.Y)

        content = tk.Frame(self.crop_win, bg=C["bg"])
        content.pack(expand=True, fill=tk.BOTH, padx=18, pady=14)
        controls = themed_card(content)
        controls.pack(fill=tk.X, pady=(0, 8))

        text_box = tk.Frame(controls, bg=C["card"])
        text_box.pack(side=tk.LEFT, padx=(14, 18), pady=8)
        section_label(text_box, "Manual regions").pack(anchor="w")
        tk.Label(text_box, text="Two 4:1 deskewed crops per camera",
                 fg=C["text_soft"], bg=C["card"], font=(FONT, 9)).pack(anchor="w")

        self.crop_camera_buttons = {}
        for camera in (1, 2):
            button = themed_button(
                controls, f"CAMERA {camera}",
                lambda value=camera: self._select_crop_camera(value),
                role="selected" if camera == 1 else "secondary", width=9, pady=8,
            )
            button.pack(side=tk.LEFT, padx=3, pady=9)
            self.crop_camera_buttons[camera] = button

        self.crop_index_buttons = {}
        for crop_index in (0, 1):
            button = themed_button(
                controls, f"CROP {crop_index + 1}",
                lambda value=crop_index: self._select_crop_index(value),
                role="selected" if crop_index == 0 else "secondary", width=8, pady=8,
            )
            button.pack(side=tk.LEFT, padx=3, pady=9)
            self.crop_index_buttons[crop_index] = button

        themed_button(controls, "CAPTURE", self.capture_crop_frame,
                      role="blue", width=9, pady=8).pack(side=tk.RIGHT, padx=(3, 12), pady=9)
        themed_button(controls, "SAVE CROP", self.save_crop_region,
                      role="primary", width=10, pady=8).pack(side=tk.RIGHT, padx=3, pady=9)
        themed_button(controls, "CLEAR", self.clear_crop_region,
                      role="danger", width=7, pady=8).pack(side=tk.RIGHT, padx=3, pady=9)
        themed_button(controls, "SET LINE", self.start_crop_rotation_line,
                      role="purple", width=9, pady=8).pack(side=tk.RIGHT, padx=3, pady=9)

        self.crop_status = tk.Label(
            content, text="Live preview is raw; CAPTURE freezes and undistorts one frame",
            bg=C["surface_2"], fg=C["text_soft"], font=(FONT, 9, "bold"),
            anchor="w", padx=12, pady=7,
        )
        self.crop_status.pack(fill=tk.X, pady=(0, 8))

        self.crop_canvas = tk.Canvas(
            content, bg=C["camera"], highlightthickness=1,
            highlightbackground=C["border"], cursor="crosshair",
        )
        self.crop_canvas.pack(fill=tk.BOTH, expand=True)
        self.crop_canvas.bind("<ButtonPress-1>", self.on_crop_press)
        self.crop_canvas.bind("<B1-Motion>", self.on_crop_drag)
        self.crop_canvas.bind("<ButtonRelease-1>", self.on_crop_release)

    def _select_crop_camera(self, camera):
        self.crop_selected_camera = camera
        self.crop_frozen = False
        self.crop_live_frame = None
        self.crop_frozen_frame = None
        self.crop_edit = None
        self.crop_line_points = []
        for number, button in self.crop_camera_buttons.items():
            set_button_role(button, "selected" if number == camera else "secondary")
        self.crop_status.configure(
            text=f"Camera {camera} selected — live preview is raw; CAPTURE applies calibration"
        )

    def _select_crop_index(self, crop_index):
        self.crop_selected_index = crop_index
        for number, button in self.crop_index_buttons.items():
            set_button_role(button, "selected" if number == crop_index else "secondary")
        if self.crop_frozen_frame is not None:
            self._load_selected_crop_for_edit()
            self._display_crop_setup_frame(self.crop_frozen_frame)
        self.crop_status.configure(
            text=f"Camera {self.crop_selected_camera}, Crop {crop_index + 1} selected"
        )

    def capture_crop_frame(self):
        if self.crop_live_frame is None:
            self._show_error_popup("No camera frame is available.")
            return
        camera = self.camera1 if self.crop_selected_camera == 1 else self.camera2
        if camera is None or camera.undistorter is None:
            self._show_error_popup(
                f"Camera {self.crop_selected_camera} must be calibrated before crop setup."
            )
            return
        # Keep the moving preview inexpensive. Lens correction is applied only
        # once, to the exact rotation-adjusted raw frame frozen by CAPTURE.
        try:
            self.crop_frozen_frame = camera.undistorter.undistort(
                self.crop_live_frame.copy()
            )
        except (cv2.error, ValueError, TypeError) as error:
            self._show_error_popup(f"Could not undistort the captured frame:\n{error}")
            return
        self.crop_frozen = True
        self.crop_edit_mode = "draw"
        self._load_selected_crop_for_edit()
        self._display_crop_setup_frame(self.crop_frozen_frame)
        self.crop_status.configure(
            text="Drag to create a 4:1 crop. Drag inside to move; drag a corner to resize."
        )

    def _load_selected_crop_for_edit(self):
        saved = self.crop_definitions["cameras"][str(self.crop_selected_camera)][
            self.crop_selected_index
        ]
        if saved is None or self.crop_frozen_frame is None:
            self.crop_edit = None
            self.crop_line_points = []
            return
        height, width = self.crop_frozen_frame.shape[:2]
        self.crop_edit = pixel_definition(saved, (width, height), CROP_RATIO)
        self.crop_line_points = [list(point) for point in self.crop_edit["line"]]

    def start_crop_rotation_line(self):
        if not self.crop_frozen or self.crop_edit is None:
            self._show_error_popup("Capture a frame and draw the crop first.")
            return
        self.crop_edit_mode = "line"
        self.crop_line_points = []
        self.crop_status.configure(text="Click two points along the direction that must become horizontal")
        self._redraw_crop_overlay()

    def clear_crop_region(self):
        self.crop_edit = None
        self.crop_line_points = []
        self.crop_edit_mode = "draw"
        camera_crops = self.crop_definitions["cameras"][str(self.crop_selected_camera)]
        camera_crops[self.crop_selected_index] = None
        save_crop_store(CROP_DEFINITIONS_FILE, self.crop_definitions)
        self._redraw_crop_overlay()
        self.crop_status.configure(text="Crop cleared — drag to create a new 4:1 region")

    def save_crop_region(self):
        if not self.crop_frozen or self.crop_frozen_frame is None or self.crop_edit is None:
            self._show_error_popup("Capture a frame and draw the crop first.")
            return
        if len(self.crop_line_points) != 2:
            self._show_error_popup("Select SET LINE and click two rotation points first.")
            return
        height, width = self.crop_frozen_frame.shape[:2]
        definition = normalized_definition(
            self.crop_edit["center"], self.crop_edit["width"],
            self.crop_edit["angle_degrees"], self.crop_line_points, (width, height),
        )
        if not definition_fits_image(definition, (width, height), CROP_RATIO):
            self._show_error_popup("The rotated crop must remain completely inside the image.")
            return
        self.crop_definitions["cameras"][str(self.crop_selected_camera)][
            self.crop_selected_index
        ] = definition
        save_crop_store(CROP_DEFINITIONS_FILE, self.crop_definitions)
        self.crop_status.configure(
            text=(f"Saved Camera {self.crop_selected_camera}, Crop "
                  f"{self.crop_selected_index + 1} at {self.crop_edit['angle_degrees']:.1f}°")
        )

    def _display_crop_setup_frame(self, frame):
        if not hasattr(self, "crop_canvas") or not self.crop_canvas.winfo_exists():
            return
        self.crop_canvas.update_idletasks()
        canvas_width = max(2, self.crop_canvas.winfo_width())
        canvas_height = max(2, self.crop_canvas.winfo_height())
        image_height, image_width = frame.shape[:2]
        scale = min(canvas_width / image_width, canvas_height / image_height)
        display_size = (max(1, round(image_width * scale)),
                        max(1, round(image_height * scale)))
        resized = cv2.resize(frame, display_size, interpolation=cv2.INTER_AREA)
        self.crop_display_scale = scale
        self.crop_display_offset = ((canvas_width - display_size[0]) / 2.0,
                                    (canvas_height - display_size[1]) / 2.0)
        image = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)))
        self.tk_image_crop = image
        self.crop_canvas.delete("all")
        self.crop_canvas.create_image(
            self.crop_display_offset[0], self.crop_display_offset[1], image=image,
            anchor=tk.NW, tags="crop_image",
        )
        self._redraw_crop_overlay()

    def _canvas_to_crop_image(self, x, y):
        if self.crop_frozen_frame is None or not hasattr(self, 'crop_display_scale'):
            return None
        offset_x, offset_y = self.crop_display_offset
        image_x = (x - offset_x) / self.crop_display_scale
        image_y = (y - offset_y) / self.crop_display_scale
        height, width = self.crop_frozen_frame.shape[:2]
        if not (0 <= image_x < width and 0 <= image_y < height):
            return None
        return np.array([image_x, image_y], dtype=np.float64)

    def _crop_image_to_canvas(self, point):
        offset_x, offset_y = self.crop_display_offset
        return np.asarray(point) * self.crop_display_scale + np.array([offset_x, offset_y])

    def _redraw_crop_overlay(self):
        if not hasattr(self, 'crop_canvas') or not self.crop_canvas.winfo_exists():
            return
        self.crop_canvas.delete("crop_overlay")
        if self.crop_edit is not None:
            corners = rotated_crop_corners(
                self.crop_edit["center"], self.crop_edit["width"], CROP_RATIO,
                self.crop_edit.get("angle_degrees", 0.0),
            )
            canvas_corners = np.asarray(
                [self._crop_image_to_canvas(point) for point in corners]
            )
            flattened = canvas_corners.reshape(-1).tolist()
            self.crop_canvas.create_polygon(
                *flattened, outline=C["danger"], fill="", width=3,
                tags="crop_overlay",
            )
            for point in canvas_corners:
                x, y = point
                self.crop_canvas.create_rectangle(
                    x - 5, y - 5, x + 5, y + 5, fill=C["danger"], outline="white",
                    tags="crop_overlay",
                )
        if self.crop_line_points:
            canvas_points = [self._crop_image_to_canvas(point)
                             for point in self.crop_line_points]
            for point in canvas_points:
                x, y = point
                self.crop_canvas.create_oval(
                    x - 6, y - 6, x + 6, y + 6, fill=C["warning"], outline="white",
                    tags="crop_overlay",
                )
            if len(canvas_points) == 2:
                self.crop_canvas.create_line(
                    *canvas_points[0], *canvas_points[1], fill=C["warning"],
                    width=3, arrow=tk.LAST, tags="crop_overlay",
                )

    def on_crop_press(self, event):
        if not self.crop_frozen:
            return
        point = self._canvas_to_crop_image(event.x, event.y)
        if point is None:
            return
        if self.crop_edit_mode == "line":
            self.crop_line_points.append(point.tolist())
            if len(self.crop_line_points) == 2:
                first, second = map(np.asarray, self.crop_line_points)
                delta = second - first
                if np.linalg.norm(delta) < 5:
                    self.crop_line_points = []
                    self.crop_status.configure(text="Rotation points are too close — click two wider points")
                else:
                    self.crop_edit["angle_degrees"] = parallel_line_angle(first, second)
                    self.crop_edit["line"] = [first.tolist(), second.tolist()]
                    self.crop_edit_mode = "draw"
                    self.crop_status.configure(
                        text=f"Deskew angle {self.crop_edit['angle_degrees']:.1f}° — save this crop"
                    )
            self._redraw_crop_overlay()
            return

        self.crop_drag = {"start": point, "kind": "new"}
        if self.crop_edit is not None:
            corners = rotated_crop_corners(
                self.crop_edit["center"], self.crop_edit["width"], CROP_RATIO,
                self.crop_edit.get("angle_degrees", 0.0),
            )
            threshold = 12.0 / self.crop_display_scale
            distances = np.linalg.norm(corners - point, axis=1)
            if float(distances.min()) <= threshold:
                self.crop_drag = {"start": point, "kind": "resize"}
            elif cv2.pointPolygonTest(corners.astype(np.float32), tuple(point), False) >= 0:
                self.crop_drag = {
                    "start": point, "kind": "move",
                    "center": np.asarray(self.crop_edit["center"], dtype=np.float64),
                }

    def on_crop_drag(self, event):
        if not self.crop_frozen or self.crop_drag is None or self.crop_edit_mode == "line":
            return
        point = self._canvas_to_crop_image(event.x, event.y)
        if point is None:
            return
        start = self.crop_drag["start"]
        kind = self.crop_drag["kind"]
        if kind == "move":
            self.crop_edit["center"] = (self.crop_drag["center"] + point - start).tolist()
        elif kind == "resize":
            center = np.asarray(self.crop_edit["center"], dtype=np.float64)
            radians = math.radians(self.crop_edit.get("angle_degrees", 0.0))
            rotation_inverse = np.array([
                [math.cos(radians), math.sin(radians)],
                [-math.sin(radians), math.cos(radians)],
            ])
            local = rotation_inverse @ (point - center)
            self.crop_edit["width"] = max(40.0, 2.0 * max(abs(local[0]), CROP_RATIO * abs(local[1])))
            self.crop_edit["height"] = self.crop_edit["width"] / CROP_RATIO
        else:
            delta = point - start
            width = max(abs(delta[0]), CROP_RATIO * abs(delta[1]), 40.0)
            adjusted = np.array([
                math.copysign(width, delta[0] if delta[0] else 1),
                math.copysign(width / CROP_RATIO, delta[1] if delta[1] else 1),
            ])
            self.crop_edit = {
                "center": (start + adjusted / 2.0).tolist(),
                "width": width, "height": width / CROP_RATIO,
                "angle_degrees": 0.0, "line": [],
            }
            self.crop_line_points = []
        self._redraw_crop_overlay()

    def on_crop_release(self, _event):
        self.crop_drag = None

    # ==============================================================
    # image panel (dual canvases + zoom toolbars)
    # ==============================================================
    def _create_image_panel(self):
        self.image_panel = tk.Frame(self.main_frame, bg=C["bg"])
        self.image_panel.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.images_container = tk.Frame(self.image_panel, bg=C["bg"])
        self.images_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 6))

        # ---- left ----
        self.left_frame = themed_card(self.images_container, bg=C["surface"])
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        left_header = tk.Frame(self.left_frame, bg=C["surface"], height=38)
        left_header.pack(fill=tk.X, padx=12)
        left_header.pack_propagate(False)
        status_dot(left_header).pack(side=tk.LEFT, pady=14, padx=(0, 7))
        tk.Label(left_header, text="LEFT CAMERA", bg=C["surface"], fg=C["text_soft"],
                 font=(FONT, 9, "bold")).pack(side=tk.LEFT, pady=9)
        tk.Label(left_header, text="LIVE", bg=C["surface"], fg=C["accent"],
                 font=(FONT, 8, "bold")).pack(side=tk.RIGHT, pady=10)

        self.canvas_1 = tk.Canvas(self.left_frame, bg=C["camera"], highlightthickness=0)
        self.canvas_1.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))

        self.toolbar_1 = tk.Frame(self.canvas_1, bg=C["surface"], bd=0,
                                  highlightbackground=C["border_strong"], highlightthickness=1)
        themed_button(self.toolbar_1, "+", self.zoom_in_1, role="quiet", width=2,
                      padx=4, pady=5).pack(side=tk.TOP, padx=2, pady=(2, 0))
        themed_button(self.toolbar_1, "−", self.zoom_out_1, role="quiet", width=2,
                      padx=4, pady=5).pack(side=tk.TOP, padx=2)
        themed_button(self.toolbar_1, "↺", self.reset_view_1, role="quiet", width=2,
                      padx=4, pady=5).pack(side=tk.TOP, padx=2, pady=(0, 2))
        # Store the canvas window ID for later repositioning
        self.toolbar_win_id_1 = self.canvas_1.create_window(0, 0, anchor="ne", window=self.toolbar_1)

        # ---- right ----
        self.right_frame = themed_card(self.images_container, bg=C["surface"])
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        right_header = tk.Frame(self.right_frame, bg=C["surface"], height=38)
        right_header.pack(fill=tk.X, padx=12)
        right_header.pack_propagate(False)
        status_dot(right_header).pack(side=tk.LEFT, pady=14, padx=(0, 7))
        tk.Label(right_header, text="RIGHT CAMERA", bg=C["surface"], fg=C["text_soft"],
                 font=(FONT, 9, "bold")).pack(side=tk.LEFT, pady=9)
        tk.Label(right_header, text="LIVE", bg=C["surface"], fg=C["accent"],
                 font=(FONT, 8, "bold")).pack(side=tk.RIGHT, pady=10)

        self.canvas_2 = tk.Canvas(self.right_frame, bg=C["camera"], highlightthickness=0)
        self.canvas_2.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))

        self.toolbar_2 = tk.Frame(self.canvas_2, bg=C["surface"], bd=0,
                                  highlightbackground=C["border_strong"], highlightthickness=1)
        themed_button(self.toolbar_2, "+", self.zoom_in_2, role="quiet", width=2,
                      padx=4, pady=5).pack(side=tk.TOP, padx=2, pady=(2, 0))
        themed_button(self.toolbar_2, "−", self.zoom_out_2, role="quiet", width=2,
                      padx=4, pady=5).pack(side=tk.TOP, padx=2)
        themed_button(self.toolbar_2, "↺", self.reset_view_2, role="quiet", width=2,
                      padx=4, pady=5).pack(side=tk.TOP, padx=2, pady=(0, 2))
        self.toolbar_win_id_2 = self.canvas_2.create_window(0, 0, anchor="ne", window=self.toolbar_2)

        # Bind configure events to reposition toolbars when canvas resizes
        self.canvas_1.bind("<Configure>", self._reposition_toolbar_1)
        self.canvas_2.bind("<Configure>", self._reposition_toolbar_2)

        # ---- progress bar ----
        self.loading_container = tk.Frame(self.image_panel, bg=C["bg"], height=24)
        self.loading_container.pack(side=tk.BOTTOM, fill=tk.X)

        self.progress_label = tk.Label(self.loading_container, text="Live Feed",
                                       fg=C["muted"], bg=C["bg"], font=(FONT, 9))
        self.progress_label.pack(side=tk.LEFT, padx=(14, 10))

        self.progress_bar = ttk.Progressbar(self.loading_container, orient=tk.HORIZONTAL,
                                            mode='determinate', style="App.Horizontal.TProgressbar")
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 14), pady=8)

        # Store original pack options for restoration
        self.left_frame_pack_opts = {'side': tk.LEFT, 'fill': tk.BOTH, 'expand': True, 'padx': (0, 5)}
        self.right_frame_pack_opts = {'side': tk.RIGHT, 'fill': tk.BOTH, 'expand': True, 'padx': (5, 0)}

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
    def _refresh_inspection_availability(self):
        """Enable inspection only for cameras with usable calibration."""
        if getattr(self, 'inspection_busy', False):
            self.detect_btn_L.config(state=tk.DISABLED)
            self.detect_btn_R.config(state=tk.DISABLED)
            return
        left_ready = bool(
            self.camera1 is not None
            and getattr(self.camera1, 'calibration_available', False)
        )
        right_ready = bool(
            self.camera2 is not None
            and getattr(self.camera2, 'calibration_available', False)
        )
        self.detect_btn_L.config(state=tk.NORMAL if left_ready else tk.DISABLED)
        self.detect_btn_R.config(state=tk.NORMAL if right_ready else tk.DISABLED)

        missing = []
        if not left_ready:
            missing.append("left")
        if not right_ready:
            missing.append("right")
        if missing:
            camera_text = " and ".join(missing)
            self.set_pass_fail("SETUP")
            self.update_progress(
                100,
                f"Camera setup required for {camera_text}; live preview is available, inspection is disabled",
            )
        else:
            self.set_pass_fail("LIVE")
            self.update_progress(100, "Live Feed")

    def start_video_stream(self):
        if self.camera1 is not None and self.camera2 is not None:
            self.video_streaming = True
            self.video_paused = False
            self.update_video_feed()
            self._refresh_inspection_availability()
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
        cycle_started = time.perf_counter()
        if self.video_paused:
            self._video_job = self.root.after(CONFIG['preview']['interval_ms'], self.update_video_feed)
            return

        # Read frames from both cameras (we need both to keep them alive)
        ret1, raw1 = self.camera1.get_raw_frame_with_ret()
        ret2, raw2 = self.camera2.get_raw_frame_with_ret()

        if getattr(self, "crop_setup_active", False):
            if not self.crop_frozen:
                # CameraStream has already applied the configured rotation.
                # Show the immutable latest snapshot directly; CAPTURE copies
                # and performs lens correction once.
                selected_ok = ret1 if self.crop_selected_camera == 1 else ret2
                frame = raw1 if self.crop_selected_camera == 1 else raw2
                if selected_ok and frame is not None:
                    self.crop_live_frame = frame
                    self._display_crop_setup_frame(frame)
        elif self.maximized_camera is None:
            # Update each available camera independently. One delayed device
            # should not prevent the other preview from refreshing.
            if ret1:
                self._display_video_frame(raw1, 1)
            if ret2:
                self._display_video_frame(raw2, 2)
        elif self.maximized_camera == 1:
            if ret1:
                self._display_video_frame(raw1, 1)
        elif ret2:
            self._display_video_frame(raw2, 2)

        # Tk's after() delay begins only after this callback returns. Subtract
        # processing time so the configured interval represents frame-to-frame
        # cadence instead of processing time plus another full interval.
        elapsed_ms = (time.perf_counter() - cycle_started) * 1000.0
        delay_ms = max(1, round(CONFIG['preview']['interval_ms'] - elapsed_ms))
        self._video_job = self.root.after(delay_ms, self.update_video_feed)

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

        frame_height, frame_width = frame.shape[:2]
        preview_scale = min(cw / frame_width, ch / frame_height)
        frame_resized = cv2.resize(
            frame,
            (max(1, int(frame_width * preview_scale)),
             max(1, int(frame_height * preview_scale))),
            interpolation=cv2.INTER_AREA if preview_scale < 1 else cv2.INTER_LINEAR,
        )
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
        colours = {"PASS": C["success"], "FAIL": C["danger"],
                   "READY": C["muted"], "LIVE": C["accent"],
                   "WARNING": C["warning"], "SETUP": C["warning"]}
        self.status_result_label.config(text=status, fg=colours.get(status, C["muted"]))

    def update_progress(self, value, text):
        self.progress_bar['value'] = value
        self.progress_label.config(text=text)
        self.root.update_idletasks()

    # ==============================================================
    # detection trigger – with maximize & highlight
    # ==============================================================
    def start_detect_thread(self, side):
        if getattr(self, 'inspection_busy', False):
            return False
        if getattr(self, 'calibration_page', None) is not None:
            return False
        if self.camera1 is None or self.camera2 is None:
            return False
        button = self.detect_btn_L if side == "L" else self.detect_btn_R
        camera = self.camera1 if side == "L" else self.camera2
        if (button['state'] == tk.DISABLED
                or not getattr(camera, 'calibration_available', False)):
            self._refresh_inspection_availability()
            return False
            
        if not self.active_size:
            self._show_error_popup("Set Size First")
            return False

        camera_number = 1 if side == "L" else 2
        saved_crops = self.crop_definitions["cameras"][str(camera_number)]
        if len(saved_crops) != 2 or any(crop is None for crop in saved_crops):
            self._show_error_popup(
                f"Set up both crops for Camera {camera_number} before inspection."
            )
            return False

        # Disable both detection buttons while processing
        self.inspection_busy = True
        self.detect_btn_L.config(state=tk.DISABLED)
        self.detect_btn_R.config(state=tk.DISABLED)
        self.video_paused = True

        # Start a thread that does the detection
        threading.Thread(target=self._run_detection, args=(side,), daemon=True).start()
        return True

    def _run_detection(self, side):
        try:
            self._simulate_detection(side)
        except Exception as error:
            print(f"Inspection {side} failed: {error}")
            self.root.after(0, self._inspection_failed)

    def _inspection_failed(self):
        self.restore_dual_view()
        self._finish_detection(completed=False)
        self.set_pass_fail("WARNING")
        self.update_progress(100, "Inspection could not be completed")

    def _simulate_detection(self, side):
        """Save full frames and prepare two independent deskewed model inputs."""
        if side == "L":
            camera_num = 1
            canvas = self.canvas_1
            cam_name = "Camera 1"
            _, frame = self.camera1.get_raw_frame_with_ret()
            frame_undist = self.camera1.get_undistorted_frame()
        else:
            camera_num = 2
            canvas = self.canvas_2
            cam_name = "Camera 2"
            _, frame = self.camera2.get_raw_frame_with_ret()
            frame_undist = self.camera2.get_undistorted_frame()

        if frame is None:
            raise RuntimeError(f"Camera {camera_num} did not return a frame")
        if frame_undist is None:
            raise RuntimeError(f"Camera {camera_num} could not produce an undistorted frame")

        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        camera_dir = os.path.join("Dataset_capture", f"Camera{camera_num}")
        original_dir = os.path.join(camera_dir, "Original")
        undistorted_dir = os.path.join(camera_dir, "Undistorted")
        crop_dirs = [os.path.join(camera_dir, "Crop1"),
                     os.path.join(camera_dir, "Crop2")]
        for folder in (original_dir, undistorted_dir, *crop_dirs):
            os.makedirs(folder, exist_ok=True)

        if not cv2.imwrite(os.path.join(original_dir, f"frame_{timestamp}.jpg"), frame):
            raise RuntimeError("The original camera frame could not be saved")
        if not cv2.imwrite(
                os.path.join(undistorted_dir, f"frame_{timestamp}.png"), frame_undist):
            raise RuntimeError("The undistorted camera frame could not be saved")

        definitions = self.crop_definitions["cameras"][str(camera_num)]
        if len(definitions) != 2 or any(definition is None for definition in definitions):
            raise RuntimeError(f"Camera {camera_num} requires two saved crop regions")

        crops = []
        for crop_index, (definition, folder) in enumerate(zip(definitions, crop_dirs), start=1):
            crop = extract_rotated_crop(
                frame_undist, definition, CROP_OUTPUT_SIZE, CROP_RATIO,
            )
            crop_path = os.path.join(folder, f"crop_{timestamp}.png")
            if not cv2.imwrite(crop_path, crop):
                raise RuntimeError(f"Camera {camera_num} Crop {crop_index} could not be saved")
            crops.append(crop)
            print(f"Camera {camera_num} Crop {crop_index} saved: {crop_path}")

        display_crops, detection_warnings = self._detect_crops(
            camera_num, timestamp, crops, frame_undist.shape[:2][::-1],
        )
        display_frame = stack_crop_results(display_crops)
        self.root.after(
            0, lambda f=display_frame.copy(), c=camera_num: self._display_video_frame(f, c)
        )
        self.root.after(
            0, self.update_progress, 100,
            f"Camera {camera_num}: two deskewed crops prepared and saved",
        )

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

        print(f"Crop preprocessing completed for {cam_name} (size: {self.active_size})")

        # Restore dual view on main thread
        self.root.after(0, self.restore_dual_view)

        # Re-enable buttons and resume video on main thread. Detection warnings
        # are applied afterwards because _finish_detection resets the status
        # card and progress text back to LIVE.
        def finish():
            self._finish_detection()
            if detection_warnings:
                self.set_pass_fail("WARNING")
                self.update_progress(100, " | ".join(detection_warnings))

        self.root.after(0, finish)

    def _strip_scale(self, camera_num, crop_index, frame_size, warnings):
        """The crop's millimetre scale, or None when it cannot be built.

        Built once per crop region and reused: it depends only on the saved
        region and the calibration files, which do not change mid-session.
        Returns None when no scale is available, so the measurement falls back
        to pixels and the reason is reported rather than silently dropped.
        """
        key = (camera_num, crop_index)
        if key not in self.strip_scales:
            self.strip_scales[key] = self._build_strip_scale(
                key, frame_size, warnings,
            )
        return self.strip_scales[key]

    def _build_strip_scale(self, key, frame_size, warnings):
        camera_num, crop_index = key
        if not CONFIG['sam_detection']['measure_in_mm']:
            return None
        size = frame_size or self._undistorted_size(camera_num)
        if size is None:
            warnings.append(
                f"Crop {crop_index}: no millimetre scale (the undistorted frame "
                "size is unknown)")
            return None
        try:
            definitions = self.crop_definitions['cameras'][str(camera_num)]
            return plane_scale.load_plane_scale(
                camera_num, definitions[crop_index - 1], size,
            )
        except (plane_scale.PlaneScaleError, IndexError, KeyError) as error:
            print(f"Camera {camera_num} Crop {crop_index}: measuring in pixels "
                  f"({error})")
            warnings.append(f"Crop {crop_index}: measuring in pixels ({error})")
            return None

    def _undistorted_size(self, camera_num):
        """The size of the frame the crops are taken from."""
        camera = self.camera1 if camera_num == 1 else self.camera2
        undistorter = getattr(camera, 'undistorter', None)
        size = getattr(undistorter, 'calibration_image_size', None)
        if not size:
            return None
        return (int(size[0]), int(size[1]))

    def _detect_crops(self, camera_num, timestamp, crops, frame_size=None):
        """Run SAM detection on the configured crops; return what to display.

        Returns the crop images to stack (annotated where detection ran) and a
        list of warning strings. This never raises: a failed call leaves the raw
        crop in place and is reported instead, so a network problem can never
        lose an inspection.
        """
        settings = CONFIG['sam_detection']
        selected = settings['send_crops'][str(camera_num)]
        if not settings['enabled'] or not any(selected):
            return crops, []

        prompt = settings['prompt']
        quality = settings['jpeg_quality']
        timeout = settings['timeout_seconds']
        detected_dir = os.path.join("Dataset_capture", f"Camera{camera_num}", "Detected")
        os.makedirs(detected_dir, exist_ok=True)
        self.root.after(0, self.update_progress, 100,
                        f"Camera {camera_num}: detecting objects…")

        display = list(crops)
        warnings = []
        for crop_index, crop in enumerate(crops, start=1):
            if not selected[crop_index - 1]:
                continue
            image_size = (crop.shape[1], crop.shape[0])
            detection = sam_detection.detect_crop(
                crop, prompt, jpeg_quality=quality, timeout_seconds=timeout,
            )
            print(f"Camera {camera_num} Crop {crop_index}: "
                  f"{len(detection.polygons)} polygon(s) from "
                  f"{detection.upload_bytes} byte upload")

            measurement = None
            if detection.polygons and settings['analyze_strip']:
                measurement = sam_detection.analyze_detection(
                    detection.polygons, crop.shape, settings['strip_segments'],
                    self._strip_scale(camera_num, crop_index, frame_size, warnings),
                )
            if measurement is not None:
                summary = measurement.analysis
                length = (f"{summary.total_length_mm:.2f}mm"
                          if summary.metric else f"{summary.total_length_px:.0f}px")
                width = (f"{summary.average_width_mm:.2f}mm"
                         if summary.metric else f"{summary.average_width_px:.1f}px")
                print(f"Camera {camera_num} Crop {crop_index}: strip {length} long, "
                      f"{width} average width over "
                      f"{len(summary.segments)} segment(s)")

            overlay_saved = False
            if detection.polygons and settings['save_overlay']:
                annotated = sam_detection.draw_analysis(
                    crop, detection.polygons, measurement)
                overlay_path = os.path.join(
                    detected_dir, f"crop_{timestamp}_{crop_index}.png")
                if cv2.imwrite(overlay_path, annotated):
                    overlay_saved = True
                    display[crop_index - 1] = annotated
                else:
                    warnings.append(
                        f"Crop {crop_index}: the overlay image could not be saved")

            if settings['save_polygons']:
                record = sam_detection.detection_record(
                    camera_num, crop_index, timestamp, self.active_size, prompt,
                    image_size, detection, quality, overlay_saved, measurement,
                )
                record_path = os.path.join(
                    detected_dir, f"crop_{timestamp}_{crop_index}.json")
                try:
                    with open(record_path, "w", encoding="utf-8") as stream:
                        json.dump(record, stream, indent=2)
                        stream.write("\n")
                except OSError as error:
                    warnings.append(
                        f"Crop {crop_index}: the result file could not be saved ({error})")

            if detection.error:
                warnings.append(f"Crop {crop_index}: {detection.error}")

        return display, warnings

    def _finish_detection(self, completed=True):
        self.video_paused = False
        self.inspection_busy = False
        serial_side = getattr(self, '_serial_pending_side', None)
        self._serial_pending_side = None
        if serial_side is not None:
            self._send_serial_status(
                f"{serial_side}_{'DONE' if completed else 'ERROR'}"
            )
        self._refresh_inspection_availability()

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

        app.video_paused = False
        app._refresh_inspection_availability()

    # ------------------------------------------------------------------
    app = IndustrialDashboard(root, reset_system)
    load_initial()
    root.mainloop()

if __name__ == "__main__":
    main()

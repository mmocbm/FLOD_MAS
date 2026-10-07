# Finish settings before importing any modules that cache configuration.
# Saving starts a fresh dashboard below; no interpreter relaunch is needed.
import os
import sys

# Passed when the app relaunches itself after RESET. The operator already chose
# settings for this session, so the startup countdown is not shown a second time.
NO_STARTUP_FLAG = '--no-startup-window'

if __name__ == '__main__':
    if NO_STARTUP_FLAG not in sys.argv:
        from startup_settings import run_startup
        run_startup()

import tkinter as tk
from tkinter import ttk, messagebox
import time
import math
import copy
import subprocess
from PIL import Image, ImageTk
import cv2
from camera_handler import CameraHandler
import datetime
from concurrent.futures import ThreadPoolExecutor
from app_config import CONFIG, ACTIVE_CAMERAS, CAMERA_COUNT, project_path
from CalibrateAPP.calibration_ui import CalibrationApp, CalibrationCheckApp
from inspection.dashboard import AutomaticDashboard
from ui_theme import (
    COLORS as C, FONT, button as themed_button, card as themed_card,
    configure_ttk, draw_tracking_overlay, preview_transform, section_label,
    set_button_role, status_dot,
)

# ---------------- CONFIG ----------------
WINDOW_WIDTH = 1366
WINDOW_HEIGHT = 768
BOTTOM_PANEL_HEIGHT = 118
TITLE_BAR_HEIGHT = 44

# Camera configuration
CAMERA_INDEX_1 = ACTIVE_CAMERAS[0]['index']
CAMERA_INDEX_2 = ACTIVE_CAMERAS[1]['index'] if CAMERA_COUNT == 2 else None
CALIB_FILE_1 = project_path(ACTIVE_CAMERAS[0]['calibration_file'])
CALIB_FILE_2 = project_path(ACTIVE_CAMERAS[1]['calibration_file']) if CAMERA_COUNT == 2 else None
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
class IndustrialDashboard(AutomaticDashboard):
    # Calibration tools invalidate cached scales when returning to inspection.
    strip_scales = {}

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
        # Hand and bed-marker overlays on the live preview. Session-only: the
        # button below flips this, the config value is only the starting point.
        self.show_overlays = bool(CONFIG['preview'].get('show_overlay', True))

        self.result_image_1 = None
        self.result_image_2 = None

        self.strip_scales = {}

        # --- maximized camera mode ---
        self.maximized_camera = None  # None, 1, or 2

        # --- settings variables ---
        self.size_var = tk.StringVar()
        self.strip_width_var = tk.StringVar(
            value=f"{float(CONFIG['inspection']['strip_width_mm']):g}")
        self.strip_width_tolerance_var = tk.StringVar(
            value=f"{float(CONFIG['inspection']['strip_width_tolerance_mm']):g}")
        self.end_exclusion_var = tk.StringVar(
            value=f"{float(CONFIG['inspection'].get('end_exclusion_percent', 5.0)):g}")
        self.active_end_exclusion_percent = float(self.end_exclusion_var.get())
        self.result_display_seconds_var = tk.StringVar(
            value=f"{float(CONFIG['inspection'].get('result_display_seconds', 5.0)):g}")

        # Set default size
        self.size_var.set(CONFIG['inspection']['default_size'])
        self.available_sizes = FIXED_SIZES

        # --- active (saved) values used during detection ---
        self.active_size = self.size_var.get()
        self.active_strip_width = self._validate_strip_width(self.strip_width_var.get())
        self.active_strip_width_tolerance = self._validate_strip_width_tolerance(
            self.strip_width_tolerance_var.get())
        self.active_result_display_seconds = self._validate_result_display_seconds(
            self.result_display_seconds_var.get())
        self.session_start_time = ""

        self.inspection_busy = False

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
        self._init_automatic()

    # ==============================================================
    # helpers
    # ==============================================================


    def _validate_strip_width(self, value):
        try:
            w = float(value)
            return w if w > 0 else 4.0
        except (ValueError, TypeError):
            return 4.0

    def _validate_strip_width_tolerance(self, value):
        try:
            tolerance = float(value)
            return tolerance if tolerance > 0 else 1.0
        except (ValueError, TypeError):
            return 1.0

    def _validate_result_display_seconds(self, value):
        try:
            seconds = float(value)
            return seconds if seconds > 0 else 5.0
        except (ValueError, TypeError):
            return 5.0

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
        themed_button(self.title_bar, "SETTINGS", self.open_settings_selector, role="quiet",
                      padx=16, pady=6).pack(side=tk.RIGHT, fill=tk.Y)
        self.auto_pause_button = themed_button(
            self.title_bar, "PAUSE", self._toggle_automatic_pause, role="primary", padx=16, pady=6)
        self.auto_pause_button.pack(side=tk.RIGHT, fill=tk.Y)
        # Resets restart the whole application, so they belong with the other
        # session-wide controls rather than in the bottom panel.
        themed_button(self.title_bar, "RESET", self.reset_dashboard, role="danger",
                      padx=16, pady=6).pack(side=tk.RIGHT, fill=tk.Y)
        self.overlay_button = themed_button(
            self.title_bar, "", self._toggle_overlays, role="secondary", padx=16, pady=6)
        self.overlay_button.pack(side=tk.RIGHT, fill=tk.Y)
        self._refresh_overlay_button()
        self.title_bar.bind("<ButtonPress-1>", self._start_move)
        self.title_bar.bind("<B1-Motion>", self._do_move)

    def _refresh_overlay_button(self):
        self.overlay_button.configure(text="OVERLAY ON" if self.show_overlays else "OVERLAY OFF")
        set_button_role(self.overlay_button, "secondary" if self.show_overlays else "quiet")

    def _toggle_overlays(self):
        self.show_overlays = not self.show_overlays
        self._refresh_overlay_button()
        if not self.show_overlays:
            # Clear at once rather than waiting for the next tick to find None.
            self.canvas_1.delete("overlay")
            self._overlay_shown_1 = False
            view = getattr(self, 'result_view', None)
            if view is not None:
                view.clear_live_overlay()

    def _shutdown_session(self):
        """Stop every worker and hand the cameras back to the driver."""
        self._closing = True
        self.auto_controller.stop()
        self._dismiss_result_view()
        for future in getattr(self, '_camera_futures', []):
            def release_when_ready(done):
                if not done.cancelled() and done.exception() is None:
                    done.result().release()
            future.add_done_callback(release_when_ready)
        self.video_streaming = False
        if self.camera1: self.camera1.release()
        if self.camera2: self.camera2.release()

    def close_application(self):
        self._shutdown_session()
        self.root.destroy()

    def restart_application(self):
        """RESET: relaunch the process rather than only clearing the view.

        The session is torn down first, so the replacement process opens cameras
        nobody else holds. The startup countdown is skipped because the operator
        already chose settings for this session and a reset is not a new one.
        """
        self._shutdown_session()
        script = os.path.abspath(__file__)
        subprocess.Popen([sys.executable, script, NO_STARTUP_FLAG],
                         cwd=os.path.dirname(script))
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

        # SETTINGS lives only in the title bar, and RESET beside PAUSE there, so
        # the bottom panel is left with the read-only status displays.

    # ==============================================================
    # settings windows
    # ==============================================================
    def open_settings_selector(self):
        self.auto_controller.set_active(False)
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
            size_card, text="Set strip width, tolerance, and fabric tab cycling time.",
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
        if self.auto_controller.busy:
            self._show_error_popup('Wait for the current inspection to finish before changing calibration.')
            return
        self.auto_controller.set_active(False)
        # The result view lives inside main_frame, which this page hides. Leaving
        # while an inspection is held would strand a frozen result over an
        # invisible live feed, so the operator resumes first.
        if getattr(self, 'inspection_busy', False):
            return
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
            # Build the controls offscreen and settle their geometry before
            # exposing the new page, avoiding a partially painted transition.
            self.calibration_page.update_idletasks()
            self.calibration_page.pack(fill=tk.BOTH, expand=True)
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
        self.strip_scales.clear()
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
            content, text="Set adhesive width, tolerance, and the glue-line ends to exclude.",
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
        for column in range(2):
            settings.columnconfigure(column, weight=1, uniform="limits")
        section_label(settings, "Glue-line inspection limits").grid(
            row=0, column=0, columnspan=2, sticky="w", padx=20, pady=(17, 12))

        label_options = {"fg": C["text_soft"], "bg": C["card"], "font": (FONT, 10, "bold")}
        tk.Label(settings, text="Required width (mm)", **label_options).grid(
            row=1, column=0, sticky="w", padx=20)
        self.strip_width_entry = tk.Entry(
            settings, textvariable=self.strip_width_var, bg=C["surface_2"],
            fg=C["text"], insertbackground=C["text"], font=(FONT, 12),
            relief=tk.FLAT, highlightthickness=1,
            highlightbackground=C["border"], justify=tk.CENTER,
        )
        self.strip_width_entry.grid(
            row=2, column=0, padx=20, pady=(7, 20), sticky="ew", ipady=8)

        tk.Label(settings, text="Allowed tolerance (± mm)", **label_options).grid(
            row=1, column=1, sticky="w", padx=20)
        self.strip_width_tolerance_entry = tk.Entry(
            settings, textvariable=self.strip_width_tolerance_var,
            bg=C["surface_2"], fg=C["text"], insertbackground=C["text"],
            font=(FONT, 12), relief=tk.FLAT, highlightthickness=1,
            highlightbackground=C["border"], justify=tk.CENTER,
        )
        self.strip_width_tolerance_entry.grid(
            row=2, column=1, padx=20, pady=(7, 20), sticky="ew", ipady=8)

        tk.Label(settings, text="Exclude from EACH end (%)", **label_options).grid(
            row=3, column=0, sticky="w", padx=20)
        self.end_exclusion_entry = tk.Entry(
            settings, textvariable=self.end_exclusion_var, bg=C["surface_2"],
            fg=C["text"], insertbackground=C["text"], font=(FONT, 12),
            relief=tk.FLAT, highlightthickness=1, highlightbackground=C["border"], justify=tk.CENTER)
        self.end_exclusion_entry.grid(row=4, column=0, padx=20, pady=(7, 16), sticky="ew", ipady=8)
        tk.Label(settings, text="5% per end keeps the middle 90%.\n0 disables exclusion; must be below 50%.",
                 fg=C["muted"], bg=C["card"], font=(FONT, 10), justify=tk.LEFT).grid(
            row=4, column=1, sticky="w", padx=20)

        footer = tk.Frame(content, bg=C["bg"])
        footer.pack(fill=tk.X, pady=(4, 0))
        themed_button(footer, "SAVE PROFILE", self.save_settings,
                      role="primary", width=14).pack(side=tk.RIGHT)

        self._select_size(self.size_var.get())

    # ---------- Helper methods for size buttons ----------
    def _select_size(self, size):
        self.size_var.set(size)
        for s, btn in self._size_buttons.items():
            set_button_role(btn, "selected" if s == size else "secondary")

    def save_settings(self):
        try:
            trim = float(self.end_exclusion_var.get())
            if not math.isfinite(trim) or not 0 <= trim < 50:
                raise ValueError('End exclusion must be at least 0 and less than 50 percent.')
            from startup_settings import save_config
            candidate = copy.deepcopy(CONFIG)
            candidate['inspection']['end_exclusion_percent'] = trim
            save_config(candidate)
        except (ValueError, OSError) as error:
            messagebox.showerror('Settings not saved', str(error), parent=self.size_win)
            return
        CONFIG['inspection']['end_exclusion_percent'] = trim
        self.active_end_exclusion_percent = trim
        self.end_exclusion_var.set(f'{trim:g}')
        strip_value = self.strip_width_var.get()
        validated_width = self._validate_strip_width(strip_value)
        tolerance_value = self.strip_width_tolerance_var.get()
        validated_tolerance = self._validate_strip_width_tolerance(tolerance_value)
        display_seconds = self._validate_result_display_seconds(
            self.result_display_seconds_var.get())
        self.strip_width_var.set(f"{validated_width:g}")
        self.strip_width_tolerance_var.set(f"{validated_tolerance:g}")
        self.result_display_seconds_var.set(f"{display_seconds:g}")

        # Commit to active values
        self.active_size              = self.size_var.get()
        self.active_strip_width       = validated_width
        self.active_strip_width_tolerance = validated_tolerance
        self.active_result_display_seconds = display_seconds

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

    # ==============================================================
    # image panel (dual canvases + zoom toolbars)
    # ==============================================================
    def _create_image_panel(self):
        self.image_panel = tk.Frame(self.main_frame, bg=C["bg"])
        self.image_panel.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.images_container = tk.Frame(self.image_panel, bg=C["bg"])
        self.images_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 6))
        # The inspection result swaps the whole container out for a tabbed crop
        # viewer, so the options are stored rather than repeated at restore time.
        self.images_container_pack_opts = {
            'fill': tk.BOTH, 'expand': True, 'padx': 10, 'pady': (10, 6),
        }
        # Built lazily on the first inspection, so startup pays nothing for it.
        self.result_view = None

        # ---- left ----
        self.left_frame = themed_card(self.images_container, bg=C["surface"])
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        left_header = tk.Frame(self.left_frame, bg=C["surface"], height=38)
        left_header.pack(fill=tk.X, padx=14, pady=(5, 0))
        left_header.pack_propagate(False)
        status_dot(left_header).pack(side=tk.LEFT, pady=14, padx=(0, 7))
        tk.Label(left_header, text="CAMERA" if CAMERA_COUNT == 1 else "LEFT CAMERA", bg=C["surface"], fg=C["text_soft"],
                 font=(FONT, 9, "bold")).pack(side=tk.LEFT, pady=9)
        tk.Label(left_header, text="LIVE", bg=C["surface"], fg=C["accent"],
                 font=(FONT, 8, "bold")).pack(side=tk.RIGHT, pady=10)

        self.canvas_1 = tk.Canvas(self.left_frame, bg=C["camera"], highlightthickness=0)
        self.canvas_1.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 12))

        # ---- right ----
        self.right_frame = themed_card(self.images_container, bg=C["surface"])
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        right_header = tk.Frame(self.right_frame, bg=C["surface"], height=38)
        right_header.pack(fill=tk.X, padx=14, pady=(5, 0))
        right_header.pack_propagate(False)
        status_dot(right_header).pack(side=tk.LEFT, pady=14, padx=(0, 7))
        tk.Label(right_header, text="RIGHT CAMERA", bg=C["surface"], fg=C["text_soft"],
                 font=(FONT, 9, "bold")).pack(side=tk.LEFT, pady=9)
        tk.Label(right_header, text="LIVE", bg=C["surface"], fg=C["accent"],
                 font=(FONT, 8, "bold")).pack(side=tk.RIGHT, pady=10)

        self.canvas_2 = tk.Canvas(self.right_frame, bg=C["camera"], highlightthickness=0)
        self.canvas_2.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 12))
        if CAMERA_COUNT == 1:
            self.right_frame.pack_forget()
            self.left_frame.pack_configure(padx=0)

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
    # Maximize / restore functions
    # --------------------------------------------------------------
    def maximize_camera(self, camera_num):
        """Show the selected camera in the center of the container."""
        if CAMERA_COUNT == 1:
            return
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
        if CAMERA_COUNT == 2:
            self.right_frame.pack(**self.right_frame_pack_opts)
        self.maximized_camera = None
        self.root.update_idletasks()

    # ==============================================================
    # video streaming – using raw frames
    # ==============================================================

    def start_video_stream(self):
        if self.camera1 is not None and (CAMERA_COUNT == 1 or self.camera2 is not None):
            # Resume must not create a second independently scheduled feed loop.
            if getattr(self, '_video_job', None) is not None:
                self.root.after_cancel(self._video_job)
                self._video_job = None
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
        executor = ThreadPoolExecutor(max_workers=CAMERA_COUNT)
        futures = [executor.submit(CameraHandler, spec['index'], project_path(spec['calibration_file']))
                   for spec in ACTIVE_CAMERAS]
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
            self.camera1 = cameras[0]
            self.camera2 = cameras[1] if CAMERA_COUNT == 2 else None
            if getattr(self, 'calibration_page', None) is None:
                self.start_video_stream()
        self.root.after(50, finish)


    def update_video_feed(self):
        self._video_job = None
        if not self.video_streaming:
            return
        if not self.video_paused and self.camera1 is not None:
            raw, sequence = self.camera1.stream.read_snapshot()
            if raw is not None:
                self._submit_automatic_frame(raw, sequence)
                # Resolved once per tick so both views agree, and so the result
                # view never has to reach for the controller itself.
                geometry = self._overlay_geometry()
                if self.result_view is None:
                    self._display_video_frame(raw, 1, geometry)
                else:
                    self.result_view.update_live_preview(raw, geometry)
        self._video_job = self.root.after(CONFIG['preview']['interval_ms'], self.update_video_feed)

    def _display_video_frame(self, frame, canvas_num, geometry=None):
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
        preview_scale, offset_x, offset_y = preview_transform(
            (frame_width, frame_height), cw, ch,
            CONFIG['preview']['max_width'], CONFIG['preview']['max_height'])
        previous = getattr(self, f'_preview_snapshot_{canvas_num}', None)
        if not (previous is not None and previous[0] is frame
                and previous[1:] == (canvas, cw, ch) and canvas.find_withtag("img")):
            frame_resized = cv2.resize(
                frame,
                (max(1, int(frame_width * preview_scale)),
                 max(1, int(frame_height * preview_scale))),
                # Linear sampling touches a few source pixels per preview pixel;
                # area averaging scans the entire full-resolution source image.
                interpolation=cv2.INTER_LINEAR,
            )
            pil_img = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))

            photo = ImageTk.PhotoImage(pil_img)
            setattr(self, f"tk_image_{canvas_num}", photo)
            items = canvas.find_withtag("img")
            if items:
                canvas.itemconfigure(items[0], image=photo)
                canvas.coords(items[0], cw // 2, ch // 2)
            else:
                canvas.create_image(cw // 2, ch // 2, image=photo, anchor=tk.CENTER, tags="img")
            canvas.tag_lower("img")
            setattr(self, f'_preview_snapshot_{canvas_num}', (frame, canvas, cw, ch))
        # Redrawn even when the frame above was reused: the image can repeat
        # while the detections do not, and the early return must not skip them.
        if geometry:
            draw_tracking_overlay(canvas, geometry, preview_scale, offset_x, offset_y)
            setattr(self, f'_overlay_shown_{canvas_num}', True)
        elif getattr(self, f'_overlay_shown_{canvas_num}', False):
            canvas.delete("overlay")
            setattr(self, f'_overlay_shown_{canvas_num}', False)

    # ==============================================================
    # inspection result – crops on screen until the operator resumes
    # ==============================================================


    # ==============================================================
    # status / progress helpers
    # ==============================================================
    def set_pass_fail(self, status):
        colours = {"PASS": C["success"], "FAIL": C["danger"],
                   "READY": C["muted"], "LIVE": C["accent"],
                   "RESULT": C["accent"],
                   "WARNING": C["warning"], "SETUP": C["warning"]}
        self.status_result_label.config(text=status, fg=colours.get(status, C["muted"]))

    def update_progress(self, value, text):
        self.progress_bar['value'] = value
        self.progress_label.config(text=text)
        self.root.update_idletasks()

    # ==============================================================
    # detection trigger – with maximize & highlight
    # ==============================================================


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
        # A reset is a fresh start, not a cleared view: the whole application is
        # relaunched, so cameras, calibration and the inspection session all
        # begin again from the state the app boots into.
        app.restart_application()

    # ------------------------------------------------------------------
    app = IndustrialDashboard(root, reset_system)
    load_initial()
    root.mainloop()

if __name__ == "__main__":
    main()

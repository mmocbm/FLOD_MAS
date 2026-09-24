"""Themed two-stage ChArUco calibration for the industrial vision app."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app_config import CONFIG, camera_config, project_path
from camera_handler import CameraStream

import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
from PIL import Image, ImageTk
from ui_theme import (
    COLORS as C, FONT, MONO_FONT, button as themed_button, card as themed_card,
    configure_ttk, section_label, set_button_role, status_dot,
)

try:
    from .calibration_math import (
        coverage_cell, coverage_percent, estimate_planar_pose,
        next_coverage_cell, reprojection_metrics, scale_camera_matrix,
    )
    from .measurement_accuracy import (
        measure_board_accuracy, offset_plane_tvec, summarize_accuracy,
    )
except ImportError:
    from calibration_math import (
        coverage_cell, coverage_percent, estimate_planar_pose,
        next_coverage_cell, reprojection_metrics, scale_camera_matrix,
    )
    from measurement_accuracy import (
        measure_board_accuracy, offset_plane_tvec, summarize_accuracy,
    )


WINDOW_WIDTH = 1366
WINDOW_HEIGHT = 768
TITLE_BAR_HEIGHT = 44

BG = C["bg"]
PANEL = C["card"]
TITLE_BG = C["surface"]
CANVAS_BG = C["camera"]
TEXT = C["text"]
MUTED = C["muted"]
CYAN = C["accent"]
BLUE = C["blue"]
GREEN = C["accent_dark"]
AMBER = C["warning"]
RED = C["danger"]
PURPLE = C["purple"]

SQUARE_LENGTH_MM = CONFIG['board']['square_length_mm']
MARKER_LENGTH_MM = CONFIG['board']['marker_length_mm']
SQUARES_X = CONFIG['board']['squares_x']
SQUARES_Y = CONFIG['board']['squares_y']
DICT_TYPE = getattr(cv2.aruco, CONFIG['board']['dictionary'])
NUM_CAPTURES = CONFIG['calibration']['photo_count']
MIN_CORNERS = CONFIG['calibration']['minimum_corners']
MIN_EXTRINSIC_CORNERS = CONFIG['calibration']['surface_minimum_corners']
MAX_INTRINSIC_RMS_PX = CONFIG['calibration']['camera_warning_rms_px']
MAX_EXTRINSIC_RMS_PX = CONFIG['calibration']['surface_maximum_rms_px']
MAX_VERIFICATION_RMS_PX = CONFIG['calibration']['verification_max_rms_px']
COVERAGE_ROWS = CONFIG['calibration']['coverage_grid_rows']
COVERAGE_COLUMNS = CONFIG['calibration']['coverage_grid_columns']
CAMERA_IDS = [c['index'] for c in CONFIG['cameras']]
SURFACE_SETUP_ENABLED = CONFIG['measurement_surface']['enabled']
TWO_BOARD_DEFAULT = CONFIG['two_board']['enabled']

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FILES_DIR = PROJECT_ROOT / "Files"
TEMP_ROOT = PROJECT_ROOT / "temp_calibration_images"
REFERENCE_DIR = PROJECT_ROOT / "calibration_images"


class CalibrationApp:
    def __init__(self, root, on_close=None, host=None, camera_provider=None,
                 on_open_checks=None):
        self.root = root
        self.host = host if host is not None else root
        self.on_close = on_close
        self.on_open_checks = on_open_checks
        self.camera_provider = camera_provider
        self.closed = False
        self.preview_job = None
        self.starting_camera = False
        self.ui_style = configure_ttk(self.root)
        if host is None:
            self.root.overrideredirect(True)
            self.root.configure(bg=BG)
            self.root.geometry(self._centered_geometry(WINDOW_WIDTH, WINDOW_HEIGHT))
            self.root.minsize(1100, 650)

        self.cap = None
        self.camera_running = False
        self.camera_index = None
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.actual_size = (0, 0)
        self.captured_images = []
        self.object_views = []
        self.image_views = []
        self.capture_image_size = None
        self.capture_centers = []
        self.capture_point_sets = []
        self.coverage_counts = np.zeros((COVERAGE_ROWS, COVERAGE_COLUMNS), dtype=np.int32)
        self.processing_capture = False
        self.processing_verification = False
        self.check_panel_visible = False
        self.check_preview_frozen = False
        thickness_config = CONFIG['measurement_surface'].get('board_thickness', {})
        self.board_thickness_enabled_var = tk.BooleanVar(
            value=bool(thickness_config.get('enabled', False))
        )
        self.board_thickness_mm_var = tk.StringVar(
            value=str(thickness_config.get('thickness_mm', 2.0))
        )
        self.is_calibrating = False
        self.stage = "select"
        self.camera_matrix = None
        self.dist_coeffs = None
        self.calibration_image_size = None
        self.use_two_boards = TWO_BOARD_DEFAULT
        self.pending_dual_capture = None

        FILES_DIR.mkdir(exist_ok=True)
        TEMP_ROOT.mkdir(exist_ok=True)
        REFERENCE_DIR.mkdir(exist_ok=True)

        self._init_detector()
        self._create_ui()
        self._refresh_stage_ui()
        self.update_frame()

    def _centered_geometry(self, width, height):
        x = max(0, (self.root.winfo_screenwidth() - width) // 2)
        y = max(0, (self.root.winfo_screenheight() - height) // 2)
        return f"{width}x{height}+{x}+{y}"

    def _init_detector(self):
        if self.use_two_boards:
            dictionary = cv2.aruco.getPredefinedDictionary(
                getattr(cv2.aruco, CONFIG['two_board']['dictionary'])
            )
            marker_count = (SQUARES_X * SQUARES_Y) // 2
            second_start = CONFIG['two_board']['second_board_start_id']
            id_sets = (
                np.arange(marker_count, dtype=np.int32),
                np.arange(second_start, second_start + marker_count, dtype=np.int32),
            )
        else:
            dictionary = cv2.aruco.getPredefinedDictionary(DICT_TYPE)
            id_sets = (None,)

        self.boards = []
        self.charuco_detectors = []
        for marker_ids in id_sets:
            board = cv2.aruco.CharucoBoard(
                (SQUARES_X, SQUARES_Y),
                SQUARE_LENGTH_MM / 1000.0,
                MARKER_LENGTH_MM / 1000.0,
                dictionary,
                marker_ids if marker_ids is not None else None,
            )
            detector = cv2.aruco.CharucoDetector(
                board,
                cv2.aruco.CharucoParameters(),
                cv2.aruco.DetectorParameters(),
            )
            self.boards.append(board)
            self.charuco_detectors.append(detector)

        # The first board remains the reference board for the optional surface
        # step and for compatibility with the original one-board workflow.
        self.board = self.boards[0]
        self.charuco_detector = self.charuco_detectors[0]

    def _button(self, parent, text, command, color=BLUE, width=16, state=tk.NORMAL):
        role = {BLUE: "blue", GREEN: "primary", RED: "danger", PURPLE: "purple"}.get(
            color, "secondary")
        return themed_button(parent, text, command, role=role, width=width,
                             state=state, font_size=10, pady=9)

    def _create_ui(self):
        self._create_title_bar()
        body = tk.Frame(self.host, bg=BG)
        body.pack(fill=tk.BOTH, expand=True, padx=18, pady=(14, 16))

        heading = tk.Frame(body, bg=BG)
        heading.pack(fill=tk.X, pady=(0, 12))
        tk.Label(heading, text="Camera Setup", bg=BG, fg=TEXT,
                 font=(FONT, 26, "bold")).pack(side=tk.LEFT)
        tk.Label(
            heading,
            text="Simple guided setup for accurate camera measurement",
            bg=BG, fg=MUTED, font=(FONT, 10),
        ).pack(side=tk.LEFT, padx=16, pady=(9, 0))

        stage_row = tk.Frame(body, bg=BG)
        stage_row.pack(fill=tk.X, pady=(0, 10))
        self.stage_labels = []
        for number, title in (
            (1, "Select camera"), (2, f"Capture {NUM_CAPTURES} photos"),
            (3, "Calibrate lens"),
            (4, "Save surface" if SURFACE_SETUP_ENABLED else "Camera ready"),
        ):
            label = tk.Label(
                stage_row, text=f"{number}   {title}", bg=C["surface_2"], fg=MUTED,
                font=(FONT, 9, "bold"), padx=14, pady=8,
                highlightbackground=C["border"], highlightthickness=1,
            )
            label.pack(side=tk.LEFT, padx=(0, 8))
            self.stage_labels.append(label)

        workspace = tk.Frame(body, bg=BG)
        workspace.pack(fill=tk.BOTH, expand=True)
        workspace.columnconfigure(0, weight=1, minsize=0)
        workspace.columnconfigure(1, weight=0, minsize=410)
        workspace.rowconfigure(0, weight=1)

        preview_card = themed_card(workspace, bg=C["surface"])
        preview_card.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        preview_card.pack_propagate(False)
        preview_header = tk.Frame(preview_card, bg=C["surface"])
        preview_header.pack(fill=tk.X, padx=14, pady=10)
        status_dot(preview_header).pack(side=tk.LEFT, padx=(0, 7))
        self.preview_title = tk.Label(
            preview_header, text="CAMERA PREVIEW", bg=C["surface"],
            fg=C["text_soft"], font=(FONT, 9, "bold"),
        )
        self.preview_title.pack(side=tk.LEFT)
        self.resolution_label = tk.Label(
            preview_header, text="No camera connected", bg=C["surface"], fg=MUTED,
            font=(FONT, 9),
        )
        self.resolution_label.pack(side=tk.RIGHT)
        # The viewport owns the available space; image requests cannot resize it.
        viewport = tk.Frame(preview_card, bg=CANVAS_BG)
        viewport.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.video_label = tk.Label(
            viewport, text="Select a camera and start the live feed",
            bg=CANVAS_BG, fg=MUTED, font=(FONT, 14), bd=0,
            highlightthickness=0, padx=0, pady=0,
        )
        self.video_label.place(x=0, y=0, relwidth=1, relheight=1)

        controls_shell = themed_card(workspace, bg=PANEL, width=410)
        controls_shell.grid(row=0, column=1, sticky="nsew")
        controls_shell.grid_propagate(False)
        controls_shell.rowconfigure(0, weight=1)
        controls_shell.columnconfigure(0, weight=1)
        self.controls_canvas = tk.Canvas(
            controls_shell, bg=PANEL, bd=0, highlightthickness=0,
            takefocus=False,
        )
        self.controls_canvas.grid(row=0, column=0, sticky="nsew")
        self.controls_scrollbar = ttk.Scrollbar(
            controls_shell, orient=tk.VERTICAL, command=self.controls_canvas.yview,
        )
        self.controls_scrollbar.grid(row=0, column=1, sticky="ns")
        self.controls_canvas.configure(yscrollcommand=self.controls_scrollbar.set)
        controls = tk.Frame(self.controls_canvas, bg=PANEL)
        self.controls_content = controls
        self._controls_window = self.controls_canvas.create_window(
            (0, 0), window=controls, anchor="nw",
        )
        controls.bind("<Configure>", self._update_controls_scrollregion)
        self.controls_canvas.bind("<Configure>", self._resize_controls_content)
        self._controls_wheel_binding = self.root.bind(
            "<MouseWheel>", self._scroll_controls_with_mouse, add="+",
        )
        self.status_banner = tk.Label(
            controls, text="Select a camera below", bg=C["surface_2"], fg=CYAN,
            wraplength=360, justify=tk.LEFT, font=(FONT, 11, "bold"),
            padx=14, pady=12,
        )
        self.status_banner.pack(fill=tk.X, padx=12, pady=12)

        camera_box = tk.Frame(controls, bg=PANEL)
        camera_box.pack(fill=tk.X, padx=12)
        section_label(camera_box, "1  Select camera").pack(anchor="w")
        cam_buttons = tk.Frame(camera_box, bg=PANEL)
        cam_buttons.pack(fill=tk.X, pady=6)
        self.btn_cam0 = self._button(cam_buttons, f"CAMERA {CAMERA_IDS[0]}", lambda: self.select_camera(CAMERA_IDS[0]), GREEN, 13)
        self.btn_cam0.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))
        self.btn_cam1 = self._button(cam_buttons, f"CAMERA {CAMERA_IDS[1]}", lambda: self.select_camera(CAMERA_IDS[1]), BLUE, 13)
        self.btn_cam1.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(4, 0))
        stream_buttons = tk.Frame(camera_box, bg=PANEL)
        stream_buttons.pack(fill=tk.X, pady=(0, 12))
        self.start_btn = self._button(stream_buttons, "START CAMERA", self.start_camera, GREEN, 15, tk.DISABLED)
        self.start_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))
        self.stop_btn = self._button(stream_buttons, "STOP", self.stop_camera, RED, 9, tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=(4, 0))
        mode_buttons = tk.Frame(camera_box, bg=PANEL)
        mode_buttons.pack(fill=tk.X, pady=(0, 10))
        tk.Label(mode_buttons, text="BOARD METHOD", bg=PANEL, fg=MUTED,
                 font=(FONT, 8, "bold")).pack(side=tk.LEFT, padx=(0, 8))
        self.one_board_btn = self._button(
            mode_buttons, "ONE BOARD", lambda: self.select_board_mode(False),
            BLUE, 11,
        )
        self.one_board_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 3))
        self.two_board_btn = self._button(
            mode_buttons, "TWO BOARDS", lambda: self.select_board_mode(True),
            BLUE, 11,
        )
        self.two_board_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(3, 0))
        self._refresh_board_mode_ui()
        tk.Frame(controls, bg=C["border"], height=1).pack(fill=tk.X, padx=12)

        capture_box = tk.Frame(controls, bg=PANEL)
        capture_box.pack(fill=tk.X, padx=12, pady=10)
        section_label(capture_box, "2  Setup photos").pack(anchor="w")
        self.capture_info = tk.Label(
            capture_box, text=f"0 / {NUM_CAPTURES} accepted", bg=PANEL, fg=TEXT,
            font=(FONT, 15, "bold"),
        )
        self.capture_info.pack(anchor="w", pady=(4, 2))
        self.guidance_info = tk.Label(
            capture_box, text="Start with the board near the center",
            bg=PANEL, fg=AMBER, font=(FONT, 9, "bold"),
            justify=tk.LEFT, wraplength=370,
        )
        self.guidance_info.pack(anchor="w", pady=(0, 4))
        self.ui_style.configure(
            "Calibration.Horizontal.TProgressbar", troughcolor=TITLE_BG,
            background=PURPLE, bordercolor=PANEL, lightcolor=PURPLE,
            darkcolor=PURPLE, thickness=8,
        )
        self.progress = ttk.Progressbar(
            capture_box, maximum=NUM_CAPTURES,
            style="Calibration.Horizontal.TProgressbar",
        )
        self.progress.pack(fill=tk.X, pady=(2, 7))
        self.capture_btn = self._button(
            capture_box, "CAPTURE PHOTO", self.manual_capture,
            BLUE, 28, tk.DISABLED,
        )
        self.capture_btn.pack(fill=tk.X)
        tk.Label(
            capture_box,
            text="Follow the highlighted area. Also tilt the board differently after each photo.",
            bg=PANEL, fg=MUTED, wraplength=370, justify=tk.LEFT,
            font=(FONT, 9),
        ).pack(anchor="w", pady=(5, 0))

        plane_box = tk.Frame(controls, bg=C["surface_2"], highlightbackground=C["border_strong"], highlightthickness=1)
        plane_box.pack(fill=tk.X, padx=12, pady=(0, 10))
        plane_title = ("4  MEASUREMENT SURFACE" if SURFACE_SETUP_ENABLED
                       else "4  MEASUREMENT MARKER")
        tk.Label(plane_box, text=plane_title, bg=C["surface_2"], fg=PURPLE,
                 font=(FONT, 9, "bold")).pack(anchor="w", padx=10, pady=(8, 2))
        tk.Label(
            plane_box,
            text=("Place board flat on measurement surface" if SURFACE_SETUP_ENABLED
                  else "No surface photo is required"),
            bg=C["surface_2"], fg=TEXT, font=(FONT, 11, "bold"),
        ).pack(anchor="w", padx=10)
        tk.Label(
            plane_box,
            text=("Keep the camera fixed. The board must be fully flat at the same height as the product."
                  if SURFACE_SETUP_ENABLED else
                  "Marker 0 will set the measurement surface each time you inspect."),
            bg=C["surface_2"], fg=MUTED, wraplength=360, justify=tk.LEFT,
            font=(FONT, 9),
        ).pack(anchor="w", padx=10, pady=(3, 7))
        thickness_row = tk.Frame(plane_box, bg=C["surface_2"])
        thickness_row.pack(fill=tk.X, padx=10, pady=(0, 7))
        tk.Checkbutton(
            thickness_row, text="Correct board thickness to bed plane",
            variable=self.board_thickness_enabled_var,
            bg=C["surface_2"], fg=C["text_soft"], activebackground=C["surface_2"],
            activeforeground=TEXT, selectcolor=TITLE_BG, font=(FONT, 9),
            bd=0, highlightthickness=0,
        ).pack(side=tk.LEFT)
        tk.Entry(
            thickness_row, textvariable=self.board_thickness_mm_var, width=6,
            bg=TITLE_BG, fg=TEXT, insertbackground=TEXT, relief=tk.FLAT,
            justify=tk.RIGHT, font=(FONT, 9),
        ).pack(side=tk.RIGHT, padx=(4, 0))
        tk.Label(thickness_row, text="mm", bg=C["surface_2"], fg=MUTED,
                 font=(FONT, 9)).pack(side=tk.RIGHT)
        self.extrinsic_btn = self._button(
            plane_box,
            "SAVE MEASUREMENT SURFACE" if SURFACE_SETUP_ENABLED else "NOT REQUIRED",
            self.capture_extrinsic_reference,
            PURPLE, 29, tk.DISABLED,
        )
        self.extrinsic_btn.pack(fill=tk.X, padx=10, pady=(0, 10))

        log_box = tk.Frame(controls, bg=PANEL)
        log_box.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))
        section_label(log_box, "Activity").pack(anchor="w")
        self.status_text = scrolledtext.ScrolledText(
            log_box, height=7, bg=TITLE_BG, fg=C["text_soft"],
            insertbackground="white", relief=tk.FLAT,
            font=(MONO_FONT, 9), wrap=tk.WORD,
        )
        self.status_text.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
        self.status_text.configure(state=tk.DISABLED)
        self.log("Follow the four steps shown above.")
        self.log("Keep the board clear, flat and well lit.")

    def _update_controls_scrollregion(self, _event=None):
        self.controls_canvas.configure(scrollregion=self.controls_canvas.bbox("all"))

    def _resize_controls_content(self, event):
        self.controls_canvas.itemconfigure(self._controls_window, width=event.width)

    def _scroll_controls_with_mouse(self, event):
        """Scroll the setup controls only while the pointer is over that panel."""
        widget = self.root.winfo_containing(event.x_root, event.y_root)
        while widget is not None:
            if widget in (self.controls_canvas, self.controls_content):
                direction = -1 if event.delta > 0 else 1
                self.controls_canvas.yview_scroll(direction * 3, "units")
                return "break"
            widget = getattr(widget, 'master', None)
        return None

    def _refresh_board_mode_ui(self):
        set_button_role(self.one_board_btn, "selected" if not self.use_two_boards else "secondary")
        set_button_role(self.two_board_btn, "selected" if self.use_two_boards else "secondary")
        if hasattr(self, 'stage_labels'):
            unit = "photo sets" if self.use_two_boards else "photos"
            self.stage_labels[1].configure(text=f"2   Capture {NUM_CAPTURES} {unit}")
        if hasattr(self, 'capture_info') and not self.captured_images:
            unit = " photo sets" if self.use_two_boards else ""
            self.capture_info.configure(text=f"0 / {NUM_CAPTURES}{unit} accepted")

    def select_board_mode(self, enabled):
        """Choose the original one-board flow or the optional two-board flow."""
        if (self.processing_capture or self.is_calibrating or self.captured_images or
                self.pending_dual_capture is not None):
            self._set_status("Restart this camera setup before changing the board method", AMBER)
            return
        self.use_two_boards = bool(enabled)
        self._close_calibration_checks()
        self._init_detector()
        self._refresh_board_mode_ui()
        if self.use_two_boards:
            message = "Two-board mode selected — use the separately numbered Board 1 and Board 2"
        else:
            message = "One-board mode selected"
        self.guidance_info.configure(text=message, fg=AMBER)
        self._set_status(message)
        self.log(message)

    def _create_title_bar(self):
        title_bar = tk.Frame(self.host, bg=TITLE_BG, height=TITLE_BAR_HEIGHT,
                             highlightbackground=C["border"], highlightthickness=1)
        title_bar.pack(fill=tk.X)
        title_bar.pack_propagate(False)
        if self.on_close is not None:
            themed_button(title_bar, "←  DASHBOARD", self.on_closing, role="quiet",
                          padx=16, pady=6).pack(side=tk.LEFT, fill=tk.Y)
        tk.Label(
            title_bar, text="CAMERA SETUP", bg=TITLE_BG, fg=MUTED,
            font=(FONT, 10, "bold"),
        ).pack(side=tk.LEFT, padx=12)
        if self.on_open_checks is not None:
            themed_button(
                title_bar, "CALIBRATION CHECK", self.on_open_checks,
                role="quiet", padx=16, pady=6,
            ).pack(side=tk.LEFT, fill=tk.Y)
        themed_button(title_bar, "✕", self.on_closing, role="quiet",
                      padx=16, pady=6, font_size=12).pack(side=tk.RIGHT, fill=tk.Y)
        themed_button(title_bar, "—", self.minimize_window, role="quiet",
                      padx=16, pady=6, font_size=12).pack(side=tk.RIGHT, fill=tk.Y)
        title_bar.bind("<ButtonPress-1>", self._start_move)
        title_bar.bind("<B1-Motion>", self._do_move)

    def _start_move(self, event):
        self.drag_x, self.drag_y = event.x, event.y

    def _do_move(self, event):
        self.root.geometry(
            f"+{self.root.winfo_x() + event.x - self.drag_x}"
            f"+{self.root.winfo_y() + event.y - self.drag_y}"
        )

    def minimize_window(self):
        self.root.overrideredirect(False)
        self.root.iconify()
        self.root.bind("<FocusIn>", self._restore_frameless)

    def _restore_frameless(self, _event):
        if self.root.state() == "normal":
            self.root.overrideredirect(True)
            self.root.unbind("<FocusIn>")

    def log(self, message):
        self.status_text.configure(state=tk.NORMAL)
        self.status_text.insert(tk.END, message + "\n")
        self.status_text.see(tk.END)
        self.status_text.configure(state=tk.DISABLED)

    def _set_status(self, message, color=CYAN):
        self.status_banner.configure(text=message, fg=color)

    def _refresh_stage_ui(self):
        stage_index = {
            "select": 0, "capture": 1, "calibrating": 2,
            "extrinsic_ready": 3, "extrinsic_capturing": 3, "complete": 3,
        }.get(self.stage, 0)
        for index, label in enumerate(self.stage_labels):
            if index < stage_index:
                label.configure(bg=C["success_dark"], fg=C["text"])
            elif index == stage_index:
                label.configure(bg="#7558D6", fg="white")
            else:
                label.configure(bg=C["surface_2"], fg=MUTED)
        self.capture_btn.configure(
            state=(tk.NORMAL if self.camera_running and self.stage == "capture"
                   and not self.processing_capture else tk.DISABLED)
        )
        self.extrinsic_btn.configure(
            state=(tk.NORMAL if SURFACE_SETUP_ENABLED and self.camera_running
                   and self.stage == "extrinsic_ready" else tk.DISABLED)
        )
        self._refresh_check_panel_buttons()

    def _calibration_path(self, camera_index=None):
        index = self.camera_index if camera_index is None else camera_index
        return Path(project_path(camera_config(index)['calibration_file']))

    def _extrinsics_path(self, camera_index=None):
        index = self.camera_index if camera_index is None else camera_index
        return Path(project_path(camera_config(index)['extrinsics_file']))

    def _load_saved_intrinsics(self):
        """Load and validate copied/saved lens calibration for surface setup."""
        calibration_path = self._calibration_path()
        saved = json.loads(calibration_path.read_text(encoding="utf-8"))
        camera_matrix = np.asarray(saved["camera_matrix"], dtype=np.float64)
        dist_coeffs = np.asarray(saved["dist_coeffs"], dtype=np.float64)
        image_size = tuple(int(value) for value in saved["image_size"])
        if camera_matrix.shape != (3, 3) or not np.isfinite(camera_matrix).all():
            raise ValueError("camera_matrix must be a finite 3 x 3 matrix")
        if dist_coeffs.size < 4 or not np.isfinite(dist_coeffs).all():
            raise ValueError("dist_coeffs must contain finite lens coefficients")
        if len(image_size) != 2 or min(image_size) <= 0:
            raise ValueError("image_size must contain a positive width and height")
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.calibration_image_size = image_size
        capture_method = saved.get("capture_method")
        if capture_method is None and "two_board" in saved:
            capture_method = "two_boards"
        if capture_method in ("one_board", "two_boards"):
            self.use_two_boards = capture_method == "two_boards"
            self._init_detector()
            self._refresh_board_mode_ui()
        return saved

    def _preferred_resolution(self):
        spec = camera_config(self.camera_index)
        return spec['width'], spec['height']

    def select_camera(self, index):
        if (self.starting_camera or self.processing_capture or self.processing_verification or
                self.stage in ("calibrating", "extrinsic_capturing")):
            return
        if self.camera_running:
            self.stop_camera()
        self._close_calibration_checks()
        self.camera_index = index
        self.stage = "select"
        self.captured_images = []
        self.object_views = []
        self.image_views = []
        self.capture_image_size = None
        self.capture_centers = []
        self.capture_point_sets = []
        self.coverage_counts.fill(0)
        self.camera_matrix = None
        self.dist_coeffs = None
        self.calibration_image_size = None
        self.pending_dual_capture = None
        self.processing_verification = False
        unit = " photo sets" if self.use_two_boards else ""
        self.capture_info.configure(text=f"0 / {NUM_CAPTURES}{unit} accepted")
        self.guidance_info.configure(
            text=("Show both numbered boards, or show Board 1 first"
                  if self.use_two_boards else "Start with the board near the center"),
            fg=AMBER,
        )
        self.progress["value"] = 0
        set_button_role(self.btn_cam0, "selected" if index == CAMERA_IDS[0] else "secondary")
        set_button_role(self.btn_cam1, "selected" if index == CAMERA_IDS[1] else "secondary")
        self.start_btn.configure(state=tk.NORMAL)
        self._set_status(f"Camera {index} selected — start the live feed")
        self.log(f"Selected Camera {index}")
        self._refresh_stage_ui()

    def start_camera(self):
        if (self.starting_camera or self.processing_capture or self.processing_verification or
                self.stage in ("calibrating", "extrinsic_capturing")):
            return
        if self.camera_index is None:
            messagebox.showwarning("Camera required", "Select a camera first.")
            return
        self.start_btn.configure(state=tk.DISABLED, text="STARTING...")
        self.starting_camera = True
        self._set_status(f"Opening Camera {self.camera_index}…", AMBER)
        threading.Thread(target=self._start_camera_worker, daemon=True).start()

    def _start_camera_worker(self):
        try:
            width, height = self._preferred_resolution()
            cap = (self.camera_provider(self.camera_index) if self.camera_provider
                   else CameraStream(self.camera_index))
            if not cap.isOpened():
                if not self.camera_provider: cap.release()
                raise RuntimeError("Camera could not be opened")
            self.cap = cap
            self.actual_size = cap.size
            self.camera_running = True
            self.root.after(0, self._camera_started, width, height)
        except Exception as error:
            self.root.after(0, self._camera_failed, str(error))

    def _camera_started(self, requested_width, requested_height):
        self.starting_camera = False
        self.stage = "capture"
        self.start_btn.configure(text="START CAMERA", state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)
        self.resolution_label.configure(
            text=f"Camera {self.camera_index}  •  {self.actual_size[0]} × {self.actual_size[1]}"
        )
        method = "two numbered boards" if self.use_two_boards else "the board"
        if self._calibration_path().is_file():
            try:
                self._load_saved_intrinsics()
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                self._set_status(
                    "Saved calibration is invalid — create a new camera calibration", RED,
                )
                self.log(f"Saved calibration could not be loaded: {error}")
            else:
                if SURFACE_SETUP_ENABLED:
                    # A copied intrinsic calibration is sufficient for solving a
                    # new camera-to-bed pose. Do not force another 20-photo lens
                    # calibration just because this is a fresh repository.
                    self.stage = "extrinsic_ready"
                    self.capture_info.configure(text="Saved lens calibration loaded")
                    self.guidance_info.configure(
                        text="Place the ChArUco board flat on the measurement surface",
                        fg=GREEN,
                    )
                    self._set_status(
                        "Saved lens calibration loaded — save the measurement surface",
                        GREEN,
                    )
                    self.log(
                        "A saved lens calibration was loaded. The measurement surface "
                        "can now be saved without recalibrating the lens."
                    )
                else:
                    self._set_status(
                        "Saved calibration found — open Calibration Checks to verify it"
                    )
                    self.log("A saved calibration was found. Test it before recalibrating.")
        else:
            self._set_status(
                f"Take {NUM_CAPTURES} photo sets while moving {method} around the camera view"
            )
        self.log(
            f"Camera {self.camera_index} started at {self.actual_size[0]} × {self.actual_size[1]} "
            f"(requested {requested_width} × {requested_height})"
        )
        session_dir = TEMP_ROOT / f"camera_{self.camera_index}"
        session_dir.mkdir(parents=True, exist_ok=True)
        for pattern in ("calib_*.jpg", "calib_*.png"):
            for old_image in session_dir.glob(pattern):
                old_image.unlink()
        self._refresh_stage_ui()

    def _camera_failed(self, error):
        self.starting_camera = False
        self.start_btn.configure(text="START CAMERA", state=tk.NORMAL)
        self._set_status("Camera could not be opened", RED)
        self.log(f"Camera error: {error}")
        messagebox.showerror("Camera Error", error)

    def stop_camera(self):
        if (self.starting_camera or self.processing_capture or self.processing_verification or
                self.stage in ("calibrating", "extrinsic_capturing")):
            return
        self.camera_running = False
        time.sleep(0.05)
        if self.cap is not None:
            if not self.camera_provider: self.cap.release()
            self.cap = None
        self.start_btn.configure(text="START CAMERA", state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        self.video_label.configure(image="", text="Camera stopped")
        self.resolution_label.configure(text="No camera connected")
        if self.stage != "complete":
            self.stage = "select"
        self._refresh_stage_ui()

    def update_frame(self):
        if self.closed:
            return
        cycle_started = time.perf_counter()
        # Keep the last displayed capture visible while its board points are
        # processed. Board detection never runs in the live preview.
        if (not self.processing_capture and self.camera_running and
                self.cap is not None and self.cap.isOpened()):
            ok, frame = self.cap.read()
            if ok:
                with self.frame_lock:
                    self.current_frame = frame
                scale = min(1.0, CONFIG['preview']['max_width'] / frame.shape[1],
                            CONFIG['preview']['max_height'] / frame.shape[0])
                display = cv2.resize(frame, (max(1, int(frame.shape[1] * scale)),
                                             max(1, int(frame.shape[0] * scale))))
                self._draw_coverage_guide(display)
                self._draw_capture_history(display)
                if not (self.check_panel_visible and self.check_preview_frozen):
                    self._show_frame(display)
        elapsed_ms = (time.perf_counter() - cycle_started) * 1000.0
        delay_ms = max(1, round(CONFIG['preview']['interval_ms'] - elapsed_ms))
        self.preview_job = self.root.after(delay_ms, self.update_frame)

    def _recommended_cell(self):
        last_cell = None
        if self.capture_centers:
            x, y = self.capture_centers[-1]
            last_cell = (
                min(COVERAGE_ROWS - 1, int(y * COVERAGE_ROWS)),
                min(COVERAGE_COLUMNS - 1, int(x * COVERAGE_COLUMNS)),
            )
        return next_coverage_cell(self.coverage_counts, last_cell)

    @staticmethod
    def _cell_name(cell):
        row, column = cell
        vertical = ("TOP", "MIDDLE", "BOTTOM")
        horizontal = ("LEFT", "CENTER", "RIGHT")
        row_name = vertical[row] if len(vertical) == COVERAGE_ROWS else f"ROW {row + 1}"
        column_name = horizontal[column] if len(horizontal) == COVERAGE_COLUMNS else f"COLUMN {column + 1}"
        return f"{row_name} {column_name}"

    def _draw_coverage_guide(self, frame):
        """Overlay captured areas and the recommended location for the next photo."""
        height, width = frame.shape[:2]
        drawing_scale = max(1.0, width / 880.0)
        line_width = max(1, round(drawing_scale))
        strong_line_width = max(2, round(3 * drawing_scale))
        overlay = frame.copy()
        target_row, target_column = self._recommended_cell()
        for row in range(COVERAGE_ROWS):
            for column in range(COVERAGE_COLUMNS):
                x1 = int(column * width / COVERAGE_COLUMNS)
                x2 = int((column + 1) * width / COVERAGE_COLUMNS)
                y1 = int(row * height / COVERAGE_ROWS)
                y2 = int((row + 1) * height / COVERAGE_ROWS)
                if self.coverage_counts[row, column] > 0:
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), self._hex_to_bgr(GREEN), -1)
                if (row, column) == (target_row, target_column):
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), self._hex_to_bgr(AMBER), -1)
        cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)

        for row in range(1, COVERAGE_ROWS):
            y = int(row * height / COVERAGE_ROWS)
            cv2.line(frame, (0, y), (width, y), (110, 120, 135), line_width, cv2.LINE_AA)
        for column in range(1, COVERAGE_COLUMNS):
            x = int(column * width / COVERAGE_COLUMNS)
            cv2.line(frame, (x, 0), (x, height), (110, 120, 135), line_width, cv2.LINE_AA)

        x1 = int(target_column * width / COVERAGE_COLUMNS)
        x2 = int((target_column + 1) * width / COVERAGE_COLUMNS)
        y1 = int(target_row * height / COVERAGE_ROWS)
        y2 = int((target_row + 1) * height / COVERAGE_ROWS)
        cv2.rectangle(frame, (x1 + 2, y1 + 2), (x2 - 2, y2 - 2),
                      self._hex_to_bgr(AMBER), strong_line_width, cv2.LINE_AA)
        cv2.putText(
            frame, "NEXT AREA", (x1 + round(12 * drawing_scale),
                                 min(y2 - 12, y1 + round(30 * drawing_scale))),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65 * drawing_scale,
            self._hex_to_bgr(AMBER), max(2, round(2 * drawing_scale)), cv2.LINE_AA,
        )
        for x_normalized, y_normalized in self.capture_centers:
            point = (int(x_normalized * width), int(y_normalized * height))
            cv2.circle(frame, point, round(8 * drawing_scale),
                       self._hex_to_bgr(GREEN), -1, cv2.LINE_AA)
            cv2.circle(frame, point, round(11 * drawing_scale),
                       self._hex_to_bgr(TEXT), max(2, round(2 * drawing_scale)), cv2.LINE_AA)

    def _annotate_detected_board(self, frame, corners, ids):
        """Draw detected board points on the captured (not live) image."""
        annotated = frame.copy()
        if corners is None:
            return annotated
        points = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
        if not len(points):
            return annotated
        drawing_scale = max(1.0, annotated.shape[1] / 880.0)
        # Draw directly instead of drawDetectedCornersCharuco. OpenCV 5 can
        # reject otherwise valid detector output when the NumPy corner/ID
        # layouts differ, and a display-only error must never stop capture.
        point_radius = max(3, round(4 * drawing_scale))
        point_thickness = max(1, round(2 * drawing_scale))
        flat_ids = (np.asarray(ids, dtype=np.int32).reshape(-1)
                    if ids is not None else np.empty(0, dtype=np.int32))
        for index, point in enumerate(points):
            location = tuple(np.round(point).astype(int))
            cv2.circle(
                annotated, location, point_radius, self._hex_to_bgr(CYAN),
                point_thickness, cv2.LINE_AA,
            )
            if index < len(flat_ids):
                cv2.putText(
                    annotated, str(int(flat_ids[index])),
                    (location[0] + point_radius + 2, location[1] - point_radius - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28 * drawing_scale,
                    self._hex_to_bgr(TEXT), max(1, round(drawing_scale)), cv2.LINE_AA,
                )
        if len(points) >= 3:
            hull = cv2.convexHull(points.astype(np.int32))
            cv2.polylines(
                annotated, [hull], True, self._hex_to_bgr(CYAN),
                max(2, round(2 * drawing_scale)), cv2.LINE_AA,
            )
        center = tuple(np.round(points.mean(axis=0)).astype(int))
        cv2.drawMarker(
            annotated, center, self._hex_to_bgr(TEXT), cv2.MARKER_CROSS,
            round(22 * drawing_scale), max(2, round(3 * drawing_scale)), cv2.LINE_AA,
        )
        return annotated

    def _draw_capture_history(self, frame):
        """Show points found in earlier captures without running live detection."""
        if not self.capture_point_sets:
            return
        height, width = frame.shape[:2]
        drawing_scale = max(1.0, width / 880.0)
        for capture_index, normalized_points in enumerate(self.capture_point_sets):
            is_latest = capture_index == len(self.capture_point_sets) - 1
            color = self._hex_to_bgr(CYAN if is_latest else GREEN)
            radius = max(2, round((4 if is_latest else 3) * drawing_scale))
            thickness = -1 if is_latest else max(1, round(drawing_scale))
            for x_normalized, y_normalized in normalized_points:
                point = (int(x_normalized * width), int(y_normalized * height))
                cv2.circle(frame, point, radius, color, thickness, cv2.LINE_AA)

    @staticmethod
    def _hex_to_bgr(value):
        value = value.lstrip("#")
        red, green, blue = (int(value[i:i + 2], 16) for i in (0, 2, 4))
        return blue, green, red

    def _show_frame(self, frame):
        width = max(2, self.video_label.winfo_width())
        height = max(2, self.video_label.winfo_height())
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (width, height), CANVAS_BG)
        canvas.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
        photo = ImageTk.PhotoImage(canvas)
        self.video_label.image = photo
        self.video_label.configure(image=photo, text="")

    def manual_capture(self):
        if (self.current_frame is None or self.stage != "capture" or
                self.processing_capture or
                getattr(self, 'processing_verification', False)):
            return
        with self.frame_lock:
            frame = self.current_frame.copy()
        self.processing_capture = True
        frozen = frame.copy()
        self._draw_coverage_guide(frozen)
        self._show_frame(frozen)
        self._set_status("Checking the captured photo…", AMBER)
        self._refresh_stage_ui()
        self.root.update_idletasks()
        session_dir = TEMP_ROOT / f"camera_{self.camera_index}"
        session_dir.mkdir(parents=True, exist_ok=True)
        suffix = "_b" if getattr(self, 'pending_dual_capture', None) is not None else "_a"
        file_path = session_dir / f"calib_{len(self.captured_images):03d}{suffix}.png"
        threading.Thread(
            target=self._detect_capture_worker, args=(frame, file_path), daemon=True,
        ).start()

    def _close_calibration_checks(self):
        self.check_panel_visible = False
        self.check_preview_frozen = False
        self._set_check_preview_title("CAMERA PREVIEW", C["text_soft"])

    def _refresh_check_panel_buttons(self):
        if not hasattr(self, 'check_panel'):
            return
        busy = bool(self.processing_capture or self.processing_verification)
        if self.camera_index is None:
            self.check_lens_btn.configure(state=tk.DISABLED)
            self.check_measurement_btn.configure(state=tk.DISABLED)
            self.check_live_btn.configure(state=tk.DISABLED)
            self.check_plane_info.configure(
                text="Select and start a camera to use calibration checks.", fg=MUTED,
            )
            return
        intrinsics_loaded = (
            self.camera_matrix is not None and self.dist_coeffs is not None
            and self.calibration_image_size is not None
        )
        lens_ready = (self.camera_running and self._calibration_path().is_file()
                      and intrinsics_loaded and not busy)
        metric_ready = lens_ready and self._extrinsics_path().is_file()
        self.check_lens_btn.configure(state=tk.NORMAL if lens_ready else tk.DISABLED)
        self.check_measurement_btn.configure(state=tk.NORMAL if metric_ready else tk.DISABLED)
        self.check_live_btn.configure(state=tk.DISABLED if busy else tk.NORMAL)
        if not self._extrinsics_path().is_file():
            self.check_plane_info.configure(
                text="Save the measurement surface before checking millimetre accuracy.",
                fg=AMBER,
            )
        else:
            self.check_plane_info.configure(
                text="Saved measurement plane loaded and used directly.",
                fg=C["text_soft"],
            )

    def _set_check_report(self, text):
        if not hasattr(self, 'check_report') or not self.check_report.winfo_exists():
            return
        self.check_report.configure(state=tk.NORMAL)
        self.check_report.delete("1.0", tk.END)
        self.check_report.insert(tk.END, text)
        self.check_report.configure(state=tk.DISABLED)

    def _show_check_frame(self, frame):
        # Check results use the existing Camera Setup preview; there is no
        # second preview widget or camera stream.
        self._show_frame(frame)

    def _set_check_preview_title(self, text, color):
        title = getattr(self, 'preview_title', None)
        if title is not None and title.winfo_exists():
            title.configure(text=text, fg=color)

    def resume_check_preview(self):
        """Return from a frozen result image to the current live camera view."""
        if self.processing_verification:
            return
        self.check_preview_frozen = False
        self._set_check_preview_title("●  LIVE CAMERA PREVIEW", GREEN)
        self._set_check_report(
            "Live preview resumed. Position the board flat and fully visible, "
            "then run the required check."
        )

    def verify_saved_calibration(self):
        """Check a fresh ChArUco capture against the saved intrinsic calibration."""
        if (self.current_frame is None or
                self.processing_capture or self.processing_verification):
            return
        calibration_path = self._calibration_path()
        if not calibration_path.is_file():
            self._set_status("No saved calibration exists for this camera", RED)
            return
        with self.frame_lock:
            frame = self.current_frame.copy()
        self.processing_verification = True
        self.check_preview_frozen = True
        self._set_check_preview_title("●  CAPTURED LENS-CHECK FRAME", AMBER)
        self._set_status("Testing the saved calibration…", AMBER)
        self._set_check_report("Checking lens calibration from a fresh frame…")
        self._refresh_stage_ui()
        threading.Thread(
            target=self._verify_saved_calibration_worker, args=(frame, calibration_path), daemon=True,
        ).start()

    def _verify_saved_calibration_worker(self, frame, calibration_path):
        try:
            saved = json.loads(calibration_path.read_text(encoding="utf-8"))
            camera_matrix = np.asarray(saved["camera_matrix"], dtype=np.float64)
            dist_coeffs = np.asarray(saved["dist_coeffs"], dtype=np.float64)
            calibration_size = tuple(saved["image_size"])
            if len(calibration_size) != 2:
                raise ValueError("The saved calibration has no valid image size")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_size = gray.shape[::-1]
            scaled_matrix = scale_camera_matrix(camera_matrix, calibration_size, frame_size)
            detections = []
            board_metrics = []
            total_points = 0
            detector_count = 2 if self.use_two_boards else 1
            required_corners = getattr(self, 'check_minimum_corners', MIN_CORNERS)
            for board_index in range(detector_count):
                corners, ids, _, _ = self.charuco_detectors[board_index].detectBoard(gray)
                corner_count = len(ids) if ids is not None else 0
                if corner_count < required_corners:
                    raise ValueError(
                        f"Board {board_index + 1}: only {corner_count}/{required_corners} "
                        "points were found. Show the board clearly and try again."
                    )
                rvec, tvec, metrics, object_points, image_points = estimate_planar_pose(
                    self.boards[board_index], corners, ids, scaled_matrix, dist_coeffs,
                )
                total_points += len(object_points)
                board_metrics.append((metrics, len(object_points)))
                detections.append({
                    "board_index": board_index, "corners": corners, "ids": ids,
                    "image_points": image_points, "rvec": rvec, "tvec": tvec,
                })
            annotated = self._annotate_board_views(frame, detections)
            for detection in detections:
                cv2.drawFrameAxes(
                    annotated, scaled_matrix, dist_coeffs,
                    detection["rvec"], detection["tvec"], 0.05, 4,
                )
            metrics = {
                "rms": float(np.sqrt(sum(
                    item[0]["rms"] ** 2 * item[1] for item in board_metrics
                ) / total_points)),
                "mean": float(sum(
                    item[0]["mean"] * item[1] for item in board_metrics
                ) / total_points),
                "max": float(max(item[0]["max"] for item in board_metrics)),
            }
            is_ok = metrics["rms"] <= MAX_VERIFICATION_RMS_PX
            self.root.after(
                0, self._verification_complete, annotated, metrics,
                total_points, is_ok, detector_count,
            )
        except Exception as error:
            self.root.after(0, self._verification_failed, str(error))

    def _verification_complete(self, annotated, metrics, corner_count, is_ok, board_count=1):
        self.processing_verification = False
        self._set_check_preview_title("●  LENS-CHECK RESULT", PURPLE)
        self._show_check_frame(annotated)
        result = "CALIBRATION OK" if is_ok else "RECALIBRATION RECOMMENDED"
        color = GREEN if is_ok else RED
        details = (f"{result} — RMS {metrics['rms']:.2f} px "
                   f"(mean {metrics['mean']:.2f} px, max {metrics['max']:.2f} px; "
                   f"{corner_count} points)")
        self._set_status(details, color)
        self.log(details)
        self._set_check_report(
            "LENS CALIBRATION CHECK\n"
            f"Boards checked:          {board_count}\n"
            f"Board definition:        {getattr(self, 'check_board_description', 'main configured board')}\n"
            f"Detected board points:   {corner_count}\n"
            f"Reprojection RMS:        {metrics['rms']:.3f} px\n"
            f"Mean reprojection error: {metrics['mean']:.3f} px\n"
            f"Maximum error:           {metrics['max']:.3f} px\n\n"
            "This checks the saved lens model. Use CHECK MEASUREMENT ACCURACY "
            "to validate millimetre distances on the saved plane."
        )
        self._refresh_stage_ui()

    def _verification_failed(self, error):
        self.processing_verification = False
        self.check_preview_frozen = False
        self._set_check_preview_title("●  LIVE CAMERA PREVIEW", GREEN)
        self._set_status("Calibration could not be checked — show the board clearly", RED)
        self.log(f"Calibration check failed: {error}")
        self._set_check_report(f"Lens calibration check failed.\n\n{error}")
        self._refresh_stage_ui()

    def verify_measurement_accuracy(self):
        """Measure known ChArUco spans through the saved fixed-plane calibration."""
        if (self.current_frame is None or self.processing_capture
                or self.processing_verification):
            return
        calibration_path = self._calibration_path()
        extrinsics_path = self._extrinsics_path()
        if not calibration_path.is_file() or not extrinsics_path.is_file():
            self._set_check_report(
                "Lens calibration and a saved measurement surface are required."
            )
            return
        with self.frame_lock:
            frame = self.current_frame.copy()
        self.processing_verification = True
        self.check_preview_frozen = True
        self._set_check_preview_title(
            "●  CAPTURED MEASUREMENT-CHECK FRAME", AMBER,
        )
        self._set_status("Checking real-world ChArUco distances…", AMBER)
        self._set_check_report(
            "Detecting the board and measuring short and long distances…"
        )
        self._refresh_stage_ui()
        threading.Thread(
            target=self._measurement_accuracy_worker,
            args=(frame, calibration_path, extrinsics_path), daemon=True,
        ).start()

    def _measurement_accuracy_worker(self, frame, calibration_path, extrinsics_path):
        try:
            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            extrinsics = json.loads(extrinsics_path.read_text(encoding="utf-8"))
            calibration_size = tuple(calibration["image_size"])
            frame_size = (frame.shape[1], frame.shape[0])
            camera_matrix = scale_camera_matrix(
                np.asarray(calibration["camera_matrix"], np.float64),
                calibration_size, frame_size,
            )
            dist_coeffs = np.asarray(calibration["dist_coeffs"], np.float64)
            plane_rvec = np.asarray(extrinsics["rvec"], np.float64).reshape(3, 1)
            plane_tvec = np.asarray(extrinsics["tvec"], np.float64).reshape(3, 1)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detections = []
            rows = []
            detector_count = 2 if self.use_two_boards else 1
            required_corners = getattr(self, 'check_minimum_corners', MIN_CORNERS)
            for board_index in range(detector_count):
                corners, ids, _, _ = self.charuco_detectors[board_index].detectBoard(gray)
                corner_count = len(ids) if ids is not None else 0
                if corner_count < required_corners:
                    raise ValueError(
                        f"Board {board_index + 1}: only {corner_count}/{required_corners} "
                        "points were found. Show more of the board and try again."
                    )
                board_rows = measure_board_accuracy(
                    self.boards[board_index], corners, ids,
                    camera_matrix, dist_coeffs, plane_rvec, plane_tvec,
                    getattr(self, 'check_square_length_mm', SQUARE_LENGTH_MM),
                    board_number=board_index + 1,
                )
                rows.extend(board_rows)
                detections.append({
                    "board_index": board_index, "corners": corners, "ids": ids,
                    "image_points": corners,
                })

            summary = summarize_accuracy(rows)
            annotated = self._annotate_board_views(frame, detections)
            overall = summary["overall"]
            scale = max(0.7, frame.shape[1] / 1800.0)
            cv2.putText(
                annotated,
                f"Mean abs error {overall['mean_absolute_error_mm']:.3f} mm",
                (25, 45), cv2.FONT_HERSHEY_SIMPLEX, scale,
                self._hex_to_bgr(GREEN), max(2, round(2 * scale)), cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                f"RMSE {overall['rmse_mm']:.3f} mm  Max {overall['maximum_absolute_error_mm']:.3f} mm",
                (25, 85), cv2.FONT_HERSHEY_SIMPLEX, scale,
                self._hex_to_bgr(AMBER), max(2, round(2 * scale)), cv2.LINE_AA,
            )
            self.root.after(
                0, self._measurement_accuracy_complete,
                annotated, rows, summary, extrinsics,
            )
        except Exception as error:
            self.root.after(0, self._measurement_accuracy_failed, str(error))

    @staticmethod
    def _accuracy_metrics_text(title, metrics):
        if not metrics:
            return f"{title}: no usable distances"
        return (
            f"{title}\n"
            f"  measurements: {metrics['count']}\n"
            f"  mean absolute error: {metrics['mean_absolute_error_mm']:.3f} mm\n"
            f"  RMSE:                {metrics['rmse_mm']:.3f} mm\n"
            f"  maximum error:       {metrics['maximum_absolute_error_mm']:.3f} mm\n"
            f"  mean signed error:   {metrics['mean_signed_error_mm']:+.3f} mm"
        )

    def _measurement_accuracy_complete(self, annotated, rows, summary, extrinsics):
        self.processing_verification = False
        self._set_check_preview_title("●  MEASUREMENT-CHECK RESULT", PURPLE)
        self._show_check_frame(annotated)
        report_parts = [
            "REAL-WORLD MEASUREMENT CHECK",
            f"Board mode: {'two boards' if self.use_two_boards else 'one board'}",
            f"Board definition: {getattr(self, 'check_board_description', 'main configured board')}",
            "Measurement plane: saved calibration used directly",
            "Additional checker thickness adjustment: none",
            "No PASS/FAIL limit is applied.",
            "",
            self._accuracy_metrics_text("SHORT DISTANCES", summary.get("short")),
            "",
            self._accuracy_metrics_text("LONG DISTANCES", summary.get("long")),
            "",
            self._accuracy_metrics_text("OVERALL", summary["overall"]),
            "",
            "Largest individual errors:",
        ]
        largest = sorted(
            rows, key=lambda item: item["absolute_error_mm"], reverse=True,
        )[:12]
        for row in largest:
            report_parts.append(
                f"  B{row['board']} {row['category']:<5}  "
                f"expected {row['expected_mm']:7.2f} mm  "
                f"measured {row['measured_mm']:7.2f} mm  "
                f"error {row['error_mm']:+7.3f} mm "
                f"({row['error_percent']:+6.2f}%)"
            )
        report = "\n".join(report_parts)
        self._set_check_report(report)
        self._set_status(
            f"Measurement check complete — mean error "
            f"{summary['overall']['mean_absolute_error_mm']:.3f} mm",
            GREEN,
        )
        self.log(
            f"Measurement accuracy checked: {summary['overall']['count']} distances, "
            f"mean absolute error {summary['overall']['mean_absolute_error_mm']:.3f} mm, "
            f"maximum {summary['overall']['maximum_absolute_error_mm']:.3f} mm."
        )
        self._refresh_stage_ui()

    def _measurement_accuracy_failed(self, error):
        self.processing_verification = False
        self.check_preview_frozen = False
        self._set_check_preview_title("●  LIVE CAMERA PREVIEW", GREEN)
        self._set_check_report(f"Measurement accuracy check failed.\n\n{error}")
        self._set_status("Measurement accuracy could not be checked", RED)
        self.log(f"Measurement accuracy check failed: {error}")
        self._refresh_stage_ui()

    def _detect_capture_worker(self, frame, file_path):
        """Save, reload and detect one pressed capture away from Tk's UI thread."""
        if getattr(self, 'use_two_boards', False):
            self._detect_dual_capture_worker(frame, file_path)
            return
        try:
            if not cv2.imwrite(str(file_path), frame):
                raise RuntimeError("The captured image could not be saved")
            saved_frame = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
            if saved_frame is None:
                raise RuntimeError("The saved image could not be loaded")
            gray = cv2.cvtColor(saved_frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _, _ = self.charuco_detector.detectBoard(gray)
            corner_count = len(ids) if ids is not None else 0
            if corner_count < MIN_CORNERS:
                file_path.unlink(missing_ok=True)
                self.root.after(
                    0, self._capture_rejected, saved_frame, corners, ids,
                    f"Photo not accepted — {corner_count}/{MIN_CORNERS} board points found",
                    f"Photo deleted. Found {corner_count} board points; {MIN_CORNERS} are required.",
                )
                return
            object_points, image_points = self.board.matchImagePoints(corners, ids)
            if (object_points is None or image_points is None or
                    len(object_points) < MIN_CORNERS):
                matched_count = 0 if image_points is None else len(image_points)
                file_path.unlink(missing_ok=True)
                self.root.after(
                    0, self._capture_rejected, saved_frame, corners, ids,
                    f"Photo not accepted — {matched_count}/{MIN_CORNERS} points matched",
                    f"Photo deleted. Only {matched_count} board points could be matched.",
                )
                return
            image_size = gray.shape[::-1]
            self.root.after(
                0, self._capture_detection_complete, saved_frame, file_path, corners, ids,
                np.asarray(object_points, dtype=np.float32),
                np.asarray(image_points, dtype=np.float32), image_size,
            )
        except Exception as error:
            file_path.unlink(missing_ok=True)
            self.root.after(
                0, self._capture_rejected, frame, None, None,
                "Photo could not be checked — please try again",
                f"Photo deleted after capture error: {error}",
            )

    def _detect_board_view(self, gray, board_index):
        """Return one valid board view plus the raw points for operator feedback."""
        detector = self.charuco_detectors[board_index]
        board = self.boards[board_index]
        corners, ids, _, _ = detector.detectBoard(gray)
        corner_count = len(ids) if ids is not None else 0
        if corner_count < MIN_CORNERS:
            return None, (corners, ids, corner_count)
        object_points, image_points = board.matchImagePoints(corners, ids)
        matched_count = 0 if image_points is None else len(image_points)
        if object_points is None or image_points is None or matched_count < MIN_CORNERS:
            return None, (corners, ids, matched_count)
        return {
            "board_index": board_index,
            "corners": corners,
            "ids": ids,
            "object_points": np.asarray(object_points, dtype=np.float32),
            "image_points": np.asarray(image_points, dtype=np.float32),
        }, (corners, ids, matched_count)

    @staticmethod
    def _mask_detected_board(gray, corners, padding):
        """Hide one detected board so the other detector gets a clean retry."""
        masked = gray.copy()
        points = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
        if len(points) < 3:
            return masked
        hull = cv2.convexHull(points.astype(np.int32))
        board_mask = np.zeros_like(gray)
        cv2.fillConvexPoly(board_mask, hull, 255)
        if padding > 0:
            size = 2 * int(padding) + 1
            board_mask = cv2.dilate(board_mask, np.ones((size, size), np.uint8))
        masked[board_mask > 0] = 255
        return masked

    def _detect_dual_capture_worker(self, frame, file_path):
        """Detect both uniquely numbered boards, with an automatic masked retry."""
        try:
            if not cv2.imwrite(str(file_path), frame):
                raise RuntimeError("The captured image could not be saved")
            saved_frame = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
            if saved_frame is None:
                raise RuntimeError("The saved image could not be loaded")
            gray = cv2.cvtColor(saved_frame, cv2.COLOR_BGR2GRAY)
            detections = []
            feedback = []
            for board_index in range(2):
                detection, raw = self._detect_board_view(gray, board_index)
                feedback.append(raw)
                if detection is not None:
                    detections.append(detection)

            # When one board is clear, mask its area and retry the missing board.
            # This avoids nearby markers from weakening interpolation while still
            # keeping the original full-resolution image for calibration.
            if len(detections) == 1:
                found = detections[0]
                missing_index = 1 - found["board_index"]
                retry_gray = self._mask_detected_board(
                    gray, found["corners"], CONFIG['two_board']['mask_padding_px'],
                )
                retry, retry_raw = self._detect_board_view(retry_gray, missing_index)
                feedback[missing_index] = retry_raw
                if retry is not None:
                    detections.append(retry)

            if not detections:
                best = max(feedback, key=lambda item: item[2])
                file_path.unlink(missing_ok=True)
                self.root.after(
                    0, self._capture_rejected, saved_frame, best[0], best[1],
                    f"Photo not accepted — no board has {MIN_CORNERS} clear points",
                    "Photo deleted. Show Board 1 or Board 2 clearly and try again.",
                )
                return
            self.root.after(
                0, self._dual_capture_detection_complete, saved_frame, file_path,
                detections, gray.shape[::-1],
            )
        except Exception as error:
            file_path.unlink(missing_ok=True)
            self.root.after(
                0, self._capture_rejected, frame, None, None,
                "Photo could not be checked — please try again",
                f"Photo deleted after two-board capture error: {error}",
            )

    def _annotate_board_views(self, frame, detections):
        annotated = frame.copy()
        for detection in detections:
            annotated = self._annotate_detected_board(
                annotated, detection["corners"], detection["ids"],
            )
            points = np.asarray(detection["image_points"], np.float32).reshape(-1, 2)
            center = tuple(np.round(points.mean(axis=0)).astype(int))
            cv2.putText(
                annotated, f"BOARD {detection['board_index'] + 1}",
                (center[0] + 12, center[1] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                max(0.6, annotated.shape[1] / 1500.0), self._hex_to_bgr(TEXT),
                max(2, round(annotated.shape[1] / 900.0)), cv2.LINE_AA,
            )
        return annotated

    def _dual_capture_detection_complete(self, frame, file_path, detections, image_size):
        if self.capture_image_size is not None and image_size != self.capture_image_size:
            file_path.unlink(missing_ok=True)
            first = detections[0]
            self._capture_rejected(
                frame, first["corners"], first["ids"],
                "Camera size changed — restart camera setup",
                "Photo deleted because the camera image size changed.",
            )
            return

        pending = self.pending_dual_capture
        if pending is None and len(detections) == 1:
            found = detections[0]
            self.pending_dual_capture = {
                "detection": found, "path": file_path, "image_size": image_size,
            }
            self.capture_image_size = image_size
            annotated = self._annotate_board_views(frame, [found])
            self._draw_coverage_guide(annotated)
            self._draw_capture_history(annotated)
            self._show_frame(annotated)
            missing_number = 2 if found["board_index"] == 0 else 1
            self.guidance_info.configure(
                text=f"Board {found['board_index'] + 1} saved • Now show Board {missing_number}",
                fg=AMBER,
            )
            self._set_status(
                f"First board saved — capture Board {missing_number} to complete this photo set",
                GREEN,
            )
            self.log(
                f"Board {found['board_index'] + 1} saved separately; waiting for Board {missing_number}."
            )
            self.root.after(650, self._finish_capture_review)
            return

        if pending is not None:
            missing_index = 1 - pending["detection"]["board_index"]
            missing = next(
                (item for item in detections if item["board_index"] == missing_index), None,
            )
            if missing is None:
                file_path.unlink(missing_ok=True)
                found = detections[0]
                self._capture_rejected(
                    frame, found["corners"], found["ids"],
                    f"Board {missing_index + 1} is still needed",
                    f"Photo deleted. Keep the saved first board and show Board {missing_index + 1}.",
                )
                return
            set_detections = [pending["detection"], missing]
            paths = [pending["path"], file_path]
        else:
            set_detections = sorted(detections, key=lambda item: item["board_index"])
            paths = [file_path]

        self.pending_dual_capture = None
        self._accept_dual_capture_set(frame, paths, set_detections, image_size)

    def _accept_dual_capture_set(self, frame, paths, detections, image_size):
        """Store two independent board poses as two intrinsic-calibration views."""
        self.captured_images.append(tuple(paths))
        self.capture_image_size = image_size
        for detection in detections:
            object_points = detection["object_points"]
            image_points = detection["image_points"]
            self.object_views.append(object_points)
            self.image_views.append(image_points)
            row, column, center = coverage_cell(
                image_points, image_size, COVERAGE_ROWS, COVERAGE_COLUMNS,
            )
            self.capture_centers.append(center)
            normalized = np.asarray(image_points, np.float32).reshape(-1, 2)
            self.capture_point_sets.append(
                normalized / np.asarray(image_size, dtype=np.float32)
            )
            self.coverage_counts[row, column] += 1

        annotated = self._annotate_board_views(frame, detections)
        self._draw_coverage_guide(annotated)
        self._draw_capture_history(annotated)
        self._show_frame(annotated)
        count = len(self.captured_images)
        self.capture_info.configure(text=f"{count} / {NUM_CAPTURES} photo sets accepted")
        self.progress["value"] = count
        target_name = self._cell_name(self._recommended_cell())
        self.guidance_info.configure(
            text=f"Both boards saved • Coverage {coverage_percent(self.coverage_counts)}% • Next: {target_name}",
            fg=AMBER,
        )
        self.log(
            f"Photo set {count}/{NUM_CAPTURES} accepted — two separate board views saved"
        )
        if count >= NUM_CAPTURES:
            self.stage = "calibrating"
            self._set_status("All photo sets captured — checking and saving camera setup…", AMBER)
            self._refresh_stage_ui()
            object_views = [view.copy() for view in self.object_views]
            image_views = [view.copy() for view in self.image_views]
            threading.Thread(
                target=self._calibrate_intrinsics_worker,
                args=(object_views, image_views, image_size), daemon=True,
            ).start()
            return
        self._set_status(
            f"Photo set {count} accepted — move both boards toward {target_name}", GREEN,
        )
        self.root.after(650, self._finish_capture_review)

    def _capture_rejected(self, frame, corners, ids, status, log_message):
        annotated = self._annotate_detected_board(frame, corners, ids)
        self._draw_coverage_guide(annotated)
        self._draw_capture_history(annotated)
        self._show_frame(annotated)
        self._set_status(status, RED)
        self.log(log_message)
        self.root.after(550, self._finish_capture_review)

    def _capture_detection_complete(
            self, frame, file_path, corners, ids, object_points, image_points, image_size):
        if self.capture_image_size is not None and image_size != self.capture_image_size:
            file_path.unlink(missing_ok=True)
            self._capture_rejected(
                frame, corners, ids,
                "Camera size changed — restart camera setup",
                "Photo deleted because the camera image size changed.",
            )
            return

        row, column, normalized_center = coverage_cell(
            image_points, image_size, COVERAGE_ROWS, COVERAGE_COLUMNS,
        )
        self.captured_images.append(file_path)
        self.object_views.append(np.asarray(object_points, dtype=np.float32))
        self.image_views.append(np.asarray(image_points, dtype=np.float32))
        self.capture_image_size = image_size
        self.capture_centers.append(normalized_center)
        normalized_points = np.asarray(image_points, dtype=np.float32).reshape(-1, 2)
        normalized_points = normalized_points / np.asarray(image_size, dtype=np.float32)
        self.capture_point_sets.append(normalized_points)
        self.coverage_counts[row, column] += 1
        annotated = self._annotate_detected_board(frame, corners, ids)
        self._draw_coverage_guide(annotated)
        self._draw_capture_history(annotated)
        self._show_frame(annotated)
        count = len(self.captured_images)
        self.capture_info.configure(text=f"{count} / {NUM_CAPTURES} accepted")
        self.progress["value"] = count
        target_name = self._cell_name(self._recommended_cell())
        self.guidance_info.configure(
            text=f"Coverage {coverage_percent(self.coverage_counts)}%  •  Next: {target_name}",
            fg=AMBER,
        )
        self.log(f"Photo {count}/{NUM_CAPTURES} accepted — board points saved")
        if count >= NUM_CAPTURES:
            self.stage = "calibrating"
            self._set_status("All photos captured — checking and saving camera setup…", AMBER)
            self.log("All photos captured. Saving the final camera setup now.")
            self._refresh_stage_ui()
            object_views = [view.copy() for view in self.object_views]
            image_views = [view.copy() for view in self.image_views]
            threading.Thread(
                target=self._calibrate_intrinsics_worker,
                args=(object_views, image_views, image_size), daemon=True,
            ).start()
            return
        self._set_status(
            f"Photo {count} accepted — move the board toward {target_name}", GREEN,
        )
        self.root.after(650, self._finish_capture_review)

    def _calibrate_intrinsics_worker(self, object_views, image_views, image_size):
        """Run the full camera calibration once, after all photos are accepted."""
        self.is_calibrating = True
        try:
            dual_mode = getattr(self, 'use_two_boards', False)
            if len(object_views) < CONFIG['calibration']['minimum_valid_photos']:
                raise RuntimeError("Not enough clear photos. Please restart and take clearer board photos.")
            rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                object_views, image_views, image_size, None, None,
            )
            if (not np.isfinite(camera_matrix).all() or
                    camera_matrix[0, 0] <= 0 or camera_matrix[1, 1] <= 0):
                raise RuntimeError("The camera setup could not be calculated. Please try again.")

            per_view_rms = [
                reprojection_metrics(obj, img, rv, tv, camera_matrix, dist_coeffs)["rms"]
                for obj, img, rv, tv in zip(object_views, image_views, rvecs, tvecs)
            ]
            data = {
                "camera_matrix": camera_matrix.tolist(),
                "dist_coeffs": dist_coeffs.tolist(),
                "rms_error": float(rms),
                "per_view_rms": per_view_rms,
                "image_size": list(image_size),
                "valid_views": len(object_views),
                "capture_method": "two_boards" if dual_mode else "one_board",
                "accepted_photo_sets": (len(getattr(self, 'captured_images', []))
                                        if dual_mode else len(object_views)),
                "capture_coverage": {
                    "rows": COVERAGE_ROWS,
                    "columns": COVERAGE_COLUMNS,
                    "counts": self.coverage_counts.tolist(),
                    "percent": coverage_percent(self.coverage_counts),
                    "centers_normalized": [list(center) for center in self.capture_centers],
                },
                "board": {
                    "squares_x": SQUARES_X, "squares_y": SQUARES_Y,
                    "square_length_mm": SQUARE_LENGTH_MM,
                    "marker_length_mm": MARKER_LENGTH_MM,
                    "dictionary": (CONFIG['two_board']['dictionary']
                                   if dual_mode else CONFIG['board']['dictionary']),
                },
            }
            if dual_mode:
                data["two_board"] = {
                    "board_1_marker_ids": self.boards[0].getIds().reshape(-1).tolist(),
                    "board_2_marker_ids": self.boards[1].getIds().reshape(-1).tolist(),
                    "views_per_photo_set": 2,
                }
            self._calibration_path().write_text(json.dumps(data, indent=2), encoding="utf-8")
            self.root.after(
                0, self._intrinsic_complete, camera_matrix, dist_coeffs,
                image_size, float(rms), len(object_views),
            )
        except Exception as error:
            self.root.after(0, self._calibration_failed, str(error))
        finally:
            self.is_calibrating = False

    def _finish_capture_review(self):
        self.processing_capture = False
        self._refresh_stage_ui()

    def _intrinsic_complete(self, camera_matrix, dist_coeffs, image_size, rms, valid_views):
        self.processing_capture = False
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.calibration_image_size = image_size
        dual_mode = getattr(self, 'use_two_boards', False)
        accepted_count = (len(getattr(self, 'captured_images', []))
                          if dual_mode else valid_views)
        accepted_name = "photo sets" if dual_mode else "photos"
        if not SURFACE_SETUP_ENABLED:
            self.stage = "complete"
            self.camera_running = False
            if self.cap is not None:
                if not self.camera_provider:
                    self.cap.release()
                self.cap = None
            self.stop_btn.configure(state=tk.DISABLED)
            self.start_btn.configure(state=tk.DISABLED)
            self.resolution_label.configure(
                text=f"Camera {self.camera_index}  •  setup complete"
            )
            self.capture_info.configure(
                text=f"{accepted_count} / {NUM_CAPTURES} accepted  •  Setup ready"
            )
            self.guidance_info.configure(
                text=f"Image coverage {coverage_percent(self.coverage_counts)}% complete",
                fg=GREEN,
            )
            self._set_status("Camera setup complete — marker 0 will be checked during inspection", GREEN)
            self.log(
                f"Camera setup saved using {accepted_count} clear {accepted_name} "
                f"({valid_views} board views)."
            )
            self.log("No measurement-surface photo is required in marker mode.")
            self._refresh_stage_ui()
            messagebox.showinfo(
                "Camera Setup Complete",
                f"Camera {self.camera_index} setup is complete.\n\n"
                "Keep the camera fixed and keep marker 0 visible during inspection.",
            )
            return
        self.stage = "extrinsic_ready"
        self.capture_info.configure(
            text=f"{accepted_count} / {NUM_CAPTURES} accepted  •  Setup ready"
        )
        self.guidance_info.configure(
            text=f"Image coverage {coverage_percent(self.coverage_counts)}% complete",
            fg=GREEN,
        )
        self._set_status(
            "Camera setup complete. Place the board flat on the measurement surface.",
            GREEN if rms <= MAX_INTRINSIC_RMS_PX else AMBER,
        )
        self.log(
            f"Camera setup saved using {accepted_count} clear {accepted_name} "
            f"({valid_views} board views)"
        )
        if rms > MAX_INTRINSIC_RMS_PX:
            self.log("Quality warning: results may improve with clearer photos and better lighting.")
        self.log("NEXT: Place the board flat on the final measurement surface.")
        self.log("Do not move the camera after saving the measurement surface.")
        self._refresh_stage_ui()

    def _calibration_failed(self, error):
        self.processing_capture = False
        self.stage = "capture"
        print(f"Camera setup error: {error}")
        self._set_status("Camera setup could not be completed — please try again", RED)
        self.log("Camera setup was not saved. Take clearer board photos and try again.")
        self._refresh_stage_ui()
        messagebox.showerror(
            "Camera Setup Not Completed",
            "Please take clearer board photos in different positions and try again.",
        )

    def capture_extrinsic_reference(self):
        if (not SURFACE_SETUP_ENABLED or self.stage != "extrinsic_ready"
                or self.current_frame is None):
            return
        thickness_enabled = bool(self.board_thickness_enabled_var.get())
        try:
            thickness_mm = float(self.board_thickness_mm_var.get())
            if not np.isfinite(thickness_mm) or thickness_mm < 0:
                raise ValueError
        except (TypeError, ValueError):
            messagebox.showwarning(
                "Board thickness", "Enter a non-negative board thickness in millimetres."
            )
            return
        with self.frame_lock:
            frame = self.current_frame.copy()
        self.stage = "extrinsic_capturing"
        self._set_status("Saving the measurement surface…", AMBER)
        self._refresh_stage_ui()
        threading.Thread(
            target=self._extrinsic_worker,
            args=(frame, thickness_enabled, thickness_mm), daemon=True,
        ).start()

    def _extrinsic_worker(self, frame, thickness_enabled=False, thickness_mm=0.0):
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _, _ = self.charuco_detector.detectBoard(gray)
            corner_count = len(ids) if ids is not None else 0
            if corner_count < MIN_EXTRINSIC_CORNERS:
                raise RuntimeError(
                    "The board is not clear enough. Show the full board, improve the lighting, and try again."
                )
            frame_size = (frame.shape[1], frame.shape[0])
            camera_matrix = scale_camera_matrix(
                self.camera_matrix, self.calibration_image_size, frame_size,
            )
            rvec, tvec, metrics, object_points, _ = estimate_planar_pose(
                self.board, corners, ids, camera_matrix, self.dist_coeffs,
            )
            if metrics["rms"] > MAX_EXTRINSIC_RMS_PX:
                raise RuntimeError(
                    "The photo quality is not good enough. Make sure the board is flat, clear and well lit."
                )

            reference_path = REFERENCE_DIR / f"extrinsic_reference_{self.camera_index}.jpg"
            annotated_path = REFERENCE_DIR / f"extrinsic_reference_{self.camera_index}_annotated.jpg"
            cv2.imwrite(str(reference_path), frame)
            annotated = self._annotate_detected_board(frame, corners, ids)
            cv2.drawFrameAxes(
                annotated, camera_matrix, self.dist_coeffs, rvec, tvec, 0.05, 4,
            )
            cv2.imwrite(str(annotated_path), annotated)

            reference_board_tvec = tvec.copy()
            measurement_tvec = (
                offset_plane_tvec(rvec, tvec, thickness_mm, away_from_camera=True)
                if thickness_enabled and thickness_mm > 0 else tvec.copy()
            )
            data = {
                "rvec": rvec.reshape(-1).tolist(),
                "tvec": measurement_tvec.reshape(-1).tolist(),
                "reference_board_tvec": reference_board_tvec.reshape(-1).tolist(),
                "reprojection_error": metrics,
                "image_size": list(frame_size),
                "corner_count": int(len(object_points)),
                "reference_image": str(reference_path.relative_to(PROJECT_ROOT)),
                "coordinate_system": (
                    "ChArUco XY axes on bed plane; lengths stored in metres"
                    if thickness_enabled else
                    "ChArUco board-top plane; lengths stored in metres"
                ),
                "board_thickness": {
                    "enabled": bool(thickness_enabled),
                    "thickness_mm": float(thickness_mm),
                    "measurement_plane": ("bed below board" if thickness_enabled
                                          else "top of ChArUco board"),
                },
                "board": {
                    "squares_x": SQUARES_X, "squares_y": SQUARES_Y,
                    "square_length_mm": SQUARE_LENGTH_MM,
                    "marker_length_mm": MARKER_LENGTH_MM,
                    "dictionary": (CONFIG['two_board']['dictionary']
                                   if self.use_two_boards else CONFIG['board']['dictionary']),
                },
            }
            self._extrinsics_path().write_text(json.dumps(data, indent=2), encoding="utf-8")
            self.root.after(
                0, self._extrinsic_complete, annotated, metrics,
                measurement_tvec, thickness_enabled, thickness_mm,
            )
        except Exception as error:
            self.root.after(0, self._extrinsic_failed, str(error))

    def _extrinsic_complete(
            self, annotated, metrics, tvec,
            thickness_enabled=False, thickness_mm=0.0):
        self.stage = "complete"
        self.stop_btn.configure(state=tk.NORMAL)
        self.start_btn.configure(state=tk.DISABLED)
        self.resolution_label.configure(
            text=f"Camera {self.camera_index}  •  setup complete  •  live for checks"
        )
        self._set_status("Setup complete — keep the camera fixed", GREEN)
        self.log("Measurement surface saved successfully.")
        if thickness_enabled:
            self.log(
                f"Measurement plane corrected {thickness_mm:.3f} mm from board top to bed."
            )
        else:
            self.log("Board-thickness correction is disabled.")
        self.log("Camera setup is complete and ready to use.")
        self._show_frame(annotated)
        self._refresh_stage_ui()
        messagebox.showinfo(
            "Setup Complete",
            f"Camera {self.camera_index} setup is complete.\n\n"
            "Keep the camera and measurement surface fixed.\n"
            "Open Calibration Checks to validate real-world distances.",
        )

    def _extrinsic_failed(self, error):
        self.stage = "extrinsic_ready"
        print(f"Measurement surface setup error: {error}")
        self._set_status("Surface photo not accepted — flatten the board and try again", RED)
        self.log("Measurement surface was not saved. Check the board and try again.")
        self._refresh_stage_ui()
        messagebox.showerror(
            "Measurement Surface Not Saved",
            "Make sure the board is flat, fully visible and well lit, then try again.",
        )

    def shutdown(self):
        """Stop this embedded page without deciding which main-app page follows."""
        if self.closed:
            return True
        if (self.starting_camera or self.processing_capture or self.processing_verification or
                self.stage in ("calibrating", "extrinsic_capturing")):
            self._set_status("Please wait for the current step to finish before going back", AMBER)
            return False
        self.closed = True
        self._close_calibration_checks()
        if getattr(self, '_controls_wheel_binding', None):
            self.root.unbind("<MouseWheel>", self._controls_wheel_binding)
            self._controls_wheel_binding = None
        if self.preview_job is not None:
            self.root.after_cancel(self.preview_job)
            self.preview_job = None
        self.camera_running = False
        if self.cap is not None:
            if not self.camera_provider: self.cap.release()
            self.cap = None
        return True

    def on_closing(self):
        if not self.shutdown():
            return
        if self.on_close is not None:
            self.on_close()
        else:
            self.root.destroy()


class CalibrationCheckApp(CalibrationApp):
    """Dedicated in-app page for intrinsic and millimetre accuracy checks."""

    def __init__(self, root, on_close=None, host=None, camera_provider=None,
                 on_open_setup=None):
        self.on_open_setup = on_open_setup
        self.check_square_length_mm = SQUARE_LENGTH_MM
        self.manual_measurement_active = False
        self.manual_image_bgr = None
        self.manual_image_pil = None
        self.manual_points = []
        self.manual_zoom = 1.0
        self.manual_fit_scale = 1.0
        self.manual_offset = [0.0, 0.0]
        self.manual_pan_anchor = None
        super().__init__(
            root, on_close=on_close, host=host, camera_provider=camera_provider,
        )

    def _create_title_bar(self):
        title_bar = tk.Frame(
            self.host, bg=TITLE_BG, height=TITLE_BAR_HEIGHT,
            highlightbackground=C["border"], highlightthickness=1,
        )
        title_bar.pack(fill=tk.X)
        title_bar.pack_propagate(False)
        if self.on_close is not None:
            themed_button(
                title_bar, "←  DASHBOARD", self.on_closing,
                role="quiet", padx=16, pady=6,
            ).pack(side=tk.LEFT, fill=tk.Y)
        if self.on_open_setup is not None:
            themed_button(
                title_bar, "CAMERA SETUP", self.on_open_setup,
                role="quiet", padx=16, pady=6,
            ).pack(side=tk.LEFT, fill=tk.Y)
        tk.Label(
            title_bar, text="CALIBRATION CHECK", bg=TITLE_BG, fg=PURPLE,
            font=(FONT, 10, "bold"),
        ).pack(side=tk.LEFT, padx=12)
        themed_button(
            title_bar, "✕", self.on_closing, role="quiet",
            padx=16, pady=6, font_size=12,
        ).pack(side=tk.RIGHT, fill=tk.Y)
        themed_button(
            title_bar, "—", self.minimize_window, role="quiet",
            padx=16, pady=6, font_size=12,
        ).pack(side=tk.RIGHT, fill=tk.Y)
        title_bar.bind("<ButtonPress-1>", self._start_move)
        title_bar.bind("<B1-Motion>", self._do_move)

    def _create_ui(self):
        self._create_title_bar()
        body = tk.Frame(self.host, bg=BG)
        body.pack(fill=tk.BOTH, expand=True, padx=18, pady=(14, 16))

        heading = tk.Frame(body, bg=BG)
        heading.pack(fill=tk.X, pady=(0, 12))
        tk.Label(
            heading, text="Calibration Check", bg=BG, fg=TEXT,
            font=(FONT, 26, "bold"),
        ).pack(side=tk.LEFT)
        tk.Label(
            heading,
            text="Verify the saved lens calibration and real-world millimetre accuracy",
            bg=BG, fg=MUTED, font=(FONT, 10),
        ).pack(side=tk.LEFT, padx=16, pady=(9, 0))

        workspace = tk.Frame(body, bg=BG)
        workspace.pack(fill=tk.BOTH, expand=True)
        workspace.columnconfigure(0, weight=1, minsize=0)
        workspace.columnconfigure(1, weight=0, minsize=430)
        workspace.rowconfigure(0, weight=1)

        preview_card = themed_card(workspace, bg=C["surface"])
        preview_card.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        preview_card.pack_propagate(False)
        preview_header = tk.Frame(preview_card, bg=C["surface"])
        preview_header.pack(fill=tk.X, padx=14, pady=10)
        status_dot(preview_header).pack(side=tk.LEFT, padx=(0, 7))
        self.preview_title = tk.Label(
            preview_header, text="CAMERA PREVIEW", bg=C["surface"],
            fg=C["text_soft"], font=(FONT, 9, "bold"),
        )
        self.preview_title.pack(side=tk.LEFT)
        self.resolution_label = tk.Label(
            preview_header, text="No camera connected", bg=C["surface"],
            fg=MUTED, font=(FONT, 9),
        )
        self.resolution_label.pack(side=tk.RIGHT)
        viewport = tk.Frame(preview_card, bg=CANVAS_BG)
        viewport.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.video_label = tk.Label(
            viewport, text="Select a camera and start the live feed",
            bg=CANVAS_BG, fg=MUTED, font=(FONT, 14), bd=0,
            highlightthickness=0,
        )
        self.video_label.place(x=0, y=0, relwidth=1, relheight=1)
        self.manual_canvas = tk.Canvas(
            viewport, bg=CANVAS_BG, bd=0, highlightthickness=0,
            cursor="crosshair",
        )
        self.manual_canvas.bind("<Configure>", self._manual_canvas_resized)
        self.manual_canvas.bind("<MouseWheel>", self._manual_zoom_wheel)
        self.manual_canvas.bind("<Button-1>", self._manual_select_point)
        self.manual_canvas.bind("<ButtonPress-2>", self._manual_pan_start)
        self.manual_canvas.bind("<B2-Motion>", self._manual_pan_move)
        self.manual_canvas.bind("<ButtonPress-3>", self._manual_pan_start)
        self.manual_canvas.bind("<B3-Motion>", self._manual_pan_move)

        controls_shell = themed_card(workspace, bg=PANEL, width=430)
        controls_shell.grid(row=0, column=1, sticky="nsew")
        controls_shell.grid_propagate(False)
        controls_shell.rowconfigure(0, weight=1)
        controls_shell.columnconfigure(0, weight=1)
        self.controls_canvas = tk.Canvas(
            controls_shell, bg=PANEL, bd=0, highlightthickness=0, takefocus=False,
        )
        self.controls_canvas.grid(row=0, column=0, sticky="nsew")
        self.controls_scrollbar = ttk.Scrollbar(
            controls_shell, orient=tk.VERTICAL, command=self.controls_canvas.yview,
        )
        self.controls_scrollbar.grid(row=0, column=1, sticky="ns")
        self.controls_canvas.configure(yscrollcommand=self.controls_scrollbar.set)
        controls = tk.Frame(self.controls_canvas, bg=PANEL)
        self.controls_content = controls
        self._controls_window = self.controls_canvas.create_window(
            (0, 0), window=controls, anchor="nw",
        )
        controls.bind("<Configure>", self._update_controls_scrollregion)
        self.controls_canvas.bind("<Configure>", self._resize_controls_content)
        self._controls_wheel_binding = self.root.bind(
            "<MouseWheel>", self._scroll_controls_with_mouse, add="+",
        )
        self.status_banner = tk.Label(
            controls, text="Select a camera below", bg=C["surface_2"], fg=CYAN,
            wraplength=380, justify=tk.LEFT, font=(FONT, 11, "bold"),
            padx=14, pady=11,
        )
        self.status_banner.pack(fill=tk.X, padx=12, pady=12)

        camera_box = tk.Frame(controls, bg=PANEL)
        camera_box.pack(fill=tk.X, padx=12)
        section_label(camera_box, "1  Select camera and board method").pack(anchor="w")
        camera_row = tk.Frame(camera_box, bg=PANEL)
        camera_row.pack(fill=tk.X, pady=(6, 4))
        self.btn_cam0 = self._button(
            camera_row, f"CAMERA {CAMERA_IDS[0]}",
            lambda: self.select_camera(CAMERA_IDS[0]), GREEN, 13,
        )
        self.btn_cam0.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))
        self.btn_cam1 = self._button(
            camera_row, f"CAMERA {CAMERA_IDS[1]}",
            lambda: self.select_camera(CAMERA_IDS[1]), BLUE, 13,
        )
        self.btn_cam1.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(4, 0))
        stream_row = tk.Frame(camera_box, bg=PANEL)
        stream_row.pack(fill=tk.X, pady=(0, 7))
        self.start_btn = self._button(
            stream_row, "START CAMERA", self.start_camera, GREEN, 15, tk.DISABLED,
        )
        self.start_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))
        self.stop_btn = self._button(
            stream_row, "STOP", self.stop_camera, RED, 9, tk.DISABLED,
        )
        self.stop_btn.pack(side=tk.LEFT, padx=(4, 0))
        mode_row = tk.Frame(camera_box, bg=PANEL)
        mode_row.pack(fill=tk.X, pady=(0, 9))
        self.one_board_btn = self._button(
            mode_row, "ONE BOARD", lambda: self.select_board_mode(False), BLUE, 12,
        )
        self.one_board_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 3))
        self.two_board_btn = self._button(
            mode_row, "TWO BOARDS", lambda: self.select_board_mode(True), BLUE, 12,
        )
        self.two_board_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(3, 0))
        self.guidance_info = tk.Label(
            camera_box, text="Use the same ChArUco board method as calibration",
            bg=PANEL, fg=AMBER, wraplength=380, justify=tk.LEFT, font=(FONT, 9, "bold"),
        )
        self.guidance_info.pack(anchor="w", pady=(0, 8))
        self._refresh_board_mode_ui()

        board_settings = tk.Frame(
            controls, bg=C["surface_2"], highlightbackground=C["border_strong"],
            highlightthickness=1,
        )
        board_settings.pack(fill=tk.X, padx=12, pady=(0, 10))
        section_label(board_settings, "2  ChArUco board used for this check").grid(
            row=0, column=0, columnspan=4, sticky="w", padx=10, pady=(9, 5),
        )
        self.check_dictionary_var = tk.StringVar(value=CONFIG['board']['dictionary'])
        self.check_squares_x_var = tk.StringVar(value=str(SQUARES_X))
        self.check_squares_y_var = tk.StringVar(value=str(SQUARES_Y))
        self.check_square_mm_var = tk.StringVar(value=str(SQUARE_LENGTH_MM))
        self.check_marker_mm_var = tk.StringVar(value=str(MARKER_LENGTH_MM))
        self.check_board1_start_var = tk.StringVar(value="0")
        self.check_board2_start_var = tk.StringVar(
            value=str(CONFIG['two_board']['second_board_start_id'])
        )
        tk.Label(
            board_settings, text="Dictionary", bg=C["surface_2"], fg=MUTED,
            font=(FONT, 8, "bold"),
        ).grid(row=1, column=0, sticky="w", padx=(10, 4), pady=3)
        dictionary_values = (
            "DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000",
            "DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250", "DICT_5X5_1000",
            "DICT_6X6_50", "DICT_6X6_100", "DICT_6X6_250", "DICT_6X6_1000",
            "DICT_7X7_50", "DICT_7X7_100", "DICT_7X7_250", "DICT_7X7_1000",
        )
        ttk.Combobox(
            board_settings, textvariable=self.check_dictionary_var,
            values=dictionary_values, state="readonly", width=18,
        ).grid(row=1, column=1, columnspan=3, sticky="ew", padx=(4, 10), pady=3)

        def setting_entry(label, variable, row, column):
            tk.Label(
                board_settings, text=label, bg=C["surface_2"], fg=MUTED,
                font=(FONT, 8, "bold"),
            ).grid(row=row, column=column, sticky="w", padx=(10 if column == 0 else 6, 3), pady=3)
            tk.Entry(
                board_settings, textvariable=variable, width=7, bg=TITLE_BG,
                fg=TEXT, insertbackground=TEXT, relief=tk.FLAT,
                justify=tk.CENTER, font=(FONT, 9),
            ).grid(row=row, column=column + 1, sticky="ew", padx=(3, 6), pady=3)

        setting_entry("Squares X", self.check_squares_x_var, 2, 0)
        setting_entry("Squares Y", self.check_squares_y_var, 2, 2)
        setting_entry("Square mm", self.check_square_mm_var, 3, 0)
        setting_entry("Marker mm", self.check_marker_mm_var, 3, 2)
        setting_entry("Board 1 ID", self.check_board1_start_var, 4, 0)
        setting_entry("Board 2 ID", self.check_board2_start_var, 4, 2)
        for column in range(4):
            board_settings.columnconfigure(column, weight=1)
        themed_button(
            board_settings, "APPLY BOARD SETTINGS", self._apply_checker_board_settings,
            role="secondary", pady=7,
        ).grid(row=5, column=0, columnspan=4, sticky="ew", padx=10, pady=(7, 10))

        checks = tk.Frame(
            controls, bg=C["surface_2"], highlightbackground=C["border_strong"],
            highlightthickness=1,
        )
        checks.pack(fill=tk.X, padx=12, pady=(0, 10))
        self.check_panel = checks
        self.check_panel_visible = True
        section_label(checks, "3  Run a fresh verification capture").pack(
            anchor="w", padx=10, pady=(9, 3),
        )
        tk.Label(
            checks,
            text=("One-board mode detects Board 1. Two-board mode requires both "
                  "separately numbered boards for both checks."),
            bg=C["surface_2"], fg=MUTED, wraplength=370, justify=tk.LEFT,
            font=(FONT, 9),
        ).pack(anchor="w", padx=10, pady=(0, 7))
        self.check_lens_btn = themed_button(
            checks, "CHECK LENS CALIBRATION", self.verify_saved_calibration,
            role="blue", width=29, pady=9,
        )
        self.check_lens_btn.pack(fill=tk.X, padx=10, pady=(0, 6))
        self.check_measurement_btn = themed_button(
            checks, "CHECK MEASUREMENT ACCURACY", self.verify_measurement_accuracy,
            role="purple", width=29, pady=9,
        )
        self.check_measurement_btn.pack(fill=tk.X, padx=10, pady=(0, 6))
        self.manual_measurement_btn = themed_button(
            checks, "MANUAL TWO-POINT MEASUREMENT", self.start_manual_measurement,
            role="primary", width=29, pady=9,
        )
        self.manual_measurement_btn.pack(fill=tk.X, padx=10, pady=(0, 6))
        self.check_live_btn = themed_button(
            checks, "RESUME LIVE PREVIEW", self.resume_check_preview,
            role="secondary", width=29, pady=7,
        )
        self.check_live_btn.pack(fill=tk.X, padx=10, pady=(0, 7))
        self.check_plane_info = tk.Label(
            checks, text="", bg=TITLE_BG, fg=C["text_soft"], justify=tk.LEFT,
            wraplength=370, font=(FONT, 9), padx=9, pady=6,
        )
        self.check_plane_info.pack(fill=tk.X, padx=10, pady=(0, 8))
        manual_actions = tk.Frame(checks, bg=C["surface_2"])
        manual_actions.pack(fill=tk.X, padx=10, pady=(0, 8))
        themed_button(
            manual_actions, "UNDO POINT", self.undo_manual_point,
            role="quiet", width=13, pady=6,
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 3))
        themed_button(
            manual_actions, "RESET POINTS", self.reset_manual_points,
            role="quiet", width=13, pady=6,
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(3, 0))

        self.check_report = scrolledtext.ScrolledText(
            controls, height=9, bg=TITLE_BG, fg=C["text_soft"],
            insertbackground=TEXT, font=(MONO_FONT, 9), wrap=tk.WORD,
            relief=tk.FLAT,
        )
        self.check_report.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 8))
        self.check_report.configure(state=tk.DISABLED)
        self.status_text = scrolledtext.ScrolledText(
            controls, height=3, bg=TITLE_BG, fg=C["text_soft"],
            insertbackground=TEXT, font=(MONO_FONT, 8), wrap=tk.WORD,
            relief=tk.FLAT,
        )
        self.status_text.pack(fill=tk.X, padx=12, pady=(0, 12))
        self.status_text.configure(state=tk.DISABLED)
        self._set_check_report(
            "Select a camera and the same one-board or two-board method used during "
            "calibration, then start the live preview."
        )

    def _refresh_stage_ui(self):
        if not hasattr(self, 'start_btn'):
            return
        selected = self.camera_index is not None
        busy = self.starting_camera or self.processing_verification
        self.start_btn.configure(
            state=(tk.NORMAL if selected and not self.camera_running and not busy
                   else tk.DISABLED)
        )
        self.stop_btn.configure(
            state=(tk.NORMAL if self.camera_running and not busy else tk.DISABLED)
        )
        self._refresh_check_panel_buttons()

    def _refresh_check_panel_buttons(self):
        super()._refresh_check_panel_buttons()
        if not hasattr(self, 'manual_measurement_btn'):
            return
        ready = (
            self.camera_index is not None and self.camera_running
            and self.camera_matrix is not None and self.dist_coeffs is not None
            and self._extrinsics_path().is_file()
            and not self.processing_verification
        )
        self.manual_measurement_btn.configure(
            state=tk.NORMAL if ready else tk.DISABLED,
        )

    def _load_saved_intrinsics(self):
        saved = super()._load_saved_intrinsics()
        board = saved.get("board", {})
        self.check_dictionary_var.set(
            board.get("dictionary", self.check_dictionary_var.get())
        )
        self.check_squares_x_var.set(str(board.get("squares_x", SQUARES_X)))
        self.check_squares_y_var.set(str(board.get("squares_y", SQUARES_Y)))
        self.check_square_mm_var.set(str(board.get("square_length_mm", SQUARE_LENGTH_MM)))
        self.check_marker_mm_var.set(str(board.get("marker_length_mm", MARKER_LENGTH_MM)))
        self.check_board1_start_var.set("0")
        two_board = saved.get("two_board", {})
        board_1_ids = two_board.get("board_1_marker_ids", [])
        board_2_ids = two_board.get("board_2_marker_ids", [])
        if board_1_ids:
            self.check_board1_start_var.set(str(min(map(int, board_1_ids))))
        if board_2_ids:
            self.check_board2_start_var.set(str(min(map(int, board_2_ids))))
        self._apply_checker_board_settings(announce=False)
        return saved

    def _apply_checker_board_settings(self, announce=True):
        """Build session-only ChArUco boards from the checker controls."""
        if self.processing_verification:
            if announce:
                self._set_status("Wait for the current check to finish", AMBER)
            return False
        try:
            dictionary_name = self.check_dictionary_var.get().strip()
            dictionary_code = getattr(cv2.aruco, dictionary_name)
            dictionary = cv2.aruco.getPredefinedDictionary(dictionary_code)
            squares_x = int(self.check_squares_x_var.get())
            squares_y = int(self.check_squares_y_var.get())
            square_mm = float(self.check_square_mm_var.get())
            marker_mm = float(self.check_marker_mm_var.get())
            board_1_start = int(self.check_board1_start_var.get())
            board_2_start = int(self.check_board2_start_var.get())
            if squares_x < 3 or squares_y < 3:
                raise ValueError("Squares X and Y must both be at least 3")
            if not np.isfinite(square_mm) or not np.isfinite(marker_mm):
                raise ValueError("Square and marker dimensions must be finite")
            if square_mm <= 0 or marker_mm <= 0 or marker_mm >= square_mm:
                raise ValueError("Marker size must be positive and smaller than square size")
            if board_1_start < 0 or board_2_start < 0:
                raise ValueError("Board start IDs cannot be negative")

            template = cv2.aruco.CharucoBoard(
                (squares_x, squares_y), square_mm / 1000.0,
                marker_mm / 1000.0, dictionary,
            )
            marker_count = int(len(template.getIds()))
            capacity = int(dictionary.bytesList.shape[0])
            starts = [board_1_start]
            if self.use_two_boards:
                starts.append(board_2_start)
            id_sets = [
                np.arange(start, start + marker_count, dtype=np.int32)
                for start in starts
            ]
            for board_number, marker_ids in enumerate(id_sets, start=1):
                if int(marker_ids[-1]) >= capacity:
                    raise ValueError(
                        f"Board {board_number} needs IDs through {int(marker_ids[-1])}, "
                        f"but {dictionary_name} ends at {capacity - 1}"
                    )
            if len(id_sets) == 2 and set(id_sets[0]).intersection(map(int, id_sets[1])):
                raise ValueError("Board 1 and Board 2 marker-ID ranges overlap")

            boards = []
            detectors = []
            for marker_ids in id_sets:
                board = cv2.aruco.CharucoBoard(
                    (squares_x, squares_y), square_mm / 1000.0,
                    marker_mm / 1000.0, dictionary, marker_ids,
                )
                detector = cv2.aruco.CharucoDetector(
                    board, cv2.aruco.CharucoParameters(),
                    cv2.aruco.DetectorParameters(),
                )
                boards.append(board)
                detectors.append(detector)
            self.boards = boards
            self.charuco_detectors = detectors
            self.board = boards[0]
            self.charuco_detector = detectors[0]
            self.check_square_length_mm = square_mm
            self.check_board_description = (
                f"{dictionary_name}, {squares_x}×{squares_y}, "
                f"square {square_mm:g} mm, marker {marker_mm:g} mm"
            )
            self.check_minimum_corners = min(
                MIN_CORNERS, max(4, (squares_x - 1) * (squares_y - 1))
            )
        except (AttributeError, TypeError, ValueError, cv2.error) as error:
            self._set_status("Board settings were not applied", RED)
            self._set_check_report(f"Invalid ChArUco board settings.\n\n{error}")
            return False

        self._refresh_board_mode_ui()
        if announce:
            ranges = [f"B{index + 1}: {ids[0]}–{ids[-1]}"
                      for index, ids in enumerate(id_sets)]
            message = (
                f"Applied {dictionary_name}, {squares_x}×{squares_y}, "
                f"square {square_mm:g} mm, marker {marker_mm:g} mm; "
                + ", ".join(ranges)
            )
            self._set_status("ChArUco checker settings applied", GREEN)
            self._set_check_report(message)
            self.log(message)
        return True

    def select_camera(self, index):
        if self.starting_camera or self.processing_verification:
            return
        self._hide_manual_measurement()
        if self.camera_running:
            self.stop_camera()
        self.camera_index = index
        self.stage = "select"
        self.current_frame = None
        self.camera_matrix = None
        self.dist_coeffs = None
        self.calibration_image_size = None
        set_button_role(self.btn_cam0, "selected" if index == CAMERA_IDS[0] else "secondary")
        set_button_role(self.btn_cam1, "selected" if index == CAMERA_IDS[1] else "secondary")
        self._set_status(f"Camera {index} selected — start the live feed")
        self.log(f"Selected Camera {index} for calibration checks")
        self._refresh_stage_ui()

    def stop_camera(self):
        self._hide_manual_measurement()
        super().stop_camera()

    def _camera_started(self, requested_width, requested_height):
        self.starting_camera = False
        self.stage = "complete"
        self.start_btn.configure(text="START CAMERA", state=tk.DISABLED)
        self.resolution_label.configure(
            text=f"Camera {self.camera_index}  •  {self.actual_size[0]} × {self.actual_size[1]}"
        )
        try:
            self._load_saved_intrinsics()
        except (OSError, KeyError, TypeError, ValueError) as error:
            self._set_status("Saved lens calibration is missing or invalid", RED)
            self.log(f"Calibration checks unavailable: {error}")
        else:
            self._set_status("Live preview ready — position the ChArUco board", GREEN)
            self.log(
                f"Camera {self.camera_index} ready for "
                f"{'two-board' if self.use_two_boards else 'one-board'} checks"
            )
        self._refresh_stage_ui()

    def select_board_mode(self, enabled):
        if self.processing_verification:
            return
        previous_mode = self.use_two_boards
        self.use_two_boards = bool(enabled)
        if not self._apply_checker_board_settings(announce=False):
            self.use_two_boards = previous_mode
            self._refresh_board_mode_ui()
            return
        message = (
            "Two-board mode — show both separately numbered ChArUco boards"
            if self.use_two_boards else
            "One-board mode — show Board 1"
        )
        self.guidance_info.configure(text=message, fg=AMBER)
        self._set_status(message)
        self._set_check_report(
            message + ". This selection is used for both lens and measurement checks."
        )

    def start_manual_measurement(self):
        """Capture one frame and prepare an undistorted zoomable point picker."""
        if (self.current_frame is None or self.processing_verification
                or self.camera_index is None):
            return
        calibration_path = self._calibration_path()
        extrinsics_path = self._extrinsics_path()
        if not calibration_path.is_file() or not extrinsics_path.is_file():
            self._set_check_report(
                "Manual measurement requires saved lens and measurement-surface calibration."
            )
            return
        with self.frame_lock:
            frame = self.current_frame.copy()
        self.processing_verification = True
        self.check_preview_frozen = True
        self._set_check_preview_title("●  PREPARING UNDISTORTED CAPTURE", AMBER)
        self._set_status("Preparing manual measurement image…", AMBER)
        self._refresh_stage_ui()
        threading.Thread(
            target=self._prepare_manual_measurement_worker,
            args=(frame, calibration_path, extrinsics_path), daemon=True,
        ).start()

    def _prepare_manual_measurement_worker(self, frame, calibration_path, extrinsics_path):
        try:
            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            extrinsics = json.loads(extrinsics_path.read_text(encoding="utf-8"))
            frame_size = (frame.shape[1], frame.shape[0])
            camera_matrix = scale_camera_matrix(
                np.asarray(calibration["camera_matrix"], dtype=np.float64),
                tuple(calibration["image_size"]), frame_size,
            )
            dist_coeffs = np.asarray(calibration["dist_coeffs"], dtype=np.float64)
            undistorted = cv2.undistort(
                frame, camera_matrix, dist_coeffs, None, camera_matrix,
            )
            rvec = np.asarray(extrinsics["rvec"], dtype=np.float64).reshape(3, 1)
            tvec = np.asarray(extrinsics["tvec"], dtype=np.float64).reshape(3, 1)
            self.root.after(
                0, self._manual_measurement_ready,
                undistorted, camera_matrix, rvec, tvec,
            )
        except Exception as error:
            self.root.after(0, self._manual_measurement_failed, str(error))

    def _manual_measurement_ready(self, image, camera_matrix, rvec, tvec):
        self.processing_verification = False
        self.manual_measurement_active = True
        self.manual_image_bgr = image
        self.manual_image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        self.manual_camera_matrix = camera_matrix
        self.manual_plane_rvec = rvec
        self.manual_plane_tvec = tvec
        self.manual_points = []
        self.manual_distance_mm = None
        self.video_label.place_forget()
        self.manual_canvas.place(x=0, y=0, relwidth=1, relheight=1)
        self.manual_canvas.lift()
        self._set_check_preview_title("●  MANUAL TWO-POINT MEASUREMENT", PURPLE)
        self._set_status("Select Point 1 and Point 2 on the saved measurement plane", GREEN)
        self._set_check_report(
            "MANUAL TWO-POINT MEASUREMENT\n\n"
            "Left-click: select Point 1 and Point 2\n"
            "Mouse wheel: zoom at cursor\n"
            "Right-drag or middle-drag: pan\n\n"
            "The captured image is undistorted. The saved measurement-plane "
            "calibration is used directly; no additional thickness adjustment is applied."
        )
        self._refresh_stage_ui()
        self.root.after_idle(self._reset_manual_view)

    def _manual_measurement_failed(self, error):
        self.processing_verification = False
        self.check_preview_frozen = False
        self.manual_measurement_active = False
        self._set_check_preview_title("●  LIVE CAMERA PREVIEW", GREEN)
        self._set_status("Manual measurement could not be prepared", RED)
        self._set_check_report(f"Manual measurement failed.\n\n{error}")
        self._refresh_stage_ui()

    def _manual_canvas_resized(self, _event=None):
        if self.manual_measurement_active:
            self._reset_manual_view()

    def _reset_manual_view(self):
        if not self.manual_measurement_active or self.manual_image_pil is None:
            return
        canvas_width = max(2, self.manual_canvas.winfo_width())
        canvas_height = max(2, self.manual_canvas.winfo_height())
        image_width, image_height = self.manual_image_pil.size
        self.manual_fit_scale = min(
            canvas_width / image_width, canvas_height / image_height,
        )
        self.manual_zoom = 1.0
        scale = self.manual_fit_scale
        self.manual_offset = [
            (canvas_width - image_width * scale) / 2.0,
            (canvas_height - image_height * scale) / 2.0,
        ]
        self._render_manual_measurement()

    def _manual_scale(self):
        return max(1e-9, self.manual_fit_scale * self.manual_zoom)

    def _clamp_manual_offset(self):
        if self.manual_image_pil is None:
            return
        canvas_width = max(2, self.manual_canvas.winfo_width())
        canvas_height = max(2, self.manual_canvas.winfo_height())
        image_width, image_height = self.manual_image_pil.size
        scale = self._manual_scale()
        scaled_width = image_width * scale
        scaled_height = image_height * scale
        if scaled_width <= canvas_width:
            self.manual_offset[0] = (canvas_width - scaled_width) / 2.0
        else:
            self.manual_offset[0] = min(0.0, max(canvas_width - scaled_width, self.manual_offset[0]))
        if scaled_height <= canvas_height:
            self.manual_offset[1] = (canvas_height - scaled_height) / 2.0
        else:
            self.manual_offset[1] = min(0.0, max(canvas_height - scaled_height, self.manual_offset[1]))

    def _render_manual_measurement(self):
        if not self.manual_measurement_active or self.manual_image_pil is None:
            return
        self._clamp_manual_offset()
        canvas_width = max(2, self.manual_canvas.winfo_width())
        canvas_height = max(2, self.manual_canvas.winfo_height())
        image_width, image_height = self.manual_image_pil.size
        scale = self._manual_scale()
        offset_x, offset_y = self.manual_offset
        source_left = max(0, int(np.floor(-offset_x / scale)))
        source_top = max(0, int(np.floor(-offset_y / scale)))
        source_right = min(image_width, int(np.ceil((canvas_width - offset_x) / scale)))
        source_bottom = min(image_height, int(np.ceil((canvas_height - offset_y) / scale)))
        self.manual_canvas.delete("all")
        if source_right <= source_left or source_bottom <= source_top:
            return
        crop = self.manual_image_pil.crop(
            (source_left, source_top, source_right, source_bottom)
        )
        display_width = max(1, round((source_right - source_left) * scale))
        display_height = max(1, round((source_bottom - source_top) * scale))
        crop = crop.resize(
            (display_width, display_height),
            Image.Resampling.LANCZOS if scale < 1.0 else Image.Resampling.BILINEAR,
        )
        photo = ImageTk.PhotoImage(crop)
        self.manual_canvas.image = photo
        self.manual_canvas.create_image(
            offset_x + source_left * scale,
            offset_y + source_top * scale,
            image=photo, anchor="nw",
        )
        canvas_points = []
        for index, (image_x, image_y) in enumerate(self.manual_points, start=1):
            canvas_x = offset_x + image_x * scale
            canvas_y = offset_y + image_y * scale
            canvas_points.append((canvas_x, canvas_y))
            radius = 7
            self.manual_canvas.create_oval(
                canvas_x - radius, canvas_y - radius,
                canvas_x + radius, canvas_y + radius,
                outline="#FFFFFF", fill=RED if index == 1 else GREEN, width=2,
            )
            self.manual_canvas.create_text(
                canvas_x + 11, canvas_y - 11, text=f"P{index}", anchor="sw",
                fill="#FFFFFF", font=(FONT, 10, "bold"),
            )
        if len(canvas_points) == 2:
            (x1, y1), (x2, y2) = canvas_points
            self.manual_canvas.create_line(x1, y1, x2, y2, fill=AMBER, width=3)
            if self.manual_distance_mm is not None:
                self.manual_canvas.create_text(
                    (x1 + x2) / 2.0, (y1 + y2) / 2.0 - 12,
                    text=f"{self.manual_distance_mm:.3f} mm",
                    fill="#FFFFFF", font=(FONT, 12, "bold"),
                )

    def _manual_zoom_wheel(self, event):
        if not self.manual_measurement_active:
            return "break"
        old_scale = self._manual_scale()
        image_x = (event.x - self.manual_offset[0]) / old_scale
        image_y = (event.y - self.manual_offset[1]) / old_scale
        factor = 1.25 if event.delta > 0 else 0.8
        self.manual_zoom = min(20.0, max(1.0, self.manual_zoom * factor))
        new_scale = self._manual_scale()
        self.manual_offset = [
            event.x - image_x * new_scale,
            event.y - image_y * new_scale,
        ]
        self._render_manual_measurement()
        return "break"

    def _manual_pan_start(self, event):
        if self.manual_measurement_active:
            self.manual_pan_anchor = (
                event.x, event.y, self.manual_offset[0], self.manual_offset[1],
            )
        return "break"

    def _manual_pan_move(self, event):
        if not self.manual_measurement_active or self.manual_pan_anchor is None:
            return "break"
        start_x, start_y, offset_x, offset_y = self.manual_pan_anchor
        self.manual_offset = [
            offset_x + event.x - start_x,
            offset_y + event.y - start_y,
        ]
        self._render_manual_measurement()
        return "break"

    def _manual_select_point(self, event):
        if not self.manual_measurement_active or self.manual_image_pil is None:
            return "break"
        scale = self._manual_scale()
        image_x = (event.x - self.manual_offset[0]) / scale
        image_y = (event.y - self.manual_offset[1]) / scale
        image_width, image_height = self.manual_image_pil.size
        if not (0 <= image_x < image_width and 0 <= image_y < image_height):
            return "break"
        if len(self.manual_points) >= 2:
            self.manual_points = []
            self.manual_distance_mm = None
        self.manual_points.append((float(image_x), float(image_y)))
        if len(self.manual_points) == 2:
            self._calculate_manual_distance()
        else:
            self._set_check_report(
                f"Point 1 selected at ({image_x:.1f}, {image_y:.1f}) px.\n\n"
                "Zoom and pan if needed, then select Point 2."
            )
        self._render_manual_measurement()
        return "break"

    def _manual_pixels_to_plane_mm(self, points):
        inverse_camera = np.linalg.inv(self.manual_camera_matrix)
        rotation, _ = cv2.Rodrigues(self.manual_plane_rvec)
        normal = rotation[:, 2]
        translation = self.manual_plane_tvec.reshape(3)
        plane_d = -float(normal.dot(translation))
        results = []
        for image_x, image_y in points:
            ray = inverse_camera.dot(np.array([image_x, image_y, 1.0], dtype=np.float64))
            denominator = float(normal.dot(ray))
            if abs(denominator) < 1e-10:
                raise ValueError("Selected ray is parallel to the saved measurement plane")
            camera_point = (-plane_d / denominator) * ray
            plane_xy = rotation[:, :2].T.dot(camera_point - translation)
            results.append(plane_xy * 1000.0)
        return np.asarray(results, dtype=np.float64)

    def _calculate_manual_distance(self):
        try:
            world_points = self._manual_pixels_to_plane_mm(self.manual_points)
            self.manual_distance_mm = float(np.linalg.norm(world_points[1] - world_points[0]))
        except (cv2.error, np.linalg.LinAlgError, TypeError, ValueError) as error:
            self.manual_distance_mm = None
            self._set_check_report(f"Manual distance could not be calculated.\n\n{error}")
            return
        p1, p2 = self.manual_points
        pixel_distance = float(np.linalg.norm(np.subtract(p2, p1)))
        self._set_status(
            f"Manual distance: {self.manual_distance_mm:.3f} mm", GREEN,
        )
        self._set_check_report(
            "MANUAL TWO-POINT MEASUREMENT\n\n"
            f"Point 1:       ({p1[0]:.2f}, {p1[1]:.2f}) px\n"
            f"Point 2:       ({p2[0]:.2f}, {p2[1]:.2f}) px\n"
            f"Pixel distance: {pixel_distance:.3f} px\n"
            f"Linear distance: {self.manual_distance_mm:.3f} mm\n\n"
            "Saved measurement-plane calibration used directly. "
            "No additional board-thickness adjustment was applied."
        )

    def undo_manual_point(self):
        if not self.manual_measurement_active or not self.manual_points:
            return
        self.manual_points.pop()
        self.manual_distance_mm = None
        self._set_check_report(
            "Last point removed. Select the remaining measurement point."
        )
        self._render_manual_measurement()

    def reset_manual_points(self):
        if not self.manual_measurement_active:
            return
        self.manual_points = []
        self.manual_distance_mm = None
        self._set_check_report(
            "Points cleared. Left-click Point 1 and Point 2 on the captured image."
        )
        self._render_manual_measurement()

    def _hide_manual_measurement(self):
        if hasattr(self, 'manual_canvas'):
            self.manual_canvas.place_forget()
        if hasattr(self, 'video_label'):
            self.video_label.place(x=0, y=0, relwidth=1, relheight=1)
        self.manual_measurement_active = False
        self.manual_image_bgr = None
        self.manual_image_pil = None
        self.manual_points = []
        self.manual_distance_mm = None

    def resume_check_preview(self):
        self._hide_manual_measurement()
        super().resume_check_preview()

    def verify_saved_calibration(self):
        self._hide_manual_measurement()
        super().verify_saved_calibration()

    def verify_measurement_accuracy(self):
        self._hide_manual_measurement()
        super().verify_measurement_accuracy()

    def _show_check_frame(self, frame):
        self._hide_manual_measurement()
        super()._show_check_frame(frame)

    def _close_calibration_checks(self):
        self._hide_manual_measurement()
        self.check_preview_frozen = False
        self._set_check_preview_title("CAMERA PREVIEW", C["text_soft"])

    def open_calibration_checks(self):
        return


def main():
    root = tk.Tk()
    CalibrationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

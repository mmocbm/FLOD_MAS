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
except ImportError:
    from calibration_math import (
        coverage_cell, coverage_percent, estimate_planar_pose,
        next_coverage_cell, reprojection_metrics, scale_camera_matrix,
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
    def __init__(self, root, on_close=None, host=None, camera_provider=None):
        self.root = root
        self.host = host if host is not None else root
        self.on_close = on_close
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
            (3, "Check camera"),
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
        tk.Label(preview_header, text="CAMERA PREVIEW", bg=C["surface"], fg=C["text_soft"],
                 font=(FONT, 9, "bold")).pack(side=tk.LEFT)
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

    def _calibration_path(self, camera_index=None):
        index = self.camera_index if camera_index is None else camera_index
        return Path(project_path(camera_config(index)['calibration_file']))

    def _extrinsics_path(self, camera_index=None):
        index = self.camera_index if camera_index is None else camera_index
        return Path(project_path(camera_config(index)['extrinsics_file']))

    def _preferred_resolution(self):
        spec = camera_config(self.camera_index)
        return spec['width'], spec['height']

    def select_camera(self, index):
        if (self.starting_camera or self.processing_capture or
                self.stage in ("calibrating", "extrinsic_capturing")):
            return
        if self.camera_running:
            self.stop_camera()
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
        if (self.starting_camera or self.processing_capture or
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
        if (self.starting_camera or self.processing_capture or
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
        # Keep the last displayed capture visible while its board points are
        # processed. Board detection never runs in the live preview.
        if (not self.processing_capture and self.camera_running and
                self.cap is not None and self.cap.isOpened()):
            ok, frame = self.cap.read()
            if ok:
                with self.frame_lock:
                    self.current_frame = frame.copy()
                scale = min(1.0, CONFIG['preview']['max_width'] / frame.shape[1],
                            CONFIG['preview']['max_height'] / frame.shape[0])
                display = cv2.resize(frame, (max(1, int(frame.shape[1] * scale)),
                                             max(1, int(frame.shape[0] * scale))))
                self._draw_coverage_guide(display)
                self._draw_capture_history(display)
                self._show_frame(display)
        self.preview_job = self.root.after(CONFIG['preview']['interval_ms'], self.update_frame)

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
        if self.current_frame is None or self.stage != "capture" or self.processing_capture:
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
        with self.frame_lock:
            frame = self.current_frame.copy()
        self.stage = "extrinsic_capturing"
        self._set_status("Saving the measurement surface…", AMBER)
        self._refresh_stage_ui()
        threading.Thread(target=self._extrinsic_worker, args=(frame,), daemon=True).start()

    def _extrinsic_worker(self, frame):
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

            data = {
                "rvec": rvec.reshape(-1).tolist(),
                "tvec": tvec.reshape(-1).tolist(),
                "reprojection_error": metrics,
                "image_size": list(frame_size),
                "corner_count": int(len(object_points)),
                "reference_image": str(reference_path.relative_to(PROJECT_ROOT)),
                "coordinate_system": "ChArUco board plane; lengths stored in metres",
                "board": {
                    "squares_x": SQUARES_X, "squares_y": SQUARES_Y,
                    "square_length_mm": SQUARE_LENGTH_MM,
                    "marker_length_mm": MARKER_LENGTH_MM,
                    "dictionary": (CONFIG['two_board']['dictionary']
                                   if self.use_two_boards else CONFIG['board']['dictionary']),
                },
            }
            self._extrinsics_path().write_text(json.dumps(data, indent=2), encoding="utf-8")
            self.root.after(0, self._extrinsic_complete, frame, metrics, tvec)
        except Exception as error:
            self.root.after(0, self._extrinsic_failed, str(error))

    def _extrinsic_complete(self, annotated, metrics, tvec):
        self.stage = "complete"
        self.camera_running = False
        if self.cap is not None:
            if not self.camera_provider: self.cap.release()
            self.cap = None
        self.stop_btn.configure(state=tk.DISABLED)
        self.start_btn.configure(state=tk.DISABLED)
        self.resolution_label.configure(text=f"Camera {self.camera_index}  •  setup complete")
        self._set_status("Setup complete — keep the camera fixed", GREEN)
        self.log("Measurement surface saved successfully.")
        self.log("Camera setup is complete and ready to use.")
        self._show_frame(annotated)
        self._refresh_stage_ui()
        messagebox.showinfo(
            "Setup Complete",
            f"Camera {self.camera_index} setup is complete.\n\n"
            "Keep the camera and measurement surface fixed.",
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

    def on_closing(self):
        if self.closed:
            return
        if (self.starting_camera or self.processing_capture or
                self.stage in ("calibrating", "extrinsic_capturing")):
            self._set_status("Please wait for the current step to finish before going back", AMBER)
            return
        self.closed = True
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
        if self.on_close is not None:
            self.on_close()
        else:
            self.root.destroy()


def main():
    root = tk.Tk()
    CalibrationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

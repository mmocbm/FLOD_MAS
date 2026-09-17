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

try:
    from .calibration_math import estimate_planar_pose, reprojection_metrics, scale_camera_matrix
except ImportError:
    from calibration_math import estimate_planar_pose, reprojection_metrics, scale_camera_matrix


WINDOW_WIDTH = 1366
WINDOW_HEIGHT = 768
TITLE_BAR_HEIGHT = 38

BG = "#1e293b"
PANEL = "#334155"
TITLE_BG = "#0f172a"
CANVAS_BG = "#020617"
TEXT = "#f8fafc"
MUTED = "#94a3b8"
CYAN = "#22d3ee"
BLUE = "#3b82f6"
GREEN = "#10b981"
AMBER = "#f59e0b"
RED = "#f43f5e"
PURPLE = "#8b5cf6"

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
CAMERA_IDS = [c['index'] for c in CONFIG['cameras']]

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
        self.is_calibrating = False
        self.stage = "select"
        self.camera_matrix = None
        self.dist_coeffs = None
        self.calibration_image_size = None

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
        dictionary = cv2.aruco.getPredefinedDictionary(DICT_TYPE)
        self.board = cv2.aruco.CharucoBoard(
            (SQUARES_X, SQUARES_Y),
            SQUARE_LENGTH_MM / 1000.0,
            MARKER_LENGTH_MM / 1000.0,
            dictionary,
        )
        self.charuco_detector = cv2.aruco.CharucoDetector(
            self.board,
            cv2.aruco.CharucoParameters(),
            cv2.aruco.DetectorParameters(),
        )

    def _button(self, parent, text, command, color=BLUE, width=16, state=tk.NORMAL):
        return tk.Button(
            parent, text=text, command=command, bg=color, fg="white",
            activebackground=color, activeforeground="white",
            disabledforeground="#cbd5e1", relief=tk.FLAT, bd=0,
            width=width, height=2, cursor="hand2",
            font=("Segoe UI", 11, "bold"), state=state,
        )

    def _create_ui(self):
        self._create_title_bar()
        body = tk.Frame(self.host, bg=BG)
        body.pack(fill=tk.BOTH, expand=True, padx=14, pady=(10, 14))

        heading = tk.Frame(body, bg=BG)
        heading.pack(fill=tk.X, pady=(0, 10))
        tk.Label(heading, text="Camera Setup", bg=BG, fg=TEXT,
                 font=("Segoe UI", 24, "bold")).pack(side=tk.LEFT)
        tk.Label(
            heading,
            text="Simple guided setup for accurate camera measurement",
            bg=BG, fg=MUTED, font=("Segoe UI", 11),
        ).pack(side=tk.LEFT, padx=18, pady=(8, 0))

        stage_row = tk.Frame(body, bg=BG)
        stage_row.pack(fill=tk.X, pady=(0, 10))
        self.stage_labels = []
        for number, title in (
            (1, "Select camera"), (2, f"Capture {NUM_CAPTURES} photos"),
            (3, "Check camera"), (4, "Save surface"),
        ):
            label = tk.Label(
                stage_row, text=f" {number}  {title} ", bg=PANEL, fg=MUTED,
                font=("Segoe UI", 10, "bold"), padx=12, pady=7,
            )
            label.pack(side=tk.LEFT, padx=(0, 8))
            self.stage_labels.append(label)

        workspace = tk.Frame(body, bg=BG)
        workspace.pack(fill=tk.BOTH, expand=True)
        workspace.columnconfigure(0, weight=1, minsize=0)
        workspace.columnconfigure(1, weight=0, minsize=410)
        workspace.rowconfigure(0, weight=1)

        preview_card = tk.Frame(workspace, bg=PANEL, highlightbackground="#475569", highlightthickness=1)
        preview_card.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        preview_card.pack_propagate(False)
        preview_header = tk.Frame(preview_card, bg=PANEL)
        preview_header.pack(fill=tk.X, padx=12, pady=9)
        tk.Label(preview_header, text="LIVE CAMERA", bg=PANEL, fg=CYAN,
                 font=("Segoe UI", 11, "bold")).pack(side=tk.LEFT)
        self.resolution_label = tk.Label(
            preview_header, text="No camera connected", bg=PANEL, fg=MUTED,
            font=("Segoe UI", 10),
        )
        self.resolution_label.pack(side=tk.RIGHT)
        # The viewport owns the available space; image requests cannot resize it.
        viewport = tk.Frame(preview_card, bg=CANVAS_BG)
        viewport.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.video_label = tk.Label(
            viewport, text="Select a camera and start the live feed",
            bg=CANVAS_BG, fg=MUTED, font=("Segoe UI", 16), bd=0,
            highlightthickness=0, padx=0, pady=0,
        )
        self.video_label.place(x=0, y=0, relwidth=1, relheight=1)

        controls = tk.Frame(workspace, bg=PANEL, width=410)
        controls.grid(row=0, column=1, sticky="nsew")
        controls.pack_propagate(False)
        self.status_banner = tk.Label(
            controls, text="Select a camera below", bg=TITLE_BG, fg=CYAN,
            wraplength=360, justify=tk.LEFT, font=("Segoe UI", 12, "bold"),
            padx=14, pady=12,
        )
        self.status_banner.pack(fill=tk.X, padx=12, pady=12)

        camera_box = tk.Frame(controls, bg=PANEL)
        camera_box.pack(fill=tk.X, padx=12)
        tk.Label(camera_box, text="1. SELECT CAMERA", bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w")
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
        tk.Frame(controls, bg="#475569", height=1).pack(fill=tk.X, padx=12)

        capture_box = tk.Frame(controls, bg=PANEL)
        capture_box.pack(fill=tk.X, padx=12, pady=10)
        tk.Label(capture_box, text="2. CAMERA SETUP PHOTOS", bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self.capture_info = tk.Label(
            capture_box, text=f"0 / {NUM_CAPTURES} accepted", bg=PANEL, fg=TEXT,
            font=("Segoe UI", 15, "bold"),
        )
        self.capture_info.pack(anchor="w", pady=(4, 2))
        style = ttk.Style()
        style.theme_use("default")
        style.configure(
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
            text="Move and tilt the board after each photo. Show it in different parts of the camera view.",
            bg=PANEL, fg=MUTED, wraplength=370, justify=tk.LEFT,
            font=("Segoe UI", 9),
        ).pack(anchor="w", pady=(5, 0))

        plane_box = tk.Frame(controls, bg=TITLE_BG, highlightbackground=PURPLE, highlightthickness=1)
        plane_box.pack(fill=tk.X, padx=12, pady=(0, 10))
        tk.Label(plane_box, text="4. MEASUREMENT SURFACE", bg=TITLE_BG, fg=PURPLE,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=10, pady=(8, 2))
        tk.Label(
            plane_box, text="Place board flat on measurement surface",
            bg=TITLE_BG, fg=TEXT, font=("Segoe UI", 12, "bold"),
        ).pack(anchor="w", padx=10)
        tk.Label(
            plane_box,
            text="Keep the camera fixed. The board must be fully flat at the same height as the product.",
            bg=TITLE_BG, fg=MUTED, wraplength=360, justify=tk.LEFT,
            font=("Segoe UI", 9),
        ).pack(anchor="w", padx=10, pady=(3, 7))
        self.extrinsic_btn = self._button(
            plane_box, "SAVE MEASUREMENT SURFACE", self.capture_extrinsic_reference,
            PURPLE, 29, tk.DISABLED,
        )
        self.extrinsic_btn.pack(fill=tk.X, padx=10, pady=(0, 10))

        log_box = tk.Frame(controls, bg=PANEL)
        log_box.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))
        tk.Label(log_box, text="STATUS LOG", bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self.status_text = scrolledtext.ScrolledText(
            log_box, height=7, bg=TITLE_BG, fg="#cbd5e1",
            insertbackground="white", relief=tk.FLAT,
            font=("Consolas", 9), wrap=tk.WORD,
        )
        self.status_text.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
        self.status_text.configure(state=tk.DISABLED)
        self.log("Follow the four steps shown above.")
        self.log("Keep the board clear, flat and well lit.")

    def _create_title_bar(self):
        title_bar = tk.Frame(self.host, bg=TITLE_BG, height=TITLE_BAR_HEIGHT)
        title_bar.pack(fill=tk.X)
        title_bar.pack_propagate(False)
        if self.on_close is not None:
            tk.Button(
                title_bar, text="← Back to dashboard", command=self.on_closing,
                bg=TITLE_BG, fg=CYAN, relief=tk.FLAT, bd=0,
                padx=12, font=("Segoe UI", 11, "bold"),
            ).pack(side=tk.LEFT, fill=tk.Y)
        tk.Label(
            title_bar, text="Industrial Vision Dashboard  /  Camera Setup",
            bg=TITLE_BG, fg=MUTED, font=("Segoe UI", 11),
        ).pack(side=tk.LEFT, padx=10)
        tk.Button(
            title_bar, text="✕", command=self.on_closing, bg=TITLE_BG, fg="white",
            activebackground="#e11d48", relief=tk.FLAT, bd=0, padx=12,
            font=("Segoe UI", 12),
        ).pack(side=tk.RIGHT, fill=tk.Y)
        tk.Button(
            title_bar, text="—", command=self.minimize_window, bg=TITLE_BG, fg="white",
            activebackground="#475569", relief=tk.FLAT, bd=0, padx=12,
            font=("Segoe UI", 12),
        ).pack(side=tk.RIGHT, fill=tk.Y)
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
                label.configure(bg="#065f46", fg="#d1fae5")
            elif index == stage_index:
                label.configure(bg=PURPLE, fg="white")
            else:
                label.configure(bg=PANEL, fg=MUTED)
        self.capture_btn.configure(
            state=tk.NORMAL if self.camera_running and self.stage == "capture" else tk.DISABLED
        )
        self.extrinsic_btn.configure(
            state=tk.NORMAL if self.camera_running and self.stage == "extrinsic_ready" else tk.DISABLED
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
        if self.starting_camera or self.stage in ("calibrating", "extrinsic_capturing"):
            return
        if self.camera_running:
            self.stop_camera()
        self.camera_index = index
        self.stage = "select"
        self.captured_images = []
        self.camera_matrix = None
        self.dist_coeffs = None
        self.calibration_image_size = None
        self.capture_info.configure(text=f"0 / {NUM_CAPTURES} accepted")
        self.progress["value"] = 0
        self.btn_cam0.configure(relief=tk.SUNKEN if index == CAMERA_IDS[0] else tk.FLAT,
                                bg=GREEN if index == CAMERA_IDS[0] else "#047857")
        self.btn_cam1.configure(relief=tk.SUNKEN if index == CAMERA_IDS[1] else tk.FLAT,
                                bg=BLUE if index == CAMERA_IDS[1] else "#1d4ed8")
        self.start_btn.configure(state=tk.NORMAL)
        self._set_status(f"Camera {index} selected — start the live feed")
        self.log(f"Selected Camera {index}")
        self._refresh_stage_ui()

    def start_camera(self):
        if self.starting_camera or self.stage in ("calibrating", "extrinsic_capturing"):
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
        self._set_status(f"Take {NUM_CAPTURES} photos while moving the board around the camera view")
        self.log(
            f"Camera {self.camera_index} started at {self.actual_size[0]} × {self.actual_size[1]} "
            f"(requested {requested_width} × {requested_height})"
        )
        session_dir = TEMP_ROOT / f"camera_{self.camera_index}"
        session_dir.mkdir(parents=True, exist_ok=True)
        for old_image in session_dir.glob("calib_*.jpg"):
            old_image.unlink()
        self._refresh_stage_ui()

    def _camera_failed(self, error):
        self.starting_camera = False
        self.start_btn.configure(text="START CAMERA", state=tk.NORMAL)
        self._set_status("Camera could not be opened", RED)
        self.log(f"Camera error: {error}")
        messagebox.showerror("Camera Error", error)

    def stop_camera(self):
        if self.starting_camera or self.stage in ("calibrating", "extrinsic_capturing"):
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
        if self.camera_running and self.cap is not None and self.cap.isOpened():
            ok, frame = self.cap.read()
            if ok:
                with self.frame_lock:
                    self.current_frame = frame.copy()
                scale = min(1.0, CONFIG['preview']['max_width'] / frame.shape[1],
                            CONFIG['preview']['max_height'] / frame.shape[0])
                display = cv2.resize(frame, (max(1, int(frame.shape[1] * scale)),
                                             max(1, int(frame.shape[0] * scale))))
                gray = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)
                corners, ids, _, _ = self.charuco_detector.detectBoard(gray)
                corner_count = len(ids) if ids is not None else 0
                if ids is not None and corner_count:
                    cv2.aruco.drawDetectedCornersCharuco(display, corners, ids)
                board_ready = corner_count >= MIN_CORNERS
                color = GREEN if board_ready else AMBER
                cv2.putText(
                    display, "Board: READY" if board_ready else "Board: NOT CLEAR", (28, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.15, self._hex_to_bgr(color),
                    3, cv2.LINE_AA,
                )
                self._show_frame(display)
        self.preview_job = self.root.after(CONFIG['preview']['interval_ms'], self.update_frame)

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
        if self.current_frame is None or self.stage != "capture":
            return
        with self.frame_lock:
            frame = self.current_frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _, _ = self.charuco_detector.detectBoard(gray)
        corner_count = len(ids) if ids is not None else 0
        if corner_count < MIN_CORNERS:
            self._set_status("Photo not accepted — make the board clearer", RED)
            self.log("Photo not accepted. Show more of the board and improve the lighting.")
            return

        session_dir = TEMP_ROOT / f"camera_{self.camera_index}"
        file_path = session_dir / f"calib_{len(self.captured_images):03d}.jpg"
        if not cv2.imwrite(str(file_path), frame):
            messagebox.showerror("Save Error", f"Could not save {file_path}")
            return
        self.captured_images.append(file_path)
        count = len(self.captured_images)
        self.capture_info.configure(text=f"{count} / {NUM_CAPTURES} accepted")
        self.progress["value"] = count
        self._set_status(f"View {count} accepted — move the board before the next capture", GREEN)
        self.log(f"Photo {count}/{NUM_CAPTURES} accepted")
        if count >= NUM_CAPTURES:
            self.stage = "calibrating"
            self._set_status("Checking the camera setup…", AMBER)
            self.log("All photos captured. Checking camera setup now.")
            self._refresh_stage_ui()
            threading.Thread(target=self._calibrate_intrinsics_worker, daemon=True).start()

    def _calibrate_intrinsics_worker(self):
        self.is_calibrating = True
        try:
            object_views, image_views = [], []
            image_size = None
            for number, path in enumerate(self.captured_images, start=1):
                image = cv2.imread(str(path))
                if image is None:
                    continue
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                corners, ids, _, _ = self.charuco_detector.detectBoard(gray)
                if corners is None or ids is None or len(ids) < MIN_CORNERS:
                    continue
                object_points, image_points = self.board.matchImagePoints(corners, ids)
                if object_points is None or image_points is None or len(object_points) < MIN_CORNERS:
                    continue
                current_size = gray.shape[::-1]
                if image_size is None:
                    image_size = current_size
                if current_size != image_size:
                    raise RuntimeError("All setup photos must use the same camera size")
                object_views.append(np.asarray(object_points, dtype=np.float32))
                image_views.append(np.asarray(image_points, dtype=np.float32))
                self.root.after(0, self.log, f"Checking photo {number}/{NUM_CAPTURES}")

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
                "board": {
                    "squares_x": SQUARES_X, "squares_y": SQUARES_Y,
                    "square_length_mm": SQUARE_LENGTH_MM,
                    "marker_length_mm": MARKER_LENGTH_MM,
                    "dictionary": CONFIG['board']['dictionary'],
                },
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

    def _intrinsic_complete(self, camera_matrix, dist_coeffs, image_size, rms, valid_views):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.calibration_image_size = image_size
        self.stage = "extrinsic_ready"
        self._set_status(
            "Camera setup complete. Place the board flat on the measurement surface.",
            GREEN if rms <= MAX_INTRINSIC_RMS_PX else AMBER,
        )
        self.log(f"Camera setup saved using {valid_views} clear photos")
        if rms > MAX_INTRINSIC_RMS_PX:
            self.log("Quality warning: results may improve with clearer photos and better lighting.")
        self.log("NEXT: Place the board flat on the final measurement surface.")
        self.log("Do not move the camera after saving the measurement surface.")
        self._refresh_stage_ui()

    def _calibration_failed(self, error):
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
        if self.stage != "extrinsic_ready" or self.current_frame is None:
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
            annotated = frame.copy()
            cv2.aruco.drawDetectedCornersCharuco(annotated, corners, ids)
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
                    "dictionary": CONFIG['board']['dictionary'],
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
        if self.starting_camera or self.stage in ("calibrating", "extrinsic_capturing"):
            self._set_status("Please wait for the current step to finish before going back", AMBER)
            return
        self.closed = True
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

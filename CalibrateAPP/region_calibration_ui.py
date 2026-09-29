"""Operator page for the optional per-region homography calibration mode.

This page is additive. The global lens calibration and measurement-plane workflow in
:class:`~CalibrateAPP.calibration_ui.CalibrationApp` is untouched and remains the only
thing that writes ``camera_calibration_<n>.json`` and ``camera_extrinsics_<n>.json``;
this page writes a separate per-region store and nothing else.

What the operator sees is deliberately not the dashboard's live camera preview. The two
regions being calibrated are narrow strips near the top and bottom of a portrait frame,
so in a 3456 x 4608 image a board laid in one of them is a small part of the whole, and
at any preview size that fits on screen the operator cannot see whether the card is
square to the strip. Showing the deskewed crop instead puts the region on screen at its
full 2208 x 552, with the detected corners drawn on it.

Reaching this page at all requires both crop regions to have been marked for the selected
camera: a local homography is defined on the crop raster, so without a crop there is
nothing for it to map.

Two frames are in play here and they are not interchangeable. The *raw* frame is what the
camera delivers; the *undistorted* frame is that frame put through the saved lens model,
and it is the one the saved crop regions were marked on and the one
:func:`crop_processing.extract_rotated_crop` deskews. Detection, the fit and the plane
alignment all work in undistorted-frame pixels, so this page undistorts on every preview
tick rather than only at capture -- which also means the crop on screen is exactly the
raster the homography will be fitted to.
"""

from __future__ import annotations

import json
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
import numpy as np

from app_config import CONFIG, project_path
from crop_processing import extract_rotated_crop, load_crop_store
from measure.makeUndistored import ImageUndistorter
import plane_scale
import region_calibration

try:
    from .calibration_math import scale_camera_matrix
    from .calibration_ui import (
        AMBER, BG, BLUE, CANVAS_BG, CYAN, GREEN, MUTED, PANEL, PURPLE, RED, TEXT,
        TITLE_BAR_HEIGHT, CalibrationApp, C,
    )
except ImportError:
    from calibration_math import scale_camera_matrix
    from calibration_ui import (
        AMBER, BG, BLUE, CANVAS_BG, CYAN, GREEN, MUTED, PANEL, PURPLE, RED, TEXT,
        TITLE_BAR_HEIGHT, CalibrationApp, C,
    )

from ui_theme import (
    FONT, button as themed_button, card as themed_card, section_label,
    set_button_role, status_dot,
)


CROP_RATIO = float(CONFIG['crop_setup']['aspect_ratio'][0]) / float(
    CONFIG['crop_setup']['aspect_ratio'][1])
CROP_OUTPUT_SIZE = tuple(CONFIG['crop_setup']['output_size'])

REGION_SETTINGS = CONFIG.get('region_homography', {})

REGION_NAMES = {1: "REGION 01", 2: "REGION 02"}
CAMERA_IDS = [camera['index'] for camera in CONFIG['cameras']]

# Every dictionary OpenCV exposes, so a board bought later can be typed in rather than
# waiting for a code change.
DICTIONARIES = sorted(
    name for name in dir(cv2.aruco)
    if name.startswith("DICT_") and isinstance(getattr(cv2.aruco, name), int)
)


def region_store_path():
    return project_path(REGION_SETTINGS.get("store_file",
                                            "Files/region_homographies.json"))


def board_profiles_path():
    return project_path(REGION_SETTINGS.get("board_profiles_file",
                                            "Files/board_definitions.json"))


def crop_definitions_path():
    return project_path(CONFIG['crop_setup']['definitions_file'])


def scale_to_viewport(image, size):
    """Fit an image inside a viewport without changing its aspect ratio.

    Shrinking uses ``INTER_AREA`` because the crop is being averaged down by a large
    factor and nearest or linear sampling would alias the board's own edges, which are
    the thing the operator is looking at.
    """
    if size is None:
        return None
    width = max(2, int(size[0]))
    height = max(2, int(size[1]))
    scale = min(1.0, width / float(image.shape[1]), height / float(image.shape[0]))
    if scale >= 1.0:
        return image
    return cv2.resize(image, (max(1, int(image.shape[1] * scale)),
                             max(1, int(image.shape[0] * scale))),
                      interpolation=cv2.INTER_AREA)


class RegionCalibrationApp(CalibrationApp):
    """Fit one local homography per crop region from a board laid flat inside it."""

    def __init__(self, root, on_close=None, host=None, camera_provider=None,
                 on_open_setup=None, on_open_checks=None):
        self.on_open_setup = on_open_setup
        self.crop_definitions = load_crop_store(crop_definitions_path())
        self.profiles = region_calibration.load_board_profiles(board_profiles_path())
        self.profile_index = self._default_profile_index()
        self.region_index = 1
        self.crop_ready = False
        self.undistorter = None
        # The fit awaiting a SAVE, with everything that has to be stored beside it.
        self.pending = None
        self.capture_busy = False
        self.refinement_views = []
        self.region_status = {1: None, 2: None}
        super().__init__(
            root, on_close=on_close, host=host, camera_provider=camera_provider,
            on_open_checks=on_open_checks,
        )

    # -- configuration ------------------------------------------------------------------

    def _default_profile_index(self):
        wanted = REGION_SETTINGS.get("default_profile")
        for index, profile in enumerate(self.profiles):
            if profile.name == wanted:
                return index
        return 0

    def _starting_profile(self):
        if self.profiles:
            return self.profiles[self.profile_index]
        return region_calibration.REGION_BOARD_PROFILE

    def current_profile(self):
        """The board as the entry fields currently describe it.

        Built from the fields rather than read back from the store, because the operator
        is allowed to edit a profile in place and capture with it before saving it.
        """
        return region_calibration.BoardDefinition(
            name=self.profile_var.get().strip() or self._starting_profile().name,
            dictionary=self.dictionary_var.get().strip(),
            squares_x=self._whole(self.squares_x_var.get()),
            squares_y=self._whole(self.squares_y_var.get()),
            square_length_mm=self._positive(self.square_var.get()),
            marker_length_mm=self._positive(self.marker_var.get()),
        )

    @staticmethod
    def _whole(text):
        try:
            return int(str(text).strip())
        except ValueError:
            return 0

    @staticmethod
    def _positive(text):
        try:
            return float(str(text).strip())
        except ValueError:
            return 0.0

    def _crop_for(self, region_index):
        """The saved definition for one region, or None when it has not been marked."""
        try:
            definitions = self.crop_definitions['cameras'][str(self.camera_number())]
            definition = definitions[region_index - 1]
        except (KeyError, IndexError, TypeError):
            return None
        return definition if isinstance(definition, dict) else None

    def camera_number(self, camera_index=None):
        """1 or 2 by position, which is how the rest of the app names the sides.

        The ``index`` in config is the device's capture index, not the side's number, so
        matching on it would hand camera 1 the other camera's calibration.
        """
        index = self.camera_index if camera_index is None else camera_index
        for position, camera in enumerate(CONFIG['cameras'], start=1):
            if camera['index'] == index:
                return position
        return 1

    def _calibration_file(self):
        return region_calibration.camera_paths(self.camera_number())[0]

    def _extrinsics_file(self):
        return region_calibration.camera_paths(self.camera_number())[1]

    @staticmethod
    def _read_json(path):
        if not path:
            return None
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return None

    def _refresh_gate(self):
        """Everything on this page is unusable until both crop regions exist."""
        self.crop_ready = all(self._crop_for(index) is not None for index in (1, 2))
        if self.crop_ready:
            self.gate_banner.pack_forget()
        else:
            missing = [REGION_NAMES[i] for i in (1, 2) if self._crop_for(i) is None]
            self.gate_banner.configure(
                text=("Mark both crop regions for this camera before calibrating them. "
                      f"Missing: {', '.join(missing)}.   "
                      "Settings → VIEW SIZE → CROP SETUP"))
            self.gate_banner.pack(fill=tk.X, pady=(0, 10), before=self.workspace)
        self._refresh_controls()

    # -- UI ------------------------------------------------------------------------------

    def _create_title_bar(self):
        title_bar = tk.Frame(
            self.host, bg=C["surface"], height=TITLE_BAR_HEIGHT,
            highlightbackground=C["border"], highlightthickness=1,
        )
        title_bar.pack(fill=tk.X)
        title_bar.pack_propagate(False)
        if self.on_close is not None:
            themed_button(title_bar, "←  DASHBOARD", self.on_closing,
                          role="quiet", padx=16, pady=6).pack(side=tk.LEFT, fill=tk.Y)
        if self.on_open_setup is not None:
            themed_button(title_bar, "CAMERA SETUP", self.on_open_setup,
                          role="quiet", padx=16, pady=6).pack(side=tk.LEFT, fill=tk.Y)
        tk.Label(title_bar, text="REGION CALIBRATION", bg=C["surface"], fg=BLUE,
                 font=(FONT, 10, "bold")).pack(side=tk.LEFT, padx=12)
        if self.on_open_checks is not None:
            themed_button(title_bar, "CALIBRATION CHECK", self.on_open_checks,
                          role="quiet", padx=16, pady=6).pack(side=tk.LEFT, fill=tk.Y)
        themed_button(title_bar, "✕", self.on_closing, role="quiet",
                      padx=16, pady=6, font_size=12).pack(side=tk.RIGHT, fill=tk.Y)
        themed_button(title_bar, "—", self.minimize_window, role="quiet",
                      padx=16, pady=6, font_size=12).pack(side=tk.RIGHT, fill=tk.Y)
        title_bar.bind("<ButtonPress-1>", self._start_move)
        title_bar.bind("<B1-Motion>", self._do_move)

    def _create_ui(self):
        self._create_title_bar()
        body = tk.Frame(self.host, bg=BG)
        body.pack(fill=tk.BOTH, expand=True, padx=18, pady=(14, 16))

        heading = tk.Frame(body, bg=BG)
        heading.pack(fill=tk.X, pady=(0, 10))
        tk.Label(heading, text="Region Calibration", bg=BG, fg=TEXT,
                 font=(FONT, 26, "bold")).pack(side=tk.LEFT)
        tk.Label(heading,
                 text="One local homography per crop region, fitted from a board laid "
                      "flat inside it",
                 bg=BG, fg=MUTED, font=(FONT, 10)).pack(side=tk.LEFT, padx=16,
                                                        pady=(9, 0))

        self.gate_banner = tk.Label(
            body, bg=C["surface_2"], fg=AMBER, font=(FONT, 10, "bold"),
            justify=tk.LEFT, wraplength=1240, padx=14, pady=10,
        )

        self.workspace = tk.Frame(body, bg=BG)
        self.workspace.pack(fill=tk.BOTH, expand=True)
        self.workspace.columnconfigure(0, weight=1, minsize=0)
        self.workspace.columnconfigure(1, weight=0, minsize=410)
        self.workspace.rowconfigure(0, weight=1)

        self._create_preview(self.workspace)
        self._create_controls(self.workspace)
        self._refresh_gate()

    def _create_preview(self, parent):
        preview_card = themed_card(parent, bg=C["surface"])
        preview_card.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        preview_card.pack_propagate(False)
        preview_header = tk.Frame(preview_card, bg=C["surface"])
        preview_header.pack(fill=tk.X, padx=14, pady=10)
        status_dot(preview_header).pack(side=tk.LEFT, padx=(0, 7))
        self.preview_title = tk.Label(
            preview_header, text=f"DESKEWED CROP — {REGION_NAMES[self.region_index]}",
            bg=C["surface"], fg=C["text_soft"], font=(FONT, 9, "bold"),
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

    def _create_controls(self, parent):
        controls_shell = themed_card(parent, bg=PANEL, width=410)
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

        self._create_camera_section(controls)
        self._create_board_section(controls)
        self._create_region_section(controls)
        self._create_capture_section(controls)

    def _create_camera_section(self, parent):
        section_label(parent, "Camera").pack(anchor="w", padx=12, pady=(14, 0))
        cam_buttons = tk.Frame(parent, bg=PANEL)
        cam_buttons.pack(fill=tk.X, padx=12, pady=6)
        self.btn_cam0 = self._button(
            cam_buttons, f"CAMERA {CAMERA_IDS[0]}",
            lambda: self.select_camera(CAMERA_IDS[0]), GREEN, 13)
        self.btn_cam0.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))
        self.btn_cam1 = self._button(
            cam_buttons, f"CAMERA {CAMERA_IDS[1]}",
            lambda: self.select_camera(CAMERA_IDS[1]), BLUE, 13)
        self.btn_cam1.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(4, 0))
        actions = tk.Frame(parent, bg=PANEL)
        actions.pack(fill=tk.X, padx=12, pady=(0, 8))
        self.start_btn = self._button(actions, "START CAMERA", self.start_camera,
                                      GREEN, 13)
        self.start_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))
        self.stop_btn = self._button(actions, "STOP", self.stop_camera, RED, 8,
                                     state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=(4, 0))
        self.status_banner = tk.Label(
            parent, text="Select a camera to begin", bg=C["surface_2"], fg=CYAN,
            wraplength=360, justify=tk.LEFT, font=(FONT, 11, "bold"), padx=14, pady=12,
        )
        self.status_banner.pack(fill=tk.X, padx=12, pady=(0, 8))
        self.status_text = tk.Text(
            parent, height=7, bg=C["surface"], fg=C["text_soft"], font=(FONT, 8),
            bd=0, wrap=tk.WORD, state=tk.DISABLED, highlightthickness=0,
        )
        self.status_text.pack(fill=tk.X, padx=12, pady=(0, 4))

    def _create_board_section(self, parent):
        section_label(parent, "Board").pack(anchor="w", padx=12, pady=(12, 0))
        profile = self._starting_profile()

        tk.Label(parent, text="Saved profile", bg=PANEL, fg=MUTED,
                 font=(FONT, 9)).pack(anchor="w", padx=12, pady=(6, 0))
        self.profile_var = tk.StringVar(value=profile.name)
        self.profile_box = ttk.Combobox(
            parent, textvariable=self.profile_var, state="readonly",
            values=[entry.name for entry in self.profiles])
        self.profile_box.pack(fill=tk.X, padx=12, pady=(2, 6))
        self.profile_box.bind("<<ComboboxSelected>>", self._profile_selected)

        self.dictionary_var = tk.StringVar(value=profile.dictionary)
        self.squares_x_var = tk.StringVar(value=str(profile.squares_x))
        self.squares_y_var = tk.StringVar(value=str(profile.squares_y))
        self.square_var = tk.StringVar(value=f"{profile.square_length_mm:g}")
        self.marker_var = tk.StringVar(value=f"{profile.marker_length_mm:g}")

        tk.Label(parent, text="Dictionary", bg=PANEL, fg=MUTED,
                 font=(FONT, 9)).pack(anchor="w", padx=12)
        ttk.Combobox(parent, textvariable=self.dictionary_var,
                     values=DICTIONARIES).pack(fill=tk.X, padx=12, pady=(2, 6))

        grid = tk.Frame(parent, bg=PANEL)
        grid.pack(fill=tk.X, padx=12)
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)
        fields = (("Squares X", self.squares_x_var), ("Squares Y", self.squares_y_var),
                  ("Checker mm", self.square_var), ("Marker mm", self.marker_var))
        for position, (label, variable) in enumerate(fields):
            cell = tk.Frame(grid, bg=PANEL)
            cell.grid(row=position // 2, column=position % 2, sticky="ew", pady=3,
                      padx=(0, 6) if position % 2 == 0 else (6, 0))
            tk.Label(cell, text=label, bg=PANEL, fg=MUTED,
                     font=(FONT, 8)).pack(anchor="w")
            tk.Entry(cell, textvariable=variable, bg=C["surface"], fg=TEXT, bd=0,
                     insertbackground=TEXT, font=(FONT, 10),
                     highlightthickness=0).pack(fill=tk.X, ipady=4)
        # Edited fields must re-run the footprint advisory as they are typed, not only when
        # a profile is picked, or the operator sizes a board against a stale number.
        for variable in (self.squares_x_var, self.squares_y_var, self.square_var,
                         self.marker_var, self.dictionary_var):
            variable.trace_add("write", lambda *_: self._refresh_fit_advisory())

        self.fit_advisory = tk.Label(parent, text="", bg=PANEL, fg=MUTED,
                                     font=(FONT, 9), justify=tk.LEFT, wraplength=360)
        self.fit_advisory.pack(fill=tk.X, padx=12, pady=(8, 0))
        buttons = tk.Frame(parent, bg=PANEL)
        buttons.pack(fill=tk.X, padx=12, pady=(8, 0))
        self._button(buttons, "SAVE AS PROFILE", self.save_profile, BLUE, 14).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))
        self.shrink_btn = self._button(buttons, "SHRINK TO FIT", self.shrink_to_fit,
                                       PURPLE, 14)
        self.shrink_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(4, 0))

    def _create_region_section(self, parent):
        section_label(parent, "Region — one at a time").pack(anchor="w", padx=12,
                                                            pady=(12, 0))
        row = tk.Frame(parent, bg=PANEL)
        row.pack(fill=tk.X, padx=12, pady=6)
        self.region_buttons = {}
        for index in (1, 2):
            button = self._button(row, REGION_NAMES[index],
                                  lambda i=index: self.select_region(i), BLUE, 13)
            button.pack(side=tk.LEFT, expand=True, fill=tk.X,
                        padx=(0, 4) if index == 1 else (4, 0))
            self.region_buttons[index] = button
        self.region_state = tk.Label(parent, text="", bg=C["surface_2"], fg=MUTED,
                                     font=(FONT, 9), justify=tk.LEFT, wraplength=360,
                                     padx=12, pady=8)
        self.region_state.pack(fill=tk.X, padx=12, pady=(2, 0))

    def _create_capture_section(self, parent):
        section_label(parent, "Capture & fit").pack(anchor="w", padx=12, pady=(12, 0))
        tk.Label(parent,
                 text="Lay the board flat inside this region, filling it as completely as "
                      "you can, then capture.",
                 bg=PANEL, fg=MUTED, font=(FONT, 9), justify=tk.LEFT,
                 wraplength=360).pack(anchor="w", padx=12, pady=(4, 0))
        self.capture_btn = self._button(parent, "CAPTURE & FIT", self.capture_and_fit,
                                        PURPLE, 16, state=tk.DISABLED)
        self.capture_btn.pack(fill=tk.X, padx=12, pady=(8, 0))

        self.refine_var = tk.BooleanVar(
            value=bool(REGION_SETTINGS.get("refine_intrinsics", False)))
        self.refine_check = tk.Checkbutton(
            parent, text="Also refine intrinsics (needs tilted views)",
            variable=self.refine_var, bg=PANEL, fg=MUTED, font=(FONT, 9),
            activebackground=PANEL, activeforeground=TEXT,
            selectcolor=C["surface"], anchor="w", highlightthickness=0, bd=0,
        )
        self.refine_check.pack(fill=tk.X, padx=8, pady=(8, 0))
        self.refine_note = tk.Label(
            parent, text="", bg=PANEL, fg=MUTED, font=(FONT, 8), justify=tk.LEFT,
            wraplength=360,
        )
        self.refine_note.pack(fill=tk.X, padx=12)

        self.report = tk.Text(
            parent, height=16, bg=C["surface"], fg=C["text_soft"], font=(FONT, 8),
            bd=0, wrap=tk.WORD, state=tk.DISABLED, highlightthickness=0,
        )
        self.report.pack(fill=tk.BOTH, expand=True, padx=12, pady=(8, 0))
        self.save_btn = self._button(parent, "SAVE REGION HOMOGRAPHY", self.save_region,
                                     GREEN, 20, state=tk.DISABLED)
        self.save_btn.pack(fill=tk.X, padx=12, pady=(8, 14))

    # -- refresh -------------------------------------------------------------------------

    def _refresh_controls(self):
        if not hasattr(self, 'capture_btn'):
            return
        can_capture = (self.camera_running and not self.capture_busy and self.crop_ready
                       and self.undistorter is not None)
        self.capture_btn.configure(state=tk.NORMAL if can_capture else tk.DISABLED)
        self.save_btn.configure(
            state=tk.NORMAL if self.pending is not None and not self.capture_busy
            else tk.DISABLED)
        for index, button in self.region_buttons.items():
            set_button_role(button,
                            "selected" if index == self.region_index else "secondary")
        self.shrink_btn.configure(
            state=tk.NORMAL if self._region_size_mm() is not None else tk.DISABLED)
        self.refine_note.configure(text=self._refinement_note())
        self._refresh_fit_advisory()
        self._refresh_region_state()

    def _refinement_note(self):
        minimum = int(REGION_SETTINGS.get("minimum_refinement_views", 5))
        if not self.refine_var.get():
            return (f"Refinement needs {minimum} views at different tilts; "
                    f"{len(self.refinement_views)} collected. Leave this off unless you "
                    "re-shoot the region at several angles.")
        return (f"{len(self.refinement_views)} view(s) collected for this region; "
                f"{minimum} are needed before anything is refined. Tilt the board "
                "differently for each capture.")

    def _refresh_fit_advisory(self):
        if not hasattr(self, 'fit_advisory'):
            return
        try:
            profile = self.current_profile()
        except (IndexError, ValueError, TypeError):
            self.fit_advisory.configure(text="", fg=MUTED)
            return
        report = region_calibration.board_fit_report(profile, self._region_size_mm())
        self.fit_advisory.configure(text=report.summary(),
                                    fg=MUTED if report.fits else RED)

    def _refresh_region_state(self):
        if not hasattr(self, 'region_state'):
            return
        lines = []
        for index in (1, 2):
            label = REGION_NAMES[index]
            if self._crop_for(index) is None:
                lines.append(f"{label}: no crop region marked")
                continue
            entry = self.region_status.get(index)
            if entry is None:
                lines.append(f"{label}: not calibrated")
                continue
            metrics = entry.get("metrics", {}) or {}
            alignment = entry.get("plane_alignment") or {}
            lines.append(
                f"{label}: RMS {metrics.get('rms_mm', float('nan')):.3f} mm, "
                f"{metrics.get('corners', 0)} corners, "
                f"tilt {alignment.get('tilt_degrees', float('nan')):.2f}°")
        self.region_state.configure(text="\n".join(lines))

    def _region_size_mm(self):
        """The region's extent on the measurement plane, or None when it cannot be known.

        Advisory only. It needs a saved extrinsics plane and the frame size the crop comes
        from; without either the extent is simply unknown, and the fit measures it
        afterwards anyway.
        """
        definition = self._crop_for(self.region_index)
        frame_size = self._undistorted_size()
        if definition is None or frame_size is None:
            return None
        return region_calibration.region_size_mm(
            definition, frame_size, CROP_OUTPUT_SIZE, CROP_RATIO,
            calibration=self._read_json(self._calibration_file()),
            extrinsics=self._read_json(self._extrinsics_file()),
        )

    def _undistorted_size(self):
        """The size of the frame the crops are taken from.

        The undistorted frame keeps the stream's size; the calibration's own recorded size
        is only where the intrinsics were measured, and they are scaled between the two.
        """
        if self.undistorter is None:
            return None
        size = getattr(self.cap, 'size', None) or self.undistorter.calibration_image_size
        if not size:
            return None
        return (int(size[0]), int(size[1]))

    def _scaled_matrix(self, frame_size):
        """The saved intrinsics, scaled to the frame the crop is actually taken from."""
        if self.camera_matrix is None or self.calibration_image_size is None:
            return None
        try:
            return scale_camera_matrix(self.camera_matrix, self.calibration_image_size,
                                       frame_size)
        except (ValueError, TypeError):
            return None

    def _plane_mm_per_pixel(self, definition, frame_size):
        """The global plane's millimetres per pixel, or None when there is no plane."""
        try:
            return plane_scale.load_plane_scale(
                self.camera_number(), definition, frame_size, CROP_OUTPUT_SIZE,
                ratio=CROP_RATIO,
            ).mm_per_pixel
        except (plane_scale.PlaneScaleError, KeyError, IndexError, TypeError):
            return None

    # -- actions -------------------------------------------------------------------------

    def _profile_selected(self, _event=None):
        name = self.profile_var.get()
        for index, profile in enumerate(self.profiles):
            if profile.name == name:
                self.profile_index = index
                self.dictionary_var.set(profile.dictionary)
                self.squares_x_var.set(str(profile.squares_x))
                self.squares_y_var.set(str(profile.squares_y))
                self.square_var.set(f"{profile.square_length_mm:g}")
                self.marker_var.set(f"{profile.marker_length_mm:g}")
                break
        self._refresh_fit_advisory()

    def save_profile(self):
        try:
            profile = self.current_profile()
            profile.validate()
        except ValueError as error:
            messagebox.showerror("Board profile", str(error))
            return
        self.profiles = region_calibration.upsert_board_profile(self.profiles, profile)
        region_calibration.save_board_profiles(board_profiles_path(), self.profiles)
        self.profile_index = next(
            index for index, entry in enumerate(self.profiles)
            if entry.name == profile.name)
        self.profile_var.set(profile.name)
        self.profile_box.configure(values=[entry.name for entry in self.profiles])
        self.log(f"Saved board profile '{profile.name}'")
        self._set_status(f"Board profile '{profile.name}' saved", GREEN)
        self._refresh_fit_advisory()

    def shrink_to_fit(self):
        """Set the largest whole-tenth-millimetre square whose board fits the region."""
        region = self._region_size_mm()
        if region is None:
            messagebox.showinfo(
                "Shrink to fit",
                "The region's size on the measurement plane is not known yet, which "
                "needs a saved measurement surface. Capture a board once and the fit "
                "will report the region's measured extent.")
            return
        try:
            profile = self.current_profile()
        except (IndexError, ValueError, TypeError):
            return
        if profile.squares_x <= 0 or profile.squares_y <= 0 or not profile.square_length_mm:
            return
        smaller = region_calibration.largest_fitting_square_mm(
            profile.squares_x, profile.squares_y, region,
            profile.marker_length_mm / float(profile.square_length_mm))
        if smaller is None:
            return
        self.square_var.set(f"{smaller[0]:g}")
        self.marker_var.set(f"{smaller[1]:g}")
        self.log(f"Board set to {profile.squares_x}x{profile.squares_y} at "
                 f"{smaller[0]:g} mm squares with {smaller[1]:g} mm markers, to fit a "
                 f"region of {region[0]:.0f} x {region[1]:.0f} mm")

    def select_region(self, region_index):
        if self.capture_busy:
            return
        self.region_index = region_index
        self.pending = None
        # Views belong to one region: a homography is local, so intrinsics collected
        # over region 1 say nothing about region 2's corner of the frame.
        self.refinement_views = []
        self.preview_title.configure(
            text=f"DESKEWED CROP — {REGION_NAMES[region_index]}")
        self._set_report(
            f"{REGION_NAMES[region_index]} selected.\n\n"
            "Lay the board flat inside the region and press CAPTURE & FIT.")
        self._refresh_controls()

    def _set_report(self, text):
        self.report.configure(state=tk.NORMAL)
        self.report.delete("1.0", tk.END)
        self.report.insert(tk.END, text)
        self.report.configure(state=tk.DISABLED)

    def capture_and_fit(self):
        if self.capture_busy or not self.crop_ready:
            return
        if self.current_frame is None or self.undistorter is None:
            return
        definition = self._crop_for(self.region_index)
        if definition is None:
            return
        try:
            profile = self.current_profile()
            profile.validate()
        except ValueError as error:
            messagebox.showerror("Board profile", str(error))
            return
        with self.frame_lock:
            frame = self.current_frame.copy()
        self.capture_busy = True
        # The base preview loop reads this flag and stops replacing the screen, so the
        # annotated result stays up while the operator reads it.
        self.processing_capture = True
        self._set_status("Fitting the region homography…", AMBER)
        self._refresh_controls()
        threading.Thread(
            target=self._fit_worker,
            args=(frame, definition, profile, self.region_index),
            daemon=True).start()

    def _fit_worker(self, frame, definition, profile, region_index):
        """No Tk calls here."""
        try:
            frame_size = (frame.shape[1], frame.shape[0])
            gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detection = region_calibration.detect_board_in_region(
                gray, profile, definition, frame_size, CROP_OUTPUT_SIZE, CROP_RATIO,
                int(REGION_SETTINGS.get("minimum_corners", 8)))
            camera_matrix = self._scaled_matrix(frame_size)
            fit = region_calibration.fit_region(detection, profile, camera_matrix,
                                                frame_size, CROP_OUTPUT_SIZE)

            alignment = self._plane_alignment(detection, profile, camera_matrix)
            reference = self._plane_mm_per_pixel(definition, frame_size)
            agreement = (None if reference is None
                         else region_calibration.measure_scale_agreement(fit, reference))
            refinement = self._refine(detection, camera_matrix, frame_size)
            annotated = self._annotate(frame, fit, detection, definition)
            self.root.after(0, self._fit_complete, fit, alignment, agreement,
                            refinement, annotated, region_index)
        except region_calibration.RegionCalibrationError as error:
            self.root.after(0, self._fit_failed, str(error))
        except (cv2.error, ValueError, TypeError, KeyError, IndexError,
                np.linalg.LinAlgError) as error:
            self.root.after(0, self._fit_failed, f"Unexpected fit failure: {error}")

    def _plane_alignment(self, detection, profile, camera_matrix):
        """How far the board's own plane sits from the saved measurement plane.

        ``plane_rvec`` and ``plane_tvec`` come from the saved extrinsics and are in
        metres, as is the board pose, so the two are directly comparable.
        """
        extrinsics = self._read_json(self._extrinsics_file())
        if extrinsics is None or camera_matrix is None:
            return None
        thickness = CONFIG['measurement_surface'].get('board_thickness', {})
        expected = (float(thickness.get('thickness_mm', 0.0))
                    if thickness.get('enabled') else 0.0)
        try:
            return region_calibration.measure_plane_alignment(
                profile, detection, camera_matrix,
                np.asarray(extrinsics["rvec"], np.float64).reshape(3, 1),
                np.asarray(extrinsics["tvec"], np.float64).reshape(3, 1),
                expected)
        except (KeyError, ValueError, TypeError, cv2.error):
            return None

    def _refine(self, detection, camera_matrix, frame_size):
        """Collect this view and try the opt-in refinement. The result is never applied."""
        if not self.refine_var.get():
            return None
        self.refinement_views.append((
            np.column_stack((np.asarray(detection.board_points_mm, np.float64) / 1000.0,
                             np.zeros(len(detection.board_points_mm)))),
            np.asarray(detection.frame_points, np.float64),
        ))
        try:
            return region_calibration.refine_intrinsics(
                self.refinement_views, camera_matrix, frame_size, self.dist_coeffs,
                int(REGION_SETTINGS.get("minimum_refinement_views", 5)))
        except (ValueError, TypeError, cv2.error) as error:
            return region_calibration.Refinement(
                applied=False, reason=str(error), views=len(self.refinement_views),
                maximum_tilt_degrees=float('nan'), rms_before_px=float('nan'),
                rms_after_px=float('nan'))

    def _annotate(self, frame, fit, detection, definition):
        """The captured crop with the covered corners and the fitted board drawn on it."""
        crop = extract_rotated_crop(frame, definition, CROP_OUTPUT_SIZE, CROP_RATIO)
        for point in detection.crop_points:
            cv2.drawMarker(crop, (int(round(point[0])), int(round(point[1]))),
                           (0, 255, 255), cv2.MARKER_CROSS, 16, 2)
        # Where the saved board geometry lands under the fit. On a good fit these circles
        # sit on the crosses above, so what looks like a gap is the residual, shown at the
        # crop's own scale rather than through a preview of the whole frame.
        try:
            inverse = np.linalg.inv(np.asarray(fit.homography, np.float64))
            ideal = cv2.perspectiveTransform(
                np.asarray(detection.board_points_mm, np.float64).reshape(-1, 1, 2),
                inverse).reshape(-1, 2)
        except (np.linalg.LinAlgError, cv2.error):
            return crop
        for point in ideal:
            cv2.circle(crop, (int(round(point[0])), int(round(point[1]))), 5,
                       (0, 0, 255), 1)
        return crop

    def _fit_complete(self, fit, alignment, agreement, refinement, annotated, region_index):
        self.capture_busy = False
        self.pending = (fit, alignment, agreement, refinement, region_index)
        self._show_frame(annotated)
        self._set_report(self._fit_report(fit, alignment, agreement, refinement))
        self._set_status(
            f"{REGION_NAMES[region_index]} fitted — RMS {fit.rms_mm:.3f} mm over "
            f"{fit.inliers} corners", GREEN)
        self.log(f"Region {region_index}: RMS {fit.rms_mm:.4f} mm, max {fit.max_mm:.4f} mm, "
                 f"{fit.inliers}/{fit.corners} corners, {fit.dropped} set aside")
        self._refresh_controls()

    def _limits(self):
        return {
            "rms": float(REGION_SETTINGS.get("maximum_rms_mm", 0.5)),
            "tilt": float(REGION_SETTINGS.get("maximum_tilt_degrees", 2.0)),
            "gap": float(REGION_SETTINGS.get("maximum_gap_mm", 2.0)),
            "scale": float(REGION_SETTINGS.get("maximum_scale_error_percent", 2.0)) / 100.0,
        }

    def _fit_report(self, fit, alignment, agreement, refinement):
        limits = self._limits()
        lines = [
            f"Board        {fit.board_definition.describe()}",
            f"Corners      {fit.inliers} of {fit.corners} used, {fit.dropped} set aside",
            f"Fit RMS      {fit.rms_mm:.4f} mm   (limit {limits['rms']:.2f})",
            f"Fit max      {fit.max_mm:.4f} mm      p95 {fit.p95_mm:.4f} mm",
            f"Scale        {fit.mm_per_pixel:.5f} mm/pixel at the region centre",
            f"Lens check   {fit.intrinsic_evaluation.rms_px:.3f} px RMS under the "
            f"global camera matrix",
        ]
        if alignment is not None:
            ok = alignment.within(limits["tilt"], limits["gap"])
            lines.append(f"Flatness     {alignment.describe()}   "
                         f"{'OK' if ok else 'FAILS — do not save'}")
            if not ok:
                lines.append("             The board is not lying on the measurement "
                             "plane. This would fit perfectly and measure wrongly.")
        if agreement is not None:
            ok = agreement.within(limits["scale"])
            lines.append(f"Cross-check  {agreement.describe()}   "
                         f"{'OK' if ok else 'FAILS — do not save'}")
            if not ok:
                lines.append("             The fitted scale disagrees with the saved "
                             "plane. The usual cause is a board printed or declared at "
                             "the wrong size.")
        if refinement is not None:
            lines.append(f"Refinement   {refinement.reason}")
            if refinement.applied:
                lines.append(f"             {refinement.views} views at up to "
                             f"{refinement.maximum_tilt_degrees:.1f}° tilt: "
                             f"{refinement.rms_before_px:.4f} px -> "
                             f"{refinement.rms_after_px:.4f} px")
        lines.append("")
        lines.append("The RMS above is against the board's own geometry, so it is a "
                     "residual and not a guarantee. The flatness and cross-check lines "
                     "are what stand between a good fit and a correct measurement.")
        return "\n".join(lines)

    def _fit_failed(self, message):
        self.capture_busy = False
        self.processing_capture = False
        self.pending = None
        self._set_report(f"The capture was refused.\n\n{message}")
        self._set_status(message, RED)
        self.log(f"Region fit refused: {message}")
        self._refresh_controls()

    def save_region(self):
        if self.pending is None:
            return
        fit, alignment, agreement, refinement, region_index = self.pending
        limits = self._limits()
        refusals = []
        if alignment is not None and not alignment.within(limits["tilt"], limits["gap"]):
            refusals.append(
                f"The board is not flat on the measurement plane ({alignment.describe()})."
                "\nA homography fitted here would be excellent and would measure the "
                "wrong plane.")
        if agreement is not None and not agreement.within(limits["scale"]):
            refusals.append(
                f"The fitted scale disagrees with the saved plane ({agreement.describe()})."
                "\nCheck that the profile's checker and marker sizes match the card you "
                "actually printed.")
        if refusals and not messagebox.askyesno(
                "Save anyway?",
                f"{REGION_NAMES[region_index]} failed a check:\n\n"
                + "\n\n".join(refusals) + "\n\nSave it anyway?", icon="warning"):
            self.log(f"{REGION_NAMES[region_index]} was not saved: a check failed")
            return

        definition = self._crop_for(region_index)
        frame_size = self._undistorted_size()
        signature = region_calibration.crop_signature(
            definition, frame_size, CROP_OUTPUT_SIZE, CROP_RATIO,
            self._calibration_file())
        store = region_calibration.load_region_store(region_store_path())
        region_calibration.store_region_fit(
            store, self.camera_number(), region_index, fit, signature, definition,
            alignment=alignment, refinement=refinement, agreement=agreement)
        region_calibration.save_region_store(region_store_path(), store)
        self.pending = None
        self.processing_capture = False
        self.region_status[region_index] = region_calibration.stored_region_entry(
            region_calibration.load_region_store(region_store_path()),
            self.camera_number(), region_index)
        self.log(f"Saved the local homography for camera {self.camera_number()}, "
                 f"region {region_index}")
        self._set_status(f"{REGION_NAMES[region_index]} saved", GREEN)
        self._refresh_controls()

    # -- camera plumbing ------------------------------------------------------------------

    def select_camera(self, index):
        if self.starting_camera or self.capture_busy:
            return
        if self.camera_running:
            self.stop_camera()
        self.camera_index = index
        self.stage = "select"
        self.undistorter = None
        self.pending = None
        self.refinement_views = []
        self.camera_matrix = None
        self.dist_coeffs = None
        self.calibration_image_size = None
        self.processing_capture = False
        self.region_status = {1: None, 2: None}
        set_button_role(self.btn_cam0,
                        "selected" if index == CAMERA_IDS[0] else "secondary")
        set_button_role(self.btn_cam1,
                        "selected" if index == CAMERA_IDS[1] else "secondary")
        self.start_btn.configure(state=tk.NORMAL, text="START CAMERA")
        self._set_status(f"Camera {index} selected — start the live feed")
        self.log(f"Selected Camera {index}")
        self._refresh_gate()

    def _camera_started(self, requested_width, requested_height):
        self.starting_camera = False
        self.stage = "capture"
        self.start_btn.configure(text="START CAMERA", state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)
        self.resolution_label.configure(
            text=f"Camera {self.camera_index}  •  {self.actual_size[0]} × "
                 f"{self.actual_size[1]}")
        self.log(f"Camera {self.camera_index} started at {self.actual_size[0]} × "
                 f"{self.actual_size[1]} (requested {requested_width} × "
                 f"{requested_height})")
        self._load_undistorter()
        self._refresh_gate()

    def _load_undistorter(self):
        """Load the lens model this page undistorts with, and the intrinsics it grades by."""
        try:
            self._load_saved_intrinsics()
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self.undistorter = None
            self._set_status(
                "This camera has no usable lens calibration, so its regions cannot be "
                "calibrated. Run Camera Setup for it first.", RED)
            self.log(f"Region calibration needs a lens calibration: {error}")
            return
        try:
            self.undistorter = ImageUndistorter(self._calibration_file())
        except (OSError, KeyError, TypeError, ValueError) as error:
            self.undistorter = None
            self._set_status(f"The saved lens calibration could not be used: {error}", RED)
            self.log(f"Undistorter could not be built: {error}")
            return
        self._set_status(
            "Live crop of the selected region. Lay the board flat inside it and "
            "CAPTURE & FIT.", CYAN)
        self._load_region_status()
        if self.calibration_image_size:
            self.log(f"Lens calibration loaded (captured at "
                     f"{self.calibration_image_size[0]} × {self.calibration_image_size[1]};"
                     f" live frame {self.actual_size[0]} × {self.actual_size[1]})")
        else:
            self.log("Lens calibration loaded")

    def _load_region_status(self):
        store = region_calibration.load_region_store(region_store_path())
        camera = str(self.camera_number())
        entries = (store.get("cameras", {}) or {}).get(camera, {}) or {}
        self.region_status = {index: entries.get(str(index)) for index in (1, 2)}

    def _camera_failed(self, error):
        self.starting_camera = False
        self.start_btn.configure(text="START CAMERA", state=tk.NORMAL)
        self._set_status("Camera could not be opened", RED)
        self.log(f"Camera error: {error}")
        messagebox.showerror("Camera Error", error)
        self._refresh_controls()

    def stop_camera(self):
        if self.starting_camera or self.capture_busy:
            return
        super().stop_camera()
        self.undistorter = None
        self._refresh_controls()

    # -- base-class hooks this page does not use --------------------------------------------

    def _refresh_stage_ui(self):
        """This page has no stage strip; its own controls carry the state."""
        self._refresh_controls()

    def _refresh_board_mode_ui(self):
        """Two-board capture belongs to the lens wizard and has no meaning here."""

    def _draw_coverage_guide(self, frame):
        """Coverage guidance belongs to the lens wizard, not to a region crop."""

    def _draw_capture_history(self, frame):
        """Capture history belongs to the lens wizard, not to a region crop."""

    # -- live preview ----------------------------------------------------------------------

    def _prepare_live_preview(self, source, size):
        """Read a frame, undistort it, and deskew the selected region for display.

        Runs in the preview executor, so no Tk calls. ``frame`` is the full undistorted
        frame, because that is what the fit and the crop both need; only the deskewed
        region is shown. Undistorting on every tick rather than only at capture is what
        keeps the crop on screen and the crop that gets fitted the same raster -- the
        operator is judging what will actually be measured.
        """
        ok, raw = source.read()
        if not ok or raw is None:
            return source, None, None
        undistorter = self.undistorter
        if undistorter is None:
            # No lens model yet: show the raw frame so the operator can see the camera is
            # alive, and let the banner explain why nothing can be fitted.
            return source, raw, scale_to_viewport(raw, size)
        try:
            frame = undistorter.undistort(raw)
        except (cv2.error, ValueError, TypeError):
            return source, raw, scale_to_viewport(raw, size)
        definition = self._crop_for(self.region_index)
        if definition is None:
            return source, frame, scale_to_viewport(frame, size)
        try:
            crop = extract_rotated_crop(frame, definition, CROP_OUTPUT_SIZE, CROP_RATIO)
        except (cv2.error, ValueError, TypeError):
            crop = frame
        return source, frame, scale_to_viewport(crop, size)

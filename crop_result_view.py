"""Tabbed, zoomable view of the crops an inspection produced.

Shown in place of the dual camera view while a result is held on screen, so one
crop can use the whole image panel instead of half of one column. Each tab keeps
its own zoom and pan, and the tab for a crop that is still being uploaded carries
a rotating arc drawn as a canvas item -- never baked into the pixels, so a tick
costs two Tcl calls instead of a copy, a resize and a fresh image.

The viewport arithmetic lives in module-level functions so it can be verified
without a window; the widget methods only store their results.
"""

import time
import tkinter as tk

import cv2
from PIL import Image, ImageTk

from ui_theme import (
    COLORS as C, button as themed_button, set_button_role,
)

SPINNER_PERIOD_SECONDS = 1.0
SPINNER_ARC_DEGREES = 270
ARC_RADIUS = 34
ARC_WIDTH = 6
MAX_ARC_STEP_SECONDS = 0.25

MIN_ZOOM = 1.0
MAX_ZOOM = 20.0
ZOOM_IN_FACTOR = 1.25
ZOOM_OUT_FACTOR = 0.8

# Appended to the tab label while the crop is uploading, and once it lands. A
# crop that is never uploaded gets neither, so the marker never claims a check
# that did not happen.
MARKER_BUSY = "⟳"
MARKER_DONE = "✓"

# ----------------------------------------------------------------------
# viewport arithmetic
# ----------------------------------------------------------------------
def fit_view(image_size, canvas_size):
    """The scale that shows a whole image, and the offset that centres it."""
    image_width, image_height = image_size
    canvas_width, canvas_height = canvas_size
    scale = min(canvas_width / image_width, canvas_height / image_height)
    offset = [(canvas_width - image_width * scale) / 2.0,
              (canvas_height - image_height * scale) / 2.0]
    return scale, offset


def clamp_offset(image_size, canvas_size, scale, offset):
    """Centre an image smaller than the canvas; otherwise keep the canvas full.

    Without this a zoomed-out image would drift to a corner and a zoomed-in one
    could be dragged entirely out of sight.
    """
    scaled_width = image_size[0] * scale
    scaled_height = image_size[1] * scale
    if scaled_width <= canvas_size[0]:
        offset_x = (canvas_size[0] - scaled_width) / 2.0
    else:
        offset_x = min(0.0, max(canvas_size[0] - scaled_width, offset[0]))
    if scaled_height <= canvas_size[1]:
        offset_y = (canvas_size[1] - scaled_height) / 2.0
    else:
        offset_y = min(0.0, max(canvas_size[1] - scaled_height, offset[1]))
    return [offset_x, offset_y]


def zoom_about(fit, offset, zoom, factor, cursor):
    """Zoom about a point, keeping the image pixel under it in place.

    Returns the new ``(zoom, offset)``. ``fit`` is the scale at zoom 1.0.
    """
    old_scale = max(1e-9, fit * zoom)
    image_x = (cursor[0] - offset[0]) / old_scale
    image_y = (cursor[1] - offset[1]) / old_scale
    zoom = min(MAX_ZOOM, max(MIN_ZOOM, zoom * factor))
    new_scale = max(1e-9, fit * zoom)
    return zoom, [cursor[0] - image_x * new_scale,
                  cursor[1] - image_y * new_scale]


def viewport(image_size, canvas_size, scale, offset):
    """The source rectangle of the image visible on the canvas, or None.

    Cropping the source before resizing bounds the work by the canvas rather
    than by the full-resolution crop.
    """
    left = max(0, int(-offset[0] / scale))
    top = max(0, int(-offset[1] / scale))
    right = min(image_size[0], int((canvas_size[0] - offset[0]) / scale) + 1)
    bottom = min(image_size[1], int((canvas_size[1] - offset[1]) / scale) + 1)
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def visible_centre(image_size, canvas_size, scale, offset):
    """Centre of the part of the image on screen, or None when none of it is.

    Clamped so a view panned into a corner cannot leave the arc spinning over
    empty background.
    """
    left = max(0.0, offset[0])
    top = max(0.0, offset[1])
    right = min(float(canvas_size[0]), offset[0] + image_size[0] * scale)
    bottom = min(float(canvas_size[1]), offset[1] + image_size[1] * scale)
    if right <= left or bottom <= top:
        return None
    return ((left + right) / 2.0, (top + bottom) / 2.0)


def reanchor(canvas_size, previous_size, image_size, zoom, offset):
    """Keep the image point at the canvas centre across a resize.

    Returns the new ``(fit, offset)``. The zoom factor is preserved, so a resize
    never silently discards what the operator set up.
    """
    image_width, image_height = image_size
    previous_scale = min(previous_size[0] / image_width, previous_size[1] / image_height)
    centre_x = (previous_size[0] / 2.0 - offset[0]) / max(1e-9, previous_scale * zoom)
    centre_y = (previous_size[1] / 2.0 - offset[1]) / max(1e-9, previous_scale * zoom)
    fit = min(canvas_size[0] / image_width, canvas_size[1] / image_height)
    scale = max(1e-9, fit * zoom)
    return fit, [canvas_size[0] / 2.0 - centre_x * scale,
                 canvas_size[1] / 2.0 - centre_y * scale]


def advance_angle(angle, elapsed):
    """The next arc angle, given the measured gap since the last tick."""
    step = min(elapsed, MAX_ARC_STEP_SECONDS)
    return (angle + 360.0 * step / SPINNER_PERIOD_SECONDS) % 360.0


class CropResultView(tk.Frame):
    """Two crop tabs over a wheel-zoomable, draggable canvas."""

    def __init__(self, parent, on_resume, title="INSPECTION RESULT"):
        super().__init__(parent, bg=C["bg"])
        self._on_resume = on_resume
        self._title = title
        self._tabs = []
        self._tab_buttons = []
        self._active = 0
        self._photo = None
        self._last_canvas_size = (0, 0)
        self._fit_pending = True
        self._angle = 0.0
        self._angle_at = None
        self._pan_anchor = None
        self._build()

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def _build(self):
        header = tk.Frame(self, bg=C["bg"])
        header.pack(side=tk.TOP, fill=tk.X, pady=(0, 6))

        tk.Label(header, text=self._title, bg=C["bg"], fg=C["muted"],
                 font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=(0, 14))

        for index in range(2):
            tab = themed_button(
                header, self._tab_label(index), lambda i=index: self.select_tab(i),
                role="secondary", font_size=10, padx=12, pady=6,
            )
            tab.pack(side=tk.LEFT, padx=(0, 6))
            self._tab_buttons.append(tab)

        self._resume_button = themed_button(
            header, "START LIVE PREVIEW", self._on_resume,
            role="primary", font_size=10, padx=16, pady=6, state=tk.DISABLED,
        )
        self._resume_button.pack(side=tk.RIGHT)

        themed_button(header, "RESET", self.reset_view, role="quiet",
                      font_size=10, padx=10, pady=6).pack(side=tk.RIGHT, padx=(0, 6))
        themed_button(header, "+", lambda: self.zoom_by(ZOOM_IN_FACTOR),
                      role="quiet", font_size=12, width=2, padx=6, pady=6
                      ).pack(side=tk.RIGHT, padx=(0, 4))
        themed_button(header, "−", lambda: self.zoom_by(ZOOM_OUT_FACTOR),
                      role="quiet", font_size=12, width=2, padx=6, pady=6
                      ).pack(side=tk.RIGHT, padx=(0, 4))

        self.canvas = tk.Canvas(self, bg=C["camera"], highlightthickness=0)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<ButtonPress-1>", self._on_pan_start)
        self.canvas.bind("<B1-Motion>", self._on_pan_drag)
        self.canvas.bind("<Configure>", self._on_configure)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def show_crops(self, crops, active_index=0):
        """Replace the tabs with ``crops`` (a list of BGR arrays)."""
        self._tabs = [self._blank_tab(self._to_pil(crop)) for crop in crops]
        self._active = active_index if 0 <= active_index < len(self._tabs) else 0
        self._angle = 0.0
        self._angle_at = None
        self._refresh_tab_labels()
        # The canvas may not be mapped yet; the first <Configure> then fits it.
        self._fit_pending = True
        tab = self._active_tab()
        if tab is not None and self._canvas_size()[0] > 2:
            self._fit_tab(tab)
        self._render()

    def update_crop(self, crop_index, image, busy):
        """Show ``image`` in the 1-based ``crop_index`` tab, busy or finished."""
        if not 1 <= crop_index <= len(self._tabs):
            return
        tab = self._tabs[crop_index - 1]
        tab["image"] = self._to_pil(image)
        tab["busy"] = bool(busy)
        if not busy:
            tab["done"] = True
        if tab["busy"]:
            # Restart the sweep so the arc does not jump on the first tick.
            self._angle_at = None
        self._refresh_tab_labels()
        if crop_index - 1 == self._active:
            self._render()

    def select_tab(self, index):
        if not 0 <= index < len(self._tabs) or index == self._active:
            return
        self._active = index
        self._refresh_tab_labels()
        tab = self._active_tab()
        if tab["fit"] <= 0:
            self._fit_tab(tab)
        self._render()

    def tick(self):
        """Advance the arc when the visible tab is uploading. No timer is made."""
        tab = self._active_tab()
        if tab is None or not tab["busy"] or tab["image"] is None:
            return
        now = time.perf_counter()
        elapsed = now - (self._angle_at if self._angle_at is not None else now)
        self._angle_at = now
        self._angle = advance_angle(self._angle, elapsed)
        self._draw_spinner()

    def reset_view(self):
        tab = self._active_tab()
        if tab is None:
            return
        self._fit_tab(tab)
        self._render()

    def zoom_by(self, factor):
        """Zoom about the centre of the canvas, for the -/+ buttons."""
        tab = self._active_tab()
        if tab is None:
            return
        width, height = self._canvas_size()
        self._zoom_at(tab, factor, (width / 2.0, height / 2.0))

    def mark_all_idle(self):
        """Clear outstanding spinners without claiming any crop was analysed."""
        for tab in self._tabs:
            tab["busy"] = False
        self._refresh_tab_labels()
        self._render()

    def set_resume_enabled(self, enabled):
        self._resume_button.configure(
            state=tk.NORMAL if enabled else tk.DISABLED,
        )

    # ------------------------------------------------------------------
    # tab bookkeeping
    # ------------------------------------------------------------------
    @staticmethod
    def _blank_tab(pil_image):
        return {"image": pil_image, "busy": False, "done": False,
                "zoom": 1.0, "fit": 0.0, "offset": [0.0, 0.0]}

    @staticmethod
    def _to_pil(image):
        if image is None:
            return None
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    def _active_tab(self):
        if not self._tabs or not 0 <= self._active < len(self._tabs):
            return None
        return self._tabs[self._active]

    def _tab_label(self, index):
        if not 0 <= index < len(self._tabs):
            return f"CROP {index + 1}"
        tab = self._tabs[index]
        marker = MARKER_BUSY if tab["busy"] else MARKER_DONE if tab["done"] else ""
        return f"CROP {index + 1}" + (f"  {marker}" if marker else "")

    def _refresh_tab_labels(self):
        for index, button in enumerate(self._tab_buttons):
            text = self._tab_label(index)
            if button.cget("text") != text:
                button.configure(text=text)
            set_button_role(button, "selected" if index == self._active else "secondary")

    # ------------------------------------------------------------------
    # geometry
    # ------------------------------------------------------------------
    def _canvas_size(self):
        return (max(2, self.canvas.winfo_width()),
                max(2, self.canvas.winfo_height()))

    def _scale(self, tab):
        return max(1e-9, tab["fit"] * tab["zoom"])

    def _fit_tab(self, tab):
        if tab["image"] is None:
            return
        tab["fit"], tab["offset"] = fit_view(tab["image"].size, self._canvas_size())
        tab["zoom"] = MIN_ZOOM

    def _zoom_at(self, tab, factor, cursor):
        if tab["image"] is None:
            return
        tab["zoom"], tab["offset"] = zoom_about(
            tab["fit"], tab["offset"], tab["zoom"], factor, cursor)
        self._render()

    # ------------------------------------------------------------------
    # events
    # ------------------------------------------------------------------
    def _on_wheel(self, event):
        tab = self._active_tab()
        if tab is not None:
            factor = ZOOM_IN_FACTOR if event.delta > 0 else ZOOM_OUT_FACTOR
            self._zoom_at(tab, factor, (event.x, event.y))
        return "break"

    def _on_pan_start(self, event):
        tab = self._active_tab()
        if tab is not None:
            self._pan_anchor = (
                event.x, event.y, tab["offset"][0], tab["offset"][1],
            )
        return "break"

    def _on_pan_drag(self, event):
        tab = self._active_tab()
        if tab is None or self._pan_anchor is None:
            return "break"
        start_x, start_y, offset_x, offset_y = self._pan_anchor
        tab["offset"] = [offset_x + event.x - start_x,
                         offset_y + event.y - start_y]
        self._render()
        return "break"

    def _on_configure(self, event):
        """Re-fit or re-anchor, but only when the canvas really changed size.

        A <Configure> that does not move the canvas must never discard the
        operator's zoom.
        """
        previous = self._last_canvas_size
        if (event.width, event.height) == previous:
            return
        self._last_canvas_size = (event.width, event.height)
        tab = self._active_tab()
        if tab is None or event.width <= 2 or event.height <= 2:
            return
        if self._fit_pending or tab["fit"] <= 0:
            self._fit_pending = False
            self._fit_tab(tab)
        elif tab["image"] is not None:
            tab["fit"], tab["offset"] = reanchor(
                (event.width, event.height), previous, tab["image"].size,
                tab["zoom"], tab["offset"])
        self._render()

    # ------------------------------------------------------------------
    # rendering
    # ------------------------------------------------------------------
    def _render(self):
        """Draw the visible part of the active crop, then the arc over it."""
        self.canvas.delete("all")
        self._photo = None
        tab = self._active_tab()
        if tab is None or tab["image"] is None:
            return
        canvas_size = self._canvas_size()
        scale = self._scale(tab)
        tab["offset"] = clamp_offset(
            tab["image"].size, canvas_size, scale, tab["offset"])
        box = viewport(tab["image"].size, canvas_size, scale, tab["offset"])
        if box is None:
            return
        left, top, right, bottom = box
        offset_x, offset_y = tab["offset"]
        view = tab["image"].crop(box)
        view = view.resize(
            (max(1, round((right - left) * scale)),
             max(1, round((bottom - top) * scale))),
            Image.Resampling.LANCZOS if scale < 1.0 else Image.Resampling.BILINEAR,
        )
        self._photo = ImageTk.PhotoImage(view)
        self.canvas.create_image(
            offset_x + left * scale, offset_y + top * scale,
            image=self._photo, anchor="nw",
        )
        self._draw_spinner()

    def _draw_spinner(self):
        self.canvas.delete("arc")
        tab = self._active_tab()
        if tab is None or not tab["busy"] or tab["image"] is None:
            return
        centre = visible_centre(
            tab["image"].size, self._canvas_size(), self._scale(tab), tab["offset"])
        if centre is None:
            return
        centre_x, centre_y = centre
        self.canvas.create_arc(
            centre_x - ARC_RADIUS, centre_y - ARC_RADIUS,
            centre_x + ARC_RADIUS, centre_y + ARC_RADIUS,
            start=self._angle, extent=SPINNER_ARC_DEGREES, style=tk.ARC,
            outline=C["accent"], width=ARC_WIDTH, tags="arc",
        )

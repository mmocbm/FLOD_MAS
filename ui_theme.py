"""Shared, low-overhead visual system for the desktop application."""

import tkinter as tk
from tkinter import ttk
from tkinter import font as tkfont
from PIL import Image, ImageDraw, ImageTk


COLORS = {
    "bg": "#101722",
    "surface": "#182230",
    "surface_2": "#202D3E",
    "card": "#1B2838",
    "card_hover": "#2B3E55",
    "border": "#34465D",
    "border_strong": "#526C8A",
    "text": "#F4F7FB",
    "text_soft": "#CBD5E1",
    "muted": "#8FA2BA",
    "accent": "#60A5FA",
    "accent_hover": "#3B82F6",
    "accent_dark": "#2563EB",
    "blue": "#2563EB",
    "blue_hover": "#1D4ED8",
    "purple": "#93B9E8",
    "purple_hover": "#405C80",
    "success": "#34D399",
    "success_dark": "#166B55",
    "warning": "#FBBF24",
    "danger": "#FB7185",
    "danger_hover": "#E9526A",
    "camera": "#030811",
    "disabled": "#334155",
    "disabled_text": "#8291A6",
}

FONT = "Segoe UI"
MONO_FONT = "Cascadia Mono"

BUTTON_ROLES = {
    "primary": (COLORS["accent_dark"], COLORS["accent_hover"], COLORS["text"]),
    "blue": (COLORS["blue"], COLORS["blue_hover"], "#FFFFFF"),
    "purple": (COLORS["surface_2"], COLORS["purple_hover"], COLORS["text"]),
    "secondary": (COLORS["surface_2"], COLORS["card_hover"], COLORS["text_soft"]),
    "danger": ("#BE3E55", COLORS["danger_hover"], "#FFFFFF"),
    "quiet": (COLORS["surface"], COLORS["surface_2"], COLORS["text_soft"]),
    "selected": (COLORS["accent_dark"], COLORS["accent_hover"], "#FFFFFF"),
}


def configure_ttk(root):
    """Configure native ttk controls once; no animation or polling is added."""
    if getattr(root, '_industrial_style', None) is not None:
        return root._industrial_style
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(
        "App.Horizontal.TProgressbar",
        troughcolor=COLORS["surface"], background=COLORS["accent"],
        lightcolor=COLORS["accent"], darkcolor=COLORS["accent"],
        bordercolor=COLORS["surface"], thickness=7,
    )
    style.configure(
        "App.TCombobox", fieldbackground=COLORS["surface_2"],
        background=COLORS["surface_2"], foreground=COLORS["text"],
        arrowcolor=COLORS["text_soft"], bordercolor=COLORS["border"],
        lightcolor=COLORS["border"], darkcolor=COLORS["border"], padding=8,
    )
    style.map(
        "App.TCombobox",
        fieldbackground=[("readonly", COLORS["surface_2"])],
        foreground=[("readonly", COLORS["text"])],
        selectbackground=[("readonly", COLORS["surface_2"])],
        selectforeground=[("readonly", COLORS["text"])],
    )
    root._industrial_style = style
    return style


class RoundedCard(tk.Canvas):
    def __init__(self, parent, **kwargs):
        self._surface = kwargs.pop("bg")
        self._edge = kwargs.pop("highlightbackground")
        kwargs.pop("highlightcolor", None)
        kwargs.pop("highlightthickness", None)
        super().__init__(parent, bg=parent.cget("bg"), highlightthickness=0, **kwargs)
        self._shape = self.create_polygon(0, 0, 0, 0, fill=self._surface,
                                          outline=self._edge, smooth=True, splinesteps=16)
        self.bind("<Configure>", self._resize)

    def cget(self, key):
        return self._surface if key in ("bg", "background") else super().cget(key)

    def _resize(self, event):
        w, h, r = event.width - 1, event.height - 1, 14
        self.coords(self._shape, r, 1, w-r, 1, w, 1, w, r,
                    w, h-r, w, h, w-r, h, r, h, 1, h, 1, h-r, 1, r, 1, 1)
        self.tag_lower(self._shape)


def card(parent, **kwargs):
    options = {
        "bg": COLORS["card"],
        "highlightbackground": COLORS["border"],
        "highlightcolor": COLORS["border"],
        "highlightthickness": 1,
        "bd": 0,
    }
    options.update(kwargs)
    return RoundedCard(parent, **options)


class RoundedButton(tk.Button):
    """Native button behavior with cached, antialiased rounded backgrounds.

    Retains Tk's keyboard, invocation and disabled semantics. Rendering occurs
    only on size/state changes, never on a camera frame or animation timer.
    """
    def __init__(self, parent, **options):
        self._face = options.pop("bg")
        self._hover_face = options.pop("activebackground")
        self._outer = parent.cget("bg")
        self._hovered = self._pressed = self._focused = False
        self._images = {}
        self._char_width = options.pop("width", None)
        self._padding = (options.pop("padx", 16), options.pop("pady", 10))
        super().__init__(parent, **options, bg=self._outer,
                         activebackground=self._outer, compound="center",
                         padx=0, pady=0, takefocus=True)
        for event, name, value in (("<Enter>", "_hovered", True),
                                   ("<Leave>", "_hovered", False),
                                   ("<ButtonPress-1>", "_pressed", True),
                                   ("<ButtonRelease-1>", "_pressed", False),
                                   ("<FocusIn>", "_focused", True),
                                   ("<FocusOut>", "_focused", False)):
            self.bind(event, lambda e, n=name, v=value: self._change(n, v), add="+")
        self._paint()

    def _change(self, name, value):
        setattr(self, name, value)
        if name == "_hovered" and not value:
            self._pressed = False
        self._paint()

    def _paint(self):
        metrics_key = (self.cget("font"), self.cget("text"), self._char_width, self._padding)
        if metrics_key != getattr(self, '_metrics_key', None):
            font = tkfont.Font(root=self, font=metrics_key[0])
            lines = metrics_key[1].split("\n")
            width = (font.measure("0") * self._char_width if self._char_width
                     else max(font.measure(line) for line in lines))
            self._measured_size = (width + 2 * self._padding[0] + 4,
                                   font.metrics("linespace") * len(lines) + 2 * self._padding[1] + 4)
            self._metrics_key = metrics_key
        width, height = self._measured_size
        disabled = str(self.cget("state")) == tk.DISABLED
        fill = COLORS["disabled"] if disabled else self._hover_face if self._hovered else self._face
        if self._pressed and not disabled:
            fill = COLORS["accent_dark"]
        outline = COLORS["accent"] if self._focused else COLORS["border"] if disabled or self._face in (COLORS["surface"], COLORS["surface_2"]) else fill
        key = (width, height, fill, outline)
        if key == getattr(self, '_paint_key', None):
            return
        if key not in self._images:
            # Bound memory even when labels change repeatedly during setup.
            if len(self._images) >= 16:
                self._images.clear()
            image = Image.new("RGB", (width * 2, height * 2), self._outer)
            ImageDraw.Draw(image).rounded_rectangle(
                (2, 2, width * 2 - 3, height * 2 - 3), radius=min(22, height - 3),
                fill=fill, outline=outline, width=3 if self._focused else 2)
            self._images[key] = ImageTk.PhotoImage(image.resize((width, height), Image.Resampling.LANCZOS), master=self)
        self._background_image = self._images[key]
        super().configure(image=self._background_image)
        self._paint_key = key

    def configure(self, cnf=None, **kwargs):
        if cnf is not None or not kwargs:
            return super().configure(cnf, **kwargs)
        if "bg" in kwargs:
            self._face = kwargs.pop("bg")
        if "activebackground" in kwargs:
            self._hover_face = kwargs.pop("activebackground")
        if "width" in kwargs:
            self._char_width = kwargs.pop("width")
        kwargs = {key: value for key, value in kwargs.items()
                  if str(self.cget(key)) != str(value)}
        result = super().configure(**kwargs) if kwargs else None
        self._paint()
        return result

    config = configure


def button(parent, text, command, role="secondary", width=None, state=tk.NORMAL,
           font_size=11, bold=True, padx=16, pady=10, **kwargs):
    normal, hover, foreground = BUTTON_ROLES[role]
    options = {
        "text": text,
        "command": command,
        "bg": normal,
        "fg": foreground,
        "activebackground": hover,
        "activeforeground": foreground,
        "disabledforeground": COLORS["disabled_text"],
        "relief": tk.FLAT,
        "bd": 0,
        "highlightthickness": 0,
        "cursor": "hand2",
        "font": (FONT, font_size, "bold" if bold else "normal"),
        "padx": padx,
        "pady": pady,
        "state": state,
    }
    if width is not None:
        options["width"] = width
    options.update(kwargs)
    widget = RoundedButton(parent, **options)
    widget._theme_role = role

    return widget


def set_button_role(widget, role):
    widget._theme_role = role
    normal, hover, foreground = BUTTON_ROLES[role]
    widget.configure(
        bg=normal, fg=foreground,
        activebackground=hover, activeforeground=foreground,
    )


def section_label(parent, text, **kwargs):
    options = {
        "text": text.upper(), "bg": parent.cget("bg"), "fg": COLORS["muted"],
        "font": (FONT, 9, "bold"),
    }
    options.update(kwargs)
    return tk.Label(parent, **options)


def status_dot(parent, color=None, size=10):
    canvas = tk.Canvas(
        parent, width=size, height=size, bg=parent.cget("bg"),
        bd=0, highlightthickness=0,
    )
    canvas.create_oval(1, 1, size - 1, size - 1, fill=color or COLORS["success"], outline="")
    return canvas

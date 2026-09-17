"""Shared, low-overhead visual system for the desktop application."""

import tkinter as tk
from tkinter import ttk


COLORS = {
    "bg": "#08111F",
    "surface": "#0D1829",
    "surface_2": "#122035",
    "card": "#14243A",
    "card_hover": "#192C46",
    "border": "#263A55",
    "border_strong": "#36506F",
    "text": "#F4F7FB",
    "text_soft": "#CBD5E1",
    "muted": "#8FA2BA",
    "accent": "#2DD4BF",
    "accent_hover": "#14B8A6",
    "accent_dark": "#0F766E",
    "blue": "#2563EB",
    "blue_hover": "#1D4ED8",
    "purple": "#A78BFA",
    "purple_hover": "#8B6FE8",
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
    "purple": ("#7558D6", COLORS["purple_hover"], "#FFFFFF"),
    "secondary": (COLORS["surface_2"], COLORS["card_hover"], COLORS["text_soft"]),
    "danger": ("#BE3E55", COLORS["danger_hover"], "#FFFFFF"),
    "quiet": (COLORS["surface"], COLORS["surface_2"], COLORS["text_soft"]),
    "selected": (COLORS["accent_dark"], COLORS["accent_hover"], "#FFFFFF"),
}


def configure_ttk(root):
    """Configure native ttk controls once; no animation or polling is added."""
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
    return style


def card(parent, **kwargs):
    options = {
        "bg": COLORS["card"],
        "highlightbackground": COLORS["border"],
        "highlightcolor": COLORS["border"],
        "highlightthickness": 1,
        "bd": 0,
    }
    options.update(kwargs)
    return tk.Frame(parent, **options)


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
    widget = tk.Button(parent, **options)
    widget._theme_role = role

    def enter(_event):
        if str(widget.cget("state")) != tk.DISABLED:
            widget.configure(bg=BUTTON_ROLES[widget._theme_role][1])

    def leave(_event):
        if str(widget.cget("state")) != tk.DISABLED:
            widget.configure(bg=BUTTON_ROLES[widget._theme_role][0])

    widget.bind("<Enter>", enter, add="+")
    widget.bind("<Leave>", leave, add="+")
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

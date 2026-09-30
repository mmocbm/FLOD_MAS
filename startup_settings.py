"""Startup countdown and form editor; no camera or app imports before selection."""
import copy
import json
import math
import os
from pathlib import Path
import tempfile
import time
import tkinter as tk
from tkinter import ttk, messagebox

CONFIG_PATH = Path(__file__).resolve().parent / 'config.json'


def parse_value(text, original):
    if isinstance(original, bool):
        return bool(text)
    if isinstance(original, int):
        return int(text)
    if isinstance(original, float):
        value = float(text)
        if not math.isfinite(value):
            raise ValueError('Enter a finite number')
        return value
    return text


def save_config(config, path=CONFIG_PATH):
    from config_validation import validate_config
    validate_config(config)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         delete=False, suffix='.tmp') as stream:
            temporary = stream.name
            json.dump(config, stream, indent=2, allow_nan=False)
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


class StartupWindow:
    def __init__(self, root):
        self.root = root
        self.saved = False
        self.timer = None
        self.fields = []
        root.title('Application startup')
        root.geometry('520x240')
        root.protocol('WM_DELETE_WINDOW', self.quit)
        self.content = ttk.Frame(root, padding=24)
        self.content.pack(fill='both', expand=True)
        ttk.Label(self.content, text='Ready to start', font=('Segoe UI', 20, 'bold')).pack(pady=8)
        ttk.Label(self.content, text="Press S to change application settings.").pack(pady=8)
        self.countdown = ttk.Label(self.content)
        self.countdown.pack(pady=8)
        actions = ttk.Frame(self.content)
        actions.pack(pady=8)
        ttk.Button(actions, text='Settings (S)', command=self.settings).pack(side='left', padx=8)
        ttk.Button(actions, text='Start now', command=self.start).pack(side='left', padx=8)
        root.bind('<KeyPress-s>', self.settings)
        root.bind('<KeyPress-S>', self.settings)
        root.after_idle(root.focus_force)
        self.deadline = time.monotonic() + 5
        self.tick()

    def cancel_timer(self):
        if self.timer is not None:
            self.root.after_cancel(self.timer)
            self.timer = None

    def tick(self):
        self.timer = None
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            self.start()
            return
        self.countdown.configure(text=f'Starting with current settings in {math.ceil(remaining)} seconds…')
        self.timer = self.root.after(100, self.tick)

    def start(self):
        self.cancel_timer()
        self.root.destroy()

    def quit(self):
        self.start()
        raise SystemExit(0)

    def settings(self, event=None):
        self.cancel_timer()
        self.root.unbind('<KeyPress-s>')
        self.root.unbind('<KeyPress-S>')
        try:
            self.config = json.loads(CONFIG_PATH.read_text(encoding='utf-8-sig'))
        except (OSError, ValueError) as error:
            messagebox.showerror('Cannot read settings', str(error), parent=self.root)
            self.quit()
            return
        for child in self.content.winfo_children():
            child.destroy()
        self.root.title('Application settings')
        self.root.geometry('900x700')
        ttk.Label(self.content, text='Application settings', font=('Segoe UI', 20, 'bold')).pack(anchor='w')
        ttk.Label(self.content, text='Change values, then Save & restart. Cancel starts with the saved settings.').pack(anchor='w', pady=(4, 12))
        notebook = ttk.Notebook(self.content)
        notebook.pack(fill='both', expand=True)
        for name, value in self.config.items():
            page = ttk.Frame(notebook)
            notebook.add(page, text=name.replace('_', ' ').title())
            canvas = tk.Canvas(page, highlightthickness=0)
            scroll = ttk.Scrollbar(page, orient='vertical', command=canvas.yview)
            canvas.configure(yscrollcommand=scroll.set)
            scroll.pack(side='right', fill='y')
            canvas.pack(side='left', fill='both', expand=True)
            form = ttk.Frame(canvas, padding=12)
            window = canvas.create_window((0, 0), window=form, anchor='nw')
            form.bind('<Configure>', lambda e, c=canvas: c.configure(scrollregion=c.bbox('all')))
            canvas.bind('<Configure>', lambda e, c=canvas, w=window: c.itemconfigure(w, width=e.width))
            canvas.bind('<MouseWheel>', lambda e, c=canvas: c.yview_scroll(-int(e.delta / 120), 'units'))
            form.columnconfigure(1, weight=1)
            self.add_fields(form, value, (name,), [])
        actions = ttk.Frame(self.content)
        actions.pack(fill='x', pady=(14, 0))
        ttk.Button(actions, text='Cancel / start app', command=self.start).pack(side='left')
        ttk.Button(actions, text='Save & restart', command=self.save).pack(side='right')

    def add_fields(self, form, value, path, labels):
        if isinstance(value, dict):
            for name, child in value.items():
                self.add_fields(form, child, path + (name,), labels + [name.replace('_', ' ').title()])
        elif isinstance(value, list):
            for index, child in enumerate(value):
                self.add_fields(form, child, path + (index,), labels + [str(index + 1)])
        else:
            row = len(form.grid_slaves(column=0))
            ttk.Label(form, text=' / '.join(labels) or path[0]).grid(row=row, column=0, sticky='w', padx=(0, 18), pady=5)
            variable = tk.BooleanVar(value=value) if isinstance(value, bool) else tk.StringVar(value=str(value))
            control = ttk.Checkbutton(form, variable=variable) if isinstance(value, bool) else ttk.Entry(form, textvariable=variable)
            control.grid(row=row, column=1, sticky='ew', pady=5)
            self.fields.append((path, value, variable))

    def save(self):
        candidate = copy.deepcopy(self.config)
        try:
            for path, original, variable in self.fields:
                target = candidate
                for key in path[:-1]:
                    target = target[key]
                try:
                    target[path[-1]] = parse_value(variable.get(), original)
                except ValueError as error:
                    raise ValueError(f"{' / '.join(map(str, path))}: {error}") from error
            save_config(candidate)
        except Exception as error:
            messagebox.showerror('Settings not saved', str(error), parent=self.root)
            return
        self.saved = True
        self.start()


def run_startup():
    root = tk.Tk()
    window = StartupWindow(root)
    root.mainloop()
    return window.saved

"""Developer switch and local-preview distance ratio editor."""
import tkinter as tk
from tkinter import messagebox
from ui_theme import COLORS as C, FONT, button, set_button_role
import measurement_adjustment as adjustment


class DeveloperControls(tk.Frame):
    def __init__(self, parent, on_preview=None, edit_ratio=True, compact=False):
        super().__init__(parent, bg=C['surface_2'], padx=12, pady=8,
                         highlightthickness=1, highlightbackground=C['border'])
        self.on_preview = on_preview
        self.edit_ratio = edit_ratio
        self.editing_allowed = True
        self.compact = compact
        data = adjustment.load()
        self.enabled = tk.BooleanVar(value=data['developer_mode'])
        self.ratio = tk.StringVar(value=f"{data['ratio']:.8g}")
        heading = tk.Frame(self, bg=C['surface_2'])
        heading.pack(fill='x')
        self.mode_button = button(heading, '', self._switch_mode, role='quiet', font_size=10, pady=5)
        self.mode_button.pack(side='left')
        self.summary = tk.Label(heading, bg=C['surface_2'], fg=C['muted'], font=(FONT, 9))
        if compact:
            self.summary.pack(side='left', padx=12)
        else:
            self.summary.pack(side='right', padx=4)
        self.editor = tk.Frame(self, bg=C['surface_2'])
        row = tk.Frame(self.editor, bg=C['surface_2'])
        row.pack(fill='x', pady=4)
        tk.Label(row, text='MULTIPLIER', bg=C['surface_2'], fg=C['text_soft'],
                 font=(FONT, 9, 'bold')).pack(side='left', padx=(0, 8))
        tk.Spinbox(row, from_=0.001, to=100, increment=0.001, textvariable=self.ratio,
                   width=10, bg=C['bg'], fg=C['text'], insertbackground=C['text'],
                   buttonbackground=C['border'], relief='flat', font=(FONT, 12)).pack(side='left')
        actions = row if compact else tk.Frame(self.editor, bg=C['surface_2'])
        if not compact:
            actions.pack(fill='x', pady=4)
        button(actions, 'SAVE RATIO', self.save_ratio, role='primary', font_size=9,
               pady=6, padx=10).pack(side='left', padx=(8 if compact else 0, 6))
        button(actions, 'RESET TO 1', self.reset, role='secondary', font_size=9,
               pady=6, padx=10).pack(side='left')
        self.feedback = tk.Label(self.editor, bg=C['surface_2'], fg=C['muted'],
                                 font=(FONT, 9), wraplength=800 if compact else 340)
        self.feedback.pack(anchor='w')
        self.ratio.trace_add('write', self.preview)
        self.refresh()

    def refresh(self):
        data = adjustment.load()
        self.mode_button.configure(text='DEVELOPER  ON' if self.enabled.get() else 'DEVELOPER  OFF')
        set_button_role(self.mode_button, 'primary' if self.enabled.get() else 'quiet')
        suffix = ' · Pause to edit' if not self.editing_allowed and self.enabled.get() else ''
        self.summary.config(text=f"Saved: {data['ratio']:.8g} ×{suffix}")
        if self.enabled.get() and self.edit_ratio and self.editing_allowed:
            self.editor.pack(fill='x', pady=4)
        else:
            self.editor.pack_forget()

    def value(self):
        return (adjustment.validate_ratio(self.ratio.get()) if self.enabled.get() and self.editing_allowed
                else adjustment.current_ratio())

    def _switch_mode(self):
        self.enabled.set(not self.enabled.get())
        self.toggle()

    def set_editing_allowed(self, allowed):
        self.editing_allowed = bool(allowed)
        self.reload()

    def reload(self):
        data = adjustment.load()
        self.enabled.set(data['developer_mode'])
        self.ratio.set(f"{data['ratio']:.8g}")
        self.refresh()

    def toggle(self):
        try:
            adjustment.save(developer_mode=self.enabled.get())
        except (OSError, ValueError) as error:
            self.enabled.set(not self.enabled.get())
            messagebox.showerror('Developer mode not saved', str(error), parent=self)
        self.reload()
        self.preview()

    def preview(self, *_):
        try:
            value = self.value()
        except (ValueError, TypeError):
            self.feedback.config(text='Enter a positive finite ratio; preview unchanged.')
            return
        self.feedback.config(text=f"{(value-1)*100:+.3f}% · Preview only until Save ratio")
        if self.on_preview is not None:
            self.on_preview()

    def save_ratio(self):
        if not self.enabled.get() or not self.editing_allowed:
            return
        try:
            adjustment.save(ratio=self.value())
        except (OSError, ValueError) as error:
            messagebox.showerror('Ratio not saved', str(error), parent=self)
            return
        self.refresh()
        self.feedback.config(text='Saved. Current preview and new measurements use this ratio.')
        if self.on_preview is not None:
            self.on_preview()

    def reset(self):
        if self.enabled.get() and self.editing_allowed:
            self.ratio.set('1')
            self.save_ratio()

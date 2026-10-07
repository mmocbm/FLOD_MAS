"""Large, persistent fabric tabs with the established zoom/pan image viewer."""
import tkinter as tk
import cv2
from PIL import Image, ImageTk
from crop_result_view import CropResultView
from ui_theme import COLORS as C, button as themed_button


class FabricResultView(CropResultView):
    def __init__(self, parent, show_live_preview=True, title='FABRIC RESULTS'):
        self.records = []
        self.show_live_preview = show_live_preview
        self._live_photo = None
        self._live_item = None
        super().__init__(parent, lambda: None, title=title)

    def _build(self):
        super()._build()
        self._pause_button.pack_forget()
        self._countdown_label.configure(text='Held until fabric is removed')
        header = self._tab_buttons[0].master
        for index in range(2, 4):
            tab = themed_button(header, f'FABRIC {index+1}', lambda i=index: self.select_tab(i),
                                role='secondary', font_size=12, padx=14, pady=8)
            tab.pack(side=tk.LEFT, padx=(0, 6))
            self._tab_buttons.append(tab)
        if self.show_live_preview:
            # The measured image keeps most of the screen. The optional side
            # panel is live camera context only, never part of saved results.
            self.canvas.destroy()
            body = tk.Frame(self, bg=C['bg'])
            body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            self.live_panel = tk.Frame(body, bg=C['surface'], width=240)
            self.live_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
            self.live_panel.pack_propagate(False)
            tk.Label(self.live_panel, text='LIVE CAMERA', font=('Segoe UI', 10, 'bold'),
                     bg=C['surface'], fg=C['accent']).pack(anchor='w', padx=12, pady=12)
            self.live_canvas = tk.Canvas(self.live_panel, height=180, bg=C['camera'],
                                         highlightthickness=0)
            self.live_canvas.pack(fill=tk.X, padx=10)
            tk.Label(self.live_panel, text='Hand monitoring stays active.\n'
                     'Add or remove fabric, then\nmove your hands away.',
                     bg=C['surface'], fg=C['muted'], justify='left',
                     font=('Segoe UI', 10)).pack(anchor='w', padx=12, pady=14)
            self.canvas = tk.Canvas(body, bg=C['camera'], highlightthickness=0)
            self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            self.canvas.bind('<MouseWheel>', self._on_wheel)
            self.canvas.bind('<ButtonPress-1>', self._on_pan_start)
            self.canvas.bind('<B1-Motion>', self._on_pan_drag)
            self.canvas.bind('<Configure>', self._on_configure)
        self.summary = tk.Label(self, bg=C['surface'], fg=C['text'],
                                font=('Segoe UI', 14, 'bold'), anchor='w', justify='left')
        self.summary.pack(side=tk.BOTTOM, fill=tk.X, pady=8)

    def update_live_preview(self, frame):
        if not self.show_live_preview:
            return  # no resize or PhotoImage work when the option is off
        canvas = self.live_canvas
        width, height = canvas.winfo_width(), canvas.winfo_height()
        if width <= 1 or height <= 1:
            return
        h, w = frame.shape[:2]
        scale = min(width / w, height / h)
        small = cv2.resize(frame, (max(1, round(w*scale)), max(1, round(h*scale))),
                           interpolation=cv2.INTER_LINEAR)
        self._live_photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
        if self._live_item is None:
            self._live_item = canvas.create_image(width/2, height/2, image=self._live_photo)
        else:
            canvas.itemconfigure(self._live_item, image=self._live_photo)
            canvas.coords(self._live_item, width/2, height/2)

    def _tab_label(self, index):
        return f'FABRIC {index+1}'

    def show_fabrics(self, fabrics):
        self.records = [f.record for f in fabrics]
        self.show_crops([f.image for f in fabrics])
        for index, button in enumerate(self._tab_buttons):
            if index < len(fabrics):
                button.pack(side=tk.LEFT, padx=(0, 6))
                button.configure(state=tk.NORMAL)
            else:
                button.pack_forget()
        self._show_summary()

    def select_tab(self, index):
        super().select_tab(index)
        self._show_summary()

    def _show_summary(self):
        if not self.records:
            return
        item = self.records[self._active]
        measurement = item['measurement']
        text = f"Fabric {item['fabric']}  ·  {item['status']}"
        if measurement and measurement['metric']:
            text += (f"  ·  Length {measurement['total_length_mm']:.2f} mm"
                     f"  ·  Average width {measurement['average_width_mm']:.2f} mm"
                     f"  ·  Target {measurement['target_width_mm']:g} ± "
                     f"{measurement['width_tolerance_mm']:g} mm\n")
            text += '    '.join(f"{s['index']}: {s['average_width_mm']:.2f} mm"
                                for s in measurement['segments'])
        else:
            text += '  ·  Millimetre measurement unavailable — check calibration/segmentation'
        self.summary.configure(text=text)


class LineResultView(FabricResultView):
    """One inspected boundary, with selectable full-resolution overlay and mask."""
    def __init__(self, parent, show_live_preview=True):
        self.measurement = None
        super().__init__(parent, show_live_preview, title='RIGHTMOST LINE INSPECTION')

    def _tab_label(self, index):
        return ('OVERLAY', 'MASK')[index] if index < 2 else ''

    def _build(self):
        from tkinter import ttk
        super()._build()
        for button in self._tab_buttons[2:]:
            button.pack_forget()
        self._countdown_label.configure(text='Captured result · hand monitoring active')
        table_frame = tk.Frame(self, bg=C['surface'])
        table_frame.pack(side=tk.BOTTOM, fill=tk.X)
        style = ttk.Style(self)
        style.configure('Inspection.Treeview', background=C['surface'],
                        fieldbackground=C['surface'], foreground=C['text_soft'],
                        rowheight=22, font=('Segoe UI', 10), borderwidth=0)
        style.configure('Inspection.Treeview.Heading', background=C['surface_2'],
                        foreground=C['text'], font=('Segoe UI', 10, 'bold'))
        style.map('Inspection.Treeview', background=[('selected', C['accent_dark'])],
                  foreground=[('selected', C['text'])])
        columns = ('segment', 'average_px', 'minimum_px', 'maximum_px',
                   'average_mm', 'minimum_mm', 'maximum_mm', 'grade')
        self.segment_table = ttk.Treeview(table_frame, columns=columns, show='headings', height=5,
                                         style='Inspection.Treeview')
        self.segment_table.tag_configure('PASS', foreground=C['success'])
        self.segment_table.tag_configure('FAIL', foreground=C['danger'])
        for key, label in zip(columns, ('Segment', 'Avg px', 'Min px', 'Max px',
                                       'Avg mm', 'Min mm', 'Max mm', 'Grade')):
            self.segment_table.heading(key, text=label)
            self.segment_table.column(key, width=85, anchor='center', stretch=True)
        scroll = ttk.Scrollbar(table_frame, orient='vertical', command=self.segment_table.yview)
        self.segment_table.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.segment_table.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def show_inspection(self, result):
        self.measurement = result['measurement']
        self.records = [self.measurement]
        self.show_crops([result['overlay'], cv2.cvtColor(result['mask'], cv2.COLOR_GRAY2BGR)])
        for button in self._tab_buttons[:2]:
            button.configure(state=tk.NORMAL)
        self.segment_table.delete(*self.segment_table.get_children())
        for segment in self.measurement['segments']:
            values = [segment['segment']]
            for unit in ('px', 'mm'):
                for name in ('average', 'minimum', 'maximum'):
                    value = segment[f'{name}_width_{unit}']
                    values.append('—' if value is None else f'{value:.2f}')
            grade = segment['within_tolerance']
            values.append('—' if grade is None else 'PASS' if grade else 'FAIL')
            self.segment_table.insert('', 'end', values=values, tags=(values[-1],))
        self._show_summary()

    def _show_summary(self):
        if self.measurement is None:
            return
        m = self.measurement
        mean_px = '—' if m['mean_width_px'] is None else f"{m['mean_width_px']:.2f}"
        text = (f"Rightmost line {m['selected_line']} / {m['line_count']} · {m['status']} · "
                f"{m['component_count']} adhesive component(s) · {m['seconds']:.2f}s\n"
                f"Total length {m['length_px']:.2f} px · Mean width {mean_px} px")
        if m['metric']:
            text += (f" · Length {m['length_mm']:.2f} mm · Width {m['mean_width_mm']:.2f} mm"
                     f" · Target {m['target_width_mm']:g} ± {m['tolerance_mm']:g} mm")
        else:
            text += ' · Millimetre measurement unavailable'
        text += '\nWidths and segment grades describe the longest adhesive component.'
        self.summary.configure(text=text, font=('Segoe UI', 10, 'bold'), wraplength=1000)

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
    """One persistent tab per inspected line, each retaining its zoom and pan."""
    def __init__(self, parent, show_live_preview=True):
        super().__init__(parent, show_live_preview, title='INSPECTION RESULTS')

    def _build(self):
        super()._build()
        for button in self._tab_buttons:
            button.destroy()
        self._tab_buttons = []
        self._countdown_label.configure(text='Wheel: zoom · Drag: pan')
        strip = tk.Frame(self, bg=C['bg'])
        strip.pack(side=tk.TOP, fill=tk.X,
                   before=self.canvas.master if self.show_live_preview else self.canvas)
        self.tab_strip = tk.Canvas(strip, height=42, bg=C['bg'], highlightthickness=0)
        self.tab_strip.pack(fill=tk.X)
        scroll = tk.Scrollbar(strip, orient=tk.HORIZONTAL, command=self.tab_strip.xview)
        scroll.pack(fill=tk.X)
        self.tab_strip.configure(xscrollcommand=scroll.set)
        self.tab_header = tk.Frame(self.tab_strip, bg=C['bg'])
        self.tab_strip.create_window(0, 0, window=self.tab_header, anchor='nw')
        self.tab_header.bind('<Configure>', lambda event:
                             self.tab_strip.configure(scrollregion=self.tab_strip.bbox('all')))

    def _tab_label(self, index):
        if index < len(self.records):
            return f"Fabric {index+1} · {self.records[index]['status']}"
        return f'Fabric {index+1}'

    def show_inspection(self, result):
        import numpy as np
        self.records.append(result['measurement'])
        # Fit the actual inspected line, instead of shrinking it inside the whole bed.
        ys, xs = np.nonzero(result['mask'])
        image = result['overlay']
        if len(xs):
            x0, x1 = max(0, int(xs.min())-160), min(image.shape[1], int(xs.max())+180)
            y0, y1 = max(0, int(ys.min())-40), min(image.shape[0], int(ys.max())+40)
            image = image[y0:y1, x0:x1]
        self._tabs.append(self._blank_tab(self._to_pil(image)))
        index = len(self._tabs)-1
        button = themed_button(self.tab_header, self._tab_label(index),
                               lambda i=index: self.select_tab(i), role='secondary',
                               font_size=11, padx=14, pady=8)
        button.pack(side=tk.LEFT, padx=(0, 5))
        self._tab_buttons.append(button)
        self._active = index
        self._refresh_tab_labels()
        self.tab_header.update_idletasks()
        self.tab_strip.configure(scrollregion=self.tab_strip.bbox('all'))
        self.tab_strip.xview_moveto(1.0)
        self._fit_tab(self._active_tab())
        self._render()
        self._show_summary()

    def _show_summary(self):
        if self.records:
            item = self.records[self._active]
            self.summary.configure(text=f"Fabric {self._active+1} · {item['status']}",
                                   font=('Segoe UI', 12, 'bold'))

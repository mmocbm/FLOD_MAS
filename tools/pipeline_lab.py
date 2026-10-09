"""Pipeline Lab: try pre-processing and segmentation on any image, by hand.

    .venv\\Scripts\\python.exe tools\\pipeline_lab.py [image]

Open an image, choose the region of interest, build a list of pre-processing
steps in any order, pick a segmenter and press Run. The picture after every step
and the final glue-line outline can be viewed, and the recipe saved for reuse.

The SAM choices send the image area to the Roboflow API (key in ``.env``); the
local detector and all pre-processing run on this computer.
"""
from __future__ import annotations

import json
import queue
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT), str(Path(__file__).resolve().parent)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import cv2  # noqa: E402 - after the path is prepared
import numpy as np  # noqa: E402
from PIL import Image, ImageTk  # noqa: E402

import pipeline_steps as steps  # noqa: E402
from segmentation import sam_local  # noqa: E402
from crop_result_view import clamp_offset, fit_view, viewport, zoom_about  # noqa: E402

ROI_CHOICES = (('whole', 'Whole image'), ('model', 'Panel model (trained ROI)'),
               ('box', 'Box drawn by hand'))
SEGMENTERS = (('auto', 'SAM 3 - points placed automatically'),
              ('clicks', 'SAM 3 - points I click on the image'),
              ('text', 'SAM 3 - text prompt'))
SAM2_SIZES = ('hiera_tiny', 'hiera_small', 'hiera_b_plus', 'hiera_large')
LOCAL_MODELS = tuple(sam_local.MODELS)
DEFAULT_LOCAL_MODEL = 'sam2.1-large'
SAM_RUNS = (('local', "This PC's GPU (nothing uploaded)"), ('cloud', 'Roboflow cloud'))
PREVIEW_WIDTH = 1400       # live preview works on a copy this wide
PREVIEW_DELAY_MS = 150     # wait this long after the last change before redrawing
MOUSE_MODES = (('pan', 'Pan'), ('box', 'Draw ROI box'), ('points', 'Place points'))
DEFAULT_PROMPT = 'thin glossy line'
WORKING_KEYS = ('evidence', 'rough', 'track', 'deviations')   # per-sample working data, not saved
STATE_COLOUR = {'shiny': (255, 255, 0), 'matte': (255, 0, 0), 'one-sided': (0, 0, 255),   # BGR
                'weak': (0, 140, 255), 'none': (128, 128, 128)}
OUTLINE = {'local': (0, 200, 255)}        # BGR; anything from SAM is green


def run_job(image, recipe, box, points, client=None, progress=None, roi=None):
    """Everything Run does, without a window: returns a dict of results.

    ``recipe`` holds ``roi``, ``pipeline``, ``segmenter``, ``sam2_size``,
    ``point_source`` and ``prompt``. ``box`` is ``(x0, y0, x1, y1)`` or None and
    ``points`` a list of ``(x, y, positive)``, both in pixels of the original image.
    """
    started = time.perf_counter()
    notes = []
    say = progress or (lambda text: None)
    say('Finding the region of interest...')
    if roi is not None:
        pass                                # already worked out by the caller
    elif recipe['roi'] == 'model':
        roi = steps.model_roi(image, on_gpu=recipe.get('sam_runs') == 'local')
    elif recipe['roi'] == 'box' and box is not None:
        roi = np.zeros(image.shape[:2], np.uint8)
        x0, y0, x1, y1 = (int(round(value)) for value in box)
        roi[max(0, min(y0, y1)):max(y0, y1), max(0, min(x0, x1)):max(x0, x1)] = 1
    else:
        roi = np.ones(image.shape[:2], np.uint8)
        if recipe['roi'] == 'box':
            notes.append('No box drawn: the whole image was used.')
    say('Applying the pre-processing steps...')
    processed, roi_now, stages = steps.run_pipeline(image, roi, recipe['pipeline'], keep_stages=True)
    factor = processed.shape[1] / image.shape[1]
    moved = [(x * factor, y * factor, positive) for x, y, positive in points]

    kind = recipe['segmenter']
    local = recipe.get('sam_runs') == 'local'
    if local:                               # on this computer's GPU: nothing is uploaded
        model = sam_local.PREFIX + recipe.get('local_model', DEFAULT_LOCAL_MODEL)
    else:
        model = 'sam3' if kind == 'sam3' else 'sam2:' + recipe.get('sam2_size', SAM2_SIZES[-1])
    if local and kind == 'sam3_text':
        raise ValueError('The text prompt needs the Roboflow cloud; choose points, or cloud.')
    guided = recipe['roi'] == 'model'
    def rough_lines():
        # Found on the original picture, whatever the steps did to it: it is what
        # the detector was built for, and an enlarged picture would take minutes.
        say('Finding the lines roughly (local detector)...')
        found = steps.segment_local(image, roi, guided)
        for line in found:
            line['ring'] = line['ring'] * factor
        return found

    if kind == 'local':
        lines = rough_lines()
    elif kind == 'sam3_text':
        say('Asking SAM 3 (text prompt)...')
        lines = steps.segment_sam_text(processed, roi_now, recipe.get('prompt') or DEFAULT_PROMPT,
                                       client)
    elif recipe.get('point_source') == 'clicks':
        say('Asking SAM with your points...')
        lines = steps.segment_sam_from_clicks(processed, moved, model, client)
    else:
        rough = rough_lines()
        if not rough:
            notes.append('No line was found to place points on. Place points by hand instead.')
        say(f'Asking SAM to outline {len(rough)} line(s)...')
        # A bead is wider in pixels on an enlarged picture; so is the limit on what
        # still counts as a bead.
        lines = steps.segment_sam_from_lines(
            processed, roi_now, rough, model, client, max_width_px=80.0 * max(1.0, factor),
            scale=max(1.0, factor), progress=say,
            second_looks=() if recipe.get('ask_once') else steps.SECOND_LOOKS)
    if recipe.get('bead_edges') and kind != 'sam3_text' and recipe.get('point_source') != 'clicks':
        say("Finding the bead's own edges...")
        lines = steps.finish_on_bead_edges(image, roi, lines, factor)
    if recipe.get('ribbon') and kind != 'sam3_text' and recipe.get('point_source') != 'clicks':
        say('Fitting the smooth ribbon...')
        lines = steps.finish_as_ribbon(image, roi, lines, factor,
                                       edge_method=recipe.get('edge_method', 'slope'))
    if local:
        notes.append(f"SAM ran on this PC ({recipe.get('local_model', DEFAULT_LOCAL_MODEL)}); nothing was uploaded.")
    say('Measuring...')
    for line in lines:
        line.update(steps.ring_stats(line['ring']))

    overlay = processed.copy()
    for contour in cv2.findContours((roi_now > 0).astype(np.uint8), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)[0]:
        cv2.drawContours(overlay, [contour], -1, (0, 0, 255), max(2, overlay.shape[1] // 1200))
    for line in lines:                       # the outline before the edges were found, thin
        if 'first_ring' in line:
            cv2.polylines(overlay, [np.round(line['first_ring']).astype(np.int32)], True,
                          (255, 0, 255), 1)
    thick = max(2, overlay.shape[1] // 1500)
    for line in lines:
        # What the picture is like along the line (dots beside it), and stretches
        # where the evidence is steadily wider or narrower than the ribbon (red).
        if 'track' in line:
            for point, state in zip(line['track']['points'][::6], line['track']['state'][::6]):
                cv2.circle(overlay, (int(point[0]) + 14 * thick, int(point[1])), thick,
                           STATE_COLOUR[str(state)], -1)
        for points, _ in line.get('deviations', ()):
            cv2.polylines(overlay, [np.round(points).astype(np.int32)], False, (0, 0, 255), 3 * thick)
    for number, line in enumerate(lines, 1):
        ring = np.round(line['ring']).astype(np.int32)
        colour = (0, 255, 0) if 'edges_seen' in line or 'backed' in line else OUTLINE.get(line['source'].split(' ')[0], (0, 255, 0))
        cv2.polylines(overlay, [ring], True, colour, max(2, overlay.shape[1] // 1500))
        x, y = int(ring[:, 0].min()), int(ring[:, 1].min())
        cv2.putText(overlay, str(number), (x, max(30, y - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.8, overlay.shape[1] / 2500), colour, 3, cv2.LINE_AA)
    return {'processed': processed, 'roi': roi_now, 'stages': stages, 'lines': lines,
            'fabric': steps.fabric_colour(image, roi),
            'overlay': overlay, 'notes': notes, 'seconds': time.perf_counter() - started,
            'scale': factor}


def describe(result):
    rows = [f"{len(result['lines'])} line(s) in {result['seconds']:.1f} s, {result.get('fabric', '?')} fabric"
            + (f", image scaled x{result['scale']:g}" if result['scale'] != 1 else '')]
    for number, line in enumerate(result['lines'], 1):
        rows.append(f"  {number}: {line['source']:<28} length {line['length_px']:.0f} px, "
                    f"width {line['width_px']:.1f} px ({line['width_p10']:.0f}-{line['width_p90']:.0f})"
                    + (f", SAM outlined {line['seen']:.0%}" if 'seen' in line else '')
                    + (f" (asked {line['asked']}x)" if line.get('asked', 1) > 1 else '')
                    + (f", edges steady {line['edges_seen']:.0%}" if 'edges_seen' in line else '')
                    + (f", edges {line['edges']}" if 'edges' in line else '')
                    + (f", ribbon backed by evidence {line['backed']:.0%}, misfit {line['misfit']:.1f} px"
                       + (' - CHECK BY EYE' if line.get('doubtful') else '')
                       + (f", {len(line['deviations'])} width deviation(s): "
                          + ', '.join(f'{pixels:+.0f} px' for _, pixels in line['deviations'])
                          if line.get('deviations') else '') if 'backed' in line else '')
                    + (f", ribbon {line['ribbon']}" if 'ribbon' in line else ''))
    return '\n'.join(rows + result['notes'])


class PipelineLab:
    def __init__(self, root, image_path=None):
        self.root = root
        root.title('Pipeline Lab - glue line')
        root.geometry('1500x900')
        self.image = None
        self.image_path = None
        self.result = None
        self.pipeline = []
        self.box = None
        self.points = []
        self.views = {}
        self.jobs = queue.Queue()
        self.busy = False
        self._photo = None
        self._view = {'fit': 0.0, 'zoom': 1.0, 'offset': [0.0, 0.0]}
        self._drag = None
        self._setting_widgets = []
        self._preview_timer = None
        self.preview_token = 0
        self.image_token = 0
        self.model_roi = None
        self.preview_base = None
        self.preview_factor = 1.0
        self.run_stage = ''
        self.run_started = 0.0
        self._build()
        if image_path:
            self.open_image(image_path)
        root.after(100, self._poll)

    # ---------------------------------------------------------------- layout
    def _build(self):
        left = ttk.Frame(self.root, padding=8, width=390)
        left.pack(side=tk.LEFT, fill=tk.Y)
        left.pack_propagate(False)
        right = ttk.Frame(self.root, padding=(0, 8, 8, 8))
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        ttk.Button(left, text='Open image...', command=self.open_image).pack(fill=tk.X)
        self.image_label = ttk.Label(left, text='No image', wraplength=360)
        self.image_label.pack(anchor='w', pady=(2, 8))

        box = ttk.LabelFrame(left, text='1. Region of interest', padding=6)
        box.pack(fill=tk.X)
        self.roi_var = tk.StringVar(value='model')
        for value, label in ROI_CHOICES:
            ttk.Radiobutton(box, text=label, value=value, variable=self.roi_var,
                            command=self._changed).pack(anchor='w')

        box = ttk.LabelFrame(left, text='2. Pre-processing steps (top to bottom)', padding=6)
        box.pack(fill=tk.BOTH, expand=True, pady=8)
        self.step_list = tk.Listbox(box, height=8, exportselection=False, activestyle='none')
        self.step_list.pack(fill=tk.X)
        self.step_list.bind('<<ListboxSelect>>', lambda event: self._show_settings())
        row = ttk.Frame(box)
        row.pack(fill=tk.X, pady=4)
        self.add_var = tk.StringVar(value='Add step...')
        menu = ttk.OptionMenu(row, self.add_var, 'Add step...',
                              *[step.label for step in steps.STEPS.values()],
                              command=self._add_step)
        menu.pack(side=tk.LEFT, fill=tk.X, expand=True)
        for text, command in (('Up', lambda: self._move(-1)), ('Down', lambda: self._move(1)),
                              ('On/off', self._toggle), ('Remove', self._remove)):
            ttk.Button(row, text=text, width=7, command=command).pack(side=tk.LEFT, padx=1)
        self.settings_frame = ttk.Frame(box)
        self.settings_frame.pack(fill=tk.BOTH, expand=True)

        box = ttk.LabelFrame(left, text='3. Segmenter (SAM)', padding=6)
        box.pack(fill=tk.X)
        self.segmenter_var = tk.StringVar(value='auto')
        for value, label in SEGMENTERS:
            ttk.Radiobutton(box, text=label, value=value, variable=self.segmenter_var).pack(anchor='w')
        row = ttk.Frame(box)
        row.pack(fill=tk.X)
        ttk.Label(row, text='Text prompt').pack(side=tk.LEFT)
        self.runs_var = tk.StringVar(value='local' if steps.gpu_available() else 'cloud')
        row = ttk.Frame(box)
        row.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(row, text='SAM runs on').pack(side=tk.LEFT)
        for value, label in SAM_RUNS:
            ttk.Radiobutton(row, text=label, value=value, variable=self.runs_var).pack(side=tk.LEFT, padx=4)
        row = ttk.Frame(box)
        row.pack(fill=tk.X)
        ttk.Label(row, text='Local model').pack(side=tk.LEFT)
        self.local_model_var = tk.StringVar(value=DEFAULT_LOCAL_MODEL)
        ttk.OptionMenu(row, self.local_model_var, DEFAULT_LOCAL_MODEL, *LOCAL_MODELS).pack(side=tk.LEFT, padx=6)
        self.once_var = tk.BooleanVar(value=steps.gpu_available())
        ttk.Checkbutton(box, text='Ask SAM once per line (faster; use with the ribbon)',
                        variable=self.once_var).pack(anchor='w')
        self.edges_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(box, text="Finish on the bead's own edges (automatic points)",
                        variable=self.edges_var).pack(anchor='w')
        self.ribbon_var = tk.BooleanVar(value=steps.gpu_available())   # asking once relies on it
        ttk.Checkbutton(box, text='Smooth ribbon from the true shape (automatic points)',
                        variable=self.ribbon_var).pack(anchor='w')
        row = ttk.Frame(box)
        row.pack(fill=tk.X)
        ttk.Label(row, text='Edges placed by').pack(side=tk.LEFT)
        choices = steps.edge_methods_available()
        self.edge_method_var = tk.StringVar(value=choices[0])
        ttk.OptionMenu(row, self.edge_method_var, choices[0], *choices).pack(side=tk.LEFT, padx=6)
        self.prompt_var = tk.StringVar(value=DEFAULT_PROMPT)
        ttk.Entry(row, textvariable=self.prompt_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        ttk.Label(box, text="Cloud: Run uploads the image area to Roboflow. This PC's GPU: nothing is uploaded.",
                  foreground='#a05a00').pack(anchor='w')

        row = ttk.Frame(left)
        row.pack(fill=tk.X, pady=8)
        self.run_button = ttk.Button(row, text='Run', command=self.run)
        self.run_button.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(row, text='Save result', command=self.save_result).pack(side=tk.LEFT, padx=2)
        row = ttk.Frame(left)
        row.pack(fill=tk.X)
        ttk.Button(row, text='Save recipe', command=self.save_recipe).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(row, text='Load recipe', command=self.load_recipe).pack(side=tk.LEFT, fill=tk.X, expand=True)

        top = ttk.Frame(right)
        top.pack(fill=tk.X)
        ttk.Label(top, text='Show').pack(side=tk.LEFT)
        self.view_var = tk.StringVar(value='Original')
        self.view_box = ttk.Combobox(top, textvariable=self.view_var, values=['Original'],
                                     state='readonly', width=34)
        self.view_box.pack(side=tk.LEFT, padx=6)
        self.view_box.bind('<<ComboboxSelected>>', lambda event: self._render())
        ttk.Label(top, text='Mouse').pack(side=tk.LEFT, padx=(14, 0))
        self.mouse_var = tk.StringVar(value='pan')
        for value, label in MOUSE_MODES:
            ttk.Radiobutton(top, text=label, value=value, variable=self.mouse_var).pack(side=tk.LEFT, padx=3)
        ttk.Button(top, text='Clear points', command=self._clear_points).pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text='Fit', command=self._fit).pack(side=tk.LEFT)

        self.canvas = tk.Canvas(right, bg='#202020', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, pady=6)
        self.canvas.bind('<Configure>', lambda event: self._fit())
        self.canvas.bind('<ButtonPress-1>', self._press)
        self.canvas.bind('<B1-Motion>', self._motion)
        self.canvas.bind('<ButtonRelease-1>', self._release)
        self.canvas.bind('<ButtonPress-3>', lambda event: self._place_point(event, False))
        self.canvas.bind('<MouseWheel>', self._wheel)
        self.canvas.bind('<Button-4>', lambda event: self._wheel(event, 1))
        self.canvas.bind('<Button-5>', lambda event: self._wheel(event, -1))

        self.status = tk.Text(right, height=7, wrap='word', state='disabled', font=('TkFixedFont', 10))
        self.status.pack(fill=tk.X)
        self._say('Open an image, choose the ROI and add steps: the picture updates as you change '
                  'them.\nThen press Run to outline the glue line with SAM 3.\n'
                  'Place points: left click = glue, right click = not glue.')

    # ------------------------------------------------------------- pipeline
    def _refresh_steps(self, select=None):
        self.step_list.delete(0, tk.END)
        for number, entry in enumerate(self.pipeline, 1):
            mark = 'x' if entry['enabled'] else ' '
            self.step_list.insert(tk.END, f"[{mark}] {number}. {steps.STEPS[entry['step']].label}")
        if select is not None and 0 <= select < len(self.pipeline):
            self.step_list.selection_set(select)
        self._show_settings()

    def _selected(self):
        chosen = self.step_list.curselection()
        return chosen[0] if chosen else None

    def _add_step(self, label):
        self.add_var.set('Add step...')
        name = next(step.name for step in steps.STEPS.values() if step.label == label)
        self.pipeline.append(steps.new_entry(name))
        self._refresh_steps(len(self.pipeline) - 1)
        self._changed()

    def _move(self, by):
        index = self._selected()
        if index is None or not 0 <= index + by < len(self.pipeline):
            return
        self.pipeline[index], self.pipeline[index + by] = self.pipeline[index + by], self.pipeline[index]
        self._refresh_steps(index + by)
        self._changed()

    def _toggle(self):
        index = self._selected()
        if index is not None:
            self.pipeline[index]['enabled'] = not self.pipeline[index]['enabled']
            self._refresh_steps(index)
            self._changed()

    def _remove(self):
        index = self._selected()
        if index is not None:
            del self.pipeline[index]
            self._refresh_steps(min(index, len(self.pipeline) - 1))
            self._changed()

    def _show_settings(self):
        for widget in self.settings_frame.winfo_children():
            widget.destroy()
        index = self._selected()
        if index is None:
            ttk.Label(self.settings_frame, text='Select a step to change its settings.',
                      foreground='#666').pack(anchor='w', pady=6)
            return
        entry = self.pipeline[index]
        step = steps.STEPS[entry['step']]
        ttk.Label(self.settings_frame, text=step.help, wraplength=350,
                  foreground='#444').pack(anchor='w', pady=(6, 2))
        for setting in step.settings:
            row = ttk.Frame(self.settings_frame)
            row.pack(fill=tk.X, pady=1)
            ttk.Label(row, text=setting.label, width=14).pack(side=tk.LEFT)
            shown = ttk.Label(row, width=6)
            shown.pack(side=tk.RIGHT)

            current = entry['params'].get(setting.name, setting.default)
            shown.configure(text=f'{current:g}')
            ready = []          # the slider reports values while it is being laid out

            def changed(raw, entry=entry, setting=setting, shown=shown, ready=ready):
                if not ready:
                    return
                value = setting.low + round((float(raw) - setting.low) / setting.step) * setting.step
                value = int(round(value)) if setting.whole else round(value, 3)
                if entry['params'].get(setting.name) != value:
                    entry['params'][setting.name] = value
                    shown.configure(text=f'{value:g}')
                    self._changed()
            scale = ttk.Scale(row, from_=setting.low, to=setting.high, value=current,
                              command=changed)
            scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.root.after_idle(lambda ready=ready: ready.append(True))

    # ---------------------------------------------------------------- image
    def open_image(self, path=None):
        path = path or filedialog.askopenfilename(
            title='Open image', filetypes=[('Images', '*.jpg *.jpeg *.png *.bmp'), ('All files', '*')])
        if not path:
            return
        image = cv2.imread(str(path))
        if image is None:
            messagebox.showerror('Cannot open', f'Not a readable image:\n{path}', parent=self.root)
            return
        self.image, self.image_path = image, Path(path)
        self.result, self.box, self.points = None, None, []
        self.image_label.configure(text=f'{self.image_path.name}  ({image.shape[1]} x {image.shape[0]})')
        self._set_views({'Original': image})
        self._fit()
        # A reduced copy for the live preview, and the panel model's ROI worked out
        # once in the background (it takes a second or two).
        factor = min(1.0, PREVIEW_WIDTH / image.shape[1])
        self.preview_factor = factor
        self.preview_base = (image if factor == 1.0 else
                             cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA))
        self.model_roi = None
        token = self.image_token = self.image_token + 1

        def find_roi():
            try:
                self.jobs.put(('roi', (token, steps.model_roi(image, on_gpu=self.runs_var.get() == 'local'))))
            except Exception as error:  # noqa: BLE001 - the preview falls back to the whole image
                self.jobs.put(('note', f'Panel model unavailable: {type(error).__name__}: {error}'))
        threading.Thread(target=find_roi, daemon=True).start()
        self._changed()

    # -------------------------------------------------------------- preview
    def _roi_mask(self, shape):
        """The chosen ROI as a mask of the given (height, width)."""
        choice = self.roi_var.get()
        if choice == 'model' and self.model_roi is not None:
            return cv2.resize(self.model_roi, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = np.ones(shape[:2], np.uint8)
        if choice == 'box' and self.box is not None:
            factor = shape[1] / self.image.shape[1]
            x0, y0, x1, y1 = (int(round(value * factor)) for value in self.box)
            mask[:] = 0
            mask[max(0, min(y0, y1)):max(y0, y1), max(0, min(x0, x1)):max(x0, x1)] = 1
        return mask

    def _changed(self):
        """Something that affects the picture changed: redraw the preview shortly."""
        if self.image is None:
            return
        if self._preview_timer is not None:
            self.root.after_cancel(self._preview_timer)
        self._preview_timer = self.root.after(PREVIEW_DELAY_MS, self._start_preview)

    def _start_preview(self):
        self._preview_timer = None
        token = self.preview_token = self.preview_token + 1
        pipeline = json.loads(json.dumps(self.pipeline))
        base, factor = self.preview_base, self.preview_factor
        roi = self._roi_mask(base.shape)

        def work():
            try:
                picture, mask, _ = steps.run_pipeline(base, roi, pipeline, pixel_scale=factor)
                if picture.shape[1] > 2 * PREVIEW_WIDTH:        # a large resize step
                    shrink = 2 * PREVIEW_WIDTH / picture.shape[1]
                    picture = cv2.resize(picture, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_AREA)
                    mask = cv2.resize(mask, (picture.shape[1], picture.shape[0]), interpolation=cv2.INTER_NEAREST)
                picture = picture.copy()
                if not mask.all():
                    for contour in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
                        cv2.drawContours(picture, [contour], -1, (0, 0, 255), 2)
                self.jobs.put(('preview', (token, picture)))
            except Exception as error:  # noqa: BLE001 - shown, never fatal
                self.jobs.put(('note', f'Preview failed: {type(error).__name__}: {error}'))
        threading.Thread(target=work, daemon=True).start()

    def _show_preview(self, picture):
        keep = self.view_var.get() in ('Preview', 'Original') or self.view_var.get() not in self.views
        previous = self.views.get('Preview')
        views = {'Original': self.image, 'Preview': picture}
        views.update({name: image for name, image in self.views.items() if name not in views})
        self.views = views
        self.view_box.configure(values=list(views))
        if keep:
            same_size = (self.view_var.get() == 'Preview' and previous is not None
                         and previous.shape == picture.shape)
            self.view_var.set('Preview')
            if same_size:
                self._render()          # keep the zoom and position while sliding
            else:
                self._fit()

    def _set_views(self, views, show=None):
        self.views = views
        self.view_box.configure(values=list(views))
        self.view_var.set(show if show in views else next(iter(views)))

    def _shown(self):
        return self.views.get(self.view_var.get())

    def _canvas_size(self):
        return max(2, self.canvas.winfo_width()), max(2, self.canvas.winfo_height())

    def _fit(self):
        image = self._shown()
        if image is None:
            return
        fit, offset = fit_view((image.shape[1], image.shape[0]), self._canvas_size())
        self._view = {'fit': fit, 'zoom': 1.0, 'offset': offset}
        self._render()

    def _scale(self):
        return max(1e-9, self._view['fit'] * self._view['zoom'])

    def _to_original(self, x, y):
        """Canvas position -> pixels of the original image."""
        image = self._shown()
        factor = self.image.shape[1] / image.shape[1]
        return ((x - self._view['offset'][0]) / self._scale() * factor,
                (y - self._view['offset'][1]) / self._scale() * factor)

    def _render(self):
        self.canvas.delete('all')
        image = self._shown()
        if image is None:
            return
        size, canvas = (image.shape[1], image.shape[0]), self._canvas_size()
        if self._view['fit'] <= 0:
            self._view['fit'], self._view['offset'] = fit_view(size, canvas)
        scale = self._scale()
        self._view['offset'] = clamp_offset(size, canvas, scale, self._view['offset'])
        visible = viewport(size, canvas, scale, self._view['offset'])
        if visible is None:
            return
        left, top, right, bottom = visible
        part = cv2.resize(image[top:bottom, left:right],
                          (max(1, round((right - left) * scale)), max(1, round((bottom - top) * scale))),
                          interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_NEAREST)
        self._photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(part, cv2.COLOR_BGR2RGB)))
        ox, oy = self._view['offset']
        self.canvas.create_image(ox + left * scale, oy + top * scale, image=self._photo, anchor='nw')
        factor = image.shape[1] / self.image.shape[1] * scale     # original pixels -> canvas
        if self.box is not None and self.roi_var.get() == 'box':
            x0, y0, x1, y1 = self.box
            self.canvas.create_rectangle(ox + x0 * factor, oy + y0 * factor, ox + x1 * factor,
                                         oy + y1 * factor, outline='#ff3030', width=2)
        for x, y, positive in self.points:
            cx, cy = ox + x * factor, oy + y * factor
            self.canvas.create_oval(cx - 5, cy - 5, cx + 5, cy + 5, outline='white',
                                    fill='#20d040' if positive else '#3060ff')

    # ---------------------------------------------------------------- mouse
    def _press(self, event):
        if self.image is None:
            return
        mode = self.mouse_var.get()
        if mode == 'points':
            self._place_point(event, True)
        elif mode == 'box':
            x, y = self._to_original(event.x, event.y)
            self.box = (x, y, x, y)
            self.roi_var.set('box')
            self._drag = 'box'
        else:
            self._drag = (event.x, event.y, *self._view['offset'])

    def _motion(self, event):
        if self._drag == 'box':
            x, y = self._to_original(event.x, event.y)
            self.box = (self.box[0], self.box[1], x, y)
            self._render()
        elif self._drag:
            start_x, start_y, offset_x, offset_y = self._drag
            self._view['offset'] = [offset_x + event.x - start_x, offset_y + event.y - start_y]
            self._render()

    def _release(self, event):
        if self._drag == 'box':
            self._changed()
        self._drag = None

    def _place_point(self, event, positive):
        if self.image is None or self.mouse_var.get() != 'points':
            return
        x, y = self._to_original(event.x, event.y)
        if 0 <= x < self.image.shape[1] and 0 <= y < self.image.shape[0]:
            self.points.append((x, y, positive))
            self.segmenter_var.set('clicks')
            self._render()

    def _clear_points(self):
        self.points = []
        self._render()

    def _wheel(self, event, direction=None):
        if self._shown() is None:
            return
        direction = direction if direction is not None else (1 if event.delta > 0 else -1)
        self._view['zoom'], self._view['offset'] = zoom_about(
            self._view['fit'], self._view['offset'], self._view['zoom'],
            1.25 if direction > 0 else 0.8, (event.x, event.y))
        self._render()

    # ------------------------------------------------------------------ run
    def recipe(self):
        choice = self.segmenter_var.get()
        return {'roi': self.roi_var.get(), 'pipeline': self.pipeline,
                'segmenter': 'sam3_text' if choice == 'text' else 'sam3',
                'point_source': 'clicks' if choice == 'clicks' else 'lines',
                'prompt': self.prompt_var.get(), 'bead_edges': bool(self.edges_var.get()),
                'ribbon': bool(self.ribbon_var.get()), 'sam_runs': self.runs_var.get(),
                'local_model': self.local_model_var.get(), 'ask_once': bool(self.once_var.get()),
                'edge_method': self.edge_method_var.get()}

    def run(self):
        if self.image is None or self.busy:
            return
        self.busy = True
        self.run_started = time.perf_counter()
        self.run_stage = 'Starting...'
        self.run_button.configure(state='disabled')
        recipe = json.loads(json.dumps(self.recipe()))      # a private copy for the worker
        image, box, points = self.image, self.box, list(self.points)
        roi = self.model_roi if recipe['roi'] == 'model' else None

        def work():
            try:
                self.jobs.put(('done', run_job(
                    image, recipe, box, points, roi=roi,
                    progress=lambda text: self.jobs.put(('stage', text)))))
            except Exception as error:  # noqa: BLE001 - shown to the user
                self.jobs.put(('error', f'{type(error).__name__}: {error}'))
        threading.Thread(target=work, daemon=True).start()

    def _poll(self):
        """Take what the worker threads have finished. Nothing here may stop the
        polling: a failure is shown and the Run button is given back."""
        try:
            while True:
                try:
                    kind, payload = self.jobs.get_nowait()
                except queue.Empty:
                    break
                self._handle(kind, payload)
            if self.busy:
                self._say(f'{self.run_stage}   ({time.perf_counter() - self.run_started:.0f} s)')
        except Exception as error:  # noqa: BLE001
            self.busy = False
            self.run_button.configure(state='normal')
            self._say(f'Failed: {type(error).__name__}: {error}')
        self.root.after(100, self._poll)

    def _handle(self, kind, payload):
        if kind == 'stage':
            self.run_stage = payload
        elif kind == 'note':
            if not self.busy:
                self._say(payload)
        elif kind == 'roi':
            token, mask = payload
            if token == self.image_token:
                self.model_roi = mask
                self._changed()
        elif kind == 'preview':
            token, picture = payload
            if token == self.preview_token and self.image is not None:
                self._show_preview(picture)
        elif kind == 'error':
            self.busy = False
            self.run_button.configure(state='normal')
            self._say('Failed: ' + payload)
        elif kind == 'done':
            self.busy = False
            self.run_button.configure(state='normal')
            self.result = payload
            views = {'Original': self.image}
            if 'Preview' in self.views:
                views['Preview'] = self.views['Preview']
            views.update({f'After {label}': image for label, image in payload['stages']})
            views['Result'] = payload['overlay']
            self._set_views(views, 'Result')
            self._fit()
            self._say(describe(payload))

    def _say(self, text):
        self.status.configure(state='normal')
        self.status.delete('1.0', tk.END)
        self.status.insert('1.0', text)
        self.status.configure(state='disabled')

    # ----------------------------------------------------------------- files
    def save_recipe(self):
        path = filedialog.asksaveasfilename(title='Save recipe', defaultextension='.json',
                                            filetypes=[('Recipe', '*.json')])
        if path:
            steps.save_recipe(path, self.recipe())

    def load_recipe(self, path=None):
        path = path or filedialog.askopenfilename(title='Load recipe', filetypes=[('Recipe', '*.json')])
        if not path:
            return
        try:
            recipe = steps.load_recipe(path)
        except (OSError, ValueError) as error:
            messagebox.showerror('Cannot load recipe', str(error), parent=self.root)
            return
        self.pipeline = recipe.get('pipeline', [])
        self.roi_var.set(recipe.get('roi', 'model'))
        self.segmenter_var.set('text' if recipe.get('segmenter') == 'sam3_text' else
                               'clicks' if recipe.get('point_source') == 'clicks' else 'auto')
        self.prompt_var.set(recipe.get('prompt', DEFAULT_PROMPT))
        self.edges_var.set(bool(recipe.get('bead_edges', False)))
        self.ribbon_var.set(bool(recipe.get('ribbon', False)))
        self.runs_var.set(recipe.get('sam_runs', 'cloud'))
        self.once_var.set(bool(recipe.get('ask_once', False)))
        if recipe.get('edge_method', 'slope') in steps.edge_methods_available():
            self.edge_method_var.set(recipe.get('edge_method', 'slope'))
        self.local_model_var.set(recipe.get('local_model', DEFAULT_LOCAL_MODEL))
        self._refresh_steps()
        self._changed()

    def save_result(self, path=None):
        if self.result is None:
            self._say('Nothing to save yet: press Run first.')
            return
        path = path or filedialog.asksaveasfilename(title='Save result', defaultextension='.jpg',
                                                    filetypes=[('JPEG', '*.jpg')])
        if not path:
            return
        path = Path(path)
        cv2.imwrite(str(path), self.result['overlay'], [cv2.IMWRITE_JPEG_QUALITY, 92])
        record = {'image': str(self.image_path), 'recipe': self.recipe(),
                  'scale': self.result['scale'], 'seconds': round(self.result['seconds'], 2),
                  'lines': [{key: (np.round(value, 1).tolist() if isinstance(value, np.ndarray) else value)
                             for key, value in line.items() if key not in WORKING_KEYS}
                            for line in self.result['lines']]}
        path.with_suffix('.json').write_text(json.dumps(record, indent=1), encoding='utf-8')
        self._say(describe(self.result) + f'\nSaved {path.name} and {path.with_suffix(".json").name}')


def main(argv=None):
    arguments = sys.argv[1:] if argv is None else argv
    root = tk.Tk()
    PipelineLab(root, arguments[0] if arguments else None)
    root.mainloop()


if __name__ == '__main__':
    main()

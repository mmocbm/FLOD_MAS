"""Tk integration; cameras and calibration remain owned by the existing app."""
import tkinter as tk
from app_config import CONFIG, project_path
from auto_trigger.controller import AutoController
from segmentation import create_segmenter
from .pipeline import InspectionPipeline
from .storage import CaptureStore
from .result_view import FabricResultView


class AutomaticDashboard:
    def _init_automatic(self):
        def factory():
            store = CaptureStore(project_path(CONFIG['capture_storage']['directory']),
                                 CONFIG['capture_storage'])
            return InspectionPipeline(create_segmenter(CONFIG['segmentation']), store, CONFIG)
        self.auto_controller = AutoController(CONFIG, factory)
        self.auto_controller.start()
        self._auto_poll_job = self.root.after(100, self._poll_automatic)
        self._fabric_cycle_job = None

    def _poll_automatic(self):
        if getattr(self, '_closing', False):
            return
        controller = self.auto_controller
        active = (not self.video_paused and self.video_streaming
                  and getattr(self, 'calibration_page', None) is None)
        controller.set_active(active)
        for event, generation, payload in controller.poll():
            if event == 'result':
                if generation != controller.generation or not active:
                    continue
                self._show_automatic_results(payload)
            elif event == 'clear':
                if active:
                    self._dismiss_result_view()
                    self.set_pass_fail('READY')
                    self.update_progress(100, payload)
            elif event == 'error':
                self.set_pass_fail('WARNING')
                self.update_progress(100, payload)
            elif event == 'status':
                self.update_progress(100, payload)
        self._auto_poll_job = self.root.after(100, self._poll_automatic)

    def _submit_automatic_frame(self, raw, sequence):
        if getattr(self, 'auto_controller', None) is not None:
            self.auto_controller.submit(raw, sequence, self.camera1.undistorter,
                {'size': self.active_size, 'width': self.active_strip_width,
                 'tolerance': self.active_strip_width_tolerance})

    def _show_automatic_results(self, result):
        fabrics = result['fabrics']
        self._dismiss_result_view()
        if fabrics:
            # Fill the physical screen, retaining the existing application chrome.
            self.root.geometry(f'{self.root.winfo_screenwidth()}x{self.root.winfo_screenheight()}+0+0')
            self.bottom_panel.pack_forget()
            self.images_container.pack_forget()
            self.result_view = FabricResultView(
                self.image_panel,
                show_live_preview=CONFIG.get('result_view', {}).get('show_live_preview', True))
            self.result_view.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)
            self.result_view.show_fabrics(fabrics)
            self._cycle_fabrics()
        states = [f.record['status'] for f in fabrics]
        status = ('WARNING' if result['warnings'] or not states else
                  'FAIL' if 'FAIL' in states else 'PASS' if all(s == 'PASS' for s in states) else 'WARNING')
        self.set_pass_fail(status)
        self.update_progress(100, ' | '.join(result['warnings']) or
                             f'{len(fabrics)} fabric(s) measured — results saved — monitoring hands')

    def _cycle_fabrics(self):
        self._fabric_cycle_job = None
        if self.result_view is None:
            return
        def advance():
            view = self.result_view
            if view is not None and view.records:
                view.select_tab((view._active + 1) % len(view.records))
                self._cycle_fabrics()
        self._fabric_cycle_job = self.root.after(
            round(self.active_result_display_seconds * 1000), advance)

    def _dismiss_result_view(self):
        if getattr(self, '_fabric_cycle_job', None) is not None:
            self.root.after_cancel(self._fabric_cycle_job)
            self._fabric_cycle_job = None
        if getattr(self, 'result_view', None) is not None:
            self.result_view.destroy()
            self.result_view = None
            self.images_container.pack(**self.images_container_pack_opts)
            self.bottom_panel.pack(side=tk.BOTTOM, fill=tk.X)

    def _refresh_inspection_availability(self):
        if self.camera1 is None or not self.camera1.calibration_available:
            self.set_pass_fail('SETUP')
            self.update_progress(100, 'Camera calibration required; open Settings → Camera setup')
        else:
            self.set_pass_fail('READY')
            self.update_progress(100, 'Automatic inspection — waiting for a hand cycle')

    def reset_dashboard(self):
        self.auto_controller.set_active(False)
        self._dismiss_result_view()
        self.on_reset_callback()

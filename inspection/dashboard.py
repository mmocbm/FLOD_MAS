"""Tk integration; cameras and calibration remain owned by the existing app."""
import tkinter as tk
from app_config import CONFIG, project_path
from auto_trigger.controller import AutoController
from pathlib import Path
from .local.engine import LiveInspection
from .pipeline import InspectionPipeline
from .storage import CaptureStore
from .result_view import LineResultView


class AutomaticDashboard:
    def _init_automatic(self):
        def factory():
            store = CaptureStore(project_path(CONFIG['capture_storage']['directory']),
                                 CONFIG['capture_storage'])
            settings = CONFIG['local_inspection']
            root = Path(__file__).resolve().parents[1]
            inspector = LiveInspection(root / settings['source_model'], root / settings['glue_model'],
                                       settings['offset_pixels'], settings['source_width'])
            return InspectionPipeline(inspector, store, CONFIG)
        self.auto_controller = AutoController(CONFIG, factory)
        self.auto_controller.start()
        self._auto_poll_job = self.root.after(100, self._poll_automatic)

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
                if active and generation == controller.generation:
                    self._dismiss_result_view()
                    self.set_pass_fail('READY')
                    self.update_progress(100, payload)
            elif event == 'error':
                if not active or generation != controller.generation:
                    continue
                self._dismiss_result_view()
                self.set_pass_fail('WARNING')
                self.update_progress(100, payload)
            elif event == 'status':
                if active and generation == controller.generation:
                    self.update_progress(100, payload)
        self._auto_poll_job = self.root.after(100, self._poll_automatic)

    def _submit_automatic_frame(self, raw, sequence):
        if getattr(self, 'auto_controller', None) is not None:
            self.auto_controller.submit(raw, sequence, self.camera1.undistorter,
                {'size': self.active_size, 'width': self.active_strip_width,
                 'tolerance': self.active_strip_width_tolerance})

    def _show_automatic_results(self, result):
        measurement = result['measurement']
        self._dismiss_result_view()
        if measurement:
            # Fill the physical screen, retaining the existing application chrome.
            self.root.geometry(f'{self.root.winfo_screenwidth()}x{self.root.winfo_screenheight()}+0+0')
            self.bottom_panel.pack_forget()
            self.images_container.pack_forget()
            self.result_view = LineResultView(
                self.image_panel,
                show_live_preview=CONFIG.get('result_view', {}).get('show_live_preview', True))
            self.result_view.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)
            self.result_view.show_inspection(result)
        status = 'WARNING' if result['warnings'] else measurement['status']
        self.set_pass_fail(status)
        self.update_progress(100, ' | '.join(result['warnings']) or
                             f"Rightmost line #{measurement['selected_line']} of {measurement['line_count']} "
                             'measured — results saved — monitoring hands')

    def _dismiss_result_view(self):
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

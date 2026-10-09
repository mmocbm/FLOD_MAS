"""Saved correction affects distances, never geometry or unsaved inspection."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

import numpy as np
import measurement_adjustment as adjustment
from plane_scale import PlaneScale
from inspection.line_measurement import measurement_record
from inspection.local.width_measurement import measure_component


class DeveloperMeasurementsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'adjustment.json'
        self.patcher = patch.object(adjustment, 'settings_path', return_value=self.path)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_save_reset_and_mode_do_not_disable_saved_ratio(self):
        self.assertEqual(adjustment.load(), {'developer_mode': False, 'ratio': 1.0})
        adjustment.save(ratio=.95, developer_mode=True)
        adjustment.save(developer_mode=False)
        self.assertEqual(adjustment.current_ratio(), .95)
        adjustment.save(ratio=1)
        self.assertEqual(adjustment.current_ratio(), 1)

    def test_invalid_ratio_does_not_replace_saved_value(self):
        adjustment.save(ratio=.98)
        original = self.path.read_bytes()
        for value in [0, -1, float('nan'), float('inf'), True, 'bad']:
            with self.assertRaises(ValueError):
                adjustment.save(ratio=value)
            self.assertEqual(self.path.read_bytes(), original)

    def test_inspection_scales_once_and_grades_adjusted_width(self):
        mask = np.zeros((30, 130), np.uint8)
        mask[10:20, 5:125] = 255
        component = measure_component(mask)
        source = dict(components=[component], line_count=1, selected_line=1,
                      component_count=1, length_px=component['length_px'],
                      longest_length_px=component['length_px'], selected_boundary={}, seconds=0)
        scale = PlaneScale(np.diag([.4, .4, 1]), .4)
        homography = scale.homography.copy()
        baseline = measurement_record(source, scale, 3.6, .05, 5)
        adjustment.save(ratio=.9)
        adjusted = measurement_record(source, scale, 3.6, .05, 5)
        self.assertEqual(baseline['status'], 'FAIL')
        self.assertEqual(adjusted['status'], 'PASS')
        self.assertAlmostEqual(adjusted['mean_width_mm'], baseline['mean_width_mm']*.9)
        self.assertAlmostEqual(adjusted['length_mm'], baseline['length_mm']*.9)
        self.assertEqual(adjusted['mean_width_px'], baseline['mean_width_px'])
        self.assertEqual(adjusted['distance_ratio'], .9)
        np.testing.assert_array_equal(scale.homography, homography)
        self.assertEqual(measurement_record(source, scale, 3.6, .05, 5)['mean_width_mm'],
                         adjusted['mean_width_mm'])

    def test_checker_preserves_expected_geometry(self):
        adjustment.save(ratio=.9)
        source = dict(expected_mm=9., measured_mm=10., error_mm=1.,
                      absolute_error_mm=1., error_percent=100/9)
        row = adjustment.adjust_accuracy_rows([source])[0]
        self.assertEqual(row['measured_mm'], 9)
        self.assertEqual(row['expected_mm'], 9)
        self.assertEqual(row['error_mm'], 0)
        self.assertEqual(source['measured_mm'], 10)

    def test_manual_preview_is_not_saved_or_compounded(self):
        from CalibrateAPP.calibration_ui import CalibrationCheckApp
        page = CalibrationCheckApp.__new__(CalibrationCheckApp)
        page.manual_points = [(0, 0), (10, 0)]
        page.manual_region_scale = None
        page._manual_pixels_to_plane_mm = Mock(return_value=np.array([[0., 0.], [10., 0.]]))
        page._set_status = Mock()
        page._set_check_report = Mock()
        page.developer_controls = Mock()
        page.developer_controls.value.return_value = .9
        page._calculate_manual_distance()
        self.assertEqual(page.manual_distance_mm, 9)
        page.developer_controls.value.return_value = 1.1
        page._calculate_manual_distance()
        self.assertEqual(page.manual_distance_mm, 11)
        self.assertEqual(adjustment.current_ratio(), 1)

    def test_editor_load_save_reset_and_operator_mode(self):
        import tkinter as tk
        from developer_controls import DeveloperControls
        root = tk.Tk()
        root.withdraw()
        self.addCleanup(root.destroy)
        control = DeveloperControls(root)
        control.enabled.set(True)
        control.toggle()
        control.ratio.set('.95')
        self.assertEqual(adjustment.current_ratio(), 1)
        control.save_ratio()
        self.assertEqual(adjustment.current_ratio(), .95)
        control.ratio.set('1.2')
        control.enabled.set(False)
        control.toggle()
        self.assertEqual(control.value(), .95)
        control.enabled.set(True)
        control.toggle()
        control.reset()
        self.assertEqual(adjustment.current_ratio(), 1)

    def test_paused_results_update_all_tabs_and_preserve_geometry(self):
        import tkinter as tk
        from inspection.result_view import LineResultView
        from inspection.overlay import annotated_overlay
        root = tk.Tk()
        root.withdraw()
        self.addCleanup(root.destroy)
        adjustment.save(ratio=.95, developer_mode=True)
        mask = np.zeros((40, 150), np.uint8)
        mask[10:20, 10:140] = 255
        component = measure_component(mask)
        source = dict(components=[component], line_count=1, selected_line=1,
                      component_count=1, length_px=component['length_px'],
                      longest_length_px=component['length_px'], selected_boundary={}, seconds=0)
        original = measurement_record(source, PlaneScale(np.diag([.4, .4, 1]), .4), 4, .01, 5)
        frame = np.zeros((40, 150, 3), np.uint8)
        view = LineResultView(root, show_live_preview=False)
        view.pack(fill='both', expand=True)
        self.addCleanup(view.destroy)
        result = dict(measurement=original, source_image=frame, mask=mask,
                      overlay=annotated_overlay(frame, original, mask))
        view.show_inspection(result)
        view.show_inspection(result)
        view.set_paused(True)
        self.assertEqual(view.developer_controls.value(), .95)
        view._tabs[0]['zoom'] = 2
        view._tabs[0]['offset'] = [11, 22]
        view.developer_controls.ratio.set('1')
        view._apply_ratio_preview()
        for record in view.records:
            self.assertAlmostEqual(record['mean_width_mm'], 4)
            self.assertEqual(record['status'], 'PASS')
        self.assertEqual(original['distance_ratio'], .95)
        self.assertEqual(view._tabs[0]['zoom'], 2)
        self.assertEqual(view._tabs[0]['offset'], [11, 22])
        self.assertEqual(adjustment.current_ratio(), .95)
        view.developer_controls.ratio.set('1.1')
        view._apply_ratio_preview()
        self.assertAlmostEqual(view.records[0]['mean_width_mm'], 4.4)
        self.assertEqual(view.records[0]['status'], 'FAIL')
        view.developer_controls.save_ratio()
        self.assertEqual(adjustment.current_ratio(), 1.1)
        view.developer_controls.ratio.set('1.2')
        view.set_paused(False)
        view._apply_ratio_preview()
        self.assertAlmostEqual(view.records[0]['mean_width_mm'], 4.4)
        view.set_paused(True)
        view.developer_controls.reset()
        view._apply_ratio_preview()
        self.assertAlmostEqual(view.records[1]['mean_width_mm'], 4)

    def test_ratio_preview_handles_serialized_segments_without_double_scaling(self):
        from inspection.line_measurement import with_distance_ratio
        import json
        mask = np.zeros((30, 130), np.uint8)
        mask[10:20, 5:125] = 255
        component = measure_component(mask)
        source = dict(components=[component], line_count=1, selected_line=1,
                      component_count=1, length_px=component['length_px'],
                      longest_length_px=component['length_px'], selected_boundary={}, seconds=0)
        original = measurement_record(source, PlaneScale(np.diag([.4, .4, 1]), .4), 4, .1, 5)
        original = json.loads(json.dumps(original))
        adjusted = with_distance_ratio(original, .5)
        self.assertAlmostEqual(adjusted['segments'][0]['average_width_mm'], 2)
        self.assertAlmostEqual(adjusted['components'][0]['segments'][0]['average_width_mm'], 2)
        self.assertEqual(adjusted['target_width_mm'], 4)
        restored = with_distance_ratio(adjusted, 1)
        self.assertAlmostEqual(restored['mean_width_mm'], original['mean_width_mm'])


if __name__ == '__main__':
    unittest.main()

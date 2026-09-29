"""Defect-only result sequencing and background inspection tests."""
import unittest
from unittest.mock import MagicMock, call, mock_open, patch

import numpy as np

import main_1366 as main
import sam_detection


def make_app():
    app = main.IndustrialDashboard.__new__(main.IndustrialDashboard)
    app.active_size = 'M'
    app.active_strip_width = 4.0
    app.active_strip_width_tolerance = 1.0
    app.root = MagicMock()
    app.update_progress = MagicMock()
    app.set_pass_fail = MagicMock()
    app.strip_scales = {}
    app.crop_definitions = {'cameras': {'1': [], '2': []}}
    app.camera1 = None
    app.camera2 = None
    app._side_busy = {'L': False, 'R': False}
    app._serial_pending_sides = set()
    app._inspection_order_counter = 0
    app._next_result_order = 0
    app._inspection_epoch = 0
    app._completed_inspections = {}
    app._result_queue = []
    app._active_result = None
    app._active_inspection_side = None
    app.video_paused = False
    app._refresh_inspection_availability = MagicMock()
    app._send_serial_status = MagicMock()
    return app


def crops():
    return [np.zeros((552, 2208, 3), np.uint8) for _ in range(2)]


def detection_result(polygons=None):
    return sam_detection.CropDetection(polygons or [], 1234)


class ProgressPublishingTests(unittest.TestCase):
    def run_detection(self, send_crops, progress=True):
        app = make_app()
        images = crops()
        published = []
        callback = (lambda index, image, busy: published.append((index, busy, image))) \
            if progress else None
        settings = {
            'enabled': True, 'send_crops': send_crops, 'measure_in_mm': False,
        }
        with patch.dict(main.CONFIG['sam_detection'], settings), \
                patch.object(sam_detection, 'detect_crop',
                             return_value=detection_result()), \
                patch.object(main.os, 'makedirs'), \
                patch.object(main.cv2, 'imwrite', return_value=True), \
                patch('builtins.open', mock_open()), \
                patch.object(main.json, 'dump'):
            result = app._detect_crops(1, 'ts', images, None, progress=callback)
        return images, result, published

    def test_selected_crops_publish_busy_then_finished(self):
        _images, _result, published = self.run_detection({'1': [True, True]})
        self.assertEqual(
            [(index, busy) for index, busy, _image in published],
            [(1, True), (1, False), (2, True), (2, False)],
        )

    def test_no_callback_leaves_clean_crops_unselected_for_preview(self):
        images, (display, warnings, defects), published = self.run_detection(
            {'1': [True, True]}, progress=False)
        self.assertEqual(published, [])
        self.assertEqual(warnings, [])
        self.assertEqual(defects, [])
        self.assertIs(display[0], images[0])
        self.assertIs(display[1], images[1])


class DefectResultQueueTests(unittest.TestCase):
    def test_clean_result_is_saved_but_not_presented(self):
        app = make_app()
        app._present_detection_result = MagicMock()
        app._side_busy['L'] = True

        app._inspection_completed('L', 0, 1, [], [], [], True)

        app._present_detection_result.assert_not_called()
        self.assertFalse(app._side_busy['L'])
        self.assertFalse(app.video_paused)

    def test_completed_results_wait_for_request_order(self):
        app = make_app()
        app._present_detection_result = MagicMock()
        app._side_busy = {'L': True, 'R': True}
        right = np.ones((4, 4, 3), np.uint8)
        left = np.zeros((4, 4, 3), np.uint8)

        app._inspection_completed('R', 1, 2, [right], ['CROP 2'], [], True)
        app._present_detection_result.assert_not_called()

        app._inspection_completed('L', 0, 1, [left], ['CROP 1'], [], True)

        shown = app._present_detection_result.call_args.args[0]
        self.assertEqual(shown['side'], 'L')
        self.assertEqual([item['side'] for item in app._result_queue], ['R'])

    def test_other_side_is_shown_after_current_side_finishes(self):
        app = make_app()
        app._dismiss_result_view = MagicMock()
        app.restore_dual_view = MagicMock()
        app._present_detection_result = MagicMock()
        app._active_result = {
            'side': 'L', 'camera_num': 1, 'crops': [np.zeros((2, 2, 3))],
            'labels': ['CROP 1'], 'warnings': [], 'completed': True,
        }
        app._active_inspection_side = 'L'
        app._side_busy = {'L': True, 'R': True}
        app._result_queue = [{
            'side': 'R', 'camera_num': 2, 'crops': [np.zeros((2, 2, 3))],
            'labels': ['CROP 2'], 'warnings': [], 'completed': True,
        }]

        app.start_live_preview()

        self.assertFalse(app._side_busy['L'])
        self.assertTrue(app._side_busy['R'])
        shown = app._present_detection_result.call_args.args[0]
        self.assertEqual(shown['side'], 'R')


class ResultSequenceTests(unittest.TestCase):
    def make_timer_app(self, crop_count=2):
        app = main.IndustrialDashboard.__new__(main.IndustrialDashboard)
        app.root = MagicMock()
        app.root.after.return_value = 'timer-job'
        app.result_view = MagicMock()
        app.result_view.winfo_exists.return_value = True
        app.result_view._tabs = [{} for _ in range(crop_count)]
        app.active_result_display_seconds = 5.0
        app._result_timer_job = None
        app._result_timer_active = False
        app._result_timer_paused = False
        return app

    @patch.object(main.time, 'monotonic', return_value=10.0)
    def test_sequence_starts_on_first_defective_crop(self, _clock):
        app = self.make_timer_app()
        app._start_result_sequence()
        app.result_view.show_transition.assert_called_once_with(0, 5, paused=False)

    @patch.object(main.time, 'monotonic', return_value=20.0)
    def test_two_defective_crops_advance_then_finish(self, _clock):
        app = self.make_timer_app(2)
        app._result_timer_active = True
        app._result_sequence_index = 0
        app.start_live_preview = MagicMock()

        app._advance_result_sequence()
        app.result_view.show_transition.assert_called_once_with(1, 5, paused=False)
        app._advance_result_sequence()
        app.start_live_preview.assert_called_once_with()

    @patch.object(main.time, 'monotonic', return_value=20.0)
    def test_one_defective_crop_finishes_without_empty_second_tab(self, _clock):
        app = self.make_timer_app(1)
        app._result_timer_active = True
        app._result_sequence_index = 0
        app.start_live_preview = MagicMock()

        app._advance_result_sequence()

        app.result_view.show_transition.assert_not_called()
        app.start_live_preview.assert_called_once_with()

    @patch.object(main.time, 'monotonic', side_effect=[11.0, 12.0])
    def test_pause_and_play_keep_remaining_time(self, _clock):
        app = self.make_timer_app()
        app._result_timer_active = True
        app._result_timer_job = 'timer-job'
        app._result_sequence_index = 0
        app._result_timer_remaining = 5.0
        app._result_timer_last_tick = 10.0

        app.toggle_result_timer()
        self.assertTrue(app._result_timer_paused)
        self.assertEqual(app._result_timer_remaining, 4.0)

        app.toggle_result_timer()
        self.assertFalse(app._result_timer_paused)
        app.result_view.show_transition.assert_called_with(0, 4, paused=False)


class SerialOutcomeTests(unittest.TestCase):
    def test_each_concurrent_side_reports_its_own_result_once(self):
        app = make_app()
        app._serial_pending_sides = {'L', 'R'}

        app._report_detection_status('R', True)
        app._report_detection_status('L', False)
        app._report_detection_status('R', True)

        self.assertEqual(
            app._send_serial_status.call_args_list,
            [call('R_DONE'), call('L_ERROR')],
        )
        self.assertEqual(app._serial_pending_sides, set())

    def test_ui_started_inspection_sends_no_serial_result(self):
        app = make_app()
        app._report_detection_status('L', True)
        app._send_serial_status.assert_not_called()


if __name__ == '__main__':
    unittest.main()

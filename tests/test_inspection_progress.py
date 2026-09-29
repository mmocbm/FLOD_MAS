"""The crops go on screen before the upload starts, one tab each, with a loader
on the tab being uploaded.

`_present_detection_result` shows both crops the moment they are built, then
`_detect_crops` publishes each crop's state through the dashboard's progress
callback: busy immediately before its upload, finished once its overlay is
written. These tests drive both paths with the network and every writer stubbed
out, so nothing here reaches the disk.
"""
import unittest
from unittest.mock import MagicMock, call, mock_open, patch

import numpy as np

import main_1366 as main
import sam_detection
from crop_processing import normalized_definition


def make_app():
    """A dashboard with the attributes the progress path touches."""
    app = main.IndustrialDashboard.__new__(main.IndustrialDashboard)
    app.active_size = 'M'
    app.root = MagicMock()
    # Run marshalled callbacks immediately so the worker path is synchronous.
    app.root.after.side_effect = lambda _delay, callback, *args: callback(*args)
    app.update_progress = MagicMock()
    app.detection_generation = 0
    app.strip_scales = {}
    app.crop_definitions = {'cameras': {'1': [], '2': []}}
    app.camera1 = None
    app.camera2 = None
    return app


def crops():
    return [np.zeros((552, 2208, 3), np.uint8) for _ in range(2)]


def ring():
    return np.array([[100, 300], [1900, 300], [1900, 330], [100, 330]], np.int32)


def strip_polygon():
    return sam_detection.Polygon('strip', 0.9, ring())


def detection_result(polygons=None):
    return sam_detection.CropDetection(polygons or [], 1234)


class ProgressPublishingTests(unittest.TestCase):
    """`_detect_crops` publishing through the dashboard's progress callback."""

    def run_detection(self, app, send_crops, detect_side_effect, progress=True):
        """Drive one detection with the network and every writer stubbed.

        Pixels are measured rather than converted so no calibration file is
        needed. Returns what the callback saw as ``(index, busy, image)``.
        """
        images = crops()
        published = []
        settings = {
            'enabled': True, 'send_crops': send_crops, 'measure_in_mm': False,
        }
        callback = None
        if progress:
            def callback(crop_index, image, busy):
                published.append((crop_index, busy, image))
        with patch.dict(main.CONFIG['sam_detection'], settings), \
                patch.object(sam_detection, 'detect_crop') as detect, \
                patch.object(main.os, 'makedirs'), \
                patch.object(main.cv2, 'imwrite', return_value=True) as imwrite, \
                patch('builtins.open', mock_open()) as opener, \
                patch.object(main.json, 'dump') as dump:
            detect.side_effect = detect_side_effect
            result = app._detect_crops(1, 'ts', images, None, progress=callback)
        return images, result, published, detect, imwrite, opener, dump

    def test_each_uploaded_crop_is_marked_busy_then_finished_in_order(self):
        app = make_app()

        _images, _result, published, *_ = self.run_detection(
            app, {'1': [True, True]}, lambda *a, **k: detection_result())

        self.assertEqual(
            [(index, busy) for index, busy, _image in published],
            [(1, True), (1, False), (2, True), (2, False)],
        )

    def test_the_loader_follows_the_upload_order_not_the_tab_order(self):
        app = make_app()

        _images, _result, published, *_ = self.run_detection(
            app, {'1': [False, True]}, lambda *a, **k: detection_result())

        # The shipped configuration uploads only crop 2. Stepping by index would
        # publish a tab 3 that does not exist and spin forever on it.
        self.assertEqual(
            [(index, busy) for index, busy, _image in published],
            [(2, True), (2, False)],
        )

    def test_a_crop_that_is_never_uploaded_is_never_published(self):
        app = make_app()

        _images, _result, published, *_ = self.run_detection(
            app, {'1': [True, False]}, lambda *a, **k: detection_result())

        # No marker is better than a tick claiming a check that never happened.
        self.assertEqual([index for index, _busy, _image in published], [1, 1])

    def test_a_finished_crop_is_republished_with_its_overlay(self):
        app = make_app()
        uploads = []

        def detect(crop, prompt, **kwargs):
            uploads.append(crop)
            # Only the first upload finds a strip, so only its tab ends up
            # showing an annotated image.
            polygons = [strip_polygon()] if len(uploads) == 1 else []
            return detection_result(polygons)

        images, _result, published, _detect, imwrite, _open, dump = (
            self.run_detection(app, {'1': [True, True]}, detect)
        )

        finished = {index: image for index, busy, image in published if not busy}
        self.assertIsNot(finished[1], images[0])
        self.assertEqual(finished[1].shape[:2], (552, 2208))
        # Crop 2 found nothing, so its tab keeps the raw crop it already showed.
        self.assertIs(finished[2], images[1])
        self.assertEqual(imwrite.call_count, 1)
        self.assertEqual(dump.call_count, 2)

    def test_no_progress_callback_leaves_the_caller_untouched(self):
        app = make_app()
        images, (display, warnings), published, *_ = self.run_detection(
            app, {'1': [True, True]}, lambda *a, **k: detection_result(),
            progress=False,
        )

        self.assertEqual(published, [])
        self.assertEqual(warnings, [])
        self.assertIs(display[0], images[0])
        self.assertIs(display[1], images[1])


class ResultPresentationOrderTests(unittest.TestCase):
    """The result is on screen before the first byte is uploaded."""

    def make_simulating_app(self):
        app = make_app()
        frame = np.zeros((300, 450, 3), np.uint8)
        app.camera1 = MagicMock()
        app.camera1.get_raw_frame_with_ret.return_value = True, frame
        app.camera1.get_undistorted_frame.return_value = frame.copy()
        app.camera2 = MagicMock()
        app.crop_definitions = {
            'version': 1,
            'cameras': {
                '1': [
                    normalized_definition(
                        (225, 100), 300, 0, [(100, 100), (350, 100)], (450, 300)),
                    normalized_definition(
                        (225, 200), 300, 0, [(100, 200), (350, 200)], (450, 300)),
                ],
                '2': [None, None],
            },
        }
        app.result_view = None
        return app

    @patch.object(main.os, 'makedirs')
    @patch.object(main.cv2, 'imwrite', return_value=True)
    def test_the_crops_are_presented_before_the_first_upload(
            self, _imwrite, _makedirs):
        app = self.make_simulating_app()
        order = []
        app._present_detection_result = (
            lambda camera_num, crop_list: order.append('shown')
        )

        def detect(crop, prompt, **kwargs):
            order.append('upload')
            return detection_result()

        with patch.dict(main.CONFIG['sam_detection'],
                        {'enabled': True, 'send_crops': {'1': [True, True]},
                         'measure_in_mm': False}), \
                patch.object(sam_detection, 'detect_crop', side_effect=detect), \
                patch('builtins.open', mock_open()), \
                patch.object(main.json, 'dump'):
            app._simulate_detection('L')

        # Both crops are on screen first, then each upload marks its own tab.
        self.assertEqual(order, ['shown', 'upload', 'upload'])

    @patch.object(main.os, 'makedirs')
    @patch.object(main.cv2, 'imwrite', return_value=True)
    def test_the_result_is_held_and_the_inspection_is_not_finished(
            self, _imwrite, _makedirs):
        app = self.make_simulating_app()
        app._present_detection_result = MagicMock()
        app._finish_detection = MagicMock()

        with patch.dict(main.CONFIG['sam_detection'], {'enabled': False}):
            app._simulate_detection('L')

        app._present_detection_result.assert_called_once()
        # Only START LIVE PREVIEW may end the inspection.
        app._finish_detection.assert_not_called()
        self.assertTrue(any(
            'START LIVE PREVIEW' in str(call)
            for call in app.update_progress.call_args_list
        ))


class FirstUploadingCropTests(unittest.TestCase):
    """The tab to open on, so the spinner is visible without any interaction."""

    def test_the_first_uploading_crop_is_selected(self):
        app = make_app()
        with patch.dict(main.CONFIG['sam_detection'],
                        {'send_crops': {'1': [False, True], '2': [True, True]}}):
            self.assertEqual(app._first_uploading_crop(1), 1)
            self.assertEqual(app._first_uploading_crop(2), 0)

    def test_an_unknown_or_empty_configuration_falls_back_to_the_first_tab(self):
        app = make_app()
        with patch.dict(main.CONFIG['sam_detection'],
                        {'send_crops': {'1': [False, False]}}):
            self.assertEqual(app._first_uploading_crop(1), 0)
            self.assertEqual(app._first_uploading_crop(9), 0)


class ProgressMarshallingTests(unittest.TestCase):
    """A publish is dropped once it no longer belongs to the live view."""

    def test_a_late_publish_is_dropped_after_the_inspection_ends(self):
        app = make_app()
        app.detection_generation = 4
        app.result_view = MagicMock()

        app._apply_crop_progress(1, np.zeros((4, 4, 3), np.uint8), True, 3)

        app.result_view.update_crop.assert_not_called()

    def test_a_publish_for_a_dismissed_view_is_dropped(self):
        app = make_app()
        app.result_view = MagicMock()
        app.result_view.winfo_exists.return_value = False

        # Same generation, but the widget is gone: destroy is not atomic with
        # the generation bump, and a queued callback cannot be cancelled.
        app._apply_crop_progress(1, np.zeros((4, 4, 3), np.uint8), True, 0)

        app.result_view.update_crop.assert_not_called()

    def test_a_publish_with_no_view_at_all_is_dropped(self):
        app = make_app()
        app.result_view = None

        app._apply_crop_progress(1, np.zeros((4, 4, 3), np.uint8), True, 0)

    def test_a_current_publish_reaches_the_view(self):
        app = make_app()
        app.result_view = MagicMock()
        image = np.zeros((4, 4, 3), np.uint8)

        app._apply_crop_progress(2, image, True, 0)

        app.result_view.update_crop.assert_called_once_with(2, image, True)


class WorkerDoneTests(unittest.TestCase):
    """The resume button only goes live once the worker stops uploading."""

    def test_finishing_enables_the_resume_button_and_clears_spinners(self):
        app = make_app()
        app.result_view = MagicMock()

        app._inspection_worker_done()

        app.result_view.mark_all_idle.assert_called_once()
        app.result_view.set_resume_enabled.assert_called_once_with(True)

    def test_a_destroyed_view_is_ignored(self):
        app = make_app()
        app.result_view = MagicMock()
        app.result_view.winfo_exists.return_value = False

        app._inspection_worker_done()

        app.result_view.set_resume_enabled.assert_not_called()

    def test_no_view_is_a_no_op(self):
        app = make_app()
        app.result_view = None

        # Must not raise: the failures path tears the view down before this runs.
        app._inspection_worker_done()


class SerialOutcomeTests(unittest.TestCase):
    """The board is told the outcome once, inside its 60 second window."""

    def make_serial_app(self):
        app = make_app()
        app.video_paused = True
        app.inspection_busy = True
        app._serial_pending_side = None
        app._send_serial_status = MagicMock()
        app._refresh_inspection_availability = MagicMock()
        return app

    def test_presenting_reports_once_and_finishing_does_not_repeat_it(self):
        app = self.make_serial_app()
        app._serial_pending_side = 'L'

        app._report_detection_status(True)
        app._finish_detection()

        app._send_serial_status.assert_called_once_with('L_DONE')
        self.assertIsNone(app._serial_pending_side)

    def test_a_failure_after_the_result_still_reports_its_own_error(self):
        app = self.make_serial_app()
        app._serial_pending_side = 'L'

        app._report_detection_status(True)
        app._finish_detection(completed=False)

        # The board tolerates a red after a blue, but never a silent failure.
        self.assertEqual(
            app._send_serial_status.call_args_list,
            [call('L_DONE'), call('L_ERROR')],
        )

    def test_a_ui_started_inspection_never_reports_for_an_old_request(self):
        app = self.make_serial_app()
        app._serial_pending_side = 'L'
        app._report_detection_status(True)

        # A new run from the dashboard's own button clears what the last one
        # reported, so its failure cannot be attributed to the old request.
        app._serial_reported_side = None
        app._finish_detection(completed=False)

        app._send_serial_status.assert_called_once_with('L_DONE')


if __name__ == '__main__':
    unittest.main()

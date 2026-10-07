import time
import unittest
from pathlib import Path
from uuid import uuid4
from unittest.mock import patch
import numpy as np
import cv2
from app_config import CONFIG
from camera_handler import CameraStream, CameraHandler


class Device:
    def __init__(self, *args):
        self.properties = {}
        self.released = False

    def isOpened(self): return True
    def set(self, key, value): self.properties[key] = value
    def get(self, key): return self.properties.get(key, 0.0)
    def read(self):
        time.sleep(0.005)
        spec = CONFIG['cameras'][0]
        return True, np.zeros((spec['height'], spec['width'], 3), dtype=np.uint8)
    def release(self): self.released = True


class RecordingDevice(Device):
    """Remembers the order properties were written in."""
    def __init__(self, *args):
        super().__init__(*args)
        self.set_order = []

    def set(self, key, value):
        self.set_order.append(key)
        super().set(key, value)


class IgnoringDevice(Device):
    """Accepts set() silently but never changes state, as some drivers do."""
    def set(self, key, value): pass
    def get(self, key):
        return 0.75 if key == cv2.CAP_PROP_AUTO_EXPOSURE else 1.0


class BooleanExposureDevice(Device):
    """Spells auto exposure 1/0 rather than the DirectShow 0.75/0.25 pair."""
    def get(self, key):
        if key == cv2.CAP_PROP_AUTO_EXPOSURE:
            return 1.0 if self.properties.get(key) == 1.0 else 0.0
        return self.properties.get(key, 0.0)


class UnreadableDevice(Device):
    """A backend that cannot report property values back at all."""
    def get(self, key): raise AttributeError('this backend cannot read properties')


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.cache_path = Path(__file__).parent / f'.camera_cache_{uuid4().hex}.json'
        self.cache_patch = patch(
            'camera_handler.RESOLUTION_CACHE_PATH', self.cache_path)
        self.cache_patch.start()

    def tearDown(self):
        self.cache_patch.stop()
        self.cache_path.unlink(missing_ok=True)
        self.cache_path.with_suffix('.json.tmp').unlink(missing_ok=True)

    @patch('camera_handler.cv2.VideoCapture', side_effect=Device)
    def test_full_resolution_and_device_settings(self, _):
        spec = CONFIG['cameras'][0]
        stream = CameraStream(spec['index'])
        try:
            ok, frame = stream.read()
            self.assertTrue(ok)
            expected_size = ((spec['height'], spec['width'])
                             if spec.get('rotation', 0) in (90, 270)
                             else (spec['width'], spec['height']))
            self.assertEqual(stream.size, expected_size)
            self.assertEqual(frame.shape[:2], (expected_size[1], expected_size[0]))
            self.assertEqual(stream.cap.properties[cv2.CAP_PROP_FRAME_WIDTH], spec['width'])
            self.assertEqual(stream.cap.properties[cv2.CAP_PROP_FRAME_HEIGHT], spec['height'])
        finally:
            stream.release()
            stream.thread.join(1)
        self.assertTrue(stream.cap.released)

    @patch('camera_handler.cv2.VideoCapture', side_effect=Device)
    def test_rotation_is_applied_to_every_frame_and_reported_size(self, _):
        spec = CONFIG['cameras'][0]
        with patch.dict(spec, {'rotation': 90}):
            stream = CameraStream(spec['index'])
            try:
                ok, frame = stream.read()
                self.assertTrue(ok)
                self.assertEqual(stream.size, (spec['height'], spec['width']))
                self.assertEqual(frame.shape[:2], (spec['width'], spec['height']))
            finally:
                stream.release()
                stream.thread.join(1)

    @patch('camera_handler.cv2.VideoCapture', side_effect=Device)
    def test_dshow_can_be_enabled(self, video_capture):
        with patch.dict(CONFIG['capture'], {'use_dshow': True}):
            stream = CameraStream(CONFIG['cameras'][0]['index'])
        try:
            video_capture.assert_called_once_with(
                CONFIG['cameras'][0]['index'], cv2.CAP_DSHOW,
            )
        finally:
            stream.release()
            stream.thread.join(1)

    @patch('camera_handler.cv2.VideoCapture', side_effect=Device)
    def test_dshow_can_be_disabled_for_normal_capture(self, video_capture):
        with patch.dict(CONFIG['capture'], {'use_dshow': False}):
            stream = CameraStream(CONFIG['cameras'][0]['index'])
        try:
            video_capture.assert_called_once_with(CONFIG['cameras'][0]['index'])
        finally:
            stream.release()
            stream.thread.join(1)

    @patch('camera_handler.cv2.VideoCapture', side_effect=Device)
    @patch.object(Device, 'read', return_value=(True, np.zeros((2, 2, 3), dtype=np.uint8)))
    def test_unsupported_resolution_continues_with_warning(self, *_):
        stream = CameraStream(CONFIG['cameras'][0]['index'])
        try:
            self.assertEqual(stream.size, (2, 2))
            self.assertIn('Requested:', stream.resolution_warning)
        finally:
            stream.release()
            stream.thread.join(1)

    def test_fallback_selects_highest_actual_mode(self):
        class NegotiatingDevice(Device):
            def read(self):
                time.sleep(0.001)
                width = self.properties.get(cv2.CAP_PROP_FRAME_WIDTH)
                # Simulate a driver that returns 720p for the requested mode
                # but supports 1080p when that mode is requested explicitly.
                size = (1920, 1080) if width == 1920 else (1280, 720)
                return True, np.zeros((size[1], size[0], 3), np.uint8)
        with patch('camera_handler.cv2.VideoCapture', side_effect=NegotiatingDevice):
            stream = CameraStream(CONFIG['cameras'][0]['index'])
            try:
                expected_size = ((1080, 1920)
                                 if CONFIG['cameras'][0].get('rotation', 0) in (90, 270)
                                 else (1920, 1080))
                self.assertEqual(stream.size, expected_size)
                self.assertIn('1920 x 1080', stream.resolution_warning)
                self.assertEqual(
                    stream.read()[1].shape[:2],
                    (expected_size[1], expected_size[0]),
                )
            finally:
                stream.release()
                stream.thread.join(1)

    def test_second_start_uses_cached_resolution_without_scanning(self):
        class NegotiatingDevice(Device):
            instances = []

            def __init__(self, *args):
                super().__init__(*args)
                self.width_requests = []
                self.instances.append(self)

            def set(self, key, value):
                super().set(key, value)
                if key == cv2.CAP_PROP_FRAME_WIDTH:
                    self.width_requests.append(value)

            def read(self):
                width = self.properties.get(cv2.CAP_PROP_FRAME_WIDTH)
                size = (1920, 1080) if width == 1920 else (1280, 720)
                return True, np.zeros((size[1], size[0], 3), np.uint8)

        with patch('camera_handler.cv2.VideoCapture', side_effect=NegotiatingDevice):
            first = CameraStream(CONFIG['cameras'][0]['index'])
            first.release()
            first.thread.join(1)
            second = CameraStream(CONFIG['cameras'][0]['index'])
            try:
                self.assertGreater(len(NegotiatingDevice.instances[0].width_requests), 1)
                self.assertEqual(NegotiatingDevice.instances[1].width_requests, [1920])
                expected_size = ((1080, 1920)
                                 if CONFIG['cameras'][0].get('rotation', 0) in (90, 270)
                                 else (1920, 1080))
                self.assertEqual(second.size, expected_size)
                self.assertIn('Best available size found', second.resolution_warning)
            finally:
                second.release()
                second.thread.join(1)

    @patch('camera_handler.cv2.VideoCapture', side_effect=Device)
    def test_startup_forces_auto_white_balance_and_exposure_on(self, _):
        stream = CameraStream(CONFIG['cameras'][0]['index'])
        try:
            self.assertTrue(stream.auto_control['white_balance']['ok'])
            self.assertTrue(stream.auto_control['exposure']['ok'])
            self.assertEqual(stream.cap.properties[cv2.CAP_PROP_AUTO_WB], 1.0)
            self.assertIn(stream.cap.properties[cv2.CAP_PROP_AUTO_EXPOSURE], (0.75, 1.0))
        finally:
            stream.release()
            stream.thread.join(1)

    def test_lock_turns_white_balance_off_before_exposure(self):
        with patch('camera_handler.cv2.VideoCapture', side_effect=RecordingDevice):
            stream = CameraStream(CONFIG['cameras'][0]['index'])
            try:
                stream.cap.set_order.clear()
                result = stream.lock_automatic()
                controlled = [key for key in stream.cap.set_order
                              if key in (cv2.CAP_PROP_AUTO_WB, cv2.CAP_PROP_AUTO_EXPOSURE)]
                self.assertEqual(controlled,
                                 [cv2.CAP_PROP_AUTO_WB, cv2.CAP_PROP_AUTO_EXPOSURE])
                self.assertTrue(result['white_balance']['ok'])
                self.assertTrue(result['exposure']['ok'])
                self.assertEqual(stream.cap.properties[cv2.CAP_PROP_AUTO_WB], 0.0)
                self.assertEqual(stream.cap.properties[cv2.CAP_PROP_AUTO_EXPOSURE], 0.25)
            finally:
                stream.release()
                stream.thread.join(1)

    def test_driver_ignoring_property_reports_failure_not_success(self):
        with patch('camera_handler.cv2.VideoCapture', side_effect=IgnoringDevice):
            stream = CameraStream(CONFIG['cameras'][0]['index'])
            try:
                result = stream.lock_automatic()
                self.assertFalse(result['white_balance']['ok'])
                self.assertEqual(result['white_balance']['actual'], 1.0)
                self.assertFalse(result['exposure']['ok'])
                self.assertEqual(result['exposure']['actual'], 0.75)
            finally:
                stream.release()
                stream.thread.join(1)

    def test_exposure_alternate_convention_is_found_by_fallback(self):
        with patch('camera_handler.cv2.VideoCapture', side_effect=BooleanExposureDevice):
            stream = CameraStream(CONFIG['cameras'][0]['index'])
            try:
                result = stream.set_auto_exposure(True)
                self.assertTrue(result['ok'])
                self.assertEqual(result['actual'], 1.0)
            finally:
                stream.release()
                stream.thread.join(1)

    def test_unreadable_property_does_not_stop_camera_opening(self):
        with patch('camera_handler.cv2.VideoCapture', side_effect=UnreadableDevice):
            stream = CameraStream(CONFIG['cameras'][0]['index'])
            try:
                self.assertGreater(stream.size[0], 0)
                self.assertFalse(stream.auto_control['white_balance']['ok'])
                self.assertFalse(stream.auto_control['exposure']['ok'])
                self.assertIsNone(stream.auto_control['white_balance']['actual'])
            finally:
                stream.release()
                stream.thread.join(1)

    @patch('camera_handler.CameraStream')
    @patch('camera_handler.ImageUndistorter')
    def test_preview_does_not_undistort_and_processing_keeps_original(self, undistorter, stream):
        original = np.zeros((800, 1200, 3), dtype=np.uint8)
        stream.return_value.read.return_value = True, original
        handler = CameraHandler(CONFIG['cameras'][0]['index'], 'test.json')
        handler.read_frame()
        undistorter.return_value.undistort.assert_not_called()
        self.assertIs(handler.get_raw_frame(), original)
        handler.get_undistorted_frame()
        handler.get_undistorted_frame()
        undistorter.return_value.undistort.assert_called_once_with(original)

    @patch('camera_handler.CameraStream')
    @patch('camera_handler.ImageUndistorter', side_effect=FileNotFoundError('missing calibration'))
    def test_missing_calibration_keeps_camera_stream_available(self, undistorter, stream):
        handler = CameraHandler(CONFIG['cameras'][0]['index'], 'missing.json')

        stream.assert_called_once_with(CONFIG['cameras'][0]['index'])
        self.assertFalse(handler.calibration_available)
        self.assertIsNone(handler.undistorter)
        self.assertIn('missing calibration', handler.calibration_error)
        self.assertIsNone(handler.get_undistorted_frame())

    @patch('camera_handler.CameraStream')
    @patch('camera_handler.ImageUndistorter')
    def test_reload_enables_calibration_after_file_is_created(self, undistorter, stream):
        loaded = object()
        undistorter.side_effect = [FileNotFoundError('missing calibration'), loaded]
        handler = CameraHandler(CONFIG['cameras'][0]['index'], 'created-later.json')

        self.assertFalse(handler.calibration_available)
        self.assertTrue(handler.reload_calibration())
        self.assertTrue(handler.calibration_available)
        self.assertIs(handler.undistorter, loaded)
        self.assertIsNone(handler.calibration_error)


if __name__ == '__main__':
    unittest.main()

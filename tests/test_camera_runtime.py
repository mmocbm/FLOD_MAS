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
    def read(self):
        time.sleep(0.005)
        spec = CONFIG['cameras'][0]
        return True, np.zeros((spec['height'], spec['width'], 3), dtype=np.uint8)
    def release(self): self.released = True


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
            self.assertEqual(frame.shape[:2], (spec['height'], spec['width']))
            self.assertEqual(stream.cap.properties[cv2.CAP_PROP_FRAME_WIDTH], spec['width'])
            self.assertEqual(stream.cap.properties[cv2.CAP_PROP_FRAME_HEIGHT], spec['height'])
        finally:
            stream.release()
            stream.thread.join(1)
        self.assertTrue(stream.cap.released)

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
    def test_camera_override_uses_default_backend(self, video_capture):
        camera = CONFIG['cameras'][1]
        with patch.dict(CONFIG['capture'], {'use_dshow': True}):
            stream = CameraStream(camera['index'])
        try:
            video_capture.assert_called_once_with(camera['index'])
            self.assertEqual(stream.capture_backend, 'default')
        finally:
            stream.release()
            stream.thread.join(1)

    def test_failed_dshow_open_retries_default_backend(self):
        class ClosedDevice(Device):
            def isOpened(self): return False

        closed = ClosedDevice()
        opened = Device()
        camera = CONFIG['cameras'][0]
        with (patch.dict(CONFIG['capture'], {'use_dshow': True}),
              patch('camera_handler.cv2.VideoCapture', side_effect=[closed, opened]) as capture):
            stream = CameraStream(camera['index'])
        try:
            self.assertEqual(
                [call.args for call in capture.call_args_list],
                [(camera['index'], cv2.CAP_DSHOW), (camera['index'],)],
            )
            self.assertTrue(closed.released)
            self.assertEqual(stream.capture_backend, 'default')
            self.assertEqual(stream.size, (camera['width'], camera['height']))
        finally:
            stream.release()
            stream.thread.join(1)

    def test_backend_is_part_of_resolution_cache_signature(self):
        from camera_handler import _resolution_signature
        spec = CONFIG['cameras'][0]
        self.assertNotEqual(
            _resolution_signature(spec, 'dshow'),
            _resolution_signature(spec, 'default'),
        )

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
                self.assertEqual(stream.size, (1920, 1080))
                self.assertIn('1920 x 1080', stream.resolution_warning)
                self.assertEqual(stream.read()[1].shape[:2], (1080, 1920))
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
                self.assertEqual(second.size, (1920, 1080))
                self.assertIn('Best available size found', second.resolution_warning)
            finally:
                second.release()
                second.thread.join(1)

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

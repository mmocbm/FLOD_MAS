"""Background full-resolution capture, shared by setup and dashboard."""
import hashlib
import json
import threading
import time
import cv2
from app_config import CONFIG, ROOT, camera_config
from measure.makeUndistored import ImageUndistorter


RESOLUTION_CACHE_PATH = ROOT / CONFIG['capture'].get(
    'resolution_cache_file', 'Files/camera_resolution_cache.json')
_resolution_cache_lock = threading.Lock()


def _resolution_signature(spec):
    """Identify settings that can change the camera's best usable mode."""
    settings = {
        'requested': [spec['width'], spec['height']],
        'fps': spec['fps'],
        'fourcc': spec['fourcc'],
        'backend': CONFIG['capture']['backend'],
        'fallback_resolutions': CONFIG['capture']['fallback_resolutions'],
    }
    encoded = json.dumps(settings, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _read_cached_resolution(index, signature):
    with _resolution_cache_lock:
        try:
            cache = json.loads(RESOLUTION_CACHE_PATH.read_text(encoding='utf-8'))
            entry = cache.get(str(index), {})
            size = entry.get('size')
            if entry.get('signature') == signature and len(size) == 2 and min(size) > 0:
                return tuple(size)
        except (FileNotFoundError, OSError, ValueError, TypeError, AttributeError):
            pass
    return None


def _write_cached_resolution(index, signature, size):
    """Merge one result atomically; both cameras may finish startup together."""
    with _resolution_cache_lock:
        try:
            cache = json.loads(RESOLUTION_CACHE_PATH.read_text(encoding='utf-8'))
            if not isinstance(cache, dict):
                cache = {}
        except (FileNotFoundError, OSError, ValueError, TypeError):
            cache = {}
        cache[str(index)] = {'signature': signature, 'size': list(size)}
        try:
            RESOLUTION_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            temporary = RESOLUTION_CACHE_PATH.with_suffix(RESOLUTION_CACHE_PATH.suffix + '.tmp')
            temporary.write_text(json.dumps(cache, indent=2) + '\n', encoding='utf-8')
            temporary.replace(RESOLUTION_CACHE_PATH)
        except OSError:
            # Caching is only an optimization; never prevent camera startup.
            pass


class CameraStream:
    """One thread owns the device. Consumers read the latest full-size frame."""
    def __init__(self, index):
        self.index = index
        self.lock = threading.Lock()
        self.frame = None
        self.error = None
        self.resolution_warning = None
        self.stopped = threading.Event()
        spec = camera_config(index)
        self.cap = cv2.VideoCapture(index, getattr(cv2, CONFIG['capture']['backend']))
        try:
            if not self.cap.isOpened():
                raise RuntimeError(f"Camera {index} could not be opened")
            if spec['fourcc']:
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*spec['fourcc']))
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, CONFIG['capture']['buffer_size'])
            requested = (spec['width'], spec['height'])
            signature = _resolution_signature(spec)
            cached = _read_cached_resolution(index, signature)
            attempted = cached or requested
            frame = self._try_resolution(attempted, spec['fps'])
            actual = None if frame is None else (frame.shape[1], frame.shape[0])
            cache_is_valid = cached is not None and actual == cached
            if cache_is_valid:
                # A previously verified maximum skips the expensive mode scan.
                if actual != requested:
                    self.resolution_warning = self._resolution_message(requested, actual)
            elif actual != requested:
                # Drivers may clamp unsupported requests to a smaller mode.
                # Rank actual frames, never the requested property values.
                best = None if actual is None else (actual[0] * actual[1], attempted, actual)
                for candidate in CONFIG['capture']['fallback_resolutions']:
                    candidate = tuple(candidate)
                    tested = self._try_resolution(candidate, spec['fps'])
                    if tested is None:
                        continue
                    size = (tested.shape[1], tested.shape[0])
                    area = size[0] * size[1]
                    if best is None or area > best[0]:
                        best = (area, candidate, size)
                if best is None:
                    raise RuntimeError(f"Camera {index} did not provide a usable frame")
                frame = self._try_resolution(best[1], spec['fps'])
                if frame is None or (frame.shape[1], frame.shape[0]) != best[2]:
                    raise RuntimeError(f"Camera {index} could not keep its selected resolution. Please reconnect it.")
                actual = best[2]
                self.resolution_warning = self._resolution_message(requested, actual)
            _write_cached_resolution(index, signature, actual)
            self.size = actual
            self.frame = frame
        except Exception:
            self.cap.release()
            raise
        self.thread = threading.Thread(target=self._capture, daemon=True)
        self.thread.start()

    def _resolution_message(self, requested, actual):
        return (
            f"Camera {self.index}\nRequested: {requested[0]} x {requested[1]}\n"
            f"Best available size found: {actual[0]} x {actual[1]}"
        )

    def _try_resolution(self, size, fps):
        """Read several frames after negotiation to reject stale driver buffers."""
        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
            self.cap.set(cv2.CAP_PROP_FPS, fps)
            frame = None
            for _ in range(CONFIG['capture']['verification_frames']):
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    return None
            return frame
        except cv2.error:
            return None

    def _capture(self):
        try:
            while not self.stopped.is_set():
                ok, frame = self.cap.read()
                with self.lock:
                    self.frame = frame if ok else None
                if not ok:
                    time.sleep(0.05)
        except Exception as error:
            self.error = str(error)
            with self.lock:
                self.frame = None
        finally:
            self.cap.release()

    def read(self):
        with self.lock:
            frame = self.frame
        return (False, None) if frame is None or self.stopped.is_set() else (True, frame.copy())

    def isOpened(self):
        return not self.stopped.is_set() and self.thread.is_alive()

    def release(self):
        self.stopped.set()


class CameraHandler:
    def __init__(self, camera_index, calib_file_path):
        self.camera_index = camera_index
        self.calib_file_path = calib_file_path
        self.undistorter = ImageUndistorter(calib_file_path)
        self.stream = CameraStream(camera_index)
        self.cap = self.stream
        self.current_frame_raw = None
        self.current_frame_undistorted = None

    def reload_calibration(self):
        self.undistorter = ImageUndistorter(self.calib_file_path)
        self.current_frame_undistorted = None

    def read_frame(self):
        ok, frame = self.stream.read()
        self.current_frame_raw = frame
        self.current_frame_undistorted = None
        return ok

    def get_raw_frame(self):
        return self.current_frame_raw

    def get_undistorted_frame(self):
        if self.current_frame_raw is not None and self.current_frame_undistorted is None:
            self.current_frame_undistorted = self.undistorter.undistort(self.current_frame_raw)
        return self.current_frame_undistorted

    def release(self):
        self.stream.release()

    def get_raw_frame_with_ret(self):
        return self.read_frame(), self.current_frame_raw

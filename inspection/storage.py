"""One capture directory per detection; lossless sources and bounded retention."""
from datetime import datetime
from pathlib import Path
import json
import re
import shutil
import threading
import uuid
import cv2

SET_NAME = re.compile(r'^\d{8}_\d{6}_\d{6}_[0-9a-f]{8}$')


class CaptureStore:
    def __init__(self, root, settings):
        self.root = Path(root).resolve()
        self.settings = settings
        self.lock = threading.RLock()
        self.root.mkdir(parents=True, exist_ok=True)
        # Recover this application's interrupted sets after a restart.
        for path in self._sets():
            if not (path / 'result.json').exists():
                self.finish(path, {'status': 'interrupted'})
        self.prune()

    def _sets(self):
        return sorted(p for p in self.root.iterdir()
                      if SET_NAME.fullmatch(p.name) and p.is_dir()
                      and not p.is_symlink() and not p.is_junction()
                      and (p / '.inspection-set').is_file())

    def begin(self, original, undistorted, metadata):
        with self.lock:
            # Reserve space BEFORE creating the next set: total never exceeds limit.
            self.prune(reserve=1)
            name = datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_' + uuid.uuid4().hex[:8]
            path = self.root / name
            path.mkdir()
            (path / '.inspection-set').write_text('automatic-inspection-v1', encoding='utf-8')
            try:
                self._json(path / 'capture.json', metadata)
                self._image(path / 'original.png', original, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                if undistorted is not None:
                    self._image(path / 'undistorted.png', undistorted, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            except Exception as error:
                self.finish(path, {'status': 'save_failed', 'error': str(error)})
                raise
            return path

    def save_preview(self, path, name, image):
        h, w = image.shape[:2]
        factor = min(1.0, self.settings['preview_max_edge'] / max(h, w))
        if factor < 1:
            image = cv2.resize(image, (max(1, round(w*factor)), max(1, round(h*factor))),
                               interpolation=cv2.INTER_AREA)
        self._image(path / f'{name}.jpg', image,
                    [cv2.IMWRITE_JPEG_QUALITY, self.settings['jpeg_quality']])

    def finish(self, path, record):
        with self.lock:
            self._json(path / 'result.json', record)

    def save_mask(self, path, mask):
        self._image(path / 'mask.png', mask, [cv2.IMWRITE_PNG_COMPRESSION, 1])

    def save_overlay(self, path, overlay):
        self._image(path / 'overlay.png', overlay, [cv2.IMWRITE_PNG_COMPRESSION, 1])

    def update_metadata(self, path, metadata):
        with self.lock:
            self._json(path / 'capture.json', metadata)

    @staticmethod
    def _image(path, image, options):
        if not cv2.imwrite(str(path), image, options):
            raise OSError(f'Could not save {path.name}')

    @staticmethod
    def _json(path, record):
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(record, indent=2, allow_nan=False), encoding='utf-8')
        temporary.replace(path)

    def prune(self, reserve=0):
        with self.lock:
            paths = self._sets()
            remove_count = max(0, len(paths) + reserve - self.settings['max_sets'])
            for path in paths:
                if remove_count <= 0:
                    break
                if not (path / 'result.json').is_file():
                    continue  # never delete an inspection still being written
                resolved = path.resolve()
                if resolved.parent != self.root or path.is_symlink() or path.is_junction():
                    raise OSError('Capture retention target escaped its storage directory')
                # Reject nested links too; generated sets contain only ordinary files.
                if any(p.is_symlink() or p.is_junction() or p.is_dir() for p in path.iterdir()):
                    raise OSError(f'Unexpected content in capture set {path.name}')
                shutil.rmtree(resolved)
                remove_count -= 1
            if remove_count:
                raise OSError('Capture limit reached while another set is still being written')

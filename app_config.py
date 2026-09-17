"""Shared configuration. Edit config.json and restart the application."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
if len(CONFIG["cameras"]) != 2:
    raise ValueError("config.json must specify two cameras")
if len({c["index"] for c in CONFIG["cameras"]}) != 2:
    raise ValueError("Camera indexes must be different")
for camera in CONFIG["cameras"]:
    if camera['index'] < 0 or len(camera['fourcc']) not in (0, 4):
        raise ValueError("Camera index must be nonnegative; fourcc must be empty or four characters")
    for key in ("width", "height", "fps"):
        if camera[key] <= 0:
            raise ValueError(f"Camera {key} must be positive")
for key in ("max_width", "max_height", "interval_ms"):
    if CONFIG["preview"][key] <= 0:
        raise ValueError(f"Preview {key} must be positive")
board = CONFIG["board"]
if CONFIG['capture']['verification_frames'] < 1:
    raise ValueError("Capture verification_frames must be positive")
if not CONFIG['capture'].get('resolution_cache_file'):
    raise ValueError("Capture resolution_cache_file must not be empty")
if not CONFIG['capture']['fallback_resolutions'] or any(
        len(size) != 2 or min(size) <= 0 for size in CONFIG['capture']['fallback_resolutions']):
    raise ValueError("Fallback resolutions must be positive width/height pairs")
if not 0 < board["marker_length_mm"] < board["square_length_mm"]:
    raise ValueError("Board marker size must be smaller than square size")
if min(board["squares_x"], board["squares_y"]) < 3:
    raise ValueError("Board must have at least three squares in each direction")
if CONFIG["calibration"]["photo_count"] < CONFIG["calibration"]["minimum_valid_photos"]:
    raise ValueError("Photo count must meet minimum valid photos")
for key, value in CONFIG['calibration'].items():
    if value <= 0:
        raise ValueError(f"Calibration {key} must be positive")
if CONFIG['inspection']['default_size'] not in CONFIG['inspection']['sizes']:
    raise ValueError("Default size must be in the configured size list")


def camera_config(index):
    return next(c for c in CONFIG["cameras"] if c["index"] == index)


def project_path(path):
    return str(ROOT / path)

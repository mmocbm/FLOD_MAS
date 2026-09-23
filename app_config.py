"""Shared configuration. Edit config.json and restart the application."""
import json
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
if len(CONFIG["cameras"]) != 2:
    raise ValueError("config.json must specify two cameras")
if len({c["index"] for c in CONFIG["cameras"]}) != 2:
    raise ValueError("Camera indexes must be different")
for camera in CONFIG["cameras"]:
    if camera['index'] < 0 or len(camera['fourcc']) not in (0, 4):
        raise ValueError("Camera index must be nonnegative; fourcc must be empty or four characters")
    if camera.get('rotation', 0) not in (0, 90, 180, 270):
        raise ValueError("Camera rotation must be 0, 90, 180, or 270 degrees clockwise")
    for key in ("width", "height", "fps"):
        if camera[key] <= 0:
            raise ValueError(f"Camera {key} must be positive")
for key in ("max_width", "max_height", "interval_ms"):
    if CONFIG["preview"][key] <= 0:
        raise ValueError(f"Preview {key} must be positive")
board = CONFIG["board"]
if not isinstance(CONFIG['capture'].get('use_dshow'), bool):
    raise ValueError("Capture use_dshow must be true or false")
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
two_board = CONFIG.get('two_board', {})
if not isinstance(two_board.get('enabled'), bool):
    raise ValueError("Two-board enabled must be true or false")
if not hasattr(cv2.aruco, two_board.get('dictionary', '')):
    raise ValueError("Two-board dictionary is not supported by OpenCV")
marker_count = (board['squares_x'] * board['squares_y']) // 2
dictionary_size = cv2.aruco.getPredefinedDictionary(
    getattr(cv2.aruco, two_board['dictionary'])
).bytesList.shape[0]
second_start = two_board.get('second_board_start_id')
if (not isinstance(second_start, int) or second_start < marker_count or
        second_start + marker_count > dictionary_size):
    raise ValueError(
        "Two-board marker ID ranges must be separate and fit inside the selected dictionary"
    )
if two_board.get('mask_padding_px', 0) < 0:
    raise ValueError("Two-board mask padding cannot be negative")
maximum_board_corners = (board['squares_x'] - 1) * (board['squares_y'] - 1)
if max(CONFIG['calibration']['minimum_corners'],
       CONFIG['calibration']['surface_minimum_corners']) > maximum_board_corners:
    raise ValueError(
        f"Calibration corner requirements cannot exceed the board maximum "
        f"of {maximum_board_corners}"
    )
if CONFIG["calibration"]["photo_count"] < CONFIG["calibration"]["minimum_valid_photos"]:
    raise ValueError("Photo count must meet minimum valid photos")
if min(CONFIG['calibration']['coverage_grid_rows'],
       CONFIG['calibration']['coverage_grid_columns']) < 2:
    raise ValueError("Calibration coverage grid must have at least two rows and columns")
for key, value in CONFIG['calibration'].items():
    if value <= 0:
        raise ValueError(f"Calibration {key} must be positive")
surface = CONFIG.get('measurement_surface', {})
if not isinstance(surface.get('enabled'), bool):
    raise ValueError("Measurement surface enabled must be true or false")
if not hasattr(cv2.aruco, surface.get('aruco_dictionary', '')):
    raise ValueError("Measurement marker dictionary is not supported by OpenCV")
if not isinstance(surface.get('aruco_marker_id'), int) or surface['aruco_marker_id'] < 0:
    raise ValueError("Measurement marker ID must be a nonnegative integer")
for key in ('aruco_marker_length_mm', 'minimum_marker_side_px',
            'maximum_reprojection_error_px'):
    if surface.get(key, 0) <= 0:
        raise ValueError(f"Measurement surface {key} must be positive")
if CONFIG['inspection']['default_size'] not in CONFIG['inspection']['sizes']:
    raise ValueError("Default size must be in the configured size list")


def camera_config(index):
    return next(c for c in CONFIG["cameras"] if c["index"] == index)


def project_path(path):
    return str(ROOT / path)

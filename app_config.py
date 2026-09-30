"""Shared configuration. Edit config.json and restart the application."""
import json
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
# Avoid two camera/processing workers each recruiting all CPU cores. This
# limits OpenCV's internal pool, not acquisition or inspection worker threads.
opencv_threads = CONFIG.get('performance', {}).get('opencv_threads', 1)
if (isinstance(opencv_threads, bool) or not isinstance(opencv_threads, int)
        or opencv_threads < 1):
    raise ValueError("Performance opencv_threads must be a positive integer")
cv2.setNumThreads(opencv_threads)
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
crop_setup = CONFIG.get('crop_setup', {})
if crop_setup.get('crops_per_camera') != 2:
    raise ValueError("Crop setup must define exactly two crops per camera")
aspect_ratio = crop_setup.get('aspect_ratio')
if (not isinstance(aspect_ratio, (list, tuple)) or len(aspect_ratio) != 2
        or any(isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0
               for value in aspect_ratio)):
    raise ValueError("Crop setup aspect_ratio must be two positive numbers, for example [4, 1]")
output_size = crop_setup.get('output_size')
if (not isinstance(output_size, (list, tuple)) or len(output_size) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
               for value in output_size)):
    raise ValueError("Crop setup output_size must be two positive integers, for example [2208, 552]")
# The crop is deskewed and resized in one transform, so a mismatch here does not fail
# loudly -- it stretches every crop by a small, invisible amount. Check it instead.
if abs(output_size[0] / output_size[1] - aspect_ratio[0] / aspect_ratio[1]) > 1e-6:
    raise ValueError(
        f"Crop setup output_size {output_size[0]} x {output_size[1]} does not match "
        f"aspect_ratio {aspect_ratio[0]}:{aspect_ratio[1]}; keep the two ratios equal "
        "so crops are not stretched")
margin_percent = crop_setup.get('margin_percent', 0)
if (isinstance(margin_percent, bool) or not isinstance(margin_percent, (int, float))
        or not 0 <= margin_percent < 100):
    raise ValueError("Crop setup margin_percent must be a number from 0 up to 100")
if not crop_setup.get('definitions_file'):
    raise ValueError("Crop setup definitions_file must not be empty")
board = CONFIG["board"]
if not isinstance(CONFIG['capture'].get('use_dshow'), bool):
    raise ValueError("Capture use_dshow must be true or false")
thickness = CONFIG['measurement_surface'].get('board_thickness', {})
if not isinstance(thickness.get('enabled'), bool):
    raise ValueError("Measurement board thickness enabled must be true or false")
if thickness.get('thickness_mm', 0) < 0:
    raise ValueError("Measurement board thickness must not be negative")
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
for key in ('strip_width_mm', 'strip_width_tolerance_mm', 'result_display_seconds'):
    value = CONFIG['inspection'].get(key)
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or value <= 0):
        raise ValueError(f"Inspection {key} must be a positive number")
sam = CONFIG.get('sam_detection', {})
if not isinstance(sam.get('enabled'), bool):
    raise ValueError("SAM detection enabled must be true or false")
send_crops = sam.get('send_crops')
if not isinstance(send_crops, dict) or set(send_crops) != {"1", "2"}:
    raise ValueError("SAM detection send_crops must define cameras '1' and '2'")
for camera in ("1", "2"):
    flags = send_crops[camera]
    if (not isinstance(flags, list) or len(flags) != 2
            or not all(isinstance(flag, bool) for flag in flags)):
        raise ValueError(
            f"SAM detection send_crops['{camera}'] must be two true/false values"
        )
if sam.get('enabled'):
    if not isinstance(sam.get('prompt'), str) or not sam['prompt'].strip():
        raise ValueError("SAM detection prompt must not be empty")
    quality = sam.get('jpeg_quality')
    if isinstance(quality, bool) or not isinstance(quality, int) or not 1 <= quality <= 100:
        raise ValueError("SAM detection jpeg_quality must be an integer from 1 to 100")
    if not isinstance(sam.get('timeout_seconds'), (int, float)) or sam['timeout_seconds'] <= 0:
        raise ValueError("SAM detection timeout_seconds must be positive")
    for key in ('save_overlay', 'save_polygons', 'analyze_strip', 'measure_in_mm'):
        if not isinstance(sam.get(key), bool):
            raise ValueError(f"SAM detection {key} must be true or false")
    segments = sam.get('strip_segments')
    if isinstance(segments, bool) or not isinstance(segments, int) or segments < 1:
        raise ValueError("SAM detection strip_segments must be a positive integer")
# The optional per-region homography mode. Read with .get throughout: an absent section
# is the shipped state and must not stop the application from importing, since the whole
# point of the mode is that it is additive and off by default.
region = CONFIG.get('region_homography', {})
if not isinstance(region, dict):
    raise ValueError("region_homography must be an object")
if not isinstance(region.get('enabled', False), bool):
    raise ValueError("region_homography enabled must be true or false")
for key in ('store_file', 'board_profiles_file', 'default_profile'):
    if key in region and (not isinstance(region[key], str) or not region[key].strip()):
        raise ValueError(f"region_homography {key} must not be empty")
for key in ('minimum_corners', 'minimum_refinement_views'):
    if key in region and (isinstance(region[key], bool)
                          or not isinstance(region[key], int) or region[key] < 1):
        raise ValueError(f"region_homography {key} must be a positive whole number")
for key in ('maximum_rms_mm', 'maximum_tilt_degrees', 'maximum_gap_mm',
            'maximum_scale_error_percent'):
    if key in region and (isinstance(region[key], bool)
                          or not isinstance(region[key], (int, float))
                          or not 0 < region[key]):
        raise ValueError(f"region_homography {key} must be a positive number")
if not isinstance(region.get('refine_intrinsics', False), bool):
    raise ValueError("region_homography refine_intrinsics must be true or false")
# The floor a fit needs to exist at all; a configured minimum below it can never be met
# by a successful fit, so the two would disagree about what "enough corners" means.
if region.get('minimum_corners', 8) < 4:
    raise ValueError(
        "region_homography minimum_corners must be at least 4, the fewest points that "
        "can pin down a plane"
    )


def camera_config(index):
    return next(c for c in CONFIG["cameras"] if c["index"] == index)


def project_path(path):
    return str(ROOT / path)

import cv2
import math


def validate_automatic_config(config):
    if not isinstance(config.get('result_view', {}).get('show_live_preview', True), bool):
        raise ValueError('result_view.show_live_preview must be true or false')
    if config.get('camera_count') != 1:
        raise ValueError('Automatic inspection requires camera_count = 1')

    def number(section, key, minimum, maximum=None, integer=False):
        value = config[section][key]
        kind = int if integer else (int, float)
        if (isinstance(value, bool) or not isinstance(value, kind) or not math.isfinite(value)
                or value < minimum or (maximum is not None and value > maximum)):
            raise ValueError(f'{section}.{key} has an invalid value')

    if 'marker_confirm_seconds' in config['auto_trigger']:
        number('auto_trigger', 'marker_confirm_seconds', 0.05)
    if config.get('local_inspection', {}).get('subsequent_line', 'rightmost') not in ('leftmost', 'rightmost'):
        raise ValueError('local_inspection.subsequent_line must be leftmost or rightmost')
    for key in ('hand_absence_seconds',):
        number('auto_trigger', key, 0.05)
    number('auto_trigger', 'min_hand_present_seconds', 0)
    for key in ('debounce_frames', 'detect_width', 'max_hands'):
        number('auto_trigger', key, 1, integer=True)
    for key in ('hand_confidence',):
        number('auto_trigger', key, 0.01, 1)
    trim = config.get('inspection', {}).get('end_exclusion_percent', 5.0)
    if isinstance(trim, bool) or not isinstance(trim, (int, float)) or not math.isfinite(trim) or not 0 <= trim < 50:
        raise ValueError('inspection.end_exclusion_percent must be at least 0 and less than 50')
    if 'local_inspection' in config:
        number('local_inspection', 'source_width', 32, integer=True)
        number('local_inspection', 'offset_pixels', 0.5)
        for key in ('source_model', 'glue_model'):
            value = config['local_inspection'][key]
            if not isinstance(value, str) or not value.strip() or not value.lower().endswith('.onnx'):
                raise ValueError(f'local_inspection.{key} must name an ONNX model')
    if 'local_inspection' not in config:
        for key in ('input_width', 'input_height'):
            number('segmentation', key, 1, integer=True)
        number('segmentation', 'max_fabrics', 1, 4, integer=True)
        number('segmentation', 'max_retries', 0, integer=True)
        number('segmentation', 'timeout_seconds', 0.1)
        provider = config['segmentation']['provider']
        if not isinstance(provider, str) or (provider != 'workflow' and ':' not in provider):
            raise ValueError('segmentation.provider must be workflow or module:Factory')
    number('capture_storage', 'max_sets', 1, integer=True)
    number('capture_storage', 'jpeg_quality', 1, 100, integer=True)
    number('capture_storage', 'preview_max_edge', 64, integer=True)
    bool_fields = [('capture_storage', 'save_rejected_triggers')]
    if 'local_inspection' not in config:
        bool_fields.append(('segmentation', 'convert_to_rgb'))
    for section, key in bool_fields:
        if not isinstance(config[section][key], bool):
            raise ValueError(f'{section}.{key} must be true or false')
    directory = config['capture_storage']['directory']
    from pathlib import PureWindowsPath
    if (not isinstance(directory, str) or not directory.strip() or
            PureWindowsPath(directory).is_absolute() or '..' in PureWindowsPath(directory).parts):
        raise ValueError('capture_storage.directory must be a relative data directory')


def validate_config(CONFIG):
    if 'auto_trigger' in CONFIG:
        validate_automatic_config(CONFIG)
    # Avoid two camera/processing workers each recruiting all CPU cores. This
    # limits OpenCV's internal pool, not acquisition or inspection worker threads.
    opencv_threads = CONFIG.get('performance', {}).get('opencv_threads', 1)
    if (isinstance(opencv_threads, bool) or not isinstance(opencv_threads, int)
            or opencv_threads < 1):
        raise ValueError("Performance opencv_threads must be a positive integer")
    # Thread configuration is applied only by the running application.
    cameras = CONFIG['cameras']
    if not isinstance(cameras, list) or len(cameras) not in (1, 2):
        raise ValueError('config.json must define one or two camera configurations')
    camera_count = CONFIG.get('camera_count', len(cameras))
    if (isinstance(camera_count, bool) or not isinstance(camera_count, int)
            or camera_count not in (1, 2) or camera_count > len(cameras)):
        raise ValueError('camera_count must be 1 or 2, with enough camera configurations')
    if len({c["index"] for c in cameras}) != len(cameras):
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
    if not isinstance(CONFIG["preview"].get('show_overlay', True), bool):
        raise ValueError("Preview show_overlay must be true or false")
    # Zero keeps the display-only marker scan at full resolution. It is drawn on
    # every frame, so a negative or non-numeric value would fail inside OpenCV.
    marker_width = CONFIG["preview"].get('marker_detect_width', 0)
    if (isinstance(marker_width, bool) or not isinstance(marker_width, int)
            or marker_width < 0):
        raise ValueError("Preview marker_detect_width must be a nonnegative whole number")
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
    # Startup renders every config value as an editable field, so this could be
    # set to zero or a negative number, which would make the settle timer fire at once.
    settle_seconds = CONFIG['capture'].get('auto_lock_settle_seconds', 5.0)
    if (isinstance(settle_seconds, bool) or not isinstance(settle_seconds, (int, float))
            or settle_seconds <= 0):
        raise ValueError("Capture auto_lock_settle_seconds must be a positive number")
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
    required_cameras = {str(i + 1) for i in range(camera_count)}
    if (not isinstance(send_crops, dict) or not required_cameras.issubset(send_crops)
            or not set(send_crops).issubset({'1', '2'})):
        raise ValueError('SAM detection send_crops must define each active camera')
    for camera in send_crops:
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
    
    
    

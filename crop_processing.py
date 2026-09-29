"""Persistent rotated crop geometry shared by setup and inspection."""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

# Bounds that keep a four-point mark meaningful. These are geometry guards, not
# preferences: below them the deskew angle or the enclosed area is not trusted.
MIN_MARK_EDGE_PX = 20.0
MIN_MARK_BASELINE_PX = 20.0
MIN_MARK_SPREAD_PX = 30.0
MIN_CROP_WIDTH_PX = 120.0
MAX_END_EDGE_TILT_DEGREES = 60.0


DEFAULT_CROP_SIZE = "M"


def _empty_cameras():
    return {"1": [None, None], "2": [None, None]}


def normalized_size_name(size):
    """Canonical key used by settings, crop setup and inspection."""
    text = str(size or DEFAULT_CROP_SIZE).strip().upper()
    return text or DEFAULT_CROP_SIZE


def empty_crop_store(sizes=None):
    names = [normalized_size_name(size) for size in (sizes or [DEFAULT_CROP_SIZE])]
    return {
        "version": 2,
        "sizes": {name: _empty_cameras() for name in dict.fromkeys(names)},
    }


def crop_cameras_for_size(data, size, create=False):
    """Return the four saved regions for ``size``.

    Version-1 stores had one top-level ``cameras`` object. It is treated as the
    Medium profile so an existing installation keeps its current crops during
    migration.
    """
    name = normalized_size_name(size)
    if not isinstance(data, dict):
        return _empty_cameras() if create else None
    sizes = data.get("sizes")
    if not isinstance(sizes, dict):
        legacy = data.get("cameras")
        if (not create and name == DEFAULT_CROP_SIZE
                and isinstance(legacy, dict)):
            return legacy
        if not create:
            return None
        legacy = legacy if isinstance(legacy, dict) else _empty_cameras()
        data.pop("cameras", None)
        data["version"] = 2
        data["sizes"] = {DEFAULT_CROP_SIZE: legacy}
        sizes = data["sizes"]
    if name not in sizes:
        if not create:
            return None
        sizes[name] = _empty_cameras()
    return sizes[name]


def parallel_line_angle(first, second):
    """Return a direction-independent deskew angle in the range [-90, 90).

    Direction-independent by design: phi and phi + 180 collapse to one answer, so a
    line clicked right-to-left deskews the same as one clicked left-to-right. The
    rectangle that results is identical either way, but the *handedness* of a stored
    alignment line is not -- four-point marking therefore uses ``directed_angle``
    instead, and this stays for the callers that genuinely want the folded form.
    """
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    delta = second - first
    if np.linalg.norm(delta) < 1e-9:
        raise ValueError("Rotation points must be different")
    angle = float(np.degrees(np.arctan2(delta[1], delta[0])))
    return (angle + 90.0) % 180.0 - 90.0


def directed_angle(first, second):
    """The angle of the first-to-second direction, without folding.

    ``parallel_line_angle`` folds away a half turn; keeping it here means the second
    marked point always lands on positive local x, so "which end is which" survives
    a save and reload. ``rotated_crop_corners`` accepts any angle, so an unfurled
    value outside [-90, 90) is harmless.
    """
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    delta = second - first
    if np.linalg.norm(delta) < 1e-9:
        raise ValueError("Marked points 2 and 3 must be different")
    return float(np.degrees(np.arctan2(delta[1], delta[0])))


def _rotation(angle_degrees):
    radians = math.radians(float(angle_degrees))
    cosine, sine = math.cos(radians), math.sin(radians)
    return np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)


def _cross(origin, first, second):
    return ((first[0] - origin[0]) * (second[1] - origin[1])
            - (first[1] - origin[1]) * (second[0] - origin[0]))


def _segments_cross(first_start, first_end, second_start, second_end):
    """Whether two open segments properly cross, used to catch a bow-tie order."""
    first_side = _cross(first_start, first_end, second_start)
    second_side = _cross(first_start, first_end, second_end)
    third_side = _cross(second_start, second_end, first_start)
    fourth_side = _cross(second_start, second_end, first_end)
    return ((first_side > 0) != (second_side > 0)
            and (third_side > 0) != (fourth_side > 0))


def four_point_crop(points, image_size, ratio=4.0, margin_percent=0.0,
                    min_width=MIN_CROP_WIDTH_PX):
    """Derive a crop rectangle from four corners marked around a region.

    The clicks are taken in order around the region, so ``points[0] -> points[1]``
    is one end edge, ``points[1] -> points[2]`` is the long side that must come out
    horizontal, and ``points[2] -> points[3]`` is the opposite end edge.

    The marked corners are measured in the deskewed frame, grown by ``margin_percent``
    of the marked size on each side, then grown again -- never shrunk -- until they
    fit the requested aspect ratio. Every step only expands the box, so the marked
    region and its margin always end up inside the crop.

    Returns the same pixel-space shape as :func:`pixel_definition`, so the editor can
    hold it directly and :func:`normalized_definition` can persist it unchanged.
    """
    image_width, image_height = map(float, image_size)
    if min(image_width, image_height) <= 0:
        raise ValueError("Crop dimensions must be positive")
    if ratio <= 0:
        raise ValueError("The crop aspect ratio must be positive")
    if margin_percent < 0:
        raise ValueError("The crop margin must not be negative")
    marked = np.asarray(points, dtype=np.float64)
    if marked.shape != (4, 2):
        raise ValueError("Four marked points are required")

    edges = [marked[(index + 1) % 4] - marked[index] for index in range(4)]
    lengths = [float(np.linalg.norm(edge)) for edge in edges]
    if min(lengths) < MIN_MARK_EDGE_PX:
        raise ValueError("Two of the marked points are too close together")

    # A bow-tie order makes the long side a diagonal, so the deskew angle would come
    # from a line that does not run along the region at all.
    if (_segments_cross(marked[0], marked[1], marked[2], marked[3])
            or _segments_cross(marked[1], marked[2], marked[3], marked[0])):
        raise ValueError("Mark the four corners in order around the region")

    baseline = lengths[1]
    if baseline < MIN_MARK_BASELINE_PX:
        raise ValueError("The third point is too close to the second to set an angle")
    if baseline < 0.5 * max(lengths[0], lengths[2]):
        # Otherwise the correct crop comes out rotated a quarter turn, silently.
        raise ValueError("Points 2 and 3 must be the long side of the region")

    direction = edges[1] / baseline
    for start, end in ((marked[0], marked[1]), (marked[2], marked[3])):
        edge = end - start
        alignment = abs(float(np.dot(edge, direction))) / float(np.linalg.norm(edge))
        if alignment > math.cos(math.radians(MAX_END_EDGE_TILT_DEGREES)):
            raise ValueError(
                "The end edges must be roughly square to the long side between "
                "points 2 and 3"
            )

    angle = directed_angle(marked[1], marked[2])
    rotation = _rotation(angle)
    local = (marked - marked[1]) @ rotation
    x_min, x_max = float(local[:, 0].min()), float(local[:, 0].max())
    y_min, y_max = float(local[:, 1].min()), float(local[:, 1].max())
    width, height = x_max - x_min, y_max - y_min
    if height < MIN_MARK_SPREAD_PX:
        raise ValueError("The marked region is too thin — click four corners around it")

    margin_x = width * float(margin_percent) / 100.0
    margin_y = height * float(margin_percent) / 100.0
    width += 2.0 * margin_x
    height += 2.0 * margin_y

    # The centre of the marked box, before the ratio and floor steps grow it. Both of
    # those grow symmetrically, so they leave this point where it is.
    centre_local = np.array([(x_min + x_max) / 2.0, (y_min + y_max) / 2.0])

    # Grow the short side to meet the ratio. Dividing would let a degenerate height
    # become infinite and slip past the comparison, so the test is written as a
    # product instead.
    if width < height * float(ratio):
        width = height * float(ratio)
    else:
        height = width / float(ratio)
    if width < float(min_width):
        width = float(min_width)
        height = width / float(ratio)
    if abs(height - width / float(ratio)) > 1e-6:
        raise ValueError("The crop could not be fitted to the configured aspect ratio")

    centre = marked[1] + rotation @ centre_local
    return {
        "center": centre.tolist(),
        "width": width,
        "height": height,
        "angle_degrees": angle,
        "line": [marked[1].tolist(), marked[2].tolist()],
        # Carried so the region can be re-derived when the configured ratio or margin
        # changes, instead of asking the operator to mark it again.
        "marked_quad": marked.tolist(),
    }


def load_crop_store(path, sizes=None, default_size=DEFAULT_CROP_SIZE):
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError, TypeError):
        return empty_crop_store(sizes)
    if not isinstance(data, dict):
        return empty_crop_store(sizes)
    requested = [normalized_size_name(size) for size in (sizes or [])]
    stored_sizes = data.get("sizes")
    if isinstance(stored_sizes, dict):
        requested.extend(normalized_size_name(size) for size in stored_sizes)
    if not requested:
        requested = [normalized_size_name(default_size)]
    result = empty_crop_store(requested)
    # The ratio these regions were measured at, so a later change to the configured
    # aspect ratio is reported instead of silently reshaping every saved crop.
    stored_ratio = data.get("aspect_ratio")
    if isinstance(stored_ratio, (list, tuple)) and len(stored_ratio) == 2:
        result["aspect_ratio"] = list(stored_ratio)
    sources = stored_sizes if isinstance(stored_sizes, dict) else {
        normalized_size_name(default_size): data.get("cameras", {})
    }
    for size, cameras in sources.items():
        name = normalized_size_name(size)
        if not isinstance(cameras, dict):
            continue
        target = crop_cameras_for_size(result, name, create=True)
        for camera in ("1", "2"):
            crops = cameras.get(camera, [])
            if isinstance(crops, list):
                target[camera] = (crops[:2] + [None, None])[:2]
    return result


def save_crop_store(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def normalized_definition(center, width, angle_degrees, line, image_size, quad=None):
    """Store a crop in normalized coordinates.

    ``quad`` is the four marked corners, when the crop came from the four-point
    tool. Keeping them means a later change to the configured aspect ratio or margin
    can re-derive the region instead of asking the operator to mark it again. It is
    an extra key that older readers ignore, so no stored file needs migrating.
    """
    image_width, image_height = map(float, image_size)
    if min(image_width, image_height, width) <= 0:
        raise ValueError("Crop dimensions must be positive")
    definition = {
        "center_normalized": [float(center[0]) / image_width,
                              float(center[1]) / image_height],
        "width_normalized": float(width) / image_width,
        "angle_degrees": float(angle_degrees),
        "line_normalized": [
            [float(line[0][0]) / image_width, float(line[0][1]) / image_height],
            [float(line[1][0]) / image_width, float(line[1][1]) / image_height],
        ],
    }
    if quad is not None:
        definition["marked_quad_normalized"] = [
            [float(point[0]) / image_width, float(point[1]) / image_height]
            for point in quad
        ]
    return definition


def pixel_definition(definition, image_size, ratio=4.0):
    image_width, image_height = map(float, image_size)
    center = definition["center_normalized"]
    width = float(definition["width_normalized"]) * image_width
    line = definition.get("line_normalized", [[0.0, 0.0], [1.0, 0.0]])
    return {
        "center": [float(center[0]) * image_width, float(center[1]) * image_height],
        "width": width,
        "height": width / float(ratio),
        "angle_degrees": float(definition.get("angle_degrees", 0.0)),
        "line": [
            [float(line[0][0]) * image_width, float(line[0][1]) * image_height],
            [float(line[1][0]) * image_width, float(line[1][1]) * image_height],
        ],
    }


def crop_edit_from_saved(definition, image_size, ratio=4.0, margin_percent=0.0,
                         min_width=MIN_CROP_WIDTH_PX):
    """The editable crop for a stored definition.

    A crop marked with the four-point tool carries its corners, so it is rebuilt with
    the ratio and margin the configuration asks for now -- that is what makes those
    two settings adjustable without re-marking every region. A crop without them, an
    older one or a hand-edited entry, keeps the box it was saved with.
    """
    quad = definition.get("marked_quad_normalized")
    if quad is not None:
        image_width, image_height = map(float, image_size)
        points = [[float(point[0]) * image_width, float(point[1]) * image_height]
                  for point in quad]
        try:
            return four_point_crop(points, image_size, ratio, margin_percent, min_width)
        except ValueError:
            # A stored corner set the geometry guards now reject still opens on the
            # box that was saved, rather than leaving the editor with nothing.
            pass
    return pixel_definition(definition, image_size, ratio)


def rotated_crop_corners(center, width, ratio, angle_degrees):
    """Return TL, TR, BR, BL source corners for a rotated 4:1 rectangle."""
    height = float(width) / float(ratio)
    local = np.array([
        [-width / 2.0, -height / 2.0],
        [width / 2.0, -height / 2.0],
        [width / 2.0, height / 2.0],
        [-width / 2.0, height / 2.0],
    ], dtype=np.float64)
    radians = np.deg2rad(float(angle_degrees))
    rotation = np.array([
        [np.cos(radians), -np.sin(radians)],
        [np.sin(radians), np.cos(radians)],
    ])
    return local @ rotation.T + np.asarray(center, dtype=np.float64)


def definition_corners(definition, image_size, ratio=4.0):
    pixels = pixel_definition(definition, image_size, ratio)
    return rotated_crop_corners(
        pixels["center"], pixels["width"], ratio, pixels["angle_degrees"],
    )


def definition_fits_image(definition, image_size, ratio=4.0):
    width, height = image_size
    corners = definition_corners(definition, image_size, ratio)
    return bool(
        np.all(corners[:, 0] >= 0) and np.all(corners[:, 0] <= width - 1)
        and np.all(corners[:, 1] >= 0) and np.all(corners[:, 1] <= height - 1)
    )


def rotated_crop_transform(definition, image_size, output_size=(2208, 552), ratio=4.0):
    """The perspective transform mapping source-frame pixels to the crop.

    Exposed separately from :func:`extract_rotated_crop` because anything that
    needs to relate a measurement back to the source frame -- a millimetre
    scale, for instance -- has to use exactly this transform, not a copy of it.
    """
    source = definition_corners(definition, image_size, ratio).astype(np.float32)
    output_width, output_height = map(int, output_size)
    # Pixel-boundary coordinates preserve the exact 4:1 scale in a 2208x552
    # raster; using width-1/height-1 would introduce a small anisotropic scale.
    destination = np.array([
        [-0.5, -0.5], [output_width - 0.5, -0.5],
        [output_width - 0.5, output_height - 0.5], [-0.5, output_height - 0.5],
    ], dtype=np.float32)
    return cv2.getPerspectiveTransform(source, destination)


def extract_rotated_crop(image, definition, output_size=(2208, 552), ratio=4.0):
    """Deskew and extract one saved region at a fixed, ratio-matched size."""
    if image is None or image.size == 0:
        raise ValueError("An undistorted image is required")
    image_size = (image.shape[1], image.shape[0])
    if not definition_fits_image(definition, image_size, ratio):
        raise ValueError("Saved crop extends outside the current camera image")
    output_width, output_height = map(int, output_size)
    transform = rotated_crop_transform(definition, image_size, output_size, ratio)
    return cv2.warpPerspective(
        image, transform, (output_width, output_height),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )

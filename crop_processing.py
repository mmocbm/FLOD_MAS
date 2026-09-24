"""Persistent rotated crop geometry shared by setup and inspection."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def empty_crop_store():
    return {"version": 1, "cameras": {"1": [None, None], "2": [None, None]}}


def parallel_line_angle(first, second):
    """Return a direction-independent deskew angle in the range [-90, 90)."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    delta = second - first
    if np.linalg.norm(delta) < 1e-9:
        raise ValueError("Rotation points must be different")
    angle = float(np.degrees(np.arctan2(delta[1], delta[0])))
    return (angle + 90.0) % 180.0 - 90.0


def load_crop_store(path):
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError, TypeError):
        return empty_crop_store()
    cameras = data.get("cameras") if isinstance(data, dict) else None
    if not isinstance(cameras, dict):
        return empty_crop_store()
    result = empty_crop_store()
    for camera in ("1", "2"):
        crops = cameras.get(camera, [])
        if isinstance(crops, list):
            result["cameras"][camera] = (crops[:2] + [None, None])[:2]
    return result


def save_crop_store(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def normalized_definition(center, width, angle_degrees, line, image_size):
    image_width, image_height = map(float, image_size)
    if min(image_width, image_height, width) <= 0:
        raise ValueError("Crop dimensions must be positive")
    return {
        "center_normalized": [float(center[0]) / image_width,
                              float(center[1]) / image_height],
        "width_normalized": float(width) / image_width,
        "angle_degrees": float(angle_degrees),
        "line_normalized": [
            [float(line[0][0]) / image_width, float(line[0][1]) / image_height],
            [float(line[1][0]) / image_width, float(line[1][1]) / image_height],
        ],
    }


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


def stack_crop_results(crops):
    """Build a labelled vertical preview while keeping each crop undistorted."""
    if len(crops) != 2 or any(crop is None for crop in crops):
        raise ValueError("Exactly two crop images are required")
    labelled = []
    for index, crop in enumerate(crops, start=1):
        panel = crop.copy()
        scale = max(0.7, panel.shape[1] / 2208.0)
        cv2.putText(
            panel, f"CROP {index}", (24, max(36, int(44 * scale))),
            cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 255),
            max(2, int(round(2 * scale))), cv2.LINE_AA,
        )
        labelled.append(panel)
    separator = np.full((12, labelled[0].shape[1], 3), 28, dtype=np.uint8)
    return np.vstack([labelled[0], separator, labelled[1]])

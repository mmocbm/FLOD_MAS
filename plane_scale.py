"""Millimetre scale for a deskewed crop, from the calibrated measurement plane.

A camera's intrinsics and extrinsics together define a physical plane in front
of it: the plane the calibration board sat on. Any pixel that lands on that
plane can be turned into a real position on it, because the pixel picks out a
ray and the ray crosses the plane at exactly one point.

That is what turns a strip measured in crop pixels into millimetres. Two
transforms compose:

* crop pixels -> source pixels, by inverting the deskew warp; and
* source pixels -> millimetres on the plane, by intersecting the ray.

Both are projective, so their composition is a single 3x3 homography. It is
built here by evaluating the already-tested plane intersection at four crop
corners rather than by re-deriving the algebra, which keeps one implementation
of the projection and makes a mistake in it impossible to hide.

The measurement is only valid for objects lying on the calibration plane. An
object standing proud of it, or the plane being moved, changes the answer, and
measurements taken far outside the board's own footprint lean on the plane
being flat well beyond where it was checked.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from CalibrateAPP.calibration_math import pixel_to_plane, scale_camera_matrix
from crop_processing import rotated_crop_transform

# Metres to millimetres. The extrinsics store the pose in metres, and the plane
# intersection inherits that, matching how CalibrateAPP already reads them.
_MM_PER_METRE = 1000.0

# The crop corners used to pin the homography down. All four are needed: three
# would fit an affine map and lose the perspective the crop warp introduces.
_CORNER_INSET_PX = 0.5

# Smallest crop area, in square millimetres, still worth measuring. A real crop
# is tens of thousands; this only rejects a region that collapsed to a point.
_MIN_CROP_AREA_MM2 = 1.0


class PlaneScaleError(RuntimeError):
    """No usable millimetre scale for this camera."""


@dataclass(frozen=True)
class PlaneScale:
    """Maps deskewed-crop pixels onto the calibrated plane, in millimetres."""

    homography: np.ndarray      # 3x3, crop pixels -> plane millimetres
    mm_per_pixel: float         # at the crop centre; a summary, not a constant

    def to_mm(self, points: np.ndarray) -> np.ndarray:
        """Map crop pixels to millimetres on the plane.

        Accepts an ``(N, 2)`` or ``(N, 1, 2)`` array and returns ``(N, 2)``.
        """
        array = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(array, self.homography).reshape(-1, 2)


def load_plane_scale(
    camera_number: int,
    definition: dict,
    frame_size: tuple[int, int],
    output_size: tuple[int, int] = (2208, 552),
    calibration_path: str | Path | None = None,
    extrinsics_path: str | Path | None = None,
) -> PlaneScale:
    """Build the crop-to-millimetre scale for one camera and crop region.

    Raises :class:`PlaneScaleError` when the calibration or extrinsics are
    missing or unusable, so the caller can report why rather than silently
    measure nothing.
    """
    if calibration_path is None or extrinsics_path is None:
        calibration_path, extrinsics_path = _paths_for(camera_number)

    calibration = _read_json(calibration_path, "camera calibration")
    extrinsics = _read_json(extrinsics_path, "camera extrinsics")

    try:
        camera_matrix = np.asarray(calibration["camera_matrix"], np.float64)
        dist_coeffs = np.asarray(calibration["dist_coeffs"], np.float64)
        calibrated_size = calibration["image_size"]
        rvec = np.asarray(extrinsics["rvec"], np.float64).reshape(3, 1)
        tvec = np.asarray(extrinsics["tvec"], np.float64).reshape(3, 1)
    except KeyError as missing:
        raise PlaneScaleError(f"The calibration files are missing {missing}") from missing

    # The frame the crop is taken from; the intrinsics are scaled to match, in
    # case the camera is running at a different resolution than it calibrated at.
    camera_matrix = scale_camera_matrix(camera_matrix, calibrated_size, frame_size)

    width, height = map(int, output_size)
    inset = _CORNER_INSET_PX
    corners = np.array([
        [inset, inset], [width - 1 - inset, inset],
        [width - 1 - inset, height - 1 - inset], [inset, height - 1 - inset],
    ], dtype=np.float32)

    try:
        to_crop = rotated_crop_transform(definition, frame_size, output_size)
        source = cv2.perspectiveTransform(
            corners.reshape(-1, 1, 2), np.linalg.inv(to_crop)
        ).reshape(-1, 2)
    except (np.linalg.LinAlgError, cv2.error) as error:
        raise PlaneScaleError("The saved crop region is degenerate") from error

    try:
        on_plane = np.asarray([
            pixel_to_plane(point, camera_matrix, rvec, tvec) * _MM_PER_METRE
            for point in source
        ], dtype=np.float32)
    except ValueError as error:
        raise PlaneScaleError(str(error)) from error

    if not np.isfinite(on_plane).all():
        raise PlaneScaleError("The crop region does not project onto the measurement plane")

    # A collapsed crop region maps all four corners to the same point, which is
    # perfectly finite and would quietly make every measurement zero. A real
    # crop covers tens of thousands of square millimetres, so the floor here is
    # far below anything usable and only catches a region that is not a region.
    if abs(cv2.contourArea(on_plane.astype(np.float32))) < _MIN_CROP_AREA_MM2:
        raise PlaneScaleError("The saved crop region is degenerate")

    try:
        homography = cv2.getPerspectiveTransform(corners, on_plane)
    except cv2.error as error:
        raise PlaneScaleError("The crop cannot be mapped onto the plane") from error
    if not np.isfinite(homography).all():
        raise PlaneScaleError("The crop cannot be mapped onto the plane")

    # One pixel's worth of millimetres at the crop's centre. A summary for
    # reporting, not a conversion factor.
    centre = np.array([[width / 2.0, height / 2.0],
                       [width / 2.0 + 1.0, height / 2.0]], dtype=np.float32)
    stepped = cv2.perspectiveTransform(centre.reshape(-1, 1, 2), homography).reshape(-1, 2)
    mm_per_pixel = float(np.linalg.norm(stepped[1] - stepped[0]))

    return PlaneScale(homography=homography, mm_per_pixel=mm_per_pixel)


def _paths_for(camera_number: int) -> tuple[Path, Path]:
    """The calibration and extrinsics files named by config for a camera.

    Cameras are numbered 1 and 2 by position, matching how the rest of the app
    pairs them (``CALIB_FILE_1, CALIB_FILE_2 = [c['calibration_file'] for c in
    CONFIG['cameras']]``). The ``index`` field is the device's capture index in
    :func:`camera_config`, not the side's number, so matching on it here would
    hand camera 1 the other camera's calibration.
    """
    from app_config import CONFIG, project_path

    try:
        number = int(camera_number)
    except (TypeError, ValueError) as error:
        raise PlaneScaleError(
            f"There is no camera {camera_number} in the configuration"
        ) from error
    cameras = CONFIG["cameras"]
    # Checked rather than indexed-and-caught: camera 0 would otherwise be
    # cameras[-1], quietly handing the first side the last camera's files.
    if not 1 <= number <= len(cameras):
        raise PlaneScaleError(f"There is no camera {camera_number} in the configuration")
    camera = cameras[number - 1]
    return (
        Path(project_path(camera["calibration_file"])),
        Path(project_path(camera["extrinsics_file"])),
    )


def _read_json(path: str | Path, label: str) -> dict:
    path = Path(path)
    if not path.is_file():
        raise PlaneScaleError(f"The {label} file is missing: {path.name}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise PlaneScaleError(f"The {label} file could not be read: {error}") from error

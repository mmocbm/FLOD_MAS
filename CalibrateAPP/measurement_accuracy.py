"""Real-world ChArUco distance verification for a saved measurement plane."""

from __future__ import annotations

from itertools import combinations

import cv2
import numpy as np

try:
    from .calibration_math import pixel_to_plane
except ImportError:
    from calibration_math import pixel_to_plane


def offset_plane_tvec(rvec, tvec, distance_mm, away_from_camera=True):
    """Move a planar pose normally by a physical distance.

    ``away_from_camera`` converts a pose measured on the top of a calibration
    board into the bed plane below it. The direction is derived from the pose,
    so it remains correct if OpenCV chooses the opposite board-normal sign.
    """
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3, 1))
    translation = np.asarray(tvec, np.float64).reshape(3)
    normal = rotation[:, 2]
    direction = 1.0 if float(normal.dot(translation)) >= 0 else -1.0
    if not away_from_camera:
        direction *= -1.0
    return (translation + direction * float(distance_mm) / 1000.0 * normal).reshape(3, 1)


def _pair_rows(object_xy_mm, measured_xy_mm, square_length_mm, board_number):
    rows = []
    candidates = []
    for first, second in combinations(range(len(object_xy_mm)), 2):
        expected = float(np.linalg.norm(object_xy_mm[first] - object_xy_mm[second]))
        if expected <= 1e-6:
            continue
        measured = float(np.linalg.norm(measured_xy_mm[first] - measured_xy_mm[second]))
        candidates.append((expected, measured, first, second))

    if not candidates:
        raise ValueError("Not enough distinct ChArUco corners were detected")

    maximum = max(item[0] for item in candidates)
    long_cutoff = max(3.0 * square_length_mm, 0.65 * maximum)
    long_candidates = [item for item in candidates if item[0] >= long_cutoff]
    # Keep the report readable while retaining several directions and spans.
    long_candidates = sorted(long_candidates, reverse=True)[:24]
    selected = [
        ("short", item) for item in candidates
        if item[0] <= 1.05 * square_length_mm
    ] + [("long", item) for item in long_candidates]

    for category, (expected, measured, first, second) in selected:
        signed_error = measured - expected
        rows.append({
            "board": int(board_number),
            "category": category,
            "point_1": int(first),
            "point_2": int(second),
            "expected_mm": expected,
            "measured_mm": measured,
            "error_mm": signed_error,
            "absolute_error_mm": abs(signed_error),
            "error_percent": 100.0 * signed_error / expected,
        })
    return rows


def measure_board_accuracy(
    board,
    charuco_corners,
    charuco_ids,
    camera_matrix,
    dist_coeffs,
    plane_rvec,
    plane_tvec,
    square_length_mm,
    board_number=1,
):
    """Compare measured board distances with its known printed geometry."""
    object_points, image_points = board.matchImagePoints(charuco_corners, charuco_ids)
    if object_points is None or image_points is None or len(object_points) < 4:
        raise ValueError("At least four matched ChArUco corners are required")

    object_xy_mm = np.asarray(object_points, np.float64).reshape(-1, 3)[:, :2] * 1000.0
    image_points = np.asarray(image_points, np.float64).reshape(-1, 1, 2)
    undistorted_pixels = cv2.undistortPoints(
        image_points,
        np.asarray(camera_matrix, np.float64),
        np.asarray(dist_coeffs, np.float64),
        P=np.asarray(camera_matrix, np.float64),
    ).reshape(-1, 2)
    measured_xy_mm = np.asarray([
        pixel_to_plane(point, camera_matrix, plane_rvec, plane_tvec) * 1000.0
        for point in undistorted_pixels
    ])
    return _pair_rows(
        object_xy_mm, measured_xy_mm, float(square_length_mm), board_number,
    )


def summarize_accuracy(rows):
    """Return report statistics without assigning a pass/fail grade."""
    if not rows:
        raise ValueError("No measurement distances were available")

    def metrics(items):
        absolute = np.asarray([item["absolute_error_mm"] for item in items], np.float64)
        signed = np.asarray([item["error_mm"] for item in items], np.float64)
        return {
            "count": int(len(items)),
            "mean_absolute_error_mm": float(absolute.mean()),
            "rmse_mm": float(np.sqrt(np.mean(signed ** 2))),
            "maximum_absolute_error_mm": float(absolute.max()),
            "mean_signed_error_mm": float(signed.mean()),
        }

    summary = {"overall": metrics(rows), "boards": {}}
    for category in ("short", "long"):
        selected = [row for row in rows if row["category"] == category]
        if selected:
            summary[category] = metrics(selected)
    for board_number in sorted({row["board"] for row in rows}):
        selected = [row for row in rows if row["board"] == board_number]
        summary["boards"][str(board_number)] = metrics(selected)
    return summary

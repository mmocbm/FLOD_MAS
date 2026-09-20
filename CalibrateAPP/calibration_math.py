"""Reusable, testable camera-calibration math helpers."""

from __future__ import annotations

import cv2
import numpy as np


def coverage_cell(image_points, image_size, rows=3, columns=3):
    """Return the grid cell containing the mean detected-board position."""
    points = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    width, height = map(float, image_size)
    if len(points) == 0 or width <= 0 or height <= 0:
        raise ValueError("Image points and a positive image size are required")
    center = points.mean(axis=0)
    column = min(columns - 1, max(0, int(center[0] / width * columns)))
    row = min(rows - 1, max(0, int(center[1] / height * rows)))
    return row, column, (float(center[0] / width), float(center[1] / height))


def next_coverage_cell(counts, last_cell=None):
    """Choose a least-used cell, preferring distance from the last capture."""
    counts = np.asarray(counts)
    if counts.ndim != 2 or counts.size == 0:
        raise ValueError("Coverage counts must be a non-empty 2D array")
    candidates = np.argwhere(counts == counts.min())
    if last_cell is None:
        target = np.array([(counts.shape[0] - 1) / 2, (counts.shape[1] - 1) / 2])
        distances = np.sum((candidates - target) ** 2, axis=1)
        selected = candidates[np.argmin(distances)]
    else:
        last = np.asarray(last_cell, dtype=np.float64)
        distances = np.sum((candidates - last) ** 2, axis=1)
        selected = candidates[np.argmax(distances)]
    return int(selected[0]), int(selected[1])


def coverage_percent(counts):
    counts = np.asarray(counts)
    return int(round(100 * np.count_nonzero(counts) / counts.size)) if counts.size else 0


def scale_camera_matrix(camera_matrix, calibrated_size, target_size):
    """Scale an intrinsic matrix from calibrated_size to target_size."""
    matrix = np.asarray(camera_matrix, dtype=np.float64).copy()
    calibrated_width, calibrated_height = map(float, calibrated_size)
    target_width, target_height = map(float, target_size)

    if calibrated_width <= 0 or calibrated_height <= 0:
        raise ValueError("Calibration image size must be positive")

    scale_x = target_width / calibrated_width
    scale_y = target_height / calibrated_height
    matrix[0, 0] *= scale_x
    matrix[0, 2] *= scale_x
    matrix[1, 1] *= scale_y
    matrix[1, 2] *= scale_y
    return matrix


def reprojection_metrics(object_points, image_points, rvec, tvec, camera_matrix, dist_coeffs):
    """Return RMS, mean, and maximum reprojection error in pixels."""
    projected, _ = cv2.projectPoints(
        np.asarray(object_points, dtype=np.float32),
        np.asarray(rvec, dtype=np.float64),
        np.asarray(tvec, dtype=np.float64),
        np.asarray(camera_matrix, dtype=np.float64),
        np.asarray(dist_coeffs, dtype=np.float64),
    )
    observed = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    projected = projected.reshape(-1, 2)
    errors = np.linalg.norm(observed - projected, axis=1)
    return {
        "rms": float(np.sqrt(np.mean(errors ** 2))),
        "mean": float(np.mean(errors)),
        "max": float(np.max(errors)),
    }


def estimate_planar_pose(
    board,
    charuco_corners,
    charuco_ids,
    camera_matrix,
    dist_coeffs,
):
    """Estimate and refine the ChArUco board pose for the measurement plane."""
    object_points, image_points = board.matchImagePoints(charuco_corners, charuco_ids)
    if object_points is None or image_points is None or len(object_points) < 4:
        raise ValueError("At least four matched ChArUco corners are required")

    object_points = np.asarray(object_points, dtype=np.float32).reshape(-1, 3)
    image_points = np.asarray(image_points, dtype=np.float32).reshape(-1, 2)
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
    dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE,
    )
    if not success:
        raise RuntimeError("OpenCV could not estimate the measurement-plane pose")

    rvec, tvec = cv2.solvePnPRefineLM(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        rvec,
        tvec,
    )
    if float(tvec.reshape(-1)[2]) <= 0:
        raise RuntimeError("Estimated board pose is behind the camera")

    metrics = reprojection_metrics(
        object_points,
        image_points,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )
    return rvec, tvec, metrics, object_points, image_points


def pixel_to_plane(pixel, camera_matrix, rvec, tvec):
    """Intersect an undistorted image pixel ray with the board's Z=0 plane."""
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3)
    rotation, _ = cv2.Rodrigues(rvec)

    point = np.asarray(pixel, dtype=np.float64).reshape(1, 1, 2)
    normalized = cv2.undistortPoints(
        point,
        camera_matrix,
        np.zeros((5, 1), dtype=np.float64),
    )[0, 0]
    ray = np.array([normalized[0], normalized[1], 1.0], dtype=np.float64)

    plane_normal = rotation[:, 2]
    plane_offset = -plane_normal.dot(tvec)
    denominator = plane_normal.dot(ray)
    if abs(denominator) < 1e-12:
        raise ValueError("Pixel ray is parallel to the measurement plane")

    point_camera = (-plane_offset / denominator) * ray
    point_board = rotation.T.dot(point_camera - tvec)
    return point_board[:2]

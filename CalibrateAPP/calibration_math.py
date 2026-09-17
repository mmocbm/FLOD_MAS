"""Reusable, testable camera-calibration math helpers."""

from __future__ import annotations

import cv2
import numpy as np


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

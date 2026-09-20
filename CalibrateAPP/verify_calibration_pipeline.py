"""Deterministic numerical verification of calibration and plane projection math."""

from __future__ import annotations

from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from app_config import CONFIG

try:
    from .calibration_math import estimate_planar_pose, pixel_to_plane, scale_camera_matrix
except ImportError:
    from calibration_math import estimate_planar_pose, pixel_to_plane, scale_camera_matrix


def main():
    board_config = CONFIG['board']
    dictionary = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, board_config['dictionary'])
    )
    board = cv2.aruco.CharucoBoard(
        (board_config['squares_x'], board_config['squares_y']),
        board_config['square_length_mm'] / 1000.0,
        board_config['marker_length_mm'] / 1000.0,
        dictionary,
    )

    camera_matrix = np.array(
        [[2200.0, 0.0, 1280.0], [0.0, 2190.0, 720.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((5, 1), dtype=np.float64)
    true_rvec = np.array([[0.12], [-0.08], [0.025]], dtype=np.float64)
    true_tvec = np.array([[0.015], [-0.025], [0.72]], dtype=np.float64)

    object_points = np.asarray(board.getChessboardCorners(), dtype=np.float32)
    image_points, _ = cv2.projectPoints(
        object_points, true_rvec, true_tvec, camera_matrix, dist_coeffs
    )
    ids = np.arange(len(object_points), dtype=np.int32).reshape(-1, 1)

    estimated_rvec, estimated_tvec, metrics, _, _ = estimate_planar_pose(
        board,
        image_points.astype(np.float32),
        ids,
        camera_matrix,
        dist_coeffs,
    )
    true_rotation, _ = cv2.Rodrigues(true_rvec)
    estimated_rotation, _ = cv2.Rodrigues(estimated_rvec)
    delta_rotation, _ = cv2.Rodrigues(estimated_rotation @ true_rotation.T)
    rotation_error_degrees = float(np.linalg.norm(delta_rotation) * 180.0 / np.pi)
    translation_error_mm = float(np.linalg.norm(estimated_tvec - true_tvec) * 1000.0)

    scaled_matrix = scale_camera_matrix(camera_matrix, (2560, 1440), (1280, 720))
    scaled_projection, _ = cv2.projectPoints(
        object_points, true_rvec, true_tvec, scaled_matrix, dist_coeffs
    )
    scale_error_px = float(
        np.max(np.abs(scaled_projection.reshape(-1, 2) - image_points.reshape(-1, 2) * 0.5))
    )

    board_point = np.array([[0.075, 0.105, 0.0]], dtype=np.float32)
    board_pixel, _ = cv2.projectPoints(
        board_point, true_rvec, true_tvec, camera_matrix, dist_coeffs
    )
    recovered_point = pixel_to_plane(
        board_pixel.reshape(2), camera_matrix, true_rvec, true_tvec
    )
    plane_error_mm = float(np.linalg.norm(recovered_point - board_point[0, :2]) * 1000.0)

    checks = {
        "pose reprojection RMS < 0.001 px": metrics["rms"] < 0.001,
        "pose rotation error < 0.001 deg": rotation_error_degrees < 0.001,
        "pose translation error < 0.001 mm": translation_error_mm < 0.001,
        "resolution scaling error < 0.001 px": scale_error_px < 0.001,
        "pixel-to-plane round-trip error < 0.001 mm": plane_error_mm < 0.001,
    }

    print("Calibration pipeline numerical verification")
    print(f"  Pose reprojection RMS: {metrics['rms']:.9f} px")
    print(f"  Rotation error:        {rotation_error_degrees:.9f} deg")
    print(f"  Translation error:     {translation_error_mm:.9f} mm")
    print(f"  Resolution scale error:{scale_error_px: .9f} px")
    print(f"  Plane round-trip error:{plane_error_mm: .9f} mm")
    for name, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")

    if not all(checks.values()):
        raise SystemExit(1)
    print("RESULT: PASS")


if __name__ == "__main__":
    main()

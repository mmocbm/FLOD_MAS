"""Per-frame metric measurement plane from one configured ArUco marker."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np


class MarkerPlaneError(RuntimeError):
    """Raised when the configured measurement marker cannot define a safe plane."""


@dataclass(frozen=True)
class MarkerPlane:
    homography: np.ndarray
    corners: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    reprojection_rms_px: float
    minimum_side_px: float

    def pixels_to_mm(self, points):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(points, self.homography).reshape(-1, 2)


class ArucoPlaneEstimator:
    """Detect marker ID 0 (or configured ID) and map image pixels to plane mm."""

    def __init__(
            self, camera_matrix, dist_coeffs, calibration_image_size,
            dictionary_name="DICT_4X4_50", marker_id=0, marker_length_mm=25.0,
            minimum_side_px=30.0, maximum_reprojection_error_px=2.0):
        self.base_camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
        self.dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64)
        self.calibration_image_size = (tuple(calibration_image_size)
                                       if calibration_image_size else None)
        self.dictionary_name = dictionary_name
        self.marker_id = int(marker_id)
        self.marker_length_mm = float(marker_length_mm)
        self.minimum_side_px = float(minimum_side_px)
        self.maximum_reprojection_error_px = float(maximum_reprojection_error_px)
        dictionary_type = getattr(cv2.aruco, dictionary_name)
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_type)
        self.detector = cv2.aruco.ArucoDetector(
            dictionary, cv2.aruco.DetectorParameters(),
        )

    @classmethod
    def from_calibration_file(cls, calibration_path, marker_config):
        data = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
        return cls(
            data['camera_matrix'], data['dist_coeffs'], data.get('image_size'),
            dictionary_name=marker_config['aruco_dictionary'],
            marker_id=marker_config['aruco_marker_id'],
            marker_length_mm=marker_config['aruco_marker_length_mm'],
            minimum_side_px=marker_config['minimum_marker_side_px'],
            maximum_reprojection_error_px=marker_config['maximum_reprojection_error_px'],
        )

    def camera_matrix_for_size(self, image_size):
        width, height = image_size
        matrix = self.base_camera_matrix.copy()
        if self.calibration_image_size:
            calibrated_width, calibrated_height = self.calibration_image_size
            matrix[0, 0] *= width / calibrated_width
            matrix[0, 2] *= width / calibrated_width
            matrix[1, 1] *= height / calibrated_height
            matrix[1, 2] *= height / calibrated_height
        return matrix

    def detect(self, image, image_is_undistorted=True):
        """Return a metric image-to-marker-plane mapping for one frame."""
        if image is None or image.size == 0:
            raise MarkerPlaneError("No camera image is available")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        marker_corners, marker_ids, _ = self.detector.detectMarkers(gray)
        if marker_ids is None:
            raise MarkerPlaneError(
                f"Measurement marker {self.marker_id} was not found. "
                "Keep the complete marker visible and well lit."
            )
        matches = np.flatnonzero(marker_ids.reshape(-1) == self.marker_id)
        if not len(matches):
            found = ", ".join(map(str, marker_ids.reshape(-1).tolist()))
            raise MarkerPlaneError(
                f"Measurement marker {self.marker_id} was not found "
                f"(visible IDs: {found})."
            )

        corners = np.asarray(marker_corners[int(matches[0])], dtype=np.float64).reshape(4, 2)
        side_lengths = np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1)
        smallest_side = float(side_lengths.min())
        if smallest_side < self.minimum_side_px:
            raise MarkerPlaneError(
                f"Measurement marker is too small in the image "
                f"({smallest_side:.0f}px; need {self.minimum_side_px:.0f}px)."
            )

        image_size = (image.shape[1], image.shape[0])
        camera_matrix = self.camera_matrix_for_size(image_size)
        distortion = (np.zeros_like(self.dist_coeffs) if image_is_undistorted
                      else self.dist_coeffs)
        half = self.marker_length_mm / 2.0
        plane_corners = np.array(
            [[-half, half], [half, half], [half, -half], [-half, -half]],
            dtype=np.float32,
        )
        pose_points = np.column_stack(
            [plane_corners, np.zeros(4, dtype=np.float32)]
        )
        success, rvec, tvec = cv2.solvePnP(
            pose_points, corners.astype(np.float32), camera_matrix, distortion,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not success or float(tvec.reshape(-1)[2]) <= 0:
            raise MarkerPlaneError("The measurement marker position could not be calculated")
        projected, _ = cv2.projectPoints(
            pose_points, rvec, tvec, camera_matrix, distortion,
        )
        errors = np.linalg.norm(projected.reshape(-1, 2) - corners, axis=1)
        reprojection_rms = float(np.sqrt(np.mean(errors ** 2)))
        if reprojection_rms > self.maximum_reprojection_error_px:
            raise MarkerPlaneError(
                f"Measurement marker quality is too low ({reprojection_rms:.2f}px; "
                f"maximum {self.maximum_reprojection_error_px:.2f}px)."
            )

        homography = cv2.getPerspectiveTransform(
            corners.astype(np.float32), plane_corners,
        ).astype(np.float64)
        return MarkerPlane(
            homography=homography, corners=corners, rvec=rvec, tvec=tvec,
            reprojection_rms_px=reprojection_rms, minimum_side_px=smallest_side,
        )

    def annotate(self, image, plane, image_is_undistorted=True):
        annotated = image.copy()
        polygon = np.round(plane.corners).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(annotated, [polygon], True, (0, 220, 0), 4, cv2.LINE_AA)
        origin = tuple(polygon[0, 0])
        cv2.putText(
            annotated, f"MEASUREMENT MARKER {self.marker_id}",
            (origin[0], max(30, origin[1] - 15)), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 220, 0), 2, cv2.LINE_AA,
        )
        camera_matrix = self.camera_matrix_for_size((image.shape[1], image.shape[0]))
        distortion = (np.zeros_like(self.dist_coeffs) if image_is_undistorted
                      else self.dist_coeffs)
        cv2.drawFrameAxes(
            annotated, camera_matrix, distortion, plane.rvec, plane.tvec,
            self.marker_length_mm * 0.75, 3,
        )
        return annotated


def annotate_mask_measurements(image, mask, plane, minimum_area_px=100):
    """Measure each masked planar component through the marker homography."""
    if mask is None or mask.shape[:2] != image.shape[:2]:
        raise ValueError("Measurement mask must match the inspection image size")
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    annotated = image.copy()
    measurements = []
    inverse_homography = np.linalg.inv(plane.homography)
    for contour in contours:
        area_px = float(cv2.contourArea(contour))
        if area_px < minimum_area_px:
            continue
        image_points = contour.reshape(-1, 2).astype(np.float64)
        world_points = plane.pixels_to_mm(image_points).astype(np.float32)
        (_, _), (side_a, side_b), _ = cv2.minAreaRect(world_points)
        length_mm = float(max(side_a, side_b))
        width_mm = float(min(side_a, side_b))
        if length_mm <= 0 or width_mm <= 0:
            continue

        world_box = cv2.boxPoints(cv2.minAreaRect(world_points)).reshape(-1, 1, 2)
        image_box = cv2.perspectiveTransform(
            world_box.astype(np.float64), inverse_homography,
        ).reshape(-1, 2)
        image_box_int = np.round(image_box).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(annotated, [image_box_int], True, (0, 255, 0), 3, cv2.LINE_AA)
        center = tuple(np.round(image_box.mean(axis=0)).astype(int))
        cv2.putText(
            annotated, f"{length_mm:.1f} x {width_mm:.1f} mm",
            center, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA,
        )
        measurements.append({
            'length_mm': length_mm, 'width_mm': width_mm, 'area_px': area_px,
        })
    return annotated, measurements


def annotate_mask_pixel_measurements(image, mask, minimum_area_px=100):
    """Fallback measurement when no metric marker plane is available."""
    if mask is None or mask.shape[:2] != image.shape[:2]:
        raise ValueError("Measurement mask must match the inspection image size")
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    annotated = image.copy()
    measurements = []
    for contour in contours:
        area_px = float(cv2.contourArea(contour))
        if area_px < minimum_area_px:
            continue
        rect = cv2.minAreaRect(contour.astype(np.float32))
        side_a, side_b = rect[1]
        length_px = float(max(side_a, side_b))
        width_px = float(min(side_a, side_b))
        if length_px <= 0 or width_px <= 0:
            continue
        box = np.round(cv2.boxPoints(rect)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(annotated, [box], True, (0, 190, 255), 3, cv2.LINE_AA)
        center = tuple(np.round(np.asarray(rect[0])).astype(int))
        cv2.putText(
            annotated, f"{length_px:.1f} x {width_px:.1f} px",
            center, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 190, 255), 2, cv2.LINE_AA,
        )
        measurements.append({
            'length_px': length_px, 'width_px': width_px, 'area_px': area_px,
        })
    cv2.putText(
        annotated, "WARNING: PIXEL MEASUREMENTS ONLY - MARKER NOT FOUND",
        (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 190, 255), 2, cv2.LINE_AA,
    )
    return annotated, measurements

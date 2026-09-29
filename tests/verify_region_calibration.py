"""Numerical proof of the optional per-region homography mode.

Every fixture here is synthetic and exactly known: a pinhole camera looking at a
measurement plane, a crop region whose four corners are known millimetres on that
plane, and a board laid flat inside it. Because the truth is constructed rather
than measured, every number this prints is an error against something known, not a
residual of the code checking itself.

The first four checks are the mode working. The last three are the mode failing, on
purpose, in the ways that matter -- each one is a case where the fit is *excellent*
and the answer is *wrong*, so no residual would have caught it. Those are the reason
the module carries a flatness gate, a scale cross-check against the global plane,
and a refusal to refine intrinsics from a single view, and they are printed here
rather than asserted in a comment so the claims stay honest.

Check 7 is the one worth reading. It sweeps a deliberate focal-length error and
prints what each of the two measures reads, which is not what the module's own
comments assumed when it was first designed: the local homography is blind to the
error *exactly* (identical residual at every scale), while the lens evaluation does
see it but far too weakly to refine against -- a 15% error reads as 0.54 px, inside
detection noise. That measurement, not an assumption, is what the "evaluate always,
refine opt-in with tilted views only" decision rests on.

Run from the repository root:  .venv/Scripts/python.exe tests/verify_region_calibration.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

import plane_scale
import region_calibration
from crop_processing import rotated_crop_transform

try:
    from CalibrateAPP.calibration_math import pixel_to_plane
except ImportError:
    from calibration_math import pixel_to_plane


FRAME = (3456, 4608)
OUTPUT = (2208, 552)

CAMERA_MATRIX = np.array([
    [6812.9, 0.0, 1949.6],
    [0.0, 6812.9, 2243.1],
    [0.0, 0.0, 1.0],
], dtype=np.float64)
DISTORTION = np.zeros((5, 1), dtype=np.float64)
PLANE_RVEC = np.array([[0.62], [0.05], [0.03]], dtype=np.float64)
PLANE_TVEC = np.array([[0.04], [-0.31], [1.10]], dtype=np.float64)

# A band across the frame, the shape of a real crop region. The same definition the
# other plane fixtures use, so numbers here are comparable with tests/test_plane_scale.py.
DEFINITION = {
    "center_normalized": [0.4931506849, 0.2508561644],
    "width_normalized": 0.7762557078,
    "angle_degrees": 2.3760552180,
    "line_normalized": [[0.2283105023, 0.2859589041], [0.7785388128, 0.3030821918]],
}

CALIBRATION = {"camera_matrix": CAMERA_MATRIX.tolist(),
               "dist_coeffs": DISTORTION.tolist(),
               "image_size": list(FRAME)}
EXTRINSICS = {"rvec": PLANE_RVEC.tolist(), "tvec": PLANE_TVEC.tolist()}

TO_CROP = rotated_crop_transform(DEFINITION, FRAME, OUTPUT, 4.0)
MM_PER_METRE = 1000.0


def to_crop(frame_points):
    return cv2.perspectiveTransform(
        np.asarray(frame_points, np.float64).reshape(-1, 1, 2), TO_CROP,
    ).reshape(-1, 2)


def to_frame(crop_points):
    inverse = np.linalg.inv(TO_CROP)
    return cv2.perspectiveTransform(
        np.asarray(crop_points, np.float64).reshape(-1, 1, 2), inverse,
    ).reshape(-1, 2)


def project(plane_mm, rvec=PLANE_RVEC, tvec=PLANE_TVEC):
    """Plane millimetres -> undistorted-frame pixels, through the true camera."""
    object_points = np.column_stack([
        np.asarray(plane_mm, np.float64) / MM_PER_METRE,
        np.zeros(len(plane_mm)),
    ]).reshape(-1, 1, 3)
    image_points, _ = cv2.projectPoints(object_points, rvec, tvec, CAMERA_MATRIX,
                                        DISTORTION)
    return image_points.reshape(-1, 2)


def frame_to_plane(frame_points):
    """Undistorted-frame pixels -> millimetres on the measurement plane."""
    return np.array([
        pixel_to_plane(point, CAMERA_MATRIX, PLANE_RVEC, PLANE_TVEC) * MM_PER_METRE
        for point in np.asarray(frame_points, np.float64)
    ])


def region_corners_mm():
    """The crop's four corners as millimetres on the measurement plane."""
    width, height = OUTPUT
    corners = np.array([[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]])
    return frame_to_plane(to_frame(corners))


def fit_a_board(declared_scale=1.0, lift_mm=0.0, focal_scale=1.0, board=None):
    """Fit a local homography to a board on (or slightly off) the measurement plane.

    ``declared_scale`` multiplies the board coordinates the fit is told to expect, which
    is what a board printed at the wrong size, or a profile with the wrong checker size,
    does to the arithmetic. It does not move the board.
    """
    corners_mm = region_corners_mm()
    centre = corners_mm.mean(axis=0)
    span_x = float(np.linalg.norm(corners_mm[1] - corners_mm[0]))
    span_y = float(np.linalg.norm(corners_mm[3] - corners_mm[0]))

    if board is None:
        board = region_calibration.BoardDefinition(
            name="verification board", dictionary="DICT_5X5_100",
            squares_x=11, squares_y=3,
            square_length_mm=float(np.floor(span_x * 0.7 / 11 * 10) / 10),
            marker_length_mm=float(np.floor(span_x * 0.7 / 11 * 0.73 * 10) / 10),
        )

    local_mm = board.corner_points_mm()
    board_centre = local_mm.mean(axis=0)
    # Where the board physically sits on the plane. The fit only ever sees distances, so
    # the board's own coordinates are the mm frame -- but the *placement* has to be here
    # for the projection to land inside the region.
    on_plane_mm = local_mm - board_centre + centre

    rvec, tvec = PLANE_RVEC, PLANE_TVEC
    if lift_mm:
        # Perpendicular to the plane, not along an axis: on a plane tilted 35 degrees the
        # two differ by a third, and the gap the module reports is perpendicular.
        normal = cv2.Rodrigues(PLANE_RVEC)[0][:, 2]
        tvec = PLANE_TVEC + normal.reshape(3, 1) * (lift_mm / MM_PER_METRE)

    frame_points = project(on_plane_mm, rvec, tvec)
    camera_matrix = CAMERA_MATRIX.copy()
    if focal_scale != 1.0:
        # Focal length only. Scaling the principal point too would be a zoom -- a
        # different camera, and one a single view *can* tell apart.
        camera_matrix[0, 0] *= focal_scale
        camera_matrix[1, 1] *= focal_scale

    count = len(local_mm)
    detection = region_calibration.BoardDetection(
        frame_points=frame_points,
        crop_points=to_crop(frame_points),
        board_points_mm=local_mm * declared_scale,
        corners=frame_points.reshape(-1, 1, 2).astype(np.float32),
        ids=np.arange(count, dtype=np.int32).reshape(-1, 1),
        corner_count=count,
    )
    fit = region_calibration.fit_region(detection, board, camera_matrix, FRAME, OUTPUT)
    # The constant between plane coordinates and the board's own: the fit's millimetre
    # frame is the board's, so anything known on the plane has to cross this to be compared.
    return fit, detection, board, centre - board_centre


def main():
    checks = {}

    # The existing loader reads the calibration and extrinsics off disk, so the rig has to
    # be written down for it. TemporaryDirectory, so nothing here can reach a real camera.
    with tempfile.TemporaryDirectory() as folder:
        calibration_path = Path(folder) / "camera_calibration_1.json"
        extrinsics_path = Path(folder) / "camera_extrinsics_1.json"
        calibration_path.write_text(json.dumps(CALIBRATION), encoding="utf-8")
        extrinsics_path.write_text(json.dumps(EXTRINSICS), encoding="utf-8")
        return _verify(checks, (calibration_path, extrinsics_path))


def _verify(checks, paths):

    # -- 1. the fit is exact when the board is where it says it is ---------------------
    # 1e-4 mm is two hundred times tighter than the 0.02 mm the gates care about, and the
    # actual figure lands near 3e-6 mm -- the DLT's own conditioning, not a modelling error.
    fit, detection, board, placement = fit_a_board()
    checks["fit RMS < 1e-4 mm"] = fit.rms_mm < 1e-4
    checks["fit max < 1e-4 mm"] = fit.max_mm < 1e-4

    # -- 2. and it answers correctly for points that are not board corners -------------
    # The board's corners are what the fit was given, so reproducing them proves very
    # little. These are independent points on the same plane, expressed in the board's own
    # millimetre frame through the transform that placed it.
    corners_mm = region_corners_mm()
    probe_plane_mm = np.array([
        corners_mm.mean(axis=0),
        corners_mm[0] * 0.15 + corners_mm[2] * 0.85,
        corners_mm[1] * 0.85 + corners_mm[3] * 0.15,
        corners_mm[0] * 0.5 + corners_mm[1] * 0.5,
    ]) - placement
    probe_crop = to_crop(project(probe_plane_mm + placement))
    measured_mm = region_calibration.to_mm(fit.homography, probe_crop)
    independent_error = float(np.max(np.linalg.norm(measured_mm - probe_plane_mm, axis=1)))
    checks["independent point round-trip < 1e-4 mm"] = independent_error < 1e-4

    # -- 3. the printed geometry of the board reproduces --------------------------------
    geometry = fit.known_geometry_errors_mm()
    geometry_max = float(max(row["absolute_error_mm"] for row in geometry))
    checks["printed span error < 1e-4 mm"] = geometry_max < 1e-4

    # -- 4. and the mode agrees with the existing one -----------------------------------
    # Built through the real loader, off real files, so this is the code path the
    # application actually takes and not a re-implementation of it.
    plane = plane_scale.load_plane_scale(
        1, DEFINITION, FRAME, OUTPUT, calibration_path=paths[0], extrinsics_path=paths[1],
    )
    # Both homographies map the same crop pixels, so they are compared on the crop's own
    # corners: the plane's own construction is not the thing under test here, the
    # agreement is.
    corner_crop = np.array([[0.0, 0.0], [OUTPUT[0], 0.0],
                            [OUTPUT[0], OUTPUT[1]], [0.0, OUTPUT[1]]])
    region_mm = region_calibration.to_mm(fit.homography, corner_crop)
    existing_mm = plane.to_mm(corner_crop)
    agreement_max = float(np.max(np.abs(
        np.linalg.norm(region_mm - region_mm[0], axis=1)
        - np.linalg.norm(existing_mm - existing_mm[0], axis=1))))
    checks["agrees with the global plane within 0.2 mm"] = agreement_max < 0.2

    # -- 5. a board off the plane fits perfectly and is caught anyway -------------------
    lifted_fit, lifted_detection, lifted_board, _ = fit_a_board(lift_mm=5.0)
    lifted_alignment = region_calibration.measure_plane_alignment(
        lifted_board, lifted_detection, CAMERA_MATRIX, PLANE_RVEC, PLANE_TVEC)
    flat_alignment = region_calibration.measure_plane_alignment(
        board, detection, CAMERA_MATRIX, PLANE_RVEC, PLANE_TVEC)
    checks["a 5 mm lift still fits to < 1e-4 mm"] = lifted_fit.rms_mm < 1e-4
    checks["the lift is reported as a ~5 mm gap"] = abs(lifted_alignment.gap_mm - 5.0) < 0.01
    checks["a flat board reports a ~0 mm gap"] = abs(flat_alignment.gap_mm) < 0.001
    checks["a flat board passes the flatness gate"] = flat_alignment.within(2.0, 2.0)
    checks["a lifted board fails the flatness gate"] = not lifted_alignment.within(2.0, 2.0)

    # -- 6. a mis-declared board size fits perfectly and is caught by the cross-check ---
    wrong_fit, wrong_detection, wrong_board, _ = fit_a_board(declared_scale=1.5)
    wrong_agreement = region_calibration.measure_scale_agreement(
        wrong_fit, plane.mm_per_pixel)
    checks["a 1.5x board error still fits to < 1e-3 mm"] = wrong_fit.rms_mm < 1e-3
    checks["the 1.5x error is reported as a 1.5 scale ratio"] = (
        abs(wrong_agreement.ratio - 1.5) < 1e-3)
    checks["the 1.5x error fails the 2% scale gate"] = not wrong_agreement.within(0.02)
    checks["an honest board passes the 2% scale gate"] = (
        region_calibration.measure_scale_agreement(fit, plane.mm_per_pixel).within(0.02))

    # -- 7. one flat view sees a focal-length error, but far too weakly to refine against --
    # The measured reason the refinement is opt-in and demands tilted views. The local
    # homography is completely blind: it reproduces the board's own geometry to the same
    # ten-thousandth of a millimetre whether the focal length is right or five times wrong,
    # because a plane's pose absorbs the difference. The lens evaluation is a real sensor
    # but a feeble one, and the curve below is why "evaluate always, refine opt-in" is the
    # setting: a 15% focal error reads as half a pixel, which is inside detection noise,
    # so a refinement driven by it would be fitting noise.
    sweep = [1.0, 1.05, 1.15, 1.30, 1.50, 2.0, 3.0, 5.0]
    curve = [(scale, fit_a_board(focal_scale=scale)[0]) for scale in sweep]
    correct_fit = curve[0][1]
    distorted_fit = curve[-1][1]
    checks["a 5x focal error leaves the local fit exactly as good"] = (
        distorted_fit.rms_mm == correct_fit.rms_mm)
    checks["a 5% focal error stays under 0.25 px"] = curve[1][1].intrinsic_evaluation.rms_px < 0.25
    checks["a 15% focal error stays under the 1.0 px warning line"] = (
        curve[2][1].intrinsic_evaluation.rms_px < 1.0)
    checks["a 5x focal error is finally caught, above 1.0 px"] = (
        distorted_fit.intrinsic_evaluation.rms_px > 1.0)
    checks["the curve is monotone"] = all(
        curve[index][1].intrinsic_evaluation.rms_px
        < curve[index + 1][1].intrinsic_evaluation.rms_px
        for index in range(len(curve) - 1))
    # And the refinement refuses rather than pretending otherwise.
    refinement = region_calibration.refine_intrinsics(
        [(np.column_stack([detection.board_points_mm / MM_PER_METRE,
                           np.zeros(len(detection.board_points_mm))]),
          detection.frame_points)],
        CAMERA_MATRIX, FRAME, DISTORTION, minimum_views=5)
    checks["refinement refuses a single view"] = not refinement.applied

    print("Region homography numerical verification")
    print(f"  Region extent on the plane: "
          f"{np.linalg.norm(corners_mm[1] - corners_mm[0]):.1f} x "
          f"{np.linalg.norm(corners_mm[3] - corners_mm[0]):.1f} mm")
    print(f"  Board: {board.describe()}")
    print(f"  Fit RMS:                     {fit.rms_mm:.9f} mm")
    print(f"  Independent point error:     {independent_error:.9f} mm")
    print(f"  Printed span error:          {geometry_max:.9f} mm")
    print(f"  Agreement with plane_scale:  {agreement_max:.9f} mm")
    print(f"  Lift reported as gap:        {lifted_alignment.gap_mm:.6f} mm "
          f"(fitted to {lifted_fit.rms_mm:.9f} mm)")
    print(f"  Mis-declared size ratio:     {wrong_agreement.ratio:.6f} "
          f"(fitted to {wrong_fit.rms_mm:.9f} mm)")
    print("  Focal error vs what each measure reads:")
    print(f"    {'focal x':>8}  {'local fit (mm)':>16}  {'lens evaluation (px)':>21}")
    for scale, sweep_fit in curve:
        print(f"    {scale:>8.2f}  {sweep_fit.rms_mm:>16.9f}  "
              f"{sweep_fit.intrinsic_evaluation.rms_px:>21.6f}")
    print(f"  Refinement reason:           {refinement.reason}")
    for name, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")

    if not all(checks.values()):
        raise SystemExit(1)
    print("RESULT: PASS")


if __name__ == "__main__":
    main()

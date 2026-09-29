"""Per-crop-region local homographies, as an alternative to the global plane.

The application's original route to millimetres is one global lens calibration
(intrinsics) plus one global measurement-plane pose (extrinsics), composed in
:mod:`plane_scale` into a single homography per crop region. That model assumes the
whole bed is one plane and that the global lens model fits every part of the image
equally well -- an assumption worth avoiding near the frame edges, where the two crop
strips actually live.

This module adds a second, independent route: place a small ChArUco board flat
*inside* one crop region and fit a homography directly from the corners the camera
observes there. Intrinsics stay global; only the plane-to-pixel relationship is
measured locally, and each region gets its own. The two routes agree by construction
when the board is on the measurement plane, and that agreement is asserted in
``tests/test_region_calibration.py`` rather than assumed.

**Pixel convention: no offset.** Crop pixels are what
:func:`crop_processing.rotated_crop_transform` produces, used as they come.
``warpPerspective`` treats the integer index of a destination pixel as its continuous
coordinate, and that transform maps the source rectangle's boundary onto
``(-0.5, -0.5) .. (W-0.5, H-0.5)`` -- the raster's outer boundary. So crop array index
``i`` *is* continuous coordinate ``i``, and a point transformed through ``to_crop``
lands directly on the index of the pixel it falls in.

This was settled by measurement rather than reasoning, because the alternative is a
convincing trap. Marking a single source pixel, warping it through the real transform
for all four saved regions, and taking the intensity centroid of where it lands
reproduces ``to_crop @ point`` to within 0.06 px -- with no shift anywhere. Subtracting
a half pixel, which an earlier draft of this module did on the strength of a test that
was circular (it defined a shifted space and then confirmed the shifted fit was exact
in it), moves every mapped point by ``0.7071 x mm-per-crop-pixel``: a constant 0.117 mm
on camera 1 and 0.154 mm on camera 2. A constant translation is invisible in
repeatability, in residual reports and in strip lengths, so it would have shipped.
``CROP_PIXEL_OFFSET`` is kept at zero, and asserted, so the trap cannot be re-entered.

**The flatness gate is the check that matters most.** A board lying at an angle to the
measurement plane yields a homography for *the board's* plane. The fit is excellent,
the residuals are tiny, the save succeeds, and every subsequent measurement is wrong in
proportion to the tilt -- silently, because nothing left in the stored data can reveal
it later. :func:`measure_plane_alignment` compares the board's plane with the extrinsics
plane and is a hard gate at the point of saving.

**A wrong square size is caught against the global plane, not by the fit.** If the printed
card is declared at the wrong physical size, every corner still lines up exactly with the
coordinates the wrong profile assigns it: the fit is superb, the board's own geometry
reproduces, and every measured length is wrong by a constant factor. The intrinsics and
extrinsics give an independent millimetres-per-pixel for the same region, so
:func:`measure_scale_agreement` compares them and reports the factor directly.

**Intrinsics are evaluated, not assumed.** Every fit reports the reprojection RMS of the
board pose under the global camera matrix, so a region where the global lens model fits
badly is visible rather than silently trusted. Refinement exists but is opt-in and
deliberately hard to run by accident: one planar view carries *no* information about
intrinsics at all -- any focal-length change is exactly absorbed by a change of plane
pose -- so refinement refuses unless several views at genuinely different tilts are
supplied.

**Validity follows the lens, not the plane pose.** A stored homography maps the crop
raster to physical millimetres, and the crop raster comes from the *undistorted* frame.
Recalibrating the lens therefore invalidates it, and the signature hashes the calibration
file for exactly that reason. Recalibrating the extrinsics does *not* invalidate it --
the plane pose is no part of a local fit. That asymmetry is the point of the feature, so
it is encoded in one place rather than left to whoever edits the signature next.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from crop_processing import rotated_crop_transform
from CalibrateAPP.calibration_math import (
    estimate_planar_pose,
    reprojection_metrics,
    scale_camera_matrix,
)

# Deliberately zero; see the pixel-convention note in the module docstring. Named so
# that the tests can assert on it and the reasoning is not lost.
CROP_PIXEL_OFFSET = 0.0

# A homography needs four correspondences to exist at all. Four is where the algebra
# starts working, not where the fit stops being fragile, so this is the hard floor and
# the configured minimum (8 by default) is the operating point.
MINIMUM_CORNERS_FOR_FIT = 4

# How two-dimensional the detected corner cloud has to be. The DLT is ill-conditioned
# by a nearly collinear point set -- a board held edge-on, or one row of corners found
# -- and that failure is invisible in the residuals, because a collinear set fits its
# own line beautifully. This is the ratio of the smaller to the larger singular value of
# the mean-centred points: about 0.4 for a full 17x7 board, about 0 for a line.
MINIMUM_SPREAD_RATIO = 0.02

# Outlier gate: drop residuals beyond three sigma, but never below this floor, so that a
# genuinely exact fit does not discard corners over float noise. Everything dropped is
# counted and reported; nothing is removed silently.
OUTLIER_SIGMA = 3.0
OUTLIER_FLOOR_MM = 0.05
MAXIMUM_FIT_PASSES = 2

# Refinement is only meaningful with several views and a real spread of tilts. Five is
# the smallest set that leaves any redundancy, and 5 degrees of tilt is roughly where a
# focal-length change stops being indistinguishable from a pose change.
MINIMUM_REFINEMENT_VIEWS = 5
MINIMUM_REFINEMENT_TILT_DEGREES = 5.0


class RegionCalibrationError(RuntimeError):
    """No usable local homography for this camera and region."""


# --------------------------------------------------------------------------------------
# Board definitions
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BoardDefinition:
    """One printed ChArUco board, as the operator will actually hold it.

    Sizes are millimetres because that is how a board is specified and printed. The
    conversion to the metres OpenCV expects happens in :meth:`build_board`, in one
    place, so a stray factor of a thousand cannot hide in a caller.
    """

    name: str
    dictionary: str
    squares_x: int
    squares_y: int
    square_length_mm: float
    marker_length_mm: float

    def validate(self):
        """Raise ValueError for a definition that cannot describe a real board."""
        if not str(self.name).strip():
            raise ValueError("A board profile needs a name")
        if not hasattr(cv2.aruco, self.dictionary):
            raise ValueError(f"'{self.dictionary}' is not an OpenCV ArUco dictionary")
        for label, value in (("squares_x", self.squares_x), ("squares_y", self.squares_y)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 3:
                raise ValueError(f"Board {label} must be a whole number of at least 3")
        for label, value in (("square_length_mm", self.square_length_mm),
                             ("marker_length_mm", self.marker_length_mm)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"Board {label} must be a positive number of millimetres")
        if not 0 < self.marker_length_mm < self.square_length_mm:
            raise ValueError("Board marker size must be smaller than the square size")

        marker_count = len(self.build_board().getIds().reshape(-1))
        available = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, self.dictionary)).bytesList.shape[0]
        if marker_count > available:
            raise ValueError(
                f"A {self.squares_x}x{self.squares_y} board needs {marker_count} markers "
                f"but {self.dictionary} only has {available}; use a larger dictionary"
            )
        return self

    def build_board(self, marker_ids=None):
        """The OpenCV board, with lengths in the metres OpenCV expects."""
        return cv2.aruco.CharucoBoard(
            (int(self.squares_x), int(self.squares_y)),
            float(self.square_length_mm) / 1000.0,
            float(self.marker_length_mm) / 1000.0,
            cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.dictionary)),
            marker_ids,
        )

    def build_detector(self, marker_ids=None):
        return cv2.aruco.CharucoDetector(self.build_board(marker_ids))

    def footprint_mm(self):
        """The printed width and height of the whole board, in millimetres."""
        return (self.squares_x * float(self.square_length_mm),
                self.squares_y * float(self.square_length_mm))

    def maximum_corners(self):
        return (int(self.squares_x) - 1) * (int(self.squares_y) - 1)

    def corner_points_mm(self):
        """The board's ChArUco corners, in its own millimetre coordinates.

        Taken from OpenCV's own chessboard corner list rather than recomputed, so the
        coordinates here are in the same order as -- and provably the same points as --
        what :meth:`build_board().matchImagePoints` returns for detected corner id *i*.
        Recomputing them from the square pitch gives the same numbers today, but the
        identity that matters is the ordering, and that is OpenCV's to define.
        """
        corners = np.asarray(self.build_board().getChessboardCorners(), dtype=np.float64)
        return corners[:, :2] * 1000.0

    def describe(self):
        return (f"{self.dictionary}, {self.squares_x}x{self.squares_y}, "
                f"square {self.square_length_mm:g} mm, marker {self.marker_length_mm:g} mm")

    def as_dict(self):
        return {
            "name": self.name,
            "dictionary": self.dictionary,
            "squares_x": int(self.squares_x),
            "squares_y": int(self.squares_y),
            "square_length_mm": float(self.square_length_mm),
            "marker_length_mm": float(self.marker_length_mm),
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            name=str(data.get("name", "")),
            dictionary=str(data.get("dictionary", "")),
            squares_x=data.get("squares_x"),
            squares_y=data.get("squares_y"),
            square_length_mm=float(data.get("square_length_mm", 0.0)),
            marker_length_mm=float(data.get("marker_length_mm", 0.0)),
        )


# The region board the operator asked for, plus the board the rest of the application
# already uses. Both are only defaults: every field is editable in the UI, and profiles
# are stored on disk so a new physical board can be added without editing code.
REGION_BOARD_PROFILE = BoardDefinition(
    name="Region board 17x7",
    dictionary="DICT_5X5_100",
    squares_x=17,
    squares_y=7,
    square_length_mm=15.0,
    marker_length_mm=11.0,
)
FACTORY_BOARD_PROFILE = BoardDefinition(
    name="Factory default 7x9",
    dictionary="DICT_4X4_100",
    squares_x=7,
    squares_y=9,
    square_length_mm=28.0,
    marker_length_mm=21.0,
)


def default_board_profiles():
    return [REGION_BOARD_PROFILE, FACTORY_BOARD_PROFILE]


def empty_board_store():
    return {"version": 1, "boards": [p.as_dict() for p in default_board_profiles()]}


def load_board_profiles(path):
    """Stored profiles, falling back to the defaults when the file is absent.

    A malformed file is treated as absent rather than fatal: the defaults are always
    usable, and losing a profile is better than refusing to open the page.
    """
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError, TypeError):
        return default_board_profiles()
    boards = data.get("boards") if isinstance(data, dict) else None
    if not isinstance(boards, list):
        return default_board_profiles()
    profiles = []
    for entry in boards:
        if not isinstance(entry, dict):
            continue
        try:
            profiles.append(BoardDefinition.from_dict(entry))
        except (TypeError, ValueError):
            continue
    return profiles or default_board_profiles()


def save_board_profiles(path, profiles):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_atomically(path, {"version": 1, "boards": [p.as_dict() for p in profiles]})


def upsert_board_profile(profiles, profile):
    """Replace a profile of the same name, or append it. Returns the new list."""
    for index, existing in enumerate(profiles):
        if existing.name == profile.name:
            replaced = list(profiles)
            replaced[index] = profile
            return replaced
    return list(profiles) + [profile]


# --------------------------------------------------------------------------------------
# Does the board suit the region?
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FitReport:
    """Whether a board physically suits a region, and what to change if it does not.

    Advisory, not a gate. A board overhanging the region still yields plenty of usable
    interior corners and a perfectly good homography -- it is the *corners* that have to
    span the region, not the printed card -- so refusing on footprint alone would reject
    valid setups. What the operator needs is the arithmetic and a concrete suggestion.
    """

    fits: bool
    used_fraction: tuple
    suggestions: tuple

    def summary(self):
        if not np.isfinite(self.used_fraction[0]):
            return "Region size is not known until a board has been captured."
        lines = [f"board spans {self.used_fraction[0] * 100:.0f}% of the region width "
                 f"and {self.used_fraction[1] * 100:.0f}% of its height"]
        lines.extend(self.suggestions)
        return "; ".join(lines)


def board_fit_report(board, region_mm, minimum_squares=5):
    """Compare a board's printed footprint against the region it will sit in.

    ``region_mm`` is ``(width, height)`` on the plane, or ``None`` when the extent is not
    yet known -- it can be computed from the saved extrinsics, but only when those exist.
    """
    region_width, region_height = ((float(v) for v in region_mm) if region_mm is not None
                                   else (float("nan"), float("nan")))
    if not (np.isfinite(region_width) and np.isfinite(region_height)
            and region_width > 0 and region_height > 0):
        return FitReport(fits=True, used_fraction=(float("nan"), float("nan")),
                         suggestions=())

    width_mm, height_mm = board.footprint_mm()
    used = (width_mm / region_width, height_mm / region_height)
    fits = used[0] <= 1.0 and used[1] <= 1.0
    suggestions = []
    if not fits:
        suggestions.append(
            f"{board.squares_x}x{board.squares_y} at {board.square_length_mm:g} mm is "
            f"{width_mm:g} x {height_mm:g} mm and this region is "
            f"{region_width:.0f} x {region_height:.0f} mm"
        )
        # Keeping the printed pitch and dropping rows is usually the cheapest fix: the
        # same card can be cut down, and the marker size is unchanged.
        rows = int(region_height * 0.90 / float(board.square_length_mm))
        if rows >= 3:
            suggestions.append(
                f"cut the card to {board.squares_x}x{rows} at "
                f"{board.square_length_mm:g} mm ({width_mm:g} x "
                f"{rows * board.square_length_mm:g} mm)"
            )
        smaller = largest_fitting_square_mm(
            board.squares_x, board.squares_y, region_mm,
            board.marker_length_mm / float(board.square_length_mm))
        if smaller is not None:
            suggestions.append(
                f"or print {board.squares_x}x{board.squares_y} at {smaller[0]:g} mm "
                f"squares with {smaller[1]:g} mm markers"
            )
    if board.squares_y < int(minimum_squares):
        suggestions.append(
            f"{board.squares_y} rows is a thin card; {minimum_squares} or more gives a "
            "better-conditioned fit"
        )
    return FitReport(fits=fits, used_fraction=used, suggestions=tuple(suggestions))


def largest_fitting_square_mm(squares_x, squares_y, region_mm, marker_ratio=0.73):
    """The largest whole-tenth-millimetre square size whose board fits the region.

    Offered when the configured board is too big, so the operator can fix it in one
    action instead of guessing sizes until the fit stops complaining. The marker is
    scaled with the square at the ratio the profile already uses. Ten per cent of the
    region is reserved as placement margin: a card flush to the marked edge is not a
    realistic thing to lay down, and its outermost corners are the least trustworthy.
    """
    if region_mm is None:
        return None
    region_width, region_height = (float(v) for v in region_mm)
    if squares_x <= 0 or squares_y <= 0 or region_width <= 0 or region_height <= 0:
        return None
    square = min(0.90 * region_width / squares_x, 0.90 * region_height / squares_y)
    square = float(np.floor(square * 10.0) / 10.0)
    if square <= 0:
        return None
    return square, float(np.floor(square * marker_ratio * 10.0) / 10.0)


# --------------------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BoardDetection:
    """One board found in one frame, in both frame and crop pixel space."""

    frame_points: np.ndarray      # (N, 2) undistorted-frame pixels
    crop_points: np.ndarray       # (N, 2) crop pixels, exactly as to_crop gives them
    board_points_mm: np.ndarray   # (N, 2) the board's own metric coordinates
    corners: np.ndarray           # raw detected corners, for drawing
    ids: np.ndarray               # raw detected ids, for drawing
    corner_count: int
    outside_region: int = 0       # corners landing outside the crop raster

    def spread_ratio(self):
        return _spread_ratio(self.crop_points)


def _spread_ratio(points):
    """Smaller over larger singular value of the mean-centred points."""
    centered = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    centered = centered - centered.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    if singular[0] <= 0:
        return 0.0
    return float(singular[1] / singular[0])


def detect_board_in_region(gray, board_definition, crop_definition, frame_size,
                           output_size=(2208, 552), ratio=4.0,
                           minimum_corners=MINIMUM_CORNERS_FOR_FIT):
    """Detect the board in an undistorted frame and express it in crop coordinates.

    The frame must already be undistorted: this is the same frame
    :func:`crop_processing.extract_rotated_crop` deskews, so the crop coordinates
    derived here are in the space the crop array actually occupies.

    Corners outside the crop raster are kept, not discarded -- they are still true
    observations of the board, and they condition the fit better, since a homography is
    global -- but they are counted, because a card mostly outside the region is
    measuring somewhere the operator is not going to inspect.

    Raises :class:`RegionCalibrationError` with an operator-readable reason when the
    board cannot be used at all.
    """
    corners, ids, _, _ = board_definition.build_detector().detectBoard(gray)
    corner_count = 0 if ids is None else int(len(ids))
    if corner_count < int(minimum_corners):
        raise RegionCalibrationError(
            f"Only {corner_count} of the {minimum_corners} board points needed were "
            "found. Lay the card flat in this region and try again."
        )

    object_points, image_points = board_definition.build_board().matchImagePoints(corners, ids)
    if object_points is None or image_points is None or len(object_points) < int(minimum_corners):
        matched = 0 if image_points is None else int(len(image_points))
        raise RegionCalibrationError(
            f"Only {matched} of the detected points belong to the configured board. "
            "Check the dictionary and the square and marker sizes."
        )

    frame_points = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    to_crop = rotated_crop_transform(crop_definition, frame_size, output_size, ratio)
    crop_points = cv2.perspectiveTransform(
        frame_points.reshape(-1, 1, 2), to_crop).reshape(-1, 2) - CROP_PIXEL_OFFSET

    output_width, output_height = map(int, output_size)
    inside = ((crop_points[:, 0] >= 0.0) & (crop_points[:, 0] <= output_width)
              & (crop_points[:, 1] >= 0.0) & (crop_points[:, 1] <= output_height))

    return BoardDetection(
        frame_points=frame_points,
        crop_points=crop_points,
        # matchImagePoints returns the board's own coordinates in metres, because the
        # board was built in metres. Millimetres is how the rest of this module thinks.
        # Indexed rather than reshaped to (-1, 2): the return is (N, 1, 3) -- these are
        # 3D board points with z = 0 -- and reshaping that to two columns would quietly
        # produce one and a half times as many pairs as there are corners.
        board_points_mm=np.asarray(
            object_points, dtype=np.float64).reshape(-1, 3)[:, :2] * 1000.0,
        corners=corners,
        ids=ids,
        corner_count=corner_count,
        outside_region=int((~inside).sum()),
    )


# --------------------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class IntrinsicEvaluation:
    """How well the global lens model fits this board as seen in this region."""

    rms_px: float
    mean_px: float
    max_px: float
    corners: int


@dataclass(frozen=True)
class FitStats:
    """What the fit cost: how many corners it used and how many it set aside."""

    corners: int
    dropped: int
    spread_ratio: float

    @property
    def inliers(self):
        return self.corners - self.dropped


@dataclass(frozen=True)
class PlaneAlignment:
    """How far the region board's own plane sits from the measurement plane."""

    tilt_degrees: float
    gap_mm: float
    expected_gap_mm: float

    @property
    def gap_error_mm(self):
        return abs(self.gap_mm - self.expected_gap_mm)

    def within(self, maximum_tilt_degrees, maximum_gap_mm):
        return (abs(self.tilt_degrees) <= float(maximum_tilt_degrees)
                and self.gap_error_mm <= float(maximum_gap_mm))

    def describe(self):
        return (f"tilt {self.tilt_degrees:.2f} deg, gap {self.gap_mm:+.2f} mm "
                f"(expected {self.expected_gap_mm:+.2f})")


@dataclass(frozen=True)
class ScaleAgreement:
    """The local fit's millimetres per pixel against the global plane's.

    This is the only check that can catch a board declared at the wrong physical size,
    and that failure is otherwise perfectly silent: if the printed card is 10 mm squares
    but the profile says 15, every detected corner still lines up exactly with the
    coordinates the wrong profile assigns it -- the homography simply absorbs the factor,
    fits with residuals near zero, and reproduces the board's own geometry to a
    ten-thousandth of a millimetre. The stored data then reports a length 50% too long
    and nothing in it looks wrong. Since the intrinsics and extrinsics give an
    independent millimetres-per-pixel for the same region, comparing the two is a direct
    measurement of that factor.
    """

    local_mm_per_pixel: float
    reference_mm_per_pixel: float

    @property
    def ratio(self):
        return self.local_mm_per_pixel / self.reference_mm_per_pixel

    def relative_error(self):
        return abs(self.ratio - 1.0)

    def within(self, tolerance):
        """Whether the two agree to within a fractional tolerance, e.g. 0.02 for 2%."""
        return self.relative_error() <= float(tolerance)

    def describe(self):
        return (f"local {self.local_mm_per_pixel:.5f} mm/px vs plane "
                f"{self.reference_mm_per_pixel:.5f} mm/px "
                f"({(self.ratio - 1.0) * 100:+.2f}%)")


def measure_scale_agreement(fit, reference_mm_per_pixel):
    """Compare a fit's scale with an independent one -- the global plane, in practice.

    Returns ``None`` when the reference is not a usable number, so a caller without a
    global plane to compare against gets no opinion rather than a false alarm.
    """
    reference = float(reference_mm_per_pixel)
    if not np.isfinite(reference) or reference <= 0:
        return None
    return ScaleAgreement(local_mm_per_pixel=float(fit.mm_per_pixel),
                          reference_mm_per_pixel=reference)


@dataclass(frozen=True)
class RegionFit:
    """A fitted local homography and everything needed to judge it."""

    homography: np.ndarray                 # 3x3, crop pixels -> millimetres
    mm_per_pixel: float                    # at the region centre; a summary, not a constant
    rms_mm: float
    mean_mm: float
    max_mm: float
    p95_mm: float
    stats: FitStats
    board_definition: BoardDefinition
    intrinsic_evaluation: IntrinsicEvaluation
    image_size: tuple
    output_size: tuple
    _crop_points: np.ndarray = field(default=None, repr=False)
    _board_points_mm: np.ndarray = field(default=None, repr=False)

    @property
    def corners(self):
        return self.stats.corners

    @property
    def inliers(self):
        return self.stats.inliers

    @property
    def dropped(self):
        return self.stats.dropped

    def known_geometry_errors_mm(self):
        """Per-corner-pair distance error, measured through the fit.

        This is the check that matters most for grading a fit: it compares distances
        between the board's own detected points, mapped to millimetres, against the
        distances the printed board actually has. A homography that is wrong in scale,
        rotation or anisotropy cannot reproduce them.
        """
        return pairwise_distance_errors_mm(
            self.homography, self._crop_points, self._board_points_mm)


def fit_homography(crop_points, board_points_mm):
    """Fit crop pixels -> millimetres by least squares, with a reported outlier gate.

    Every detected ChArUco corner is an inlier by construction -- the ids give exact
    correspondence, so a "gross outlier" can only be a small detection error, never a
    mismatched point. RANSAC would therefore buy nothing while adding a random minimal
    sample that can land on a degenerate quadruple of a lattice-like board, so the fit is
    the plain normalised DLT. Robustness comes from one residual pass instead: fit,
    discard beyond three sigma, refit, and *report* how many were discarded.

    Refuses input too sparse to constrain a homography, too collinear to constrain it
    well, mirrored, or non-finite. The mirror case matters because a flipped board frame
    fits beautifully and leaves small residuals; only the determinant's sign gives it
    away, and :func:`strip_analysis.to_metric` takes norms of differences and would
    never notice.
    """
    source = np.asarray(crop_points, dtype=np.float64).reshape(-1, 2)
    target = np.asarray(board_points_mm, dtype=np.float64).reshape(-1, 2)
    if len(source) != len(target):
        raise RegionCalibrationError("There must be one board point for each crop point")
    if len(source) < MINIMUM_CORNERS_FOR_FIT:
        raise RegionCalibrationError(
            f"At least {MINIMUM_CORNERS_FOR_FIT} matched board points are required; "
            f"{len(source)} were found"
        )

    spread = _spread_ratio(source)
    if spread < MINIMUM_SPREAD_RATIO:
        raise RegionCalibrationError(
            "The detected board points are nearly in a line, which cannot pin down a "
            "plane. Lay the card flat and square to the region rather than edge-on."
        )

    homography, _ = cv2.findHomography(source, target, 0)
    if homography is None:
        raise RegionCalibrationError("The board points could not be fitted at all")
    homography = homography.reshape(3, 3)

    dropped = 0
    for _ in range(MAXIMUM_FIT_PASSES - 1):
        residuals = np.linalg.norm(to_mm(homography, source) - target, axis=1)
        sigma = float(np.sqrt(np.mean(residuals ** 2)))
        keep = residuals <= max(OUTLIER_SIGMA * sigma, OUTLIER_FLOOR_MM)
        if keep.all() or int(keep.sum()) < MINIMUM_CORNERS_FOR_FIT:
            break
        refit, _ = cv2.findHomography(source[keep], target[keep], 0)
        if refit is None:
            break
        homography = refit.reshape(3, 3)
        dropped = int((~keep).sum())

    if not np.isfinite(homography).all():
        raise RegionCalibrationError("The fitted homography is not a usable matrix")
    if determinant_sign(homography) <= 0:
        raise RegionCalibrationError(
            "The fitted homography is mirrored, which means the board was matched back "
            "to front. Check the board definition rather than the camera."
        )
    return homography, FitStats(corners=len(source), dropped=dropped, spread_ratio=spread)


def determinant_sign(homography):
    """Sign of the homography's determinant, normalised so the scale is not a factor."""
    matrix = np.asarray(homography, dtype=np.float64).reshape(3, 3)
    if abs(matrix[2, 2]) < 1e-12:
        return 0.0
    return float(np.linalg.det(matrix / matrix[2, 2]))


def pairwise_distance_errors_mm(homography, crop_points, board_points_mm):
    """Error in every measured corner-pair distance, in millimetres.

    Only distances are compared, never absolute positions, so an arbitrary mm origin is
    harmless -- the same reasoning :func:`strip_analysis.to_metric` relies on, and the
    reason two regions can report millimetres in two different board frames.

    Quadratic in the corner count, which is nothing for one board's 96 corners, and is
    called from the check page rather than from the inspection path.
    """
    measured = to_mm(homography, crop_points)
    expected = np.asarray(board_points_mm, dtype=np.float64).reshape(-1, 2)
    rows = []
    for first in range(len(expected)):
        for second in range(first + 1, len(expected)):
            expected_distance = float(np.linalg.norm(expected[second] - expected[first]))
            measured_distance = float(np.linalg.norm(measured[second] - measured[first]))
            rows.append({
                "first": first,
                "second": second,
                "expected_mm": expected_distance,
                "measured_mm": measured_distance,
                "error_mm": measured_distance - expected_distance,
                "absolute_error_mm": abs(measured_distance - expected_distance),
            })
    return rows


def to_mm(homography, points):
    """Map crop pixels to millimetres on the region's plane."""
    array = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(
        array, np.asarray(homography, dtype=np.float64).reshape(3, 3)).reshape(-1, 2)


def residual_metrics_mm(homography, crop_points, board_points_mm):
    """Forward residuals of the fit, in millimetres.

    The fit direction is the use direction, so this is the error a measurement will
    actually see. Fitting ``mm -> crop`` instead would minimise the wrong quantity.
    """
    errors = np.linalg.norm(
        to_mm(homography, crop_points)
        - np.asarray(board_points_mm, dtype=np.float64).reshape(-1, 2), axis=1)
    return {
        "rms_mm": float(np.sqrt(np.mean(errors ** 2))),
        "mean_mm": float(np.mean(errors)),
        "max_mm": float(np.max(errors)),
        "p95_mm": float(np.percentile(errors, 95)),
        "corners": int(len(errors)),
    }


def _mm_per_pixel(homography, output_size):
    """Millimetres per crop pixel at the region centre, as a reporting summary."""
    width, height = map(int, output_size)
    probe = np.array([[width / 2.0, height / 2.0],
                      [width / 2.0 + 1.0, height / 2.0]], dtype=np.float64)
    stepped = to_mm(homography, probe)
    return float(np.linalg.norm(stepped[1] - stepped[0]))


def evaluate_intrinsics(detection, board_definition, camera_matrix):
    """Reprojection RMS of this board's pose under the global camera matrix.

    Distortion is deliberately zero: the frame handed in is already undistorted, and the
    undistorter uses the camera matrix itself as the new camera matrix, so the
    undistorted image's projection *is* that matrix with no distortion. Passing the
    stored coefficients here would count the lens twice.
    """
    try:
        _, _, metrics, _, image_points = estimate_planar_pose(
            board_definition.build_board(), detection.corners, detection.ids,
            camera_matrix, np.zeros((5, 1), dtype=np.float64))
    except (ValueError, RuntimeError) as error:
        raise RegionCalibrationError(f"The lens check did not run: {error}") from error
    return IntrinsicEvaluation(
        rms_px=metrics["rms"],
        mean_px=metrics["mean"],
        max_px=metrics["max"],
        corners=int(len(image_points)),
    )


def fit_region(detection, board_definition, camera_matrix, image_size,
               output_size=(2208, 552)):
    """Fit the local homography for one region and grade it."""
    homography, stats = fit_homography(detection.crop_points, detection.board_points_mm)
    metrics = residual_metrics_mm(homography, detection.crop_points,
                                 detection.board_points_mm)
    return RegionFit(
        homography=homography,
        mm_per_pixel=_mm_per_pixel(homography, output_size),
        rms_mm=metrics["rms_mm"],
        mean_mm=metrics["mean_mm"],
        max_mm=metrics["max_mm"],
        p95_mm=metrics["p95_mm"],
        stats=stats,
        board_definition=board_definition,
        intrinsic_evaluation=evaluate_intrinsics(detection, board_definition, camera_matrix),
        image_size=tuple(int(v) for v in image_size),
        output_size=tuple(int(v) for v in output_size),
        _crop_points=np.asarray(detection.crop_points, dtype=np.float64),
        _board_points_mm=np.asarray(detection.board_points_mm, dtype=np.float64),
    )


def measure_plane_alignment(board_definition, detection, camera_matrix, plane_rvec,
                            plane_tvec, board_thickness_mm=0.0):
    """How far the region board's plane is from the measurement plane.

    The single most valuable check in this module. A board lying at an angle to the
    measurement plane yields a homography for *the board's* plane: an excellent fit with
    tiny residuals, and a wrong answer for everything measured afterwards. Nothing in the
    stored homography can reveal that later, so it has to be caught here -- at the point
    of saving, where the operator can still do something about it.

    ``plane_rvec`` and ``plane_tvec`` come from the saved extrinsics and are in metres,
    as is the board pose, so the two are directly comparable.

    The gap is the perpendicular distance from the board's printed face to the
    measurement plane. Its expected value is the card's own thickness when the extrinsics
    were taken on the same stock, and zero when they were taken on the bare plane. That
    ambiguity is why the caller supplies the expectation rather than this function
    assuming one.
    """
    try:
        board_rvec, board_tvec, _, _, _ = estimate_planar_pose(
            board_definition.build_board(), detection.corners, detection.ids,
            camera_matrix, np.zeros((5, 1), dtype=np.float64))
    except (ValueError, RuntimeError) as error:
        raise RegionCalibrationError(f"The plane check did not run: {error}") from error

    board_normal = cv2.Rodrigues(board_rvec)[0][:, 2]
    plane_normal = cv2.Rodrigues(
        np.asarray(plane_rvec, dtype=np.float64).reshape(3, 1))[0][:, 2]
    tilt_degrees = float(np.degrees(np.arccos(
        float(np.clip(abs(float(np.dot(board_normal, plane_normal))), -1.0, 1.0)))))

    # The board's origin is a point on the board plane; project it onto the measurement
    # plane's normal to get the perpendicular separation.
    difference = (np.asarray(board_tvec, dtype=np.float64).reshape(3)
                  - np.asarray(plane_tvec, dtype=np.float64).reshape(3))
    return PlaneAlignment(
        tilt_degrees=tilt_degrees,
        gap_mm=float(abs(float(np.dot(plane_normal, difference))) * 1000.0),
        expected_gap_mm=float(board_thickness_mm),
    )


# --------------------------------------------------------------------------------------
# Optional intrinsic refinement
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Refinement:
    """The outcome of an opt-in local intrinsic refinement."""

    applied: bool
    reason: str
    views: int
    maximum_tilt_degrees: float
    rms_before_px: float
    rms_after_px: float
    camera_matrix: np.ndarray | None = None
    dist_coeffs: np.ndarray | None = None

    def improved(self):
        return self.applied and self.rms_after_px < self.rms_before_px


def _view_tilt_degrees(rvec):
    """How far a view's board normal is tilted away from the camera axis."""
    normal = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))[0][:, 2]
    return float(np.degrees(np.arccos(float(np.clip(abs(normal[2]), -1.0, 1.0)))))


def refine_intrinsics(views, camera_matrix, image_size, dist_coeffs=None,
                      minimum_views=MINIMUM_REFINEMENT_VIEWS,
                      minimum_tilt_degrees=MINIMUM_REFINEMENT_TILT_DEGREES):
    """Refine the global intrinsics from several *tilted* views of one region.

    ``views`` is a sequence of ``(object_points_m, image_points_px)`` pairs, both in the
    undistorted frame's pixel space.

    This refuses far more often than it runs, on purpose. A single planar view carries no
    information about intrinsics whatsoever -- any focal-length change is exactly absorbed
    by a change of plane pose -- and several views at the *same* tilt are nearly as
    uninformative. Requiring a real spread of tilts is what separates a refinement from a
    rewrite of the lens model based on nothing.

    The result is returned, never applied: the caller decides, and the global calibration
    file is never touched.
    """
    pairs = [(np.asarray(obj, dtype=np.float32).reshape(-1, 3),
              np.asarray(img, dtype=np.float32).reshape(-1, 2)) for obj, img in views]
    if len(pairs) < int(minimum_views):
        return Refinement(
            applied=False,
            reason=(f"Refining needs at least {minimum_views} views; {len(pairs)} were "
                    "captured. One view cannot constrain the lens at all."),
            views=len(pairs), maximum_tilt_degrees=0.0,
            rms_before_px=float("nan"), rms_after_px=float("nan"),
        )

    tilts = []
    for object_points, image_points in pairs:
        success, rvec, _ = cv2.solvePnP(
            object_points, image_points, np.asarray(camera_matrix, dtype=np.float64),
            np.zeros((5, 1)), flags=cv2.SOLVEPNP_IPPE)
        if success:
            tilts.append(_view_tilt_degrees(rvec))
    maximum_tilt = float(max(tilts)) if tilts else 0.0
    if maximum_tilt < float(minimum_tilt_degrees):
        return Refinement(
            applied=False,
            reason=(f"Every view is within {maximum_tilt:.1f} degrees of flat; "
                    f"{minimum_tilt_degrees:.0f} degrees of tilt is needed before the "
                    "lens can be told apart from the pose."),
            views=len(pairs), maximum_tilt_degrees=maximum_tilt,
            rms_before_px=float("nan"), rms_after_px=float("nan"),
        )

    guess = np.asarray(camera_matrix, dtype=np.float64).copy()
    coefficients = np.resize(
        np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
        if dist_coeffs is not None else np.zeros(5), 5).astype(np.float64)

    before = _reprojection_rms(pairs, guess, coefficients)
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO
             | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3)
    try:
        _, refined_matrix, refined_coefficients, _, _ = cv2.calibrateCamera(
            [p[0] for p in pairs], [p[1] for p in pairs], tuple(map(int, image_size)),
            guess.copy(), coefficients.copy(), flags=flags)
    except cv2.error as error:
        return Refinement(
            applied=False, reason=f"The refinement did not converge: {error}",
            views=len(pairs), maximum_tilt_degrees=maximum_tilt,
            rms_before_px=before, rms_after_px=float("nan"))

    return Refinement(
        applied=True,
        reason="Refined from tilted views; the global calibration was not modified.",
        views=len(pairs),
        maximum_tilt_degrees=maximum_tilt,
        rms_before_px=before,
        rms_after_px=_reprojection_rms(pairs, refined_matrix, refined_coefficients),
        camera_matrix=refined_matrix,
        dist_coeffs=np.asarray(refined_coefficients, dtype=np.float64).reshape(-1, 1),
    )


def _reprojection_rms(pairs, camera_matrix, dist_coeffs):
    errors = []
    for object_points, image_points in pairs:
        success, rvec, tvec = cv2.solvePnP(
            object_points, image_points, np.asarray(camera_matrix, dtype=np.float64),
            np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1),
            flags=cv2.SOLVEPNP_IPPE)
        if not success:
            continue
        errors.append(reprojection_metrics(
            object_points, image_points, rvec, tvec, camera_matrix, dist_coeffs)["rms"])
    return float(np.mean(errors)) if errors else float("nan")


# --------------------------------------------------------------------------------------
# Region footprint
# --------------------------------------------------------------------------------------


def region_size_mm(crop_definition, frame_size, output_size=(2208, 552), ratio=4.0,
                   calibration=None, extrinsics=None):
    """The region's extent on the measurement plane, in millimetres, or ``None``.

    Uses the saved global plane when there is one. Without it the extent cannot be known
    before a capture, so the caller falls back to letting the fit measure it -- which it
    does anyway, because a fitted homography turns the crop's pixel size straight into
    millimetres.
    """
    if calibration is None or extrinsics is None:
        return None
    try:
        from CalibrateAPP.calibration_math import pixel_to_plane

        camera_matrix = scale_camera_matrix(
            np.asarray(calibration["camera_matrix"], dtype=np.float64),
            calibration["image_size"], frame_size)
        rvec = np.asarray(extrinsics["rvec"], dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(extrinsics["tvec"], dtype=np.float64).reshape(3, 1)

        width, height = map(int, output_size)
        corners = np.array([[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]],
                           dtype=np.float64)
        to_crop = rotated_crop_transform(crop_definition, frame_size, output_size, ratio)
        source = cv2.perspectiveTransform(
            corners.reshape(-1, 1, 2), np.linalg.inv(to_crop)).reshape(-1, 2)
        on_plane = np.array([
            pixel_to_plane(point, camera_matrix, rvec, tvec) * 1000.0 for point in source
        ], dtype=np.float64)
    except (KeyError, ValueError, TypeError, np.linalg.LinAlgError, cv2.error):
        return None
    if not np.isfinite(on_plane).all():
        return None
    return (float(np.linalg.norm(on_plane[1] - on_plane[0])),
            float(np.linalg.norm(on_plane[3] - on_plane[0])))


# --------------------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------------------


def empty_region_store():
    return {"version": 1, "cameras": {"1": {"1": None, "2": None}, "2": {"1": None, "2": None}}}


def region_key(region_index):
    """Region 1 and 2, as the store spells them."""
    try:
        number = int(region_index)
    except (TypeError, ValueError) as error:
        raise RegionCalibrationError(f"'{region_index}' is not a region number") from error
    if number not in (1, 2):
        raise RegionCalibrationError(f"There is no region {region_index}")
    return str(number)


def _file_digest(path):
    """sha256 of a file's bytes, or a sentinel when it cannot be read."""
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except (OSError, TypeError, ValueError):
        return "unreadable"


def crop_signature(crop_definition, frame_size, output_size=(2208, 552), ratio=4.0,
                   calibration_path=None):
    """A fingerprint of everything a stored homography's validity depends on.

    Included, and why:

    * the crop definition, the frame size, the output size and the aspect ratio. The
      homography maps the crop raster, so anything that changes which pixels the crop
      holds invalidates it. The ratio matters even though it is not in the definition,
      because ``pixel_definition`` derives the crop height from the *runtime* ratio while
      ``rotated_crop_transform`` takes it as an argument: editing
      ``crop_setup.aspect_ratio`` reshapes every crop without touching
      ``crop_regions.json`` at all. The raw stored definition is hashed rather than the
      editor's rebuilt one, because the raw one is what the runtime path reads.
    * a digest of the lens calibration, because the crop comes from the *undistorted*
      frame, so a lens recalibration moves every crop pixel. Without this, a stored
      homography would survive the one event that certainly invalidates it.
    * but *not* the extrinsics. The plane pose is no part of a local fit, so
      recalibrating it must leave these homographies alone. That asymmetry is the point of
      the feature, and the reason this is a signature over a specific list rather than a
      hash of everything relevant-looking.
    """
    body = {
        "definition": crop_definition,
        "frame_size": [int(v) for v in frame_size],
        "output_size": [int(v) for v in output_size],
        "ratio": float(ratio),
        "intrinsics": _file_digest(calibration_path) if calibration_path else "unspecified",
    }
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_region_store(path):
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError, TypeError):
        return empty_region_store()
    cameras = data.get("cameras") if isinstance(data, dict) else None
    if not isinstance(cameras, dict):
        return empty_region_store()
    result = empty_region_store()
    for camera in ("1", "2"):
        entries = cameras.get(camera)
        if not isinstance(entries, dict):
            continue
        for region in ("1", "2"):
            entry = entries.get(region)
            if isinstance(entry, dict):
                result["cameras"][camera][region] = entry
    return result


def save_region_store(path, store):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_atomically(path, store)


def _write_atomically(path, body):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def store_region_fit(store, camera_number, region_index, fit, signature, crop_definition,
                     alignment=None, refinement=None, agreement=None):
    """Put a fit into the store under its camera and region. Returns the store.

    The board geometry is copied in whole rather than referenced by profile name: editing
    a profile later must not silently reinterpret a homography that was fitted to a
    different physical board.
    """
    camera = str(int(camera_number))
    if camera not in store["cameras"]:
        raise RegionCalibrationError(f"There is no camera {camera_number} in the store")
    store["cameras"][camera][region_key(region_index)] = {
        "homography": np.asarray(fit.homography, dtype=np.float64).tolist(),
        "mm_per_pixel": float(fit.mm_per_pixel),
        "board": fit.board_definition.as_dict(),
        "metrics": {
            "rms_mm": float(fit.rms_mm),
            "mean_mm": float(fit.mean_mm),
            "max_mm": float(fit.max_mm),
            "p95_mm": float(fit.p95_mm),
            "corners": int(fit.corners),
            "inliers": int(fit.inliers),
            "dropped": int(fit.dropped),
            "spread_ratio": float(fit.stats.spread_ratio),
        },
        "intrinsic_evaluation": {
            "rms_px": float(fit.intrinsic_evaluation.rms_px),
            "mean_px": float(fit.intrinsic_evaluation.mean_px),
            "max_px": float(fit.intrinsic_evaluation.max_px),
            "corners": int(fit.intrinsic_evaluation.corners),
        },
        "plane_alignment": ({
            "tilt_degrees": float(alignment.tilt_degrees),
            "gap_mm": float(alignment.gap_mm),
            "expected_gap_mm": float(alignment.expected_gap_mm),
        } if alignment is not None else None),
        "scale_agreement": ({
            "local_mm_per_pixel": float(agreement.local_mm_per_pixel),
            "reference_mm_per_pixel": float(agreement.reference_mm_per_pixel),
            "ratio": float(agreement.ratio),
        } if agreement is not None else None),
        "intrinsic_refinement": ({
            "views": int(refinement.views),
            "maximum_tilt_degrees": float(refinement.maximum_tilt_degrees),
            "rms_before_px": float(refinement.rms_before_px),
            "rms_after_px": float(refinement.rms_after_px),
            "camera_matrix": np.asarray(refinement.camera_matrix, np.float64).tolist(),
            "dist_coeffs": np.asarray(
                refinement.dist_coeffs, np.float64).reshape(-1).tolist(),
        } if refinement is not None and refinement.applied else None),
        "signature": signature,
        "definition": crop_definition,
        "image_size": [int(v) for v in fit.image_size],
        "output_size": [int(v) for v in fit.output_size],
        "captured_at": datetime.now().isoformat(timespec="seconds"),
    }
    return store


def stored_region_entry(store, camera_number, region_index):
    camera = str(int(camera_number))
    entries = store.get("cameras", {}).get(camera)
    if not isinstance(entries, dict):
        return None
    entry = entries.get(region_key(region_index))
    return entry if isinstance(entry, dict) else None


# --------------------------------------------------------------------------------------
# Runtime consumption
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RegionScale:
    """Maps crop pixels onto one region's measured plane, in millimetres.

    Deliberately the same shape as :class:`plane_scale.PlaneScale` -- ``to_mm`` plus
    ``mm_per_pixel`` -- because the runtime hands either object to
    :func:`strip_analysis.to_metric` without knowing which mode produced it. That pair of
    members is the entire runtime contract; nothing elsewhere inspects the type.

    One consequence worth stating, because nothing compares positions across regions
    today and something eventually will: each region's millimetres are in *its own* board
    frame, with a different origin and rotation. Lengths are therefore comparable between
    regions and absolute coordinates are not.
    """

    homography: np.ndarray
    mm_per_pixel: float
    region: int
    board_name: str
    signature: str
    rms_mm: float = float("nan")
    max_mm: float = float("nan")
    corners: int = 0
    tilt_degrees: float = float("nan")
    gap_mm: float = float("nan")
    scale_ratio: float = float("nan")

    def to_mm(self, points):
        return to_mm(self.homography, points)


def camera_paths(camera_number):
    """The calibration and extrinsics files config names for a camera.

    Cameras are numbered 1 and 2 by position, matching how the rest of the app pairs them
    (``CALIB_FILE_1, CALIB_FILE_2 = [c['calibration_file'] for c in CONFIG['cameras']]``).
    The ``index`` field is the device's capture index, not the side's number, so matching
    on it would hand camera 1 the other camera's calibration.
    """
    from app_config import CONFIG, project_path

    cameras = CONFIG.get("cameras") or []
    try:
        number = int(camera_number)
    except (TypeError, ValueError):
        return None, None
    # Checked rather than indexed-and-caught: camera 0 would otherwise be cameras[-1].
    if not 1 <= number <= len(cameras):
        return None, None
    camera = cameras[number - 1]
    return (project_path(camera.get("calibration_file", "")),
            project_path(camera.get("extrinsics_file", "")))


def load_region_scale(camera_number, region_index, crop_definition, frame_size,
                      output_size=(2208, 552), ratio=4.0, store_path=None,
                      calibration_path=None):
    """Build the runtime scale for one region, or explain why it cannot.

    Every failure raises :class:`RegionCalibrationError` naming the specific reason, so
    the caller reports "region 2 has not been calibrated" or "this region was re-marked
    since" rather than quietly falling back to pixels with no explanation.
    """
    from app_config import CONFIG, project_path

    if store_path is None:
        store_path = project_path(CONFIG.get("region_homography", {}).get(
            "store_file", "Files/region_homographies.json"))
    if calibration_path is None:
        calibration_path = camera_paths(camera_number)[0]

    entry = stored_region_entry(load_region_store(store_path), camera_number, region_index)
    if entry is None:
        raise RegionCalibrationError(
            f"Region {region_index} of camera {camera_number} has no saved local "
            "homography; calibrate that region first")

    stored_output = entry.get("output_size")
    if [int(v) for v in stored_output or []] != [int(v) for v in output_size]:
        raise RegionCalibrationError(
            f"Region {region_index} was calibrated at crop size {stored_output} but the "
            f"crop is now {list(output_size)}; recalibrate it")

    signature = crop_signature(crop_definition, frame_size, output_size, ratio,
                               calibration_path)
    if entry.get("signature") != signature:
        raise RegionCalibrationError(
            f"Region {region_index} was calibrated against a different crop region or a "
            "different lens calibration; re-mark or recalibrate that region")

    try:
        homography = np.asarray(entry["homography"], dtype=np.float64).reshape(3, 3)
    except (KeyError, TypeError, ValueError) as error:
        raise RegionCalibrationError(
            f"The saved homography for region {region_index} is unreadable") from error
    if not np.isfinite(homography).all() or determinant_sign(homography) <= 0:
        raise RegionCalibrationError(
            f"The saved homography for region {region_index} is not a usable matrix")

    metrics = entry.get("metrics") or {}
    alignment = entry.get("plane_alignment") or {}
    agreement = entry.get("scale_agreement") or {}
    return RegionScale(
        homography=homography,
        mm_per_pixel=float(entry.get("mm_per_pixel", float("nan"))),
        region=int(region_index),
        board_name=str(entry.get("board", {}).get("name", "")),
        signature=signature,
        rms_mm=float(metrics.get("rms_mm", float("nan"))),
        max_mm=float(metrics.get("max_mm", float("nan"))),
        corners=int(metrics.get("corners", 0)),
        tilt_degrees=float(alignment.get("tilt_degrees", float("nan"))),
        gap_mm=float(alignment.get("gap_mm", float("nan"))),
        scale_ratio=float(agreement.get("ratio", float("nan"))),
    )


def region_mode_enabled():
    from app_config import CONFIG
    return bool(CONFIG.get("region_homography", {}).get("enabled", False))

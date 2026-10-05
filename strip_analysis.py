"""Centreline, segmentation and width measurement for a detected strip.

SAM3 returns one dense boundary ring that traces both edges of the adhesive
strip, so the ring alone says nothing about where the strip runs or how thick it
is. Everything here works from that ring: rasterise it, skeletonise to the
medial axis, order the skeleton into a single path, smooth it, cut it into equal
arc-length segments, and measure how wide the strip is across each one.

Every distance is in source-image pixels. Converting to millimetres is a later
stage and needs only a pixels-per-millimetre factor at the call site, because
nothing in here is expressed in any other unit.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace

import cv2
import numpy as np
from scipy.ndimage import grey_closing, median_filter, uniform_filter1d
from skimage.morphology import skeletonize

# The 8-neighbourhood used to walk the skeleton pixel by pixel.
_NEIGHBOURS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))

MIN_MASK_PIXELS = 40           # smaller than this is a speck, not a strip
MIN_CENTERLINE_PX = 30.0       # too short to cut into meaningful segments
RAY_STEP_PX = 0.5              # how finely a width ray advances
RAY_SKEW_RAD = 0.10            # extra angles tried either side of the normal (~6 deg)
MAX_CENTERLINE_SAMPLES = 500   # resampling density; also bounds ray-marching cost
SMOOTH_FRACTION = 0.02         # smoothing window, as a fraction of the centreline
END_MARGIN_FRACTION = 0.5      # end trimmed from the path, in units of strip width

# Edge refinement (refine_widths). A glue bead's edges show as thin dark lines
# even where the bead is the same colour as the fabric; these bound what counts
# as one of those lines.
REFINE_BLUR_SIGMA = 1.0        # pixel noise suppression before reading profiles
REFINE_LINE_SPAN_PX = 13       # widest dark line treated as an edge, across the strip
REFINE_MIN_DEPTH = 1.5         # gray levels a line must dip below its surroundings
REFINE_OUTLIER_WINDOW = 9      # samples; rejects a lone edge that jumped to a fold

CENTERLINE_COLOR = (0, 255, 255)  # BGR
CENTERLINE_HALO = (0, 0, 0)
TICK_COLOR = (0, 255, 255)
LABEL_TEXT_COLOR = (255, 255, 255)
LABEL_BACKGROUND = (32, 32, 32)
PASS_COLOR = (65, 190, 65)       # BGR green
FAIL_COLOR = (45, 45, 230)       # BGR red
LABEL_OFFSET_PX = 24.0
LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_SCALE = 0.5


@dataclass(frozen=True)
class Segment:
    """One equal-length slice of the strip, measured across its centreline."""

    index: int                     # 1-based, matching the label drawn on the image
    length_px: float
    average_width_px: float
    minimum_width_px: float
    maximum_width_px: float
    samples: int
    midpoint: tuple[float, float]
    normal: tuple[float, float]    # unit vector across the strip at the midpoint
    start: tuple[float, float]
    end: tuple[float, float]
    first_sample: int = 0          # centreline index range this segment covers
    last_sample: int = 0
    # Filled in by to_metric; None until a millimetre scale is applied.
    length_mm: float | None = None
    average_width_mm: float | None = None
    minimum_width_mm: float | None = None
    maximum_width_mm: float | None = None
    within_tolerance: bool | None = None


@dataclass(frozen=True)
class StripAnalysis:
    """The centreline and its division into measured segments."""

    centerline: np.ndarray         # (N, 2) float, x/y in source pixels
    normals: np.ndarray            # (N, 2) float, unit vector across the strip
    widths: np.ndarray             # (N,) float, width at each centreline point
    segments: list[Segment]
    edge_indices: list[int]        # centreline index where each segment starts
    total_length_px: float
    average_width_px: float
    total_length_mm: float | None = None
    average_width_mm: float | None = None
    target_width_mm: float | None = None
    width_tolerance_mm: float | None = None
    # Share of centreline samples whose two edges were both located in the
    # image by refine_widths; None when the widths are the polygon's own.
    refined_fraction: float | None = None

    @property
    def minimum_width_px(self) -> float:
        return min(s.minimum_width_px for s in self.segments)

    @property
    def maximum_width_px(self) -> float:
        return max(s.maximum_width_px for s in self.segments)

    @property
    def metric(self) -> bool:
        """Whether a millimetre scale has been applied."""
        return self.total_length_mm is not None

    @property
    def minimum_width_mm(self) -> float | None:
        if not self.metric:
            return None
        return min(s.minimum_width_mm for s in self.segments)

    @property
    def maximum_width_mm(self) -> float | None:
        if not self.metric:
            return None
        return max(s.maximum_width_mm for s in self.segments)


def to_metric(analysis: StripAnalysis, scale) -> StripAnalysis:
    """Restate an analysis in millimetres, keeping the pixel figures.

    Distances are recomputed from mapped points rather than scaled by a single
    factor, because the crop is a projection: a pixel is not a fixed number of
    millimetres across the whole image. Each width in particular is measured
    between its own two edge points after mapping, so it stays the width across
    the strip in the plane and not merely in the image.
    """
    centerline_mm = scale.to_mm(analysis.centerline)
    arc = np.concatenate((
        [0.0],
        np.cumsum(np.linalg.norm(np.diff(centerline_mm, axis=0), axis=1)),
    ))

    half = (analysis.widths / 2.0)[:, None]
    left_mm = scale.to_mm(analysis.centerline - analysis.normals * half)
    right_mm = scale.to_mm(analysis.centerline + analysis.normals * half)
    widths_mm = np.linalg.norm(right_mm - left_mm, axis=1)

    total_length_mm = float(arc[-1])
    segments = []
    for segment in analysis.segments:
        low, high = segment.first_sample, segment.last_sample
        chunk = widths_mm[low:high]
        segments.append(replace(
            segment,
            # Split proportionally, exactly as the pixel figure does. Measuring
            # the arc from the first sample to the last would leave a one-sample
            # gap at every boundary, so the segments would no longer add up to
            # the total -- and the centreline is evenly resampled, so the
            # proportional share is already the arc length.
            length_mm=float(total_length_mm * (high - low) / len(analysis.centerline)),
            average_width_mm=float(chunk.mean()),
            minimum_width_mm=float(chunk.min()),
            maximum_width_mm=float(chunk.max()),
        ))

    return replace(
        analysis,
        segments=segments,
        total_length_mm=total_length_mm,
        average_width_mm=float(widths_mm.mean()),
    )


def grade_widths(analysis: StripAnalysis, target_width_mm: float,
                 tolerance_mm: float) -> StripAnalysis:
    """Grade every segment's average metric width against one inclusive range."""
    target = float(target_width_mm)
    tolerance = float(tolerance_mm)
    if not np.isfinite(target) or target <= 0:
        raise ValueError("Target strip width must be a positive number")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Strip width tolerance must be a positive number")
    if not analysis.metric:
        return replace(
            analysis, target_width_mm=target, width_tolerance_mm=tolerance)
    low, high = target - tolerance, target + tolerance
    return replace(
        analysis,
        segments=[
            replace(segment, within_tolerance=(
                segment.average_width_mm is not None
                and low <= segment.average_width_mm <= high
            ))
            for segment in analysis.segments
        ],
        target_width_mm=target,
        width_tolerance_mm=tolerance,
    )


# --- centreline --------------------------------------------------------------


def _skeleton_path(skeleton: np.ndarray) -> np.ndarray | None:
    """Order the skeleton into a single open path from one end to the other.

    A skeleton is a tree (plus the odd spur), so the longest path through it is
    found by walking to the farthest pixel and then walking again from there --
    the classic two-pass tree-diameter trick. That picks the strip's length and
    ignores the stubby branches that a rasterised edge leaves behind.
    """
    ys, xs = np.nonzero(skeleton)
    if xs.size == 0:
        return None
    pixels = set(zip(xs.tolist(), ys.tolist()))

    def neighbours(point):
        x, y = point
        for dx, dy in _NEIGHBOURS:
            candidate = (x + dx, y + dy)
            if candidate in pixels:
                yield candidate

    def farthest_from(start):
        parents = {start: None}
        queue = deque([start])
        last = start
        while queue:
            point = queue.popleft()
            for nxt in neighbours(point):
                if nxt not in parents:
                    parents[nxt] = point
                    queue.append(nxt)
                    last = nxt  # breadth-first, so the last one seen is the deepest
        return last, parents

    seed = next(iter(pixels))
    end_a, _ = farthest_from(seed)
    end_b, parents = farthest_from(end_a)

    path = []
    node = end_b
    while node is not None:
        path.append(node)
        node = parents[node]
    path.reverse()
    return np.asarray(path, dtype=np.float64)


def _resample_uniformly(path: np.ndarray, count: int) -> tuple[np.ndarray, float]:
    """Redistribute a path's points at even arc-length spacing."""
    steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(steps)))
    total = float(cumulative[-1])
    if total <= 0:
        return np.empty((0, 2)), 0.0
    targets = np.linspace(0.0, total, count)
    return np.column_stack((
        np.interp(targets, cumulative, path[:, 0]),
        np.interp(targets, cumulative, path[:, 1]),
    )), total


def _tangents_and_normals(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit tangent and unit normal at every point of a smoothed path."""
    tangent = np.gradient(points, axis=0)
    lengths = np.linalg.norm(tangent, axis=1)
    lengths[lengths < 1e-6] = 1.0  # a repeated point has no direction; keep it finite
    tangent = tangent / lengths[:, None]
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    return tangent, normal


# --- width -------------------------------------------------------------------


def _widths_along(mask: np.ndarray, points: np.ndarray, normals: np.ndarray,
                  max_distance: float) -> np.ndarray:
    """Width across the mask at every point, marching along its own normal.

    Every sample advances together, one step at a time, so the cost is a few
    hundred numpy operations instead of a few hundred thousand Python steps.
    A point stops accumulating the moment it leaves the mask, which is what
    makes this the strip's own thickness rather than an axis-aligned span.
    """
    height, width = mask.shape
    steps = int(max_distance / RAY_STEP_PX)
    total = np.zeros(len(points))

    for direction in (1.0, -1.0):
        distance = np.zeros(len(points))
        for step in range(1, steps + 1):
            reach = step * RAY_STEP_PX
            columns = np.round(points[:, 0] + normals[:, 0] * direction * reach)
            rows = np.round(points[:, 1] + normals[:, 1] * direction * reach)
            inside = ((columns >= 0) & (columns < width)
                      & (rows >= 0) & (rows < height))
            columns = np.clip(columns, 0, width - 1).astype(np.int64)
            rows = np.clip(rows, 0, height - 1).astype(np.int64)
            inside &= mask[rows, columns] > 0
            distance[inside] = reach
        total += distance

    return total


def _widths(mask: np.ndarray, points: np.ndarray, normals: np.ndarray,
            max_distance: float) -> np.ndarray:
    """Width at each point, made robust to a slightly skewed normal.

    The perpendicular crossing is the shortest one, so a normal that is a few
    degrees off reads long. Measuring at three small skews and taking the middle
    value discards those over-readings without pulling the result consistently
    below the true width.
    """
    if RAY_SKEW_RAD <= 0:
        return _widths_along(mask, points, normals, max_distance)

    readings = []
    for angle in (-RAY_SKEW_RAD, 0.0, RAY_SKEW_RAD):
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        readings.append(_widths_along(mask, points, np.column_stack((
            normals[:, 0] * cos_a - normals[:, 1] * sin_a,
            normals[:, 0] * sin_a + normals[:, 1] * cos_a,
        )), max_distance))
    return np.median(np.vstack(readings), axis=0)


# --- the whole analysis ------------------------------------------------------


def analyze_ring(ring: np.ndarray, image_shape: tuple[int, ...],
                 segment_count: int = 10) -> StripAnalysis | None:
    """Measure a strip from its boundary ring.

    Returns ``None`` when the ring cannot carry a meaningful measurement -- too
    few points, an empty mask, a skeleton that is only a dot, or a centreline too
    short to divide. Callers treat that as "nothing to report" rather than an
    error, so a stray prediction can never fail an inspection.
    """
    ring = np.asarray(ring, dtype=np.int32).reshape(-1, 2)
    if len(ring) < 3 or segment_count < 1:
        return None

    height, width = image_shape[:2]
    mask = np.zeros((height, width), np.uint8)
    cv2.fillPoly(mask, [ring], 1)
    area = float(np.count_nonzero(mask))
    if area < MIN_MASK_PIXELS:
        return None

    path = _skeleton_path(skeletonize(mask.astype(bool)))
    if path is None or len(path) < 2:
        return None

    count = int(min(max(len(path), 64), MAX_CENTERLINE_SAMPLES))
    points, raw_length = _resample_uniformly(path, count)
    if raw_length < MIN_CENTERLINE_PX:
        return None

    # A skeleton zigzags at pixel resolution -- which inflates its length by up
    # to ~8% -- and sprouts short forks at the strip's flat ends, where the
    # longest path can start at a corner instead of the cap. Drop a margin equal
    # to half a width from each end, then smooth. The smoothed curve is what the
    # length, the segments and the drawing all share, so they stay consistent.
    spacing = raw_length / max(len(points) - 1, 1)
    margin = int(round(END_MARGIN_FRACTION * (area / raw_length) / spacing))
    if margin > 0 and len(points) > 2 * margin + 8:
        points = points[margin:len(points) - margin]

    window = max(5, int(round(len(points) * SMOOTH_FRACTION)))
    smoothed = np.column_stack((
        uniform_filter1d(points[:, 0], window, mode="nearest"),
        uniform_filter1d(points[:, 1], window, mode="nearest"),
    ))
    # Re-space after smoothing, so slicing by index is slicing by arc length.
    centerline, total_length = _resample_uniformly(smoothed, len(smoothed))
    if total_length < MIN_CENTERLINE_PX:
        return None
    _, normals = _tangents_and_normals(centerline)

    # The mask area over the centreline length is a good estimate of the width,
    # and it bounds the ray march without needing a hard-coded guess.
    estimate = area / total_length
    max_distance = min(4.0 * estimate + 10.0, 0.5 * max(height, width))
    widths = _widths(mask, centerline, normals, max_distance)

    # Resampling made the points evenly spaced, so slicing by index is the same
    # as slicing by arc length.
    edges = np.linspace(0, len(centerline), segment_count + 1).astype(int)
    segments: list[Segment] = []
    for index in range(segment_count):
        low, high = int(edges[index]), int(edges[index + 1])
        if high - low < 2:
            continue
        chunk = widths[low:high]
        middle = (low + high) // 2
        segments.append(Segment(
            index=index + 1,
            length_px=total_length * (high - low) / len(centerline),
            average_width_px=float(chunk.mean()),
            minimum_width_px=float(chunk.min()),
            maximum_width_px=float(chunk.max()),
            samples=int(high - low),
            midpoint=(float(centerline[middle, 0]), float(centerline[middle, 1])),
            normal=(float(normals[middle, 0]), float(normals[middle, 1])),
            start=(float(centerline[low, 0]), float(centerline[low, 1])),
            end=(float(centerline[high - 1, 0]), float(centerline[high - 1, 1])),
            first_sample=low,
            last_sample=high,
        ))

    if not segments:
        return None

    return StripAnalysis(
        centerline=centerline,
        normals=normals,
        widths=widths,
        segments=segments,
        edge_indices=[int(edge) for edge in edges],
        total_length_px=total_length,
        average_width_px=float(widths.mean()),
    )


# --- edge refinement ---------------------------------------------------------


def _trough(offsets: np.ndarray, depth: np.ndarray, index: np.ndarray) -> np.ndarray:
    """Sub-pixel position of each trough: the vertex of the parabola through it and
    the samples either side.

    A dark edge line a few pixels wide rarely has its darkest point on a pixel
    centre, so taking the deepest whole pixel alone reports every width as a whole
    number of pixels and a 28.4 px strip reads as 28 or 29 depending on where the
    fabric happens to fall. The fit is clamped to half a pixel either side of the
    pixel it started from, so noise can never walk an edge off the line that was
    found, and a trough that is not a real minimum is left on its own pixel.
    """
    rows = np.arange(depth.shape[0])
    left = np.clip(index - 1, 0, depth.shape[1] - 1)
    right = np.clip(index + 1, 0, depth.shape[1] - 1)
    y0, y1, y2 = depth[rows, left], depth[rows, index], depth[rows, right]
    curvature = y0 - 2.0 * y1 + y2          # negative only where y1 is a minimum
    usable = (depth.shape[1] > 2) & (curvature < -1e-9)
    shift = np.zeros(len(index), np.float64)
    shift[usable] = np.clip(0.5 * (y0[usable] - y2[usable]) / curvature[usable], -0.5, 0.5)
    return offsets[index] + shift


def refine_widths(gray: np.ndarray, analysis: StripAnalysis,
                  search_px: float = 10.0) -> StripAnalysis:
    """Move each strip edge from the polygon onto the edge line visible in the image.

    The polygon is predicted on a downscaled frame, so each of its edges can sit
    several full-resolution pixels from the real one -- more than the tolerance
    on a narrow strip. A glue bead on fabric of its own colour has no brightness
    step to find, but each of its edges still shows as a thin dark line. This
    reads the brightness profile across the strip at every centreline sample and,
    within ``search_px`` of where the polygon put each edge, takes the deepest
    such line as the edge.

    Glare is clipped first, so a highlight on the bead is never taken for an
    edge. An edge is moved only when a line is actually there; otherwise the
    polygon's edge is kept, so a strip with no edge lines -- a plain brightness
    step, or nothing at all -- comes back with its original widths.
    ``refined_fraction`` on the result says how much of the strip was relocated.
    """
    gray = np.asarray(gray)
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    points, normals = analysis.centerline, analysis.normals
    half = analysis.widths / 2.0
    if len(points) < 2 or search_px <= 0:
        return analysis

    # Profiles: one row per centreline sample, one column per pixel of offset
    # along that sample's normal.
    reach = int(np.ceil(float(half.max()) + search_px)) + REFINE_LINE_SPAN_PX
    offsets = np.arange(-reach, reach + 1, dtype=np.float64)
    blurred = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), REFINE_BLUR_SIGMA)
    profiles = cv2.remap(
        blurred,
        (points[:, 0:1] + normals[:, 0:1] * offsets).astype(np.float32),
        (points[:, 1:2] + normals[:, 1:2] * offsets).astype(np.float32),
        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    # An edge runs along the strip, so neighbouring profiles agree about it and
    # averaging about one strip-width of them lifts it out of the noise.
    spacing = analysis.total_length_px / max(len(points) - 1, 1)
    along = int(np.clip(round(float(half.mean()) * 2.0 / max(spacing, 1e-6)), 3, 15)) | 1
    profiles = uniform_filter1d(profiles, along, axis=0, mode="nearest")
    profiles = np.minimum(profiles, np.median(profiles, axis=1, keepdims=True))
    # How far each pixel dips below a profile with its narrow dark lines filled in.
    depth = grey_closing(profiles, size=(1, REFINE_LINE_SPAN_PX)) - profiles

    rows = np.arange(len(points))
    distances, located = [], []
    for side in (-1.0, 1.0):
        # Stay on this edge's own side of the centreline and near the polygon edge.
        window = ((np.abs(offsets[None, :] - side * half[:, None]) <= search_px)
                  & (side * offsets[None, :] >= 1.0))
        candidates = np.where(window, depth, -1.0)
        best = candidates.argmax(axis=1)
        found = candidates[rows, best] >= REFINE_MIN_DEPTH
        # Each edge sits between pixels, so read its trough to a fraction of one.
        distance = np.abs(_trough(offsets, depth, best))
        # A single sample that latched onto a fold or a neighbouring edge is
        # pulled back to what its neighbours found.
        distance = np.where(found, median_filter(
            np.where(found, distance, half), REFINE_OUTLIER_WINDOW, mode="nearest"), half)
        distances.append(distance)
        located.append(found)

    refined_fraction = float(np.mean(located[0] & located[1]))
    if not (located[0].any() or located[1].any()):
        return replace(analysis, refined_fraction=0.0)

    widths = distances[0] + distances[1]
    # The two edges no longer sit symmetrically about the old centreline, and the
    # millimetre conversion assumes they do, so the centreline moves to their middle.
    shift = uniform_filter1d((distances[1] - distances[0]) / 2.0, along, mode="nearest")
    centerline = points + normals * shift[:, None]

    segments = []
    for segment in analysis.segments:
        low, high = segment.first_sample, segment.last_sample
        chunk = widths[low:high]
        middle = (low + high) // 2
        segments.append(replace(
            segment,
            average_width_px=float(chunk.mean()),
            minimum_width_px=float(chunk.min()),
            maximum_width_px=float(chunk.max()),
            midpoint=(float(centerline[middle, 0]), float(centerline[middle, 1])),
            start=(float(centerline[low, 0]), float(centerline[low, 1])),
            end=(float(centerline[high - 1, 0]), float(centerline[high - 1, 1])),
        ))
    return replace(
        analysis,
        centerline=centerline,
        widths=widths,
        segments=segments,
        average_width_px=float(widths.mean()),
        refined_fraction=refined_fraction,
    )


# --- drawing -----------------------------------------------------------------


def _place_label(segment: Segment, shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    """Top-left corner of a label box beside ``segment``, kept inside the image.

    The box is pushed clear of the strip on whichever side has room, so a label
    never lands on the polygon it describes.
    """
    height, width = shape[:2]
    (text_w, text_h), baseline = cv2.getTextSize(
        _label_text(segment), LABEL_FONT, LABEL_SCALE, 1,
    )
    box_w, box_h = text_w + 8, text_h + baseline + 6
    reach = segment.average_width_px / 2.0 + LABEL_OFFSET_PX
    mid_x, mid_y = segment.midpoint
    normal_x, normal_y = segment.normal

    chosen = None
    for direction in (1.0, -1.0):
        x = mid_x + normal_x * direction * reach - box_w / 2.0
        y = mid_y + normal_y * direction * reach - box_h / 2.0
        if -box_w < x < width and 0 <= y < height:
            chosen = (x, y)
            break
    if chosen is None:
        chosen = (mid_x - box_w / 2.0, mid_y - box_h / 2.0)

    left = int(round(min(max(chosen[0], 2), max(2, width - box_w - 2))))
    top = int(round(min(max(chosen[1], box_h + 2), max(box_h + 2, height - 2))))
    return left, top, box_w, box_h


def _label_text(segment: Segment) -> str:
    """Show only the measured metric width; colour carries the grade."""
    if segment.average_width_mm is not None:
        return f"{segment.average_width_mm:.2f}mm"
    return ""


def _segment_color(segment: Segment) -> tuple[int, int, int]:
    if segment.within_tolerance is True:
        return PASS_COLOR
    if segment.within_tolerance is False:
        return FAIL_COLOR
    return CENTERLINE_COLOR


def draw_strip_analysis(canvas: np.ndarray, analysis: StripAnalysis) -> np.ndarray:
    """Draw the centreline, the segment boundaries and the width of each segment.

    Returns the same array it was given, drawn in place. The centreline gets a
    dark halo underneath so it stays visible over light and dark fabric alike,
    and each label carries its own background so it reads without covering the
    polygon.
    """
    # Shade the physical span of each graded segment. This makes an out-of-tolerance
    # section visible as a red portion of the strip, rather than only as red text.
    graded = any(segment.within_tolerance is not None for segment in analysis.segments)
    if graded:
        overlay = canvas.copy()
        for segment in analysis.segments:
            low, high = segment.first_sample, segment.last_sample
            points = analysis.centerline[low:high]
            normals = analysis.normals[low:high]
            widths = analysis.widths[low:high]
            if len(points) < 2:
                continue
            half = widths[:, None] / 2.0
            left = points - normals * half
            right = points + normals * half
            ring = np.round(np.vstack((left, right[::-1]))).astype(np.int32)
            cv2.fillPoly(overlay, [ring], _segment_color(segment))
        cv2.addWeighted(overlay, 0.28, canvas, 0.72, 0, dst=canvas)

    for segment in analysis.segments:
        points = analysis.centerline[segment.first_sample:segment.last_sample]
        if len(points) < 2:
            continue
        outline = np.round(points).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [outline], False, CENTERLINE_HALO, 5, cv2.LINE_AA)
        cv2.polylines(canvas, [outline], False, _segment_color(segment), 3, cv2.LINE_AA)

    # A tick at each internal boundary, spanning the strip, marks where one
    # segment ends and the next begins.
    for index in analysis.edge_indices[1:-1]:
        point = analysis.centerline[index]
        normal = analysis.normals[index]
        # Span the measured width, with a floor so a thin strip still shows a tick.
        half = max(4.0, analysis.widths[index] / 2.0)
        start = (int(round(point[0] - normal[0] * half)),
                 int(round(point[1] - normal[1] * half)))
        end = (int(round(point[0] + normal[0] * half)),
               int(round(point[1] + normal[1] * half)))
        cv2.line(canvas, start, end, CENTERLINE_HALO, 3, cv2.LINE_AA)
        cv2.line(canvas, start, end, TICK_COLOR, 1, cv2.LINE_AA)

    for segment in analysis.segments:
        if segment.average_width_mm is None:
            continue
        left, top, box_w, box_h = _place_label(segment, canvas.shape)
        cv2.rectangle(canvas, (left, top), (left + box_w, top + box_h),
                      (_segment_color(segment) if segment.within_tolerance is not None
                       else LABEL_BACKGROUND), -1)
        cv2.putText(canvas, _label_text(segment), (left + 4, top + box_h - 5),
                    LABEL_FONT, LABEL_SCALE, LABEL_TEXT_COLOR, 1, cv2.LINE_AA)

    return canvas

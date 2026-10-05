"""Local glue-line detector for pale fabric. No model and no network.

A clear glue bead on fabric of its own colour has no colour of its own. What the
camera records is its gloss and its relief: the bead is a thin bright line, and
each of its edges is a thin dark line. A strip is therefore a dark-bright-dark
triple -- two dark lines a few millimetres apart with a bright line between them
-- running a long way. A fold, a mesh edge or a seam lacks that triple, and
weave or moire texture is short.

The detector works in four steps:

1. Enhance. Remove the slow lighting gradient, then run an elongated line
   filter in many directions, keeping dark and bright lines separately along
   with the direction each line runs in. A long line keeps its response; short
   texture averages away.
2. Restrict to pale fabric, away from its outline, so panel edges, the table,
   cardboard and tape cannot be mistaken for glue.
3. At every point of every dark line, look straight across it for a second dark
   line running the same way within the strip width, with a bright line between
   the two. Where that holds, the midpoint is a point on the bead and the
   distance is its width there. This is decided point by point, so it does not
   matter how the lines happen to connect, break or meet at the bead's ends.
4. Join those points into centrelines, bridge short gaps where an edge faded,
   and report each long one as a strip: its outline is the centreline widened
   by the width measured along it.

Select it with ``segmentation.provider`` set to
``segmentation.glue_line:GlueLineSegmenter``. Its settings live in
``segmentation.glue_line`` and every distance there is in pixels of the full
camera frame.
"""
from collections import deque

import cv2
import numpy as np
from scipy.ndimage import median_filter
from skimage.morphology import skeletonize

from . import Instance, SegmentationResult

DEFAULTS = {
    'working_scale': 1.0,        # the bead is thin; reduce only for a closer camera
    'min_width_px': 8.0,         # narrowest and widest strip, full-frame pixels
    'max_width_px': 34.0,
    'min_length_px': 900.0,      # shorter than this is not reported as a strip
    'line_sensitivity': 2.0,     # line threshold, in multiples of the fabric's noise
    'min_gloss_share': 0.35,     # share of a strip that must show the bright bead
    'min_seen_share': 0.55,      # share of a strip's length that must be seen, not bridged
    'fabric_margin_px': 30.0,    # ignore this close to the fabric outline
    'max_saturation': 45,        # fabric is unsaturated; cardboard and tape are not
    'min_fabric_gray': 90,       # darker than this is not pale fabric
    'join_gap_px': 250.0,        # stretches this close, end to end, are one strip
    'track_gaps': True,          # measure a bridged stretch instead of only guessing it
}

LINE_SIGMA = 2.5        # half-thickness of the lines looked for, working pixels
LINE_ALONG = 10.0       # how far a line must persist to count, working pixels
LINE_DIRECTIONS = 12
SAMPLE_STEP = 4.0       # spacing of points along a centreline, working pixels
MIN_PIECE_PX = 40.0     # a centreline piece shorter than this is noise
BEAD_LIFT = 1.0         # gray levels the bead must stand above the fabric beside it
BEAD_CLEARANCE = 10.0   # how far outside each edge that fabric is sampled, working pixels
GAP_RELAX = 0.6         # threshold share allowed inside a bridged gap, where the
                        # edges faded but are probably still there
GAP_MARGIN = 3.0        # how far outside the joined widths the edges may sit, working px
GAP_SEEN_SHARE = 0.6    # share of a gap that must be measured before it is believed


def _settings(settings):
    merged = dict(DEFAULTS)
    merged.update(settings.get('glue_line') or {})
    return merged


def fabric_mask(image, max_saturation, min_gray):
    """Pale, unsaturated material darker than the white table it lies on."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pale = (hsv[:, :, 1] < max_saturation) & (gray > min_gray)
    values = gray[pale]
    if values.size < 100:
        return np.zeros(gray.shape, np.uint8)
    # The pale pixels are the table and the fabric; Otsu finds the level between them.
    level, _ = cv2.threshold(values.reshape(-1, 1), 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (pale & (gray < level)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    keep = np.zeros_like(mask)
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] > 0.01 * mask.size:
            keep[labels == index] = 1
    return keep


def flatten(gray):
    """Remove the slow lighting gradient, leaving local detail around zero."""
    gray = gray.astype(np.float32)
    return gray - cv2.GaussianBlur(gray, (0, 0), 25)


def line_strength(flat):
    """Long thin lines in any direction: dark strength, bright strength, and the
    index of the direction in which the dark line at each pixel runs."""
    half = int(3 * LINE_ALONG)
    ys, xs = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float32)
    dark = np.zeros_like(flat)
    bright = np.zeros_like(flat)
    direction = np.zeros(flat.shape, np.int8)
    for index in range(LINE_DIRECTIONS):
        angle = np.pi * index / LINE_DIRECTIONS
        along = xs * np.cos(angle) + ys * np.sin(angle)
        across = -xs * np.sin(angle) + ys * np.cos(angle)
        # Second derivative of a Gaussian across the line, a long Gaussian along it.
        kernel = ((across ** 2 / LINE_SIGMA ** 2 - 1)
                  * np.exp(-across ** 2 / (2 * LINE_SIGMA ** 2))
                  * np.exp(-along ** 2 / (2 * LINE_ALONG ** 2)))
        kernel -= kernel.mean()
        kernel /= np.abs(kernel).sum()
        response = cv2.filter2D(flat, cv2.CV_32F, kernel)
        direction[response > dark] = index
        np.maximum(dark, response, out=dark)
        np.maximum(bright, -response, out=bright)
    return dark, bright, direction


def _threshold(values, region, sensitivity):
    """A line threshold from the region's own noise, with a floor tied to its
    strongest lines so a very clean image does not trace faint ripple."""
    inside = values[region > 0]
    return max(sensitivity * float(np.median(inside)),
               0.1 * float(np.percentile(inside, 99.5)), 1e-6)


def bead_points(dark, bright, direction, region, low, low_bright, min_width, max_width):
    """Points on a bead's centre, with the width measured across each.

    From each dark-line pixel, step across the line: the first dark line met
    within the width range that runs the same way, with a bright line halfway,
    is the bead's other edge.
    """
    # A line must stand above its surroundings, or two edges a few pixels apart
    # are bridged by the weaker response between them and read as one.
    crest = dark >= 0.6 * cv2.dilate(dark, np.ones((9, 9), np.uint8))
    edges = (dark > low) & crest & (region > 0)
    thin = skeletonize(edges)
    near_edge = cv2.dilate(thin.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    ys, xs = np.nonzero(thin)
    if xs.size == 0:
        return np.empty((0, 2)), np.empty(0)
    own = direction[ys, xs].astype(int)
    angle = np.pi * own / LINE_DIRECTIONS
    normal = np.column_stack([-np.sin(angle), np.cos(angle)])
    height, width = dark.shape
    widths = np.arange(float(np.ceil(min_width)), float(max_width) + 1.0)

    centres, spans = [], []
    for side in (-1.0, 1.0):
        found = np.zeros(len(xs), bool)
        span = np.zeros(len(xs))
        for distance in widths:
            px = np.round(xs + side * normal[:, 0] * distance).astype(int)
            py = np.round(ys + side * normal[:, 1] * distance).astype(int)
            inside = (px >= 0) & (px < width) & (py >= 0) & (py < height)
            px, py = np.clip(px, 0, width - 1), np.clip(py, 0, height - 1)
            turn = np.abs(direction[py, px].astype(int) - own)
            same_way = np.minimum(turn, LINE_DIRECTIONS - turn) <= 1
            mx = np.clip(np.round(xs + side * normal[:, 0] * distance / 2).astype(int), 0, width - 1)
            my = np.clip(np.round(ys + side * normal[:, 1] * distance / 2).astype(int), 0, height - 1)
            hit = (~found & inside & near_edge[py, px] & same_way
                   & (bright[my, mx] > low_bright))
            span[hit] = distance
            found |= hit
        centres.append(np.column_stack([xs + side * normal[:, 0] * span / 2,
                                        ys + side * normal[:, 1] * span / 2])[found])
        spans.append(span[found])
    return np.vstack(centres), np.concatenate(spans)


def _longest_path(points):
    """The longest route through one skeleton piece, end to end."""
    pixels = set(map(tuple, points))

    def farthest(start):
        parents = {start: None}
        queue = deque([start])
        last = start
        while queue:
            x, y = queue.popleft()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    step = (x + dx, y + dy)
                    if step in pixels and step not in parents:
                        parents[step] = (x, y)
                        queue.append(step)
                        last = step
        return last, parents

    end, _ = farthest(next(iter(pixels)))
    node, parents = farthest(end)
    path = []
    while node is not None:
        path.append(node)
        node = parents[node]
    return np.asarray(path, np.float64)


def _resample(path, values):
    """Evenly spaced, lightly smoothed points along a path, with their values."""
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    if distance[-1] < 3 * SAMPLE_STEP:
        return None
    targets = np.arange(0.0, distance[-1], SAMPLE_STEP)
    points = np.column_stack([np.interp(targets, distance, path[:, 0]),
                              np.interp(targets, distance, path[:, 1])])
    window = 7
    padded = np.pad(points, ((window // 2, window // 2), (0, 0)), mode='edge')
    box = np.ones(window) / window
    smooth = np.column_stack([np.convolve(padded[:, 0], box, 'valid'),
                              np.convolve(padded[:, 1], box, 'valid')])
    return smooth, np.interp(targets, distance, values)


def centrelines(centres, spans, shape):
    """Bead centre points gathered into ordered centrelines with their widths."""
    total = np.zeros(shape, np.float32)
    count = np.zeros(shape, np.float32)
    columns = np.clip(np.round(centres[:, 0]).astype(int), 0, shape[1] - 1)
    rows = np.clip(np.round(centres[:, 1]).astype(int), 0, shape[0] - 1)
    np.add.at(total, (rows, columns), spans)
    np.add.at(count, (rows, columns), 1.0)
    # Both edges vote for the same centre a pixel or so apart; closing fuses them.
    solid = cv2.morphologyEx((count > 0).astype(np.uint8), cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    width_sum = cv2.blur(total, (7, 7))
    width_count = cv2.blur(count, (7, 7))
    pieces = []
    number, labels, stats, _ = cv2.connectedComponentsWithStats(
        skeletonize(solid > 0).astype(np.uint8), connectivity=8)
    for index in range(1, number):
        if stats[index, cv2.CC_STAT_AREA] < MIN_PIECE_PX:
            continue
        ys, xs = np.nonzero(labels == index)
        path = _longest_path(np.column_stack([xs, ys]))
        px, py = path[:, 0].astype(int), path[:, 1].astype(int)
        local = width_sum[py, px] / np.maximum(width_count[py, px], 1e-6)
        piece = _resample(path, local)
        if piece is not None:
            pieces.append(piece)
    return pieces


def _heading(points, at_end):
    reach = min(8, len(points) - 1)
    return points[-1] - points[-1 - reach] if at_end else points[0] - points[reach]


def _sample(values, grid):
    """Bilinear read of an image at an (N, W, 2) grid of points."""
    return cv2.remap(values, grid[..., 0].astype(np.float32), grid[..., 1].astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def track_gap(evidence, start, end, half):
    """Follow the bead through a stretch where its edges fell below the threshold.

    A straight bridge between two pieces is only ever a guess, and a guessed stretch
    counts against the strip's confidence. Here the two edge lines are looked for
    again along the line between the two piece ends, at a relaxed threshold, and the
    centre and width are read at every sample where both edges are there with the
    bead still brighter than the fabric between them.

    ``evidence`` is the (dark, bright, region, low, low_bright, min_width, max_width)
    the strip was found with. Returns ``(points, widths, measured)``, or None when the
    bead could not be followed and the caller should fall back to its own straight
    bridge.
    """
    dark, bright, region, low, low_bright, min_width, max_width = evidence
    jump = np.asarray(end, np.float64) - np.asarray(start, np.float64)
    length = float(np.linalg.norm(jump))
    if length < 2 * SAMPLE_STEP or half <= 0:
        return None
    steps = int(length // SAMPLE_STEP)
    fraction = (np.arange(steps + 1) / steps)[:, None]
    corridor = np.asarray(start, np.float64) + jump * fraction
    normal = np.array([-jump[1], jump[0]]) / max(length, 1e-6)
    # The pieces either side of the gap report a width measured where the bead was
    # fading, so the corridor is searched as wide as a strip may be, not as narrow
    # as those two ends claim.
    reach = max(half, max_width / 2.0) + GAP_MARGIN
    offsets = np.arange(-np.ceil(reach), np.ceil(reach) + 1, dtype=np.float64)
    grid = corridor[:, None, :] + normal[None, None, :] * offsets[None, :, None]
    inside = ((grid[..., 0] >= 0) & (grid[..., 0] <= dark.shape[1] - 1)
              & (grid[..., 1] >= 0) & (grid[..., 1] <= dark.shape[0] - 1)).all(axis=1)

    depth = _sample(dark, grid)
    fabric = _sample(region.astype(np.float32), grid)
    # The first edge on each side of the corridor is this bead's own edge; anything
    # nearer the middle than the width allows is the opposite edge coming in.
    edge = (depth > low * GAP_RELAX) & (fabric > 0.5)
    left = edge & (offsets[None, :] <= 0)
    right = edge & (offsets[None, :] > 0)
    if not (left.any() and right.any()):
        return None
    first, last = left.argmax(axis=1), edge.shape[1] - 1 - right[:, ::-1].argmax(axis=1)
    centre, span = (offsets[first] + offsets[last]) / 2.0, offsets[last] - offsets[first]
    # The bead must still be the brightest thing between its own two edges, and the
    # two edges must be as far apart as a strip of this kind is allowed to be.
    gloss = _sample(bright, (corridor + normal[None, :] * centre[:, None])[:, None, :])
    measured = (inside & left.any(axis=1) & right.any(axis=1)
                & (span >= min_width - GAP_MARGIN) & (span <= max_width + GAP_MARGIN)
                & (np.abs(centre) <= reach) & (gloss[:, 0] > low_bright * GAP_RELAX))
    if measured.mean() < GAP_SEEN_SHARE:
        return None
    return corridor + normal[None, :] * centre[:, None], span, measured


def join_pieces(pieces, join_gap, tracking=None):
    """Centreline pieces of one strip, broken where an edge faded, joined end
    to end. Returns (points, widths, seen) with ``seen`` false across a bridge.

    ``tracking`` is the :func:`track_gap` argument bundle. Where a bridge can be
    measured rather than guessed, the strip is measured through it and the samples
    count towards its confidence.
    """
    pieces = sorted(([points, widths, np.ones(len(points), bool)] for points, widths in pieces),
                    key=lambda item: -len(item[0]))
    strips = []
    while pieces:
        points, widths, seen = pieces.pop(0)
        grown = True
        while grown:
            grown = False
            best = None
            for index, (other, _, _) in enumerate(pieces):
                for flipped in (False, True):
                    candidate = other[::-1] if flipped else other
                    for at_end in (True, False):
                        if at_end:
                            jump = candidate[0] - points[-1]
                            onward = -_heading(candidate, False)
                        else:
                            jump = points[0] - candidate[-1]
                            onward = _heading(candidate, True)
                        heading = _heading(points, at_end) * (1.0 if at_end else -1.0)
                        length = float(np.linalg.norm(jump))
                        if not 0 < length <= join_gap:
                            continue
                        unit = jump / length
                        # The gap must continue the strip and lead into the next piece.
                        if (np.dot(unit, heading) / (np.linalg.norm(heading) + 1e-6) > 0.8
                                and np.dot(unit, onward) / (np.linalg.norm(onward) + 1e-6) > 0.8
                                and (best is None or length < best[0])):
                            best = (length, index, flipped, at_end)
            if best is not None:
                length, index, flipped, at_end = best
                other, other_widths, other_seen = pieces.pop(index)
                if flipped:
                    other, other_widths, other_seen = other[::-1], other_widths[::-1], other_seen[::-1]
                first = (points, widths, seen) if at_end else (other, other_widths, other_seen)
                second = (other, other_widths, other_seen) if at_end else (points, widths, seen)
                bridge, bridge_width, bridge_seen = _bridge(first, second, length, tracking)
                points = np.vstack([first[0], bridge, second[0]])
                widths = np.concatenate([first[1], bridge_width, second[1]])
                seen = np.concatenate([first[2], bridge_seen, second[2]])
                grown = True
        strips.append((points, widths, seen))
    return strips


def _bridge(first, second, length, tracking):
    """The stretch between two joined pieces, measured where it can be."""
    half = (first[1][-1] + second[1][0]) / 4.0     # half of the width both agree on
    measured = track_gap(tracking, first[0][-1], second[0][0], half) if tracking else None
    if measured is not None:
        return measured
    # Otherwise the straight line, and it is admitted as a guess.
    steps = max(1, int(length // SAMPLE_STEP))
    fraction = (np.arange(1, steps) / steps)[:, None]
    return (first[0][-1] + (second[0][0] - first[0][-1]) * fraction,
            np.interp(fraction[:, 0], [0, 1], [first[1][-1], second[1][0]]),
            np.zeros(len(fraction), bool))


def _normals(curve):
    tangent = np.gradient(curve, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-6)
    return np.column_stack([-tangent[:, 1], tangent[:, 0]])


class GlueLineSegmenter:
    def __init__(self, settings):
        self.settings = settings
        self.options = _settings(settings)
        self.last_input = None   # the enhanced line image, for audit

    def segment(self, frame):
        options = self.options
        scale = float(options['working_scale'])
        height, width = frame.shape[:2]
        small = (frame if scale == 1.0 else
                 cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))
        fabric = fabric_mask(small, options['max_saturation'], options['min_fabric_gray'])
        margin = max(3, int(round(options['fabric_margin_px'] * scale))) | 1
        interior = cv2.erode(fabric, np.ones((margin, margin), np.uint8))
        flat = flatten(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
        smooth = cv2.GaussianBlur(flat, (0, 0), 1.5)
        dark, bright, direction = line_strength(flat)
        self.last_input = self._audit_image(dark, bright, interior)
        if not interior.any():
            return SegmentationResult((), (width, height))

        sensitivity = options['line_sensitivity']
        low = _threshold(dark, interior, sensitivity)
        low_bright = _threshold(bright, interior, sensitivity)
        centres, spans = bead_points(
            dark, bright, direction, interior, low, low_bright,
            options['min_width_px'] * scale, options['max_width_px'] * scale)
        if len(centres) == 0:
            return SegmentationResult((), (width, height))

        def level(points):
            columns = np.clip(np.round(points[:, 0]).astype(int), 0, smooth.shape[1] - 1)
            rows = np.clip(np.round(points[:, 1]).astype(int), 0, smooth.shape[0] - 1)
            return smooth[rows, columns]

        instances = []
        pieces = centrelines(centres, spans, dark.shape)
        evidence = ((dark, bright, interior, low, low_bright,
                     options['min_width_px'] * scale, options['max_width_px'] * scale)
                    if options['track_gaps'] else None)
        for points, widths, seen in join_pieces(pieces, options['join_gap_px'] * scale, evidence):
            length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
            if length / scale < options['min_length_px']:
                continue
            widths = median_filter(widths, 9, mode='nearest')
            normal = _normals(points)
            # The bead itself: brighter along the centre than the fabric just
            # outside its edges. Two folds side by side have nothing brighter
            # between them.
            outward = normal * (widths / 2.0 + BEAD_CLEARANCE)[:, None]
            lift = level(points) - (level(points - outward) + level(points + outward)) / 2.0
            gloss_share = float(np.mean(lift[seen] >= BEAD_LIFT))
            if gloss_share < options['min_gloss_share']:
                continue
            half = normal * (widths / 2.0)[:, None]
            ring = np.clip(np.vstack([points - half, (points + half)[::-1]]) / scale,
                           (0, 0), (width - 1, height - 1))
            # Confidence is the share of the strip's length where both edges and
            # the bead between them were actually seen. Weave and moire can be
            # chained into something strip-shaped, but only with long bridges.
            confidence = float(seen.mean()) * gloss_share
            if confidence < options['min_seen_share']:
                continue
            if len(ring) >= 3 and np.isfinite(ring).all():
                instances.append(Instance(ring, confidence, 'glue line'))
        instances.sort(key=lambda item: (float(item.polygon[:, 0].mean()),
                                         float(item.polygon[:, 1].mean())))
        return SegmentationResult(tuple(instances), (width, height))

    @staticmethod
    def _audit_image(dark, bright, interior):
        """Dark lines in red, the bright bead in green, dimmed outside the fabric."""
        def scaled(values):
            top = float(np.percentile(values, 99.7)) or 1.0
            return np.clip(values / top * 255, 0, 255).astype(np.uint8)
        image = np.dstack([np.zeros(dark.shape, np.uint8), scaled(bright), scaled(dark)])
        image[interior == 0] //= 4
        return image

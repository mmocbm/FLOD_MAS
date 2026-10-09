"""Exact glue-line outline from SAM, prompted with points on the roughly found line.

The local detector says where each glue line runs. Its own outline is only as good
as its edge-finding, which is weak where the bead is faint. Here that rough line is
used only as a prompt: points placed along it are sent, with the camera's own
pixels, to SAM's point-prompted segmentation, which returns the bead's outline.

The frame is sent as full-resolution tiles along the line. Resized to fit the model
a whole frame leaves the bead a few pixels wide; in a 1024-pixel tile it keeps its
real width.

This calls the Roboflow serverless API (the key in ``.env``), so each tile of the
image is uploaded. A tile that fails leaves that stretch as the detector found it.
"""
import cv2
import numpy as np

from . import sam_local

TILE = 1024            # tile edge, pixels of the full frame
TILE_STRIDE = 800      # arc length of line covered per tile
POINT_SPACING = 150    # distance between prompt points along the line
MIN_POINTS = 2
MIN_COVER = 0.6          # share of the rough line SAM's outline must cover to replace it
NEGATIVE_OFFSET = 45.0   # how far to each side of the line the "not this" points go


PREP_DEFAULTS = {
    'zoom': 1.0,          # enlargement of the tile before it is sent (2 = bead twice as wide)
    'denoise': False,     # edge-preserving denoising (bilateral, then median)
    'sharpen': 0.0,       # unsharp-mask strength; 0 = none
    'contrast': False,    # multi-scale top-hat / black-hat contrast (glue_line.enhance_bead)
    'levels': 0,          # reduce to this many brightness levels (k-means); 0 = keep all
    'dim_outside': False,  # flatten everything outside the glue region to its mean
}


def prepare_tile(tile, region_tile, prep):
    """Make the bead easier for SAM to separate, using only the glue region.

    Order matters: enlarge first so later steps work at the finer scale; denoise
    before sharpening so texture is not sharpened; quantise last, so that the bead,
    its dark sides and the fabric each become a flat patch with a hard border,
    which is what a segmentation model keys on.
    """
    zoom = float(prep['zoom'])
    if zoom != 1.0:
        tile = cv2.resize(tile, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
        region_tile = cv2.resize(region_tile, (tile.shape[1], tile.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
    if not (prep['denoise'] or prep['sharpen'] or prep['contrast'] or prep['levels']
            or prep['dim_outside']):
        return tile
    gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
    if prep['denoise']:
        gray = cv2.medianBlur(cv2.bilateralFilter(gray, 9, 18, 5), 5)
    if prep['contrast']:
        from .glue_line import ENHANCE_SCALES
        result = gray.astype(np.float32)
        for size in ENHANCE_SCALES:
            size = int(round(size * zoom)) | 1
            element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            result += cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, element)
            result -= cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, element)
        inside = result[region_tile > 0] if (region_tile > 0).any() else result.ravel()
        low, high = np.percentile(inside, [0.5, 99.5])
        gray = np.clip((result - low) / max(high - low, 1e-6) * 255.0, 0, 255).astype(np.uint8)
    if prep['sharpen'] > 0:
        soft = cv2.GaussianBlur(gray, (0, 0), 2.0 * zoom)
        gray = cv2.addWeighted(gray, 1.0 + prep['sharpen'], soft, -prep['sharpen'], 0)
    if prep['levels'] and (region_tile > 0).sum() > 100:
        samples = gray[region_tile > 0].reshape(-1, 1).astype(np.float32)[::7]
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
        _, _, centres = cv2.kmeans(samples, int(prep['levels']), None, criteria, 2,
                                   cv2.KMEANS_PP_CENTERS)
        centres = np.sort(centres.ravel())
        nearest = np.abs(gray[..., None].astype(np.float32) - centres[None, None, :]).argmin(axis=2)
        gray = centres[nearest].astype(np.uint8)
    if prep['dim_outside'] and (region_tile > 0).any():
        gray = np.where(region_tile > 0, gray, int(np.median(gray[region_tile > 0]))).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def segment_points(client, image, prompts, model='sam3'):
    """One point- or box-prompted call. ``model`` is ``sam3`` or ``sam2:<size>``
    with size ``hiera_tiny``, ``hiera_small``, ``hiera_b_plus`` or ``hiera_large``
    (Roboflow cloud), or ``local:<name>`` for a model run here (``sam_local.MODELS``)."""
    if model.startswith(sam_local.PREFIX):          # on this computer's GPU: nothing is uploaded
        return sam_local.segment_points(image, prompts, model[len(sam_local.PREFIX):])
    if model.startswith('sam2'):
        version = model.partition(':')[2] or 'hiera_large'
        return client.sam2_segment_image(image, prompts=prompts, sam2_version_id=version,
                                         multimask_output=False)
    return client.sam3_visual_segment(image, prompts=prompts, multimask_output=False)


def reply_mask(reply, shape):
    """The mask and best confidence in a point-prompt reply."""
    mask = np.zeros(shape[:2], np.uint8)
    confidence = 0.0
    for prediction in reply.get('predictions', []) if isinstance(reply, dict) else []:
        _paint(mask, prediction)
        confidence = max(confidence, float(prediction.get('confidence', 0.0)))
    return mask, confidence


def _paint(mask, prediction):
    """Add one prediction to a mask: the cloud sends outlines, a local model the
    mask itself."""
    if prediction.get('mask') is not None:
        mask |= (np.asarray(prediction['mask']) > 0).astype(np.uint8)
    for ring in prediction.get('masks', []):
        if len(ring) >= 3:
            cv2.fillPoly(mask, [np.asarray(ring, np.int32)], 1)


def _client():
    import sam_detection
    return sam_detection.shared_client()


def _tiles(centre, frame_shape, tile=TILE, stride=TILE_STRIDE):
    """Top-left corners of tiles that together cover the line, and the line
    samples each one is responsible for."""
    height, width = frame_shape[:2]
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(centre, axis=0), axis=1))]
    count = max(1, int(np.ceil(arc[-1] / stride)))
    edges = np.linspace(0.0, arc[-1], count + 1)
    for start, stop in zip(edges[:-1], edges[1:]):
        members = np.flatnonzero((arc >= start) & (arc <= stop))
        if len(members) < 2:
            continue
        middle = centre[members].mean(axis=0)
        x0 = int(np.clip(middle[0] - tile / 2, 0, max(0, width - tile)))
        y0 = int(np.clip(middle[1] - tile / 2, 0, max(0, height - tile)))
        yield x0, y0, members


def refine_line(frame, centre, max_width_px=80.0, client=None, region=None, prep=None,
                model='sam3', stride=TILE_STRIDE, spacing=POINT_SPACING, prepare=None):
    """SAM's mask of the bead along one rough centreline, as a full-frame mask.

    ``prepare(tile, x0, y0)`` may return a filtered copy of a tile before it is
    sent (same size): a filter chosen for what the picture is like at that place.

    Returns ``(mask, confidences)``; the mask is empty where no tile succeeded.
    """
    if client is None and not model.startswith(sam_local.PREFIX):
        client = _client()
    prep = dict(PREP_DEFAULTS, **(prep or {}))
    zoom = float(prep['zoom'])
    if region is None:
        region = np.ones(frame.shape[:2], np.uint8)
    height, width = frame.shape[:2]
    mask = np.zeros((height, width), np.uint8)
    confidences = []
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(centre, axis=0), axis=1))]
    source = int(round(TILE / zoom))                 # frame pixels covered by one tile
    for x0, y0, members in _tiles(centre, frame.shape, source, int(stride / zoom)):
        raw = np.ascontiguousarray(frame[y0:y0 + source, x0:x0 + source])
        tile = prepare_tile(raw, region[y0:y0 + source, x0:x0 + source], prep)
        if prepare is not None:          # the caller's own filter for this place
            tile = prepare(tile, x0, y0)
        # Prompt points: evenly spaced along this tile's stretch of the line.
        targets = np.arange(arc[members[0]] + spacing / zoom / 2, arc[members[-1]],
                            spacing / zoom)
        points = np.column_stack([np.interp(targets, arc, centre[:, 0]) - x0,
                                  np.interp(targets, arc, centre[:, 1]) - y0]) * zoom
        points = points[(points[:, 0] > 4) & (points[:, 0] < tile.shape[1] - 4)
                        & (points[:, 1] > 4) & (points[:, 1] < tile.shape[0] - 4)]
        if len(points) < MIN_POINTS:
            continue
        tangent = np.gradient(points, axis=0)
        tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-6)
        beside = np.column_stack([-tangent[:, 1], tangent[:, 0]]) * NEGATIVE_OFFSET
        positive = [{'x': float(x), 'y': float(y), 'positive': True} for x, y in points]
        negative = [{'x': float(x), 'y': float(y), 'positive': False}
                    for x, y in np.vstack([points + beside, points - beside])
                    if 4 < x < tile.shape[1] - 4 and 4 < y < tile.shape[0] - 4]
        # First the bead alone. If SAM answers with something far wider than a
        # bead -- the panel, the band beside the bead, the mesh -- it is told
        # what is not wanted with points on either side, and asked again.
        for prompt in ([{'points': positive}], [{'points': positive + negative}]):
            try:
                reply = segment_points(client, tile, prompt, model)
            except Exception:  # noqa: BLE001 - this stretch keeps the detector's outline
                if model.startswith(sam_local.PREFIX):
                    raise           # a local model failing is a set-up fault, not a bad connection
                break
            piece = np.zeros(tile.shape[:2], np.uint8)
            confidence = 0.0
            for prediction in reply.get('predictions', []):
                _paint(piece, prediction)
                confidence = max(confidence, float(prediction.get('confidence', 0.0)))
            across = 2.0 * cv2.distanceTransform(piece, cv2.DIST_L2, 5)[
                np.round(points[:, 1]).astype(int), np.round(points[:, 0]).astype(int)]
            if (across > 0).mean() >= 0.5 and np.median(across[across > 0]) <= max_width_px * zoom:
                if zoom != 1.0:
                    piece = cv2.resize(piece, (raw.shape[1], raw.shape[0]),
                                       interpolation=cv2.INTER_NEAREST)
                mask[y0:y0 + raw.shape[0], x0:x0 + raw.shape[1]] |= piece
                confidences.append(confidence)
                break
    return mask, confidences


def bead_ring(mask, centre, max_width_px):
    """The outline of the mask's part that the rough line runs through, or None.

    SAM answers a prompt with *an* object. If it took the points to mean the whole
    panel or the mesh, the mask is far wider than a bead and is refused.
    """
    count, labels = cv2.connectedComponents(mask)
    if count < 2:
        return None
    columns = np.clip(np.round(centre[:, 0]).astype(int), 0, mask.shape[1] - 1)
    rows = np.clip(np.round(centre[:, 1]).astype(int), 0, mask.shape[0] - 1)
    on_line = labels[rows, columns]
    on_line = on_line[on_line > 0]
    if on_line.size < MIN_COVER * len(centre):
        return None
    if np.mean(on_line == np.bincount(on_line).argmax()) * on_line.size < MIN_COVER * len(centre):
        return None         # only scraps of the line were outlined
    chosen = (labels == np.bincount(on_line).argmax()).astype(np.uint8)
    width = 2.0 * cv2.distanceTransform(chosen, cv2.DIST_L2, 5)[rows, columns]
    width = width[width > 0]
    if width.size == 0 or np.median(width) > max_width_px:
        return None
    # A bead's outline is one even ribbon about as long as the line. One with
    # arms, bulges or a tail has taken in something beside the bead.
    x, y, w, h = cv2.boundingRect(chosen)
    part = chosen[y:y + h, x:x + w]
    from skimage.morphology import skeletonize
    spine = skeletonize(part > 0)
    across = 2.0 * cv2.distanceTransform(part, cv2.DIST_L2, 5)[spine]
    line_length = float(np.linalg.norm(np.diff(centre, axis=0), axis=1).sum())
    if (spine.sum() > 1.25 * line_length or np.percentile(across, 90) > max_width_px
            or np.percentile(across, 90) > 1.6 * np.median(across)):
        return None
    contours, _ = cv2.findContours(chosen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    ring = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    return ring if len(ring) >= 3 else None


TRIM_REACH = 70          # how far to each side of the line SAM's outline is followed
TRIM_STEP = 4.0          # spacing of the samples along the line
TRIM_TOLERANCE = 5.0     # an edge further than this from its smooth course has leaked
TRIM_NEAR = 20           # SAM's outline must pass this close to the rough line to count


def _edges(mask, centre):
    """How far SAM's mask reaches to each side of the rough line, sample by sample.

    Returns ``(points, normals, low, high)``; ``low``/``high`` are signed distances
    along the normal, NaN where the mask does not pass near the line.
    """
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(centre, axis=0), axis=1))]
    along = np.arange(0.0, arc[-1], TRIM_STEP)
    points = np.column_stack([np.interp(along, arc, centre[:, 0]), np.interp(along, arc, centre[:, 1])])
    reach = 7
    window = np.ones(2 * reach + 1) / (2 * reach + 1)
    points = np.column_stack([np.convolve(np.pad(points[:, axis], reach, mode='edge'), window, 'valid')
                              for axis in (0, 1)])
    tangent = np.gradient(points, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-6)
    normals = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    offsets = np.arange(-TRIM_REACH, TRIM_REACH + 1)
    columns = np.clip(np.round(points[:, None, 0] + offsets[None] * normals[:, None, 0]).astype(int),
                      0, mask.shape[1] - 1)
    rows = np.clip(np.round(points[:, None, 1] + offsets[None] * normals[:, None, 1]).astype(int),
                   0, mask.shape[0] - 1)
    low = np.full(len(points), np.nan)
    high = np.full(len(points), np.nan)
    for index, row in enumerate(mask[rows, columns] > 0):
        inside = np.flatnonzero(row)
        if inside.size == 0:
            continue
        nearest = inside[np.abs(offsets[inside]).argmin()]
        if abs(offsets[nearest]) > TRIM_NEAR:
            continue
        first = last = nearest
        while first > 0 and row[first - 1]:
            first -= 1
        while last < len(row) - 1 and row[last + 1]:
            last += 1
        low[index], high[index] = offsets[first], offsets[last]
    return points, normals, low, high


def trim_ring(mask, centre, expected_width=None, details=None):
    """An even ribbon from SAM's mask, with the stretches where it leaked repaired.

    On pale fabric SAM outlines the bead over most of a line and spills into the
    band or the mesh beside it over the rest, which made the whole answer unusable.
    Here the mask's two edges are read along the rough line; a stretch whose width
    or edge position departs from the line's own steady course is replaced by that
    course. ``expected_width`` (the width SAM gave the frame's clean lines) is used
    instead of the line's own median when SAM took something wider over most of it.

    ``mask`` may be a list of masks: several answers for the same line (asked with
    different tiles and points). They spill in different places, so each stretch
    takes its edges from the answers that are steady there.

    Returns ``(ring, seen, widths)`` -- ``seen`` is the share of the line where
    SAM's own edges were kept -- or None when there is nothing steady to go on.
    ``details``, a dict, receives the samples: ``points``, ``normals``, the two
    edges ``low`` / ``high`` (signed distance along the normal) and which of them
    are SAM's own (``low_seen`` / ``high_seen``).
    """
    from scipy.ndimage import median_filter
    masks = mask if isinstance(mask, (list, tuple)) else [mask]
    read = [_edges(each, centre) for each in masks]
    points, normals = read[0][0], read[0][1]
    low = np.array([each[2] for each in read])            # answers x samples
    high = np.array([each[3] for each in read])
    width = high - low
    reached = np.isfinite(width) & (low > -TRIM_REACH) & (high < TRIM_REACH)
    if reached.any(axis=0).mean() < 0.3:
        return None
    usual = float(expected_width) if expected_width else float(np.median(width[reached]))
    good = reached & (np.abs(width - usual) <= max(TRIM_TOLERANCE, 0.25 * usual))
    samples = np.arange(width.shape[1])
    low_good = high_good = good

    def merged(values, steady):
        """Per sample, the middle of the answers that are steady there."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)      # samples no answer covers
            return np.nanmedian(np.where(steady, values, np.nan), axis=0)

    def course(values, steady):
        kept = np.flatnonzero(steady.any(axis=0))
        if len(kept) < 10:
            return None
        return np.interp(samples, kept, median_filter(merged(values, steady)[kept], 31, mode='nearest'))

    # The steady course comes from stretches where both edges are right. After
    # that each edge is judged alone: a spill moves one edge and leaves the other.
    for _ in range(2):
        low_course, high_course = course(low, low_good), course(high, high_good)
        if low_course is None or high_course is None:
            return None
        low_good = reached & (np.abs(low - low_course) <= TRIM_TOLERANCE)
        high_good = reached & (np.abs(high - high_course) <= TRIM_TOLERANCE)
    low_seen, high_seen = low_good.any(axis=0), high_good.any(axis=0)
    low = np.where(low_seen, merged(low, low_good), low_course)
    high = np.where(high_seen, merged(high, high_good), high_course)
    if details is not None:
        details.update(points=points, normals=normals, low=low, high=high,
                       low_seen=low_seen, high_seen=high_seen)
    ring = np.vstack([points + normals * low[:, None], (points + normals * high[:, None])[::-1]])
    return ring, float((low_seen.mean() + high_seen.mean()) / 2.0), high - low

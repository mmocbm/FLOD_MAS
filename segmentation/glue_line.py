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
    'background': 'white',       # 'white' table as first built for, or 'any' table
    'bead_model': 'edges',       # how the bead shows: 'edges', 'line', or 'auto' by fabric colour
    'coloured_saturation': 25.0,  # with 'auto': fabric more saturated than this uses 'line'
    'roi_model': '',             # ONNX model marking the glue-bearing panel; '' = not used
    'roi_strict': False,         # the glue line is always inside the region: never search outside it
    'roi_input_width': 768,      # width the region model was trained at
    'roi_mirror_pass': False,    # average a mirrored second pass of the region model
    'roi_grow_px': 25.0,         # how far the marked region is widened, to be safe at its edge
    'cardboard_saturation': 60.0,  # warm surfaces more saturated than this are cardboard or tape
    'sam_refine': False,         # with roi_guided: outline each found line with SAM (cloud API)
    'sam_prep': {},              # tile preparation before SAM; see sam_refine.PREP_DEFAULTS
    'sam_max_width_px': 80.0,    # a SAM mask wider than this is not a bead and is refused
    'enhance_bead': False,       # with roi_guided: search on the bead-enhanced image
    'roi_guided': False,         # with roi_model: look for one line along each panel's curved side
    'guided_skip_frame_filters': False,   # with roi_guided: skip whole-frame filters it does not use
    'corridor_near_px': -55.0,   # the bead lies between these distances inside the
    'corridor_far_px': 170.0,    # curved side of its panel
}

# The known wave shape used as a condition (see glue_shape.py). Off by default.
SHAPE_DEFAULTS = {
    'enabled': False,
    'template': 'all',           # a template name from glue_shapes.json, or 'all'
    'min_scale': 0.90,           # how much smaller or larger than the template the
    'max_scale': 1.10,           # line may be in this camera's pixels
    'tolerance_px': 7.0,         # how close evidence must lie to count as on the line
    'min_seen_share': 0.5,       # share of the whole line that must be seen
    'flex_px': 12.0,             # how far the line may bend away from the pure shape
    'min_piece_px': 300.0,       # shortest stretch of evidence that may suggest a line
    'follow_px': 30.0,           # corridor either side of the shape in which the bead is followed
}

LINE_SIGMA = 2.5        # half-thickness of the lines looked for, working pixels
LINE_ALONG = 10.0       # how far a line must persist to count, working pixels
LINE_DIRECTIONS = 12
SAMPLE_STEP = 4.0       # spacing of points along a centreline, working pixels
MIN_PIECE_PX = 40.0     # a centreline piece shorter than this is noise
BEAD_LIFT = 1.0         # gray levels the bead must stand above the fabric beside it
TABLE_BORDER_SHARE = 0.42  # share of the image border a white table occupies, at least
BEAD_CLEARANCE = 10.0   # how far outside each edge that fabric is sampled, working pixels
GAP_RELAX = 0.6         # threshold share allowed inside a bridged gap, where the
                        # edges faded but are probably still there
GAP_MARGIN = 3.0        # how far outside the joined widths the edges may sit, working px
GAP_SEEN_SHARE = 0.6    # share of a gap that must be measured before it is believed


def _settings(settings):
    given = dict(settings.get('glue_line') or {})
    shape = dict(SHAPE_DEFAULTS)
    shape.update(given.pop('shape_prior', None) or {})
    merged = dict(DEFAULTS)
    merged.update(given)
    merged['shape_prior'] = shape
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


def fabric_mask_any(image, max_saturation, min_gray):
    """Fabric of any colour on any table: what is left once the dark background,
    the cardboard, the tape and a white table are taken away.

    ``fabric_mask`` assumes a white table and finds the level between table and
    fabric; on a dark table that level falls inside the fabric and cuts it apart.
    Here a white table is removed only when there is one: it is the brighter of
    the two pale classes and it runs along the image border.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hue, saturation = hsv[:, :, 0], hsv[:, :, 1]
    warm = (hue >= 8) & (hue <= 40) & (saturation > max_saturation)   # cardboard, tape, skin
    candidate = (gray > min_gray) & ~warm
    pale = candidate & (saturation < max_saturation)
    values = gray[pale]
    if values.size > 100:
        level, _ = cv2.threshold(values.reshape(-1, 1), 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bright = pale & (gray >= level)
        border = np.r_[bright[0], bright[-1], bright[:, 0], bright[:, -1]]
        if border.mean() > TABLE_BORDER_SHARE:
            candidate &= ~bright
    mask = candidate.astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    keep = np.zeros_like(mask)
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] > 0.01 * mask.size:
            keep[labels == index] = 1
    return keep


SAMPLE_STEP_PANEL = 4.0      # spacing of samples along a panel side, working pixels
ROI_INPUT = (768, 576)       # the size the region model was trained at (width, height)
ROI_MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255.0
ROI_STD = np.array([0.229, 0.224, 0.225], np.float32) * 255.0
_roi_nets = {}


def _region_logits(path, blob):
    """Run a region model on one prepared image. ONNX Runtime when it is
    installed (needed for models exported with a free input size); otherwise
    OpenCV's own ONNX reader, which handles fixed-size models."""
    key = str(path)
    if key not in _roi_nets:
        if not path.is_file():
            raise FileNotFoundError(f'Glue region model not found: {path}')
        if path.suffix.lower() == '.pt':
            _roi_nets[key] = _torch_region_net(path)
            return _roi_nets[key](blob)
        try:
            import onnxruntime
            session = onnxruntime.InferenceSession(key, providers=['CPUExecutionProvider'])
            name = session.get_inputs()[0].name
            _roi_nets[key] = lambda data: session.run(None, {name: data})[0]
        except ImportError:
            net = cv2.dnn.readNetFromONNX(key)

            def run(data, net=net):
                net.setInput(data)
                return net.forward()
            _roi_nets[key] = run
    return _roi_nets[key](blob)


def _torch_region_net(path):
    """A region model given as its PyTorch checkpoint (``config`` + ``model``, as
    the training script saves it): run with torch, on the GPU when there is one.
    The checkpoint is unpickled, so it must be a file of your own."""
    import segmentation_models_pytorch as smp
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(str(path), map_location=device, weights_only=False)
    net = smp.Unet(checkpoint['config']['encoder'], encoder_weights=None, classes=1)
    net.load_state_dict(checkpoint['model'])
    net.to(device).eval()

    def run(data):
        with torch.inference_mode(), torch.autocast(device.type, enabled=device.type == 'cuda'):
            return net(torch.from_numpy(data).to(device)).float().cpu().numpy()
    return run


def glue_region(image, model_path, grow_px=0.0, input_width=ROI_INPUT[0], mirror=False):
    """The part of each panel that carries the glue line, from the trained model.

    Glue is applied only along one band of the garment: the flat panel between its
    straight edge and the curved edge of the cup. The model marks that band on the
    whole frame, whatever the fabric colour or the table, so the search does not
    have to consider mesh, cups, folds or the background at all.

    ``input_width`` is the width the model was trained at; the frame is scaled to
    it and padded to a multiple of 32. ``mirror`` averages in a second, mirrored
    pass, as some models were validated with.
    """
    from pathlib import Path
    path = Path(model_path)
    if not path.is_absolute():
        local = Path(__file__).resolve().parent / path
        path = local if local.is_file() else Path(__file__).resolve().parents[1] / path
    height, width = image.shape[:2]
    scaled_height = int(round(height * input_width / width))
    small = cv2.resize(image, (int(input_width), scaled_height), interpolation=cv2.INTER_AREA)
    padded = cv2.copyMakeBorder(small, 0, (32 - scaled_height % 32) % 32,
                                0, (32 - int(input_width) % 32) % 32, cv2.BORDER_REFLECT)
    blob = np.ascontiguousarray(
        ((padded[:, :, ::-1].astype(np.float32) - ROI_MEAN) / ROI_STD).transpose(2, 0, 1)[None])
    logits = _region_logits(path, blob)[0, 0]
    if mirror:
        flipped = _region_logits(path, np.ascontiguousarray(blob[..., ::-1]))[0, 0][:, ::-1]
        probability = (1 / (1 + np.exp(-logits)) + 1 / (1 + np.exp(-flipped))) / 2
        mask = probability >= 0.5
    else:
        mask = logits > 0
    mask = mask[:scaled_height, :int(input_width)].astype(np.uint8)
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    keep = np.zeros_like(mask)
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] > 0.005 * mask.size:      # specks are not panels
            keep[labels == index] = 1
    if grow_px > 0:
        size = int(2 * round(grow_px) + 1)
        keep = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    return keep


def curved_sides(component, step=SAMPLE_STEP_PANEL, min_bend=15.0):
    """The curved long sides of one glue region, each ordered along the panel, with
    the unit direction that points from that side into the region.

    A panel's region has one straight side (the cut edge of the panel) and one
    curved side (the edge of the cup); the glue line runs just inside the curved
    one. Two panels lying back to back give one region with two curved sides.
    """
    ys, xs = np.nonzero(component)
    points = np.column_stack([xs, ys]).astype(np.float64)
    mean = points.mean(axis=0)
    _, _, axes = np.linalg.svd((points - mean)[::50], full_matrices=False)
    along_axis, cross_axis = axes[0], axes[1]
    along = (points - mean) @ along_axis
    across = (points - mean) @ cross_axis
    bins = np.round((along - along.min()) / step).astype(int)
    count = bins.max() + 1
    low = np.full(count, np.inf)
    high = np.full(count, -np.inf)
    np.minimum.at(low, bins, across)
    np.maximum.at(high, bins, across)
    position = along.min() + np.arange(count) * step
    keep = slice(int(0.05 * count), int(0.95 * count))      # the ends are corners, not sides
    def bend_of(side):
        values = side[keep]
        straight = np.polyval(np.polyfit(position[keep], values, 1), position[keep])
        return float(np.abs(values - straight).mean())
    # A cut edge that is a little ragged must not pass for the curved side: a
    # side counts as curved only if it bends about as much as the other does.
    most = max(bend_of(low), bend_of(high))
    sides = []
    for side, inward in ((low, 1.0), (high, -1.0)):
        values = side[keep]
        if bend_of(side) >= max(min_bend, 0.6 * most):
            window = 9
            padded = np.pad(values, window // 2, mode='edge')
            values = np.convolve(padded, np.ones(window) / window, 'valid')
            sides.append((mean + np.outer(position[keep], along_axis) + np.outer(values, cross_axis),
                          inward * cross_axis))
    return sides


def cardboard_mask(image, max_saturation, grow_px=9):
    """Cardboard, tape and skin: warm, saturated surfaces that are never fabric."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Bright as well as warm: sheer dark mesh lying over the board, or over a
    # panel, takes on its colour but stays dark, and that is fabric.
    warm = ((hsv[:, :, 0] >= 8) & (hsv[:, :, 0] <= 40) & (hsv[:, :, 1] > max_saturation)
            & (hsv[:, :, 2] > CARDBOARD_MIN_VALUE)).astype(np.uint8)
    warm = cv2.morphologyEx(warm, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))   # not mesh speckle
    return cv2.dilate(warm, np.ones((grow_px, grow_px), np.uint8))


def find_fabric(image, options):
    """The fabric mask for the configured table: white (the original) or any."""
    finder = fabric_mask_any if options.get('background', 'white') == 'any' else fabric_mask
    return finder(image, options['max_saturation'], options['min_fabric_gray'])


def flatten(gray):
    """Remove the slow lighting gradient, leaving local detail around zero."""
    gray = gray.astype(np.float32)
    return gray - cv2.GaussianBlur(gray, (0, 0), 25)


def line_strength(flat, sigma=None):
    """Long thin lines in any direction: dark strength, bright strength, and the
    index of the direction in which the dark line at each pixel runs."""
    sigma = LINE_SIGMA if sigma is None else float(sigma)
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
        kernel = ((across ** 2 / sigma ** 2 - 1)
                  * np.exp(-across ** 2 / (2 * sigma ** 2))
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


LINE_MODEL_SIGMA = 4.5     # half-thickness of a bead seen as one line, working pixels
PROFILE_REACH = 22         # how far across a line its brightness profile is read
CARDBOARD_MIN_VALUE = 100  # cardboard and tape are at least this bright (HSV value)
OUTSIDE_PROBE = 30.0       # how far beyond a panel side its neighbour is sampled, px
STRICT_NEAR = 6.0          # with a trusted region: the bead is at least this far inside its edge, px
REGION_SLACK = 60.0        # how far past the model's outline the bead may still be, px
STEP_LIMIT = 0.8           # a line whose two sides differ by more than this share of
                           # its own depth is an edge between two materials, not a bead


def choose_bead_model(image, fabric, options):
    """The bead model for this fabric colour.

    On white or pale fabric a clear bead shows as two dark edge lines with a
    bright line between ('edges'). On coloured or dark fabric it shows as one
    line, darker than the fabric and bright where it catches the light ('line').
    """
    model = options.get('bead_model', 'edges')
    if model != 'auto':
        return model
    saturation = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 1][fabric > 0]
    if saturation.size == 0:
        return 'edges'
    return 'line' if float(np.median(saturation)) > options['coloured_saturation'] else 'edges'


def line_points(ridge, region, low):
    """Centre points of single lines of bead thickness."""
    crest = ridge >= 0.8 * cv2.dilate(ridge, np.ones((9, 9), np.uint8))
    ys, xs = np.nonzero(skeletonize((ridge > low) & crest & (region > 0)))
    return np.column_stack([xs, ys]).astype(np.float64)


def line_profile(smooth, path):
    """Across a line at every sample: its width at half depth, its depth, and how
    much the brightness differs between its two sides."""
    normal = _normals(path)
    offsets = np.arange(-PROFILE_REACH, PROFILE_REACH + 1, dtype=np.float64)
    profile = _sample(smooth, path[:, None, :] + normal[:, None, :] * offsets[None, :, None])
    left = profile[:, :6].mean(axis=1)
    right = profile[:, -6:].mean(axis=1)
    deviation = profile - ((left + right) / 2.0)[:, None]
    middle = len(offsets) // 2
    core = deviation[:, middle - 3:middle + 4]
    peak = np.where(np.abs(core.max(axis=1)) >= np.abs(core.min(axis=1)),
                    core.max(axis=1), core.min(axis=1))
    above = deviation * np.sign(peak)[:, None] > np.abs(peak)[:, None] / 2.0
    # Width: the run of samples above half depth that contains the centre.
    widths = np.zeros(len(path))
    for index in range(len(path)):
        row = above[index]
        if not row[middle - 2:middle + 3].any():
            continue
        centre = middle + int(np.argmax(row[middle - 2:middle + 3])) - 2
        start = centre
        while start > 0 and row[start - 1]:
            start -= 1
        stop = centre
        while stop < len(row) - 1 and row[stop + 1]:
            stop += 1
        widths[index] = stop - start + 1
    return widths, np.abs(peak), np.abs(left - right)


LINE_ELEMENT = 25          # across-the-line size of the morphological element, px
ALONG_AVERAGE = 41         # a bead persists along its length; weave and moire do not
FLANK_NEAR, FLANK_FAR = 3, 22   # where a bead's dark sides lie from its centre, px
PROFILE_HALF = 30          # half-height of the profile read across the bead, px
PROFILE_STACK = 15         # samples of bead averaged together before measuring (x step)
GLARE_STEP = 25.0          # gray levels above the bead's usual brightness that count as shine


ENHANCE_SCALES = (9, 17, 31)   # element sizes for the bead enhancement, px


def enhance_bead(gray):
    """Make thin bead-sized detail stand out, in lit and unlit stretches alike.

    Edge-preserving denoising, then a multi-scale morphological contrast
    stretch: at each scale, what is brighter than its surroundings (top-hat) is
    added and what is darker (black-hat) is taken away. The bead's bright core
    and dark sides are pushed apart while flat fabric and slow lighting changes
    are left alone. Of the enhancements tried on the sample frames (CLAHE,
    illumination flattening, local contrast normalisation, this), this one
    roughly doubled the bead's visibility where it is not shiny and brought
    shiny and unlit stretches closest together.
    """
    smooth = cv2.medianBlur(cv2.bilateralFilter(gray, 9, 18, 5), 5)
    result = smooth.astype(np.float32)
    for size in ENHANCE_SCALES:
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        result += cv2.morphologyEx(smooth, cv2.MORPH_TOPHAT, element)
        result -= cv2.morphologyEx(smooth, cv2.MORPH_BLACKHAT, element)
    low, high = np.percentile(result[::4, ::4], [0.5, 99.5])
    return np.clip((result - low) / max(high - low, 1e-6) * 255.0, 0, 255).astype(np.uint8)


def straighten(image, side, inward, near, far):
    """The corridor beside a panel side, resampled so the side is a straight row.

    Rows are distance into the panel (``near``..``far``), columns are position
    along the side, one pixel each. In this picture the bead is a near-horizontal
    line, which is what makes one-directional filters usable on a curved bead.
    """
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(side, axis=0), axis=1))]
    positions = np.arange(0.0, distance[-1], 1.0)
    base = np.column_stack([np.interp(positions, distance, side[:, 0]),
                            np.interp(positions, distance, side[:, 1])])
    offsets = np.arange(near, far + 1.0, dtype=np.float32)
    grid_x = (base[:, 0][None, :] + inward[0] * offsets[:, None]).astype(np.float32)
    grid_y = (base[:, 1][None, :] + inward[1] * offsets[:, None]).astype(np.float32)
    strip = cv2.remap(image, grid_x, grid_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return strip, base, offsets


def corridor_filters(strip):
    """Bead evidence in a straightened corridor: (bright core, dark sides).

    1. Denoise while keeping edges (bilateral, then median): removes fabric
       weave and compression blocks, which are as strong as a faint bead.
    2. Morphological top-hat and black-hat with an element across the line:
       keeps only things thinner than the element -- the bead's bright core and
       the dark line along each of its sides -- whatever the fabric's brightness.
    3. Average along the line: the bead adds up, texture cancels.
    """
    smooth = cv2.medianBlur(cv2.bilateralFilter(strip, 9, 18, 5), 5)
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (1, LINE_ELEMENT))
    bright = cv2.morphologyEx(smooth, cv2.MORPH_TOPHAT, element).astype(np.float32)
    dark = cv2.morphologyEx(smooth, cv2.MORPH_BLACKHAT, element).astype(np.float32)
    return bright, dark


def quantise(values, levels=4):
    """Reduce a picture to a few brightness levels (k-means), darkest = 0.

    Used to read a bead's width as a count of pixels of one level instead of
    from a threshold that would have to be tuned per fabric.
    """
    samples = values.reshape(-1, 1).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, _, centres = cv2.kmeans(samples[::5], levels, None, criteria, 2, cv2.KMEANS_PP_CENTERS)
    centres = np.sort(centres.ravel())
    return np.abs(values[..., None] - centres[None, None, :]).argmin(axis=2), centres


def corridor_bead(gray, side, inward, allowed, near, far, step=4):
    """The bead in one panel's corridor: its path, its width and where it was seen.

    Returns ``(path, widths, seen)`` in image pixels at ``step`` spacing along the
    side, or None when there is nothing bead-like in the corridor.
    """
    strip, base, offsets = straighten(gray, side, inward, near, far)
    if strip.shape[1] < 10 * step:
        return None
    valid = straighten(allowed, side, inward, near, far)[0] > 0
    bright, dark = corridor_filters(strip)
    box = (ALONG_AVERAGE, 1)
    bright_along, dark_along = cv2.blur(bright, box), cv2.blur(dark, box)
    # A bead is a bright core with a dark line on each side of it. Each side is
    # the strongest dark line a little way above or below the centre.
    reach = np.ones((FLANK_FAR - FLANK_NEAR + 1, 1), np.uint8)
    shift = (FLANK_FAR + FLANK_NEAR) // 2
    spread = cv2.dilate(dark_along, reach)
    above = np.roll(spread, shift, axis=0)
    below = np.roll(spread, -shift, axis=0)
    noise_bright = float(np.median(bright_along[valid])) + 1e-3
    noise_dark = float(np.median(dark_along[valid])) + 1e-3
    score = (np.minimum(bright_along / (4.0 * noise_bright), 4.0)
             + np.minimum(np.minimum(above, below) / (4.0 * noise_dark), 4.0))
    score = np.where(valid, score, -4.0)
    columns = np.arange(0, strip.shape[1], step)
    track = _best_track(score[:, columns].T, offsets.astype(np.float64), 0.6)
    rows = np.clip(np.round(track - offsets[0]).astype(int), 0, strip.shape[0] - 1)

    # Width. The bead's two sides are the dark line on each side of its centre.
    # One column is too noisy to find them on, so the corridor is first lined
    # up on the track and averaged over a short stretch of bead: the sides add
    # up and the weave does not. The width is the distance between the two
    # dark lines, found to a fraction of a pixel.
    smooth = cv2.medianBlur(cv2.bilateralFilter(strip, 9, 18, 5), 5).astype(np.float32)
    span = np.arange(-PROFILE_HALF, PROFILE_HALF + 1)
    take = np.clip(rows[:, None] + span[None, :], 0, strip.shape[0] - 1)
    profiles = smooth[take, columns[:, None]]
    profiles = cv2.blur(profiles, (1, PROFILE_STACK))            # along the bead
    profiles -= np.median(profiles, axis=1, keepdims=True)
    middle = PROFILE_HALF
    widths = np.full(len(columns), np.nan)
    upper = profiles[:, middle - FLANK_FAR:middle - FLANK_NEAR + 1]
    lower = profiles[:, middle + FLANK_NEAR:middle + FLANK_FAR + 1]
    top = upper.argmin(axis=1)
    bottom = lower.argmin(axis=1)
    index = np.arange(len(columns))
    depth = np.minimum(-upper[index, top], -lower[index, bottom])
    noise = float(np.median(np.abs(profiles))) + 1e-3

    def refine(values, at):
        """Sub-pixel position of a minimum from its two neighbours."""
        left = values[index, np.clip(at - 1, 0, values.shape[1] - 1)]
        right = values[index, np.clip(at + 1, 0, values.shape[1] - 1)]
        centre = values[index, at]
        bend = left - 2 * centre + right
        return at + np.where(np.abs(bend) > 1e-6, 0.5 * (left - right) / np.where(bend == 0, 1, bend), 0.0)
    distance = ((middle + FLANK_NEAR + refine(lower, bottom))
                - (middle - FLANK_FAR + refine(upper, top)))
    # A side sitting at the very end of the search range was not found, only cut off.
    interior = ((top > 0) & (top < upper.shape[1] - 1)
                & (bottom > 0) & (bottom < lower.shape[1] - 1))
    sides_found = interior & (depth > 1.5 * noise)
    widths[sides_found] = distance[sides_found]

    # Shine. Where the bead reflects the lamp its centre is far brighter than
    # the same bead elsewhere. Those stretches are marked: the lamp's reflection
    # and the camera's own sharpening halo around it move the apparent sides.
    core = smooth[rows, columns]
    glare = core > np.median(core) + GLARE_STEP

    # Seen: the bead's own evidence is there under the track. Whether its two
    # sides could also be told apart, for a width, is a separate matter.
    seen = score[rows, columns] > 1.0
    measured = np.isfinite(widths) & seen
    if measured.sum() < 5:
        return None
    path = base[columns] + inward[None, :] * track[:, None]
    positions = np.arange(len(columns))
    widths = median_filter(np.interp(positions, positions[measured], widths[measured]),
                           9, mode='nearest')
    return path, widths, seen, glare, measured


def follow_line(path, ridge, low, reach=30.0, stiffness=0.6, pull=0.002):
    """As ``follow_bead``, for a bead seen as one line: stay on the ridge."""
    normal = _normals(path)
    offsets = np.arange(-np.ceil(reach), np.ceil(reach) + 1, dtype=np.float64)
    score = np.minimum(_sample(ridge, path[:, None, :] + normal[:, None, :] * offsets[None, :, None])
                       / max(low, 1e-6), 4.0)
    score -= pull * offsets[None, :] ** 2
    return path + normal * _best_track(score, offsets, stiffness)[:, None]


def _best_track(score, offsets, stiffness):
    """The smoothest sideways track through a score table (samples x offsets)."""
    count, states = score.shape
    best = np.zeros((count, states))
    back = np.zeros((count, states), np.int32)
    best[0] = score[0]
    steps = (-2, -1, 0, 1, 2)
    for index in range(1, count):
        options = np.full((len(steps), states), -np.inf)
        for row, step in enumerate(steps):
            source = np.arange(states) - step
            valid = (source >= 0) & (source < states)
            options[row, valid] = best[index - 1, source[valid]] - stiffness * step * step
        choice = options.argmax(axis=0)
        back[index] = np.arange(states) - np.asarray(steps)[choice]
        best[index] = options[choice, np.arange(states)] + score[index]
    state = int(best[-1].argmax())
    shift = np.zeros(count)
    for index in range(count - 1, -1, -1):
        shift[index] = offsets[state]
        state = back[index, state]
    return shift


def follow_bead(path, typical_width, evidence, reach=30.0, stiffness=0.6, pull=0.002):
    """Move a roughly placed line onto the bead itself, sample by sample.

    The known shape puts the line within a few tens of pixels of the bead. Inside
    that corridor the bead is the place where both edges and the bright centre are
    present together. The sideways position is chosen for the whole line at once:
    it follows the bead wherever the bead shows, may not jump between neighbouring
    samples, and where nothing shows it carries on smoothly instead of wandering.
    """
    dark, bright, region, low, low_bright, _, _ = evidence
    normal = _normals(path)
    offsets = np.arange(-np.ceil(reach), np.ceil(reach) + 1, dtype=np.float64)
    half = typical_width / 2.0

    def read(values, shift):
        grid = path[:, None, :] + normal[:, None, :] * (offsets[None, :, None] + shift)
        return _sample(values, grid)
    edges = np.minimum(read(dark, -half), read(dark, half)) / max(low, 1e-6)
    centre = read(bright, 0.0) / max(low_bright, 1e-6)
    # Both edges are required; the bright centre adds to it. Capped so that one
    # glare spot cannot outweigh a long, steady stretch of bead.
    score = np.minimum(edges, 3.0) * (edges > GAP_RELAX) + 0.5 * np.minimum(centre, 3.0)
    score -= pull * offsets[None, :] ** 2          # stay near the shape unless shown otherwise

    count, states = score.shape
    best = np.zeros((count, states))
    back = np.zeros((count, states), np.int32)
    best[0] = score[0]
    steps = (-2, -1, 0, 1, 2)                      # sideways movement between samples
    for index in range(1, count):
        options = np.full((len(steps), states), -np.inf)
        for row, step in enumerate(steps):
            source = np.arange(states) - step
            valid = (source >= 0) & (source < states)
            options[row, valid] = best[index - 1, source[valid]] - stiffness * step * step
        choice = options.argmax(axis=0)
        back[index] = np.arange(states) - np.asarray(steps)[choice]
        best[index] = options[choice, np.arange(states)] + score[index]
    state = int(best[-1].argmax())
    shift = np.zeros(count)
    for index in range(count - 1, -1, -1):
        shift[index] = offsets[state]
        state = back[index, state]
    return path + normal * shift[:, None]


def path_widths(path, centres, spans, tolerance, evidence):
    """Width at every sample of a known line, and whether it was measured there.

    Where bead evidence lies on the line its own width is used. Elsewhere the two
    edges are looked for exactly where a bead of this line's width would have
    them, at a relaxed threshold: knowing where the line runs is what makes that
    safe. What is still not found takes the width of its measured neighbours and
    is reported as not seen.
    """
    from scipy.spatial import cKDTree
    widths = np.full(len(path), np.nan)
    if len(centres):
        near = cKDTree(centres).query_ball_point(path, tolerance)
        for index, members in enumerate(near):
            if members:
                widths[index] = np.median(spans[members])
    if not np.isfinite(widths).any():
        return np.zeros(len(path)), np.zeros(len(path), bool)
    typical = float(np.nanmedian(widths))
    # Evidence that happens to lie on the line but belongs to something else --
    # a mesh edge beside a fold -- gives itself away by its width.
    widths[np.abs(widths - typical) > 0.4 * typical] = np.nan

    dark, bright, region, low, low_bright, _, _ = evidence
    missing = ~np.isfinite(widths)
    if missing.any():
        normal = _normals(path)
        reach = int(np.ceil(typical / 2.0 + GAP_MARGIN + 3))
        offsets = np.arange(-reach, reach + 1, dtype=np.float64)
        grid = path[missing][:, None, :] + normal[missing][:, None, :] * offsets[None, :, None]
        depth = _sample(dark, grid)
        window = np.abs(np.abs(offsets) - typical / 2.0) <= GAP_MARGIN + 2
        left = np.where(window & (offsets < 0), depth, -1.0)
        right = np.where(window & (offsets > 0), depth, -1.0)
        left_at, right_at = left.argmax(axis=1), right.argmax(axis=1)
        rows = np.arange(len(grid))
        middle = (offsets[left_at] + offsets[right_at]) / 2.0
        gloss = _sample(bright, (path[missing] + normal[missing] * middle[:, None])[:, None, :])[:, 0]
        on_fabric = _sample(region.astype(np.float32), path[missing][:, None, :])[:, 0] > 0.5
        found = ((left[rows, left_at] > low * GAP_RELAX) & (right[rows, right_at] > low * GAP_RELAX)
                 & (gloss > low_bright * GAP_RELAX) & on_fabric)
        measured = np.full(len(grid), np.nan)
        measured[found] = (offsets[right_at] - offsets[left_at])[found]
        widths[missing] = measured
    seen = np.isfinite(widths)
    positions = np.arange(len(path))
    widths = np.interp(positions, positions[seen], widths[seen])
    return median_filter(widths, 9, mode='nearest'), seen


def _normals(curve):
    tangent = np.gradient(curve, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-6)
    return np.column_stack([-tangent[:, 1], tangent[:, 0]])


class GlueLineSegmenter:
    def __init__(self, settings):
        self.settings = settings
        self.options = _settings(settings)
        self.last_input = None   # the enhanced line image, for audit
        self.last_details = []   # per line, panel-guided path only: shine and width figures
        self.region_override = None   # optional mask to use as the glue region

    def segment(self, frame):
        options = self.options
        scale = float(options['working_scale'])
        height, width = frame.shape[:2]
        self._frame = frame
        small = (frame if scale == 1.0 else
                 cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))
        if self.region_override is not None:
            # A region supplied by the caller (a tool that lets the region be chosen
            # by hand, or computed on a different picture than the one searched).
            fabric = (self.region_override > 0).astype(np.uint8)
            if fabric.shape != small.shape[:2]:
                fabric = cv2.resize(fabric, (small.shape[1], small.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
            self._cardboard = np.zeros_like(fabric)
            interior = fabric
        elif options['roi_model']:
            # The model already marks only where glue can be; its outline is not
            # a fabric edge to stay clear of, so nothing is trimmed from it.
            fabric = glue_region(small, options['roi_model'],
                                 0.0 if options['roi_guided'] else options['roi_grow_px'] * scale,
                                 options['roi_input_width'], options['roi_mirror_pass'])
            # The model's outline is approximate and can run over onto the board
            # the panels lie on. Cardboard is never part of the glue region.
            self._cardboard = cardboard_mask(small, options['cardboard_saturation'])
            fabric = fabric & (1 - self._cardboard)
            fabric = cv2.morphologyEx(fabric, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
            interior = fabric
        else:
            fabric = find_fabric(small, options)
            margin = max(3, int(round(options['fabric_margin_px'] * scale))) | 1
            interior = cv2.erode(fabric, np.ones((margin, margin), np.uint8))
        guided = bool((options['roi_model'] or self.region_override is not None)
                      and options['roi_guided'])
        if guided and options['guided_skip_frame_filters']:
            # The region-guided search reads the picture itself along each region's
            # side; the whole-frame line filters below are not used by it (they are
            # most of this detector's time). No audit image is made.
            self.last_input = None
            if not interior.any():
                return SegmentationResult((), (width, height))
            lines = self._panel_lines(small, None, None, None, None, None, fabric,
                                      scale, (width, height))
            return SegmentationResult(tuple(lines), (width, height))
        flat = flatten(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
        smooth = cv2.GaussianBlur(flat, (0, 0), 1.5)
        dark, bright, direction = line_strength(flat)
        self.last_input = self._audit_image(dark, bright, interior)
        if not interior.any():
            return SegmentationResult((), (width, height))

        if (options['roi_model'] or self.region_override is not None) and options['roi_guided']:
            lines = self._panel_lines(small, flat, smooth, dark, bright, direction, fabric,
                                      scale, (width, height))
            return SegmentationResult(tuple(lines), (width, height))

        if choose_bead_model(small, fabric, options) == 'line':
            # One line is far too common a thing to accept on its own; this model
            # is only used with the known shape as a condition.
            if not options['shape_prior']['enabled']:
                return SegmentationResult((), (width, height))
            lines = self._shaped_single_lines(flat, smooth, fabric, interior, scale,
                                              (width, height))
            return SegmentationResult(tuple(lines), (width, height))

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
        if options['shape_prior']['enabled']:
            lines = self._shaped_lines(pieces, centres, spans, fabric,
                                       (dark, bright, interior, low, low_bright, 0.0, 0.0),
                                       scale, (width, height))
            return SegmentationResult(tuple(lines), (width, height))
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

    def _shaped_lines(self, pieces, centres, spans, fabric, evidence, scale, frame_size):
        """Whole glue lines: the known wave laid where the evidence supports it."""
        from . import glue_shape
        prior = self.options['shape_prior']
        names = None if prior['template'] in ('', 'all') else [prior['template']]
        templates = glue_shape.load_templates(names)
        width, height = frame_size

        def on_fabric(points):
            columns = np.clip((points[:, 0] * scale).astype(int), 0, fabric.shape[1] - 1)
            rows = np.clip((points[:, 1] * scale).astype(int), 0, fabric.shape[0] - 1)
            return float(fabric[rows, columns].mean())

        # Templates are in full-frame pixels, so the evidence is taken there too.
        placements = glue_shape.find_lines(
            [points / scale for points, _ in pieces], centres / scale, templates,
            scale_range=(prior['min_scale'], prior['max_scale']),
            tolerance=prior['tolerance_px'], min_seen_share=prior['min_seen_share'],
            min_piece_px=prior['min_piece_px'], flex_px=prior['flex_px'], inside=on_fabric)
        instances = []
        for placement in placements:
            path = placement.path * scale
            widths, seen = path_widths(path, centres, spans,
                                       prior['tolerance_px'] * scale, evidence)
            if not seen.any():
                continue
            # The shape gives the line's course; the bead itself gives its exact
            # position, which is then what the width is measured on.
            path = follow_bead(path, float(np.median(widths[seen])), evidence,
                               reach=prior['follow_px'] * scale)
            widths, seen = path_widths(path, centres, spans,
                                       prior['tolerance_px'] * scale, evidence)
            # A line laid over weave or moire picks up scattered evidence but few
            # places where a bead of one width can actually be measured.
            if seen.mean() < prior['min_seen_share']:
                continue
            half = _normals(path) * (widths / 2.0)[:, None]
            ring = np.clip(np.vstack([path - half, (path + half)[::-1]]) / scale,
                           (0, 0), (width - 1, height - 1))
            # Confidence is the share of the whole line whose width was measured;
            # the rest of the line is where the shape says it must be.
            instances.append(Instance(ring, float(seen.mean()), 'glue line'))
        instances.sort(key=lambda item: (float(item.polygon[:, 0].mean()),
                                         float(item.polygon[:, 1].mean())))
        return instances

    def _panel_lines(self, small, flat, smooth, dark, bright, direction, region, scale,
                     frame_size):
        """One glue line per panel, found from where the panel says it must be.

        The glue region model marks each glue-bearing panel. The bead runs just
        inside the panel's curved side, so it is looked for only in a corridor
        along that side, as the one smooth line of bead thickness running through
        the corridor. Its width is then read with the model for the fabric colour.
        """
        options = self.options
        width, height = frame_size
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if options['enhance_bead']:
            gray = enhance_bead(gray)
            self.last_input = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)   # what the search saw
        self.last_details = []
        # The model's outline of the curved side is only approximate, and the bead
        # can sit right on it, so the search may step a little past the outline --
        # but never onto cardboard.
        if options['roi_strict']:
            # The region is trusted: the glue line is always inside it, so the
            # search may not leave it at all.
            reach = region
        else:
            grow = int(2 * round(REGION_SLACK * scale) + 1)
            reach = cv2.dilate(region, np.ones((grow, grow), np.uint8)) & (1 - self._cardboard)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(region)
        instances = []
        for index in range(1, count):
            if stats[index, cv2.CC_STAT_AREA] < 0.02 * region.size:
                continue                    # a sliver of region is not a panel
            for side, inward in curved_sides(labels == index):
                # The glue runs along the cup, so beyond the right side lies the
                # cup's mesh. Beyond a cut edge lies the board: that side is not it.
                beyond = side - inward[None, :] * (OUTSIDE_PROBE * scale)
                if _sample(self._cardboard.astype(np.float32), beyond[:, None, :]).mean() > 0.3:
                    continue
                found = corridor_bead(gray, side, inward, reach,
                                      (max(options['corridor_near_px'], STRICT_NEAR)
                                       if options['roi_strict'] else options['corridor_near_px']) * scale,
                                      options['corridor_far_px'] * scale)
                if found is None:
                    continue
                path, widths, seen, glare, measured = found
                if options['roi_strict']:
                    # Keep only the stretch that lies inside the region: where two
                    # panels touch, the side found for one runs on past its end.
                    inside = _sample(region.astype(np.float32), path[:, None, :])[:, 0] > 0.5
                    edges = np.flatnonzero(np.diff(np.r_[0, inside.astype(np.int8), 0]))
                    if edges.size == 0:
                        continue
                    runs = edges.reshape(-1, 2)
                    first, last = runs[np.argmax(runs[:, 1] - runs[:, 0])]
                    keep = slice(int(first), int(last))
                    path, widths, seen = path[keep], widths[keep], seen[keep]
                    glare, measured = glare[keep], measured[keep]
                    if len(path) < 10:
                        continue
                length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
                if (length / scale < options['min_length_px']
                        or seen.mean() < options['min_seen_share']):
                    continue
                half = _normals(path) * (widths / 2.0)[:, None]
                ring = np.clip(np.vstack([path - half, (path + half)[::-1]]) / scale,
                               (0, 0), (width - 1, height - 1))
                refined = False
                if options['sam_refine']:
                    # The rough line is only the prompt; SAM draws the outline.
                    from . import sam_refine
                    full = self._frame
                    full_region = (region if scale == 1.0 else cv2.resize(
                        region, (width, height), interpolation=cv2.INTER_NEAREST))
                    sam_mask, _ = sam_refine.refine_line(
                        full, path / scale, options['sam_max_width_px'],
                        region=full_region, prep=options['sam_prep'])
                    exact = sam_refine.bead_ring(sam_mask, path / scale,
                                                 options['sam_max_width_px'])
                    if exact is not None:
                        ring = np.clip(exact, (0, 0), (width - 1, height - 1))
                        refined = True
                instances.append(Instance(ring, float(seen.mean()), 'glue line',
                                          note=None if refined or not options['sam_refine']
                                          else 'unrefined'))
                # For whoever grades: how much of the line is under the lamp's
                # reflection, and the width with and without those stretches.
                clear = measured & ~glare
                self.last_details.append({
                    'x': float(path[:, 0].mean() / scale),
                    'shine_share': float(glare.mean()),
                    'measured_share': float(measured.mean()),
                    'width_clear_px': float(np.median(widths[clear]) / scale) if clear.sum() > 5 else float('nan'),
                    'width_shine_px': (float(np.median(widths[measured & glare]) / scale)
                                       if (measured & glare).sum() > 5 else float('nan')),
                    'width_clear_spread_px': (float(np.subtract(*np.percentile(widths[clear], [75, 25])) / scale)
                                              if clear.sum() > 5 else float('nan')),
                })
        instances.sort(key=lambda item: (float(item.polygon[:, 0].mean()),
                                         float(item.polygon[:, 1].mean())))
        return instances

    def _shaped_single_lines(self, flat, smooth, fabric, interior, scale, frame_size):
        """Glue lines on coloured fabric: single lines of bead thickness that have
        the known shape and the same fabric on both sides."""
        from scipy.spatial import cKDTree
        from . import glue_shape
        options, prior = self.options, self.options['shape_prior']
        width, height = frame_size
        dark, bright, _ = line_strength(flat, LINE_MODEL_SIGMA)
        ridge = np.maximum(dark, bright)
        low = _threshold(ridge, interior, options['line_sensitivity'] * 1.5)
        centres = line_points(ridge, interior, low)
        if len(centres) == 0:
            return []
        pieces = centrelines(centres, np.ones(len(centres)), ridge.shape)
        names = None if prior['template'] in ('', 'all') else [prior['template']]

        def on_fabric(points):
            columns = np.clip((points[:, 0] * scale).astype(int), 0, fabric.shape[1] - 1)
            rows = np.clip((points[:, 1] * scale).astype(int), 0, fabric.shape[0] - 1)
            return float(fabric[rows, columns].mean())

        placements = glue_shape.find_lines(
            [points / scale for points, _ in pieces], centres / scale,
            glue_shape.load_templates(names),
            scale_range=(prior['min_scale'], prior['max_scale']),
            tolerance=prior['tolerance_px'], min_seen_share=prior['min_seen_share'],
            min_piece_px=prior['min_piece_px'], flex_px=prior['flex_px'], inside=on_fabric)
        found = []
        for placement in placements:
            path = follow_line(placement.path * scale, ridge, low,
                               reach=prior['follow_px'] * scale)
            widths, depth, step = line_profile(smooth, path)
            seen = (widths > 0) & (widths <= options['max_width_px'] * scale)
            if seen.mean() < prior['min_seen_share']:
                continue
            # A bead has the same fabric on both sides. The edge of the mesh, or of
            # the band beside the bead, runs alongside with the same shape but
            # separates two different brightnesses.
            stepness = float(np.median(step[seen]) / max(np.median(depth[seen]), 1e-6))
            if stepness > STEP_LIMIT:
                continue
            positions = np.arange(len(path))
            filled = median_filter(np.interp(positions, positions[seen], widths[seen]),
                                   9, mode='nearest')
            found.append((stepness, path, filled, seen))
        # Of lines running side by side, the bead is the one least like an edge.
        found.sort(key=lambda item: item[0])
        kept = []
        for item in found:
            near = [cKDTree(other[1]).query(item[1], distance_upper_bound=90.0 * scale)[0]
                    for other in kept]
            if all(np.isfinite(distance).mean() < 0.5 for distance in near):
                kept.append(item)
        instances = []
        for _, path, widths, seen in kept:
            half = _normals(path) * (widths / 2.0)[:, None]
            ring = np.clip(np.vstack([path - half, (path + half)[::-1]]) / scale,
                           (0, 0), (width - 1, height - 1))
            instances.append(Instance(ring, float(seen.mean()), 'glue line'))
        instances.sort(key=lambda item: (float(item.polygon[:, 0].mean()),
                                         float(item.polygon[:, 1].mean())))
        return instances

    @staticmethod
    def _audit_image(dark, bright, interior):
        """Dark lines in red, the bright bead in green, dimmed outside the fabric."""
        def scaled(values):
            top = float(np.percentile(values, 99.7)) or 1.0
            return np.clip(values / top * 255, 0, 255).astype(np.uint8)
        image = np.dstack([np.zeros(dark.shape, np.uint8), scaled(bright), scaled(dark)])
        image[interior == 0] //= 4
        return image

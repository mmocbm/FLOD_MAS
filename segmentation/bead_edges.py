"""The glue bead's own two edges along a roughly known line, from the camera's pixels.

Seen across, the bead is a band darker than the fabric on both sides of it, about
30 pixels wide, with a thin shiny core and thin black lines beside the core where
the light catches it. On dark fabric the band is 30-50 levels darker than the
fabric; on pink and white fabric 4-7 levels, which is why an outline drawn by a
general segmentation model drifts off it there.

Here the picture is straightened along the line, so that the bead runs level:

1. averaging along the line removes weave and noise but keeps the band (this is
   what makes a 5-level step measurable);
2. a closing across the line removes the thin black lines and an opening the shiny
   core, leaving the band a plain dark stripe (a core lying at the band's edge is
   counted as part of it);
3. the stripe is followed along the line (it wanders some pixels off the rough
   line), and at each sample its two edges are the steepest darkening on one side
   and the steepest brightening on the other;
4. the ends are cut where the stripe is no longer darker than its surroundings.

Nothing is uploaded and no model is used. The sizes are in pixels of a 4608-wide
frame; the bead must be roughly 20-45 pixels wide.
"""
import cv2
import numpy as np

REACH = 70             # half-width of the straightened strip
STEP = 2.0             # spacing of samples along the line
ALONG = 21             # samples averaged along the line (42 px)
THIN_DARK = 7          # black lines beside the core narrower than this are removed
CORE = 21              # bright core narrower than this is removed
CORE_CONTRAST = 8.0    # the core is at least this much brighter than what the opening leaves
BAND = 24              # width of the stripe the tracker looks for
SEARCH = 40            # the stripe's middle lies within this of the rough line
EDGE_NEAR, EDGE_FAR = 4, 33   # an edge lies this far from the stripe's middle
STEADY = 3.0           # an edge within this of its smoothed course counts as seen
EXTEND = 120.0         # the rough line is lengthened by this at each end, then cut back
END_SHARE = 0.3        # ends are cut where darkness falls below this share of the line's


def _lengthened(centre, by):
    """The line with a straight piece added at each end."""
    head = centre[0] - centre[min(15, len(centre) - 1)]
    tail = centre[-1] - centre[max(-16, -len(centre))]
    head = head / max(np.linalg.norm(head), 1e-6)
    tail = tail / max(np.linalg.norm(tail), 1e-6)
    steps = np.arange(by, 0, -STEP)[:, None]
    return np.vstack([centre[0] + head * steps, centre, centre[-1] + tail * steps[::-1]])


def _straighten(gray, centre):
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(centre, axis=0), axis=1))]
    along = np.arange(0.0, arc[-1], STEP)
    points = np.column_stack([np.interp(along, arc, centre[:, axis]) for axis in (0, 1)])
    reach = 10
    window = np.ones(2 * reach + 1) / (2 * reach + 1)
    points = np.column_stack([np.convolve(np.pad(points[:, axis], reach, mode='edge'), window, 'valid')
                              for axis in (0, 1)])
    tangent = np.gradient(points, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-6)
    normals = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    offsets = np.arange(-REACH, REACH + 1, dtype=np.float32)
    columns = (points[:, None, 0] + offsets[None] * normals[:, None, 0]).astype(np.float32)
    rows = (points[:, None, 1] + offsets[None] * normals[:, None, 1]).astype(np.float32)
    strip = cv2.remap(gray, columns, rows, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return strip.astype(np.float32), points, normals, offsets


def _follow(score):
    """The path through the strip, one column per sample, moving at most one
    column per sample, that collects the most score."""
    count, width = score.shape
    total = score.copy()
    came = np.zeros((count, width), np.int8)
    for index in range(1, count):
        before = total[index - 1]
        best = before.copy()
        move = np.zeros(width, np.int8)
        for shift in (-1, 1):
            shifted = np.full(width, -1e9, np.float32)
            if shift < 0:
                shifted[:shift] = before[-shift:]
            else:
                shifted[shift:] = before[:-shift]
            better = shifted > best
            best = np.where(better, shifted, best)
            move = np.where(better, shift, move)
        total[index] += best
        came[index] = move
    path = np.zeros(count, int)
    column = int(total[-1].argmax())
    for index in range(count - 1, -1, -1):
        path[index] = column
        column -= int(came[index][column])
    return path


MIN_WEAVE = 0.6        # grey levels: below this the picture is too soft for the weave to help
STATES = ('none', 'one-sided', 'shiny', 'weak', 'matte')
WEAK_SHARE = 0.6       # a band less dark than this share of the line's usual is 'weak'
CORE_AT_EDGE = 6.0     # a core this close to a band edge makes that edge unreliable
MIN_RUN = 40           # a stretch of band shorter than this (80 px), apart from the rest, is a scrap
STATE_RUN = 11         # states are settled over this many samples (22 px)


def _weave(strip):
    """How strong the fabric's fine texture is at each place (a logarithm),
    averaged along the line, with the shiny core's own structure taken out."""
    from scipy.ndimage import gaussian_filter, gaussian_filter1d, uniform_filter1d
    strip = strip.astype(np.float32)
    energy = (strip - gaussian_filter(strip, 2.5)) ** 2
    energy = gaussian_filter1d(uniform_filter1d(energy, ALONG, axis=0, mode='nearest'), 2.0, axis=1)
    return cv2.morphologyEx(np.log(energy + 0.05).astype(np.float32), cv2.MORPH_OPEN,
                            np.ones((1, CORE), np.uint8))


def smoother_of(strip, shift):
    """How much weaker the weave is in a stripe BAND wide than just beside it."""
    from scipy.ndimage import uniform_filter1d
    weave = _weave(strip)
    inside = uniform_filter1d(weave, BAND - 8, axis=1, mode='nearest')
    beside = uniform_filter1d(weave, 8, axis=1, mode='nearest')
    return np.minimum(np.roll(beside, shift, axis=1), np.roll(beside, -shift, axis=1)) - inside


PAIR_WIDTHS = np.arange(20, 46, 2)     # bead widths the pair tracker considers, px


def _follow_pair(steep, allowed, extra=None):
    """The band's middle (a column per sample) as the best path of a falling and
    a rising edge a bead's width apart. The state is (middle, width); from one
    sample to the next either may change by one step."""
    count, columns = steep.shape
    spread = _spread(steep)
    index = np.arange(columns)
    score = np.empty((count, columns, len(PAIR_WIDTHS)), np.float32)
    for number, width in enumerate(PAIR_WIDTHS):
        half = width // 2
        left = np.clip(index - half, 0, columns - 1)
        right = np.clip(index + half, 0, columns - 1)
        score[:, :, number] = (steep[:, right] - steep[:, left]) / spread
        score[:, (index - half < 0) | (index + half >= columns), number] = -50.0
    if extra is not None:
        score += (extra / _spread(extra))[:, :, None]
    score[:, ~allowed, :] = -50.0
    total = score[0].copy()
    came = np.zeros((count, columns, len(PAIR_WIDTHS)), np.int8)
    for sample in range(1, count):
        padded = np.pad(total, 1, constant_values=-1e9)
        best = None
        choice = None
        for code, (move, widen) in enumerate((m, w) for m in (-1, 0, 1) for w in (-1, 0, 1)):
            shifted = padded[1 + move:1 + move + columns, 1 + widen:1 + widen + len(PAIR_WIDTHS)]
            if best is None:
                best, choice = shifted.copy(), np.zeros(shifted.shape, np.int8)
            else:
                better = shifted > best
                best = np.where(better, shifted, best)
                choice = np.where(better, code, choice)
        total = best + score[sample]
        came[sample] = choice
    column, width = np.unravel_index(int(total.argmax()), total.shape)
    path = np.zeros(count, int)
    for sample in range(count - 1, -1, -1):
        path[sample] = column
        code = int(came[sample, column, width])
        column = int(np.clip(column + code // 3 - 1, 0, columns - 1))
        width = int(np.clip(width + code % 3 - 1, 0, len(PAIR_WIDTHS) - 1))
    return path


def _weave_strength(strip):
    """The fabric's fine texture as a spread in grey levels: near zero in a soft
    picture, where there is no weave to be missing."""
    from scipy.ndimage import gaussian_filter
    strip = strip.astype(np.float32)
    return float(np.sqrt(np.median((strip - gaussian_filter(strip, 2.5)) ** 2)))


def _spread(values):
    return float(np.median(np.abs(values - np.median(values)))) / 0.6745 + 1e-6


def read_strip(strip, offsets, search=SEARCH, track='dark'):
    """The band's two edges in a straightened strip (rows = samples along the line,
    columns = ``offsets`` across it), as found sample by sample.

    ``track`` is what the band is followed by: ``dark`` (its darkness),
    ``dark+weave`` (darkness and missing weave texture together), ``pair`` (a
    falling and a rising edge a bead's width apart) or ``pair+weave``.

    Returns ``low`` / ``high`` (the edges as found), ``low_course`` / ``high_course``
    (their median courses), ``middle`` (the band's middle, column index),
    ``darkness`` (band against what lies beside it), ``core`` (how much brighter
    the shiny core is than the band's level) and ``core_at`` (its position).
    """
    from scipy.ndimage import gaussian_filter1d, median_filter, uniform_filter1d
    raw = strip
    strip = gaussian_filter1d(uniform_filter1d(strip, ALONG, axis=0, mode='nearest'), 1.0, axis=1)
    closed = cv2.morphologyEx(strip, cv2.MORPH_CLOSE, np.ones((1, THIN_DARK), np.uint8))
    plain = cv2.morphologyEx(closed, cv2.MORPH_OPEN, np.ones((1, CORE), np.uint8))
    # Where the shiny core lies at the band's edge, with fabric as bright as itself
    # beyond it, the opening leaves it at fabric level and the band would seem to
    # end before the core. The core is part of the bead: it takes the band's level.
    bright = closed - plain
    plain = np.where(bright > CORE_CONTRAST, cv2.erode(plain, np.ones((1, CORE), np.uint8)), plain)

    # How much darker a stripe BAND wide is than what lies just beside it.
    inside = uniform_filter1d(plain, BAND - 8, axis=1, mode='nearest')
    beside = uniform_filter1d(plain, 8, axis=1, mode='nearest')
    shift = BAND // 2 + 8
    darker = np.minimum(np.roll(beside, shift, axis=1), np.roll(beside, -shift, axis=1)) - inside
    following = darker
    if track == 'dark+weave' and _weave_strength(raw) >= MIN_WEAVE:
        # Glue fills the weave: the band is also where the fabric's fine texture is
        # missing. Where the band is too faint to follow by its darkness alone, the
        # two together still mark it. Each is put on the scale of its own spread.
        smoother = smoother_of(raw, shift)
        following = darker / _spread(darker) + smoother / _spread(smoother)
    following = following.copy()
    following[:, np.abs(offsets) > search] = -50.0
    darker[:, np.abs(offsets) > search] = -50.0
    if track.startswith('pair'):
        # The band as a PAIR of edges a bead's width apart, followed together: a
        # falling edge and a rising edge that keep their distance are far rarer
        # in fabric than a dark stripe, so the tracker is not drawn off by one.
        steep = gaussian_filter1d(plain, 2.0, axis=1, order=1)
        extra = None
        if track == 'pair+weave' and _weave_strength(raw) >= MIN_WEAVE:
            extra = smoother_of(raw, shift)
        middle = _follow_pair(steep, np.abs(offsets) <= search, extra)
    else:
        middle = _follow(following)
    samples = np.arange(len(plain))
    darkness = median_filter(darker[samples, middle], 25, mode='nearest')

    slope = gaussian_filter1d(plain, 2.0, axis=1, order=1)
    count = len(plain)
    low, high = np.zeros(count), np.zeros(count)
    core, core_at = np.zeros(count), np.zeros(count)
    for index, column in enumerate(middle):
        first = max(1, column - EDGE_FAR)
        last = min(slope.shape[1] - 1, column + EDGE_FAR)
        low[index] = offsets[first + int(np.argmin(slope[index, first:column - EDGE_NEAR + 1]))]
        high[index] = offsets[column + EDGE_NEAR + int(np.argmax(slope[index, column + EDGE_NEAR:last]))]
        window = bright[index, first:last]
        core[index] = float(window.max())
        core_at[index] = offsets[first + int(window.argmax())]
    low_course = median_filter(low, 15, mode='nearest')
    high_course = median_filter(high, 15, mode='nearest')

    return {'low': low, 'high': high, 'low_course': low_course, 'high_course': high_course,
            'middle': middle, 'darkness': darkness, 'core': core, 'core_at': core_at,
            'smooth': strip, 'plain': plain}


def strip_states(read, absent=None):
    """From ``read_strip``'s result: what the picture is like at each sample
    (``STATES``), which samples have both edges on their steady course, where the
    band is present, the slice from the bead's one end to the other, and the
    line's usual darkness. None when no dark stripe runs along the strip.
    ``absent`` marks samples to be taken as outside (beyond the ROI)."""
    from scipy.ndimage import median_filter
    low, high, low_course, high_course = (read[key] for key in ('low', 'high', 'low_course', 'high_course'))
    darkness, core, core_at = read['darkness'], read['core'], read['core_at']
    count = len(low)
    usual = float(np.median(darkness[len(darkness) // 4: -len(darkness) // 4]))
    if usual <= 0.5:
        return None
    present = darkness >= END_SHARE * usual
    if absent is not None:
        present &= ~absent
    kept = np.flatnonzero(present)
    if len(kept) < 3 * ALONG:
        return None
    # From one end of the bead to the other. The band may fade for a stretch (under
    # mesh, in glare) and come back, so every run of some length counts, not only
    # the longest; scraps beyond the tips do not.
    runs = np.split(kept, np.flatnonzero(np.diff(kept) > 25) + 1)
    longest = max(len(run) for run in runs)
    runs = [run for run in runs if len(run) >= min(MIN_RUN, longest)]
    span = slice(int(runs[0][0]), int(runs[-1][-1]) + 1)

    shiny = core > CORE_CONTRAST
    at_edge = shiny & ((core_at - low_course < CORE_AT_EDGE) | (high_course - core_at < CORE_AT_EDGE))
    code = np.full(count, STATES.index('matte'))
    code[darkness < WEAK_SHARE * usual] = STATES.index('weak')
    code[shiny] = STATES.index('shiny')
    code[at_edge] = STATES.index('one-sided')
    code[~present] = STATES.index('none')
    code = median_filter(code, STATE_RUN, mode='nearest')
    steady = (np.abs(low - low_course) <= STEADY) & (np.abs(high - high_course) <= STEADY)
    return np.array(STATES)[code], steady, present, span, usual


def measure(image, centre, region=None, search=SEARCH, edge_method=None):
    """Everything read from the straightened strip along ``centre``, per sample.

    ``edge_method`` replaces the way the two edges are placed (default: steepest
    slope); see ``edge_methods.METHODS``.

    ``search`` is how far from ``centre`` the band's middle may lie: wide for a
    rough line, narrow when the line is already an outline's middle, so that the
    tracker cannot wander onto another dark stripe nearby.

    Returns None when no dark stripe runs along the line, else a dict:
    ``points``, ``normals`` (the sample frame, on the lengthened line), ``low`` /
    ``high`` (the two edges as found, signed distance along the normal),
    ``low_course`` / ``high_course`` (their median courses), ``steady`` (both edges
    on course), ``darkness`` (band against the fabric beside it), ``core`` (how
    much brighter the shiny core is than the band; 0 where there is none),
    ``core_at`` (its position), ``present`` (the band is there), ``span`` (the
    slice from the bead's one end to the other) and ``state`` (see ``STATES``):

    - ``none``      no band: beyond an end, or a gap;
    - ``shiny``     the core shows: its position is exact, the band's edges are good;
    - ``one-sided`` the core runs along one edge of the band, or the band is cut
                    off on one side: that edge is not to be trusted;
    - ``weak``      the band is there but faint (under mesh, in shadow);
    - ``matte``     a plain dark band, no core: both edges good.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    centre = np.asarray(centre, np.float64)
    if len(centre) < 4:
        return None
    strip, points, normals, offsets = _straighten(gray, _lengthened(centre, EXTEND))
    if len(strip) < 3 * ALONG:
        return None
    read = read_strip(strip, offsets, search)
    if edge_method is not None:
        # Another way of placing the two edges (``edge_methods.METHODS``), given the
        # band's course from the reading above.
        from scipy.ndimage import median_filter
        low, high = edge_method(strip, offsets, read)
        read = dict(read, low=low, high=high, low_course=median_filter(low, 15, mode='nearest'),
                    high_course=median_filter(high, 15, mode='nearest'))
    low, high, low_course, high_course = (read[key] for key in ('low', 'high', 'low_course', 'high_course'))
    darkness, core, core_at = read['darkness'], read['core'], read['core_at']
    absent = None
    if region is not None:
        columns = np.clip(np.round(points[:, 0]).astype(int), 0, region.shape[1] - 1)
        rows = np.clip(np.round(points[:, 1]).astype(int), 0, region.shape[0] - 1)
        absent = region[rows, columns] == 0
    told = strip_states(read, absent)
    if told is None:
        return None
    state, steady, present, span, usual = told
    shiny = core > CORE_CONTRAST
    return {'points': points, 'normals': normals, 'low': low, 'high': high,
            'low_course': low_course, 'high_course': high_course, 'steady': steady,
            'darkness': darkness, 'usual_darkness': usual, 'core': np.where(shiny, core, 0.0),
            'core_at': core_at, 'present': present, 'span': span, 'state': state}


def bead_edges(image, centre, region=None):
    """The bead's outline along ``centre`` (an N x 2 polyline in image pixels).

    Returns ``(ring, seen, widths)``: the outline, the share of samples where both
    edges were found on their steady course, and the width at every sample; or None
    when no dark stripe runs along the line. ``region`` (the ROI mask) limits how
    far the lengthened ends may reach.
    """
    found = measure(image, centre, region)
    if found is None:
        return None
    span = found['span']
    points, normals = found['points'][span], found['normals'][span]
    low, high = found['low_course'][span], found['high_course'][span]
    ring = np.vstack([points + normals * low[:, None], (points + normals * high[:, None])[::-1]])
    return ring, float(found['steady'][span].mean()), high - low

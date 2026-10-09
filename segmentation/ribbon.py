"""One smooth ribbon of even width from the edge evidence along a glue line.

The glue line is laid as a smooth curve of nearly constant width; on black fabric
that is how it comes out of the picture. On white and pink fabric each way of
finding its edges is right over most of a line and wrong in places: SAM's outline
wobbles and spills, the edge finder steps sideways where the shiny core runs along
one edge. Drawn sample by sample, either gives a jagged outline with notches.

Here the edges found are treated as evidence, each with a weight that depends on
what the picture is like at that place (``bead_edges.measure`` says: shiny, matte,
one-sided, weak, none), and the outline is fitted to it:

1. evidence that disagrees with its neighbours is given less and less weight
   (three rounds), so that a spill or a step does not pull the fit;
2. the centre line is smoothed as a curve in the picture (a local cubic over
   about 400 pixels: it follows the omega's humps and removes wobble);
3. the width is one slowly varying value along the line.

A real change of width is not smoothed away without notice: where the evidence
departs from the ribbon the same way over a long stretch, that stretch is returned
as a deviation.
"""
import numpy as np

CENTRE_WINDOW = 201       # samples of the local cubic for the centre line (about 400 px)
WIDTH_SIGMA = 75.0        # samples over which the width may vary (about 150 px)
FIRST_SIGMA = 8.0         # samples of the first, outlier-finding smooth
OUTLIER = 3.0             # evidence further than this many spreads from the fit is dropped
MIN_SPREAD = 1.5          # pixels: evidence closer than this is never an outlier
# A width deviation: the evidence's own width departs from the line's usual width
# by more than chance allows, over a stretch. The width is averaged over windows
# of several lengths (a short defect shows in a short window, a slight one in a
# long window); each window length is judged against the spread of its own
# averages along the line.
DEVIATION = {
    'usual': 301,             # samples (600 px) of the running median that is the usual width
    'median': 5,              # samples of the median that removes single wild readings
    'windows': (8, 15, 30),   # samples averaged (16, 30 and 60 px)
    'noise': 7.0,             # report beyond this many spreads of that window's averages; on 257
                              # defect-free lines 7 gave 1 false report, 5 gave 9, 4 gave 37
    'floor_px': 1.5,          # and never less than this
}
STATE_WEIGHT = {          # how far each source is trusted in each state of the picture
    'edges': {'matte': 1.0, 'shiny': 1.0, 'weak': 0.4, 'one-sided': 0.15, 'none': 0.0},
    'sam': {'matte': 1.0, 'shiny': 1.0, 'weak': 0.6, 'one-sided': 1.0, 'none': 0.3},
}


def _smooth(values, weights, sigma):
    """Weighted Gaussian smoothing that passes over samples without evidence."""
    from scipy.ndimage import gaussian_filter1d
    total = gaussian_filter1d(weights, sigma, mode='nearest')
    return gaussian_filter1d(values * weights, sigma, mode='nearest') / np.maximum(total, 1e-9), total


def _robust(values, weights, sigma):
    """Smooth course of the evidence, and the weights left after dropping what
    disagrees with it."""
    weights = weights.copy()
    course = values
    for _ in range(3):
        course, total = _smooth(np.nan_to_num(values), weights, sigma)
        used = weights > 0
        if used.sum() < 5:
            break
        residual = np.where(used, values - course, 0.0)
        spread = max(MIN_SPREAD, 1.4826 * float(np.median(np.abs(residual[used]))))
        ratio = np.clip(np.abs(residual) / (OUTLIER * spread), 0.0, 1.0)
        weights = weights * (1.0 - ratio ** 2) ** 2
    return course, weights


def _frame(points):
    tangent = np.gradient(points, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-6)
    return np.column_stack([-tangent[:, 1], tangent[:, 0]])


def _project(points, base, normals):
    """Each point's nearest base sample and its signed distance along the normal."""
    from scipy.spatial import cKDTree
    _, nearest = cKDTree(base).query(points)
    return nearest, ((points - base[nearest]) * normals[nearest]).sum(axis=1)


def gather(measured, sam=None, weights=STATE_WEIGHT):
    """Edge evidence as points in the picture: ``(low, high)``, each an array of
    rows ``x, y, weight, source`` (source 0 = the bead's edges, 1 = SAM).

    ``measured`` is ``bead_edges.measure``'s result (it also gives the state of
    the picture at each sample); ``sam`` the ``details`` of ``sam_refine.trim_ring``.
    Either source's weight is that of the state at the place.
    """
    span = measured['span']
    base, normals = measured['points'], measured['normals']
    state = measured['state']
    low, high = [], []
    if weights.get('edges'):
        trust = np.array([weights['edges'][name] for name in state])
        trust = np.where(measured['steady'], trust, 0.3 * trust)
        keep = np.zeros(len(base), bool)
        keep[span] = True
        keep &= trust > 0
        for rows, edge in ((low, measured['low']), (high, measured['high'])):
            rows.append(np.column_stack([base[keep] + normals[keep] * edge[keep][:, None], trust[keep],
                                         np.zeros(keep.sum())]))
    if sam and weights.get('sam'):
        for rows, edge, seen in ((low, sam['low'], sam['low_seen']), (high, sam['high'], sam['high_seen'])):
            points = sam['points'][seen] + sam['normals'][seen] * edge[seen][:, None]
            if len(points) == 0:
                continue
            nearest, _ = _project(points, base, normals)
            trust = np.array([weights['sam'][state[index]] for index in nearest])
            rows.append(np.column_stack([points, trust, np.ones(len(points))]))
    empty = np.zeros((0, 4))
    return (np.vstack(low) if low else empty), (np.vstack(high) if high else empty)


def _binned(evidence, base, normals):
    """Evidence per base sample: weighted mean distance, and total weight."""
    count = len(base)
    total, summed = np.zeros(count), np.zeros(count)
    if len(evidence):
        nearest, distance = _project(evidence[:, :2], base, normals)
        np.add.at(total, nearest, evidence[:, 2])
        np.add.at(summed, nearest, evidence[:, 2] * distance)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(total > 0, summed / np.maximum(total, 1e-9), np.nan), total


def fit_ribbon(base, low_evidence, high_evidence, deviation=None):
    """The ribbon through the evidence.

    ``base`` is the line the ribbon runs along, sampled evenly (its ends are the
    ribbon's ends); the evidence comes from ``gather``. Returns None when there is
    too little evidence, else a dict: ``ring`` (the outline), ``centre``,
    ``widths``, ``backed`` (share of the line with evidence for both edges nearby,
    after outliers were dropped), ``misfit`` (median distance of the kept evidence from
    the ribbon, pixels), ``rough`` (the same for all evidence: how jagged it was)
    and ``deviations`` (list of ``(first sample, last sample, pixels)`` where the
    evidence is consistently wider or narrower than the ribbon).
    """
    from scipy.ndimage import gaussian_filter1d, median_filter, uniform_filter1d
    from scipy.signal import savgol_filter
    base = np.asarray(base, np.float64)
    count = len(base)
    if count < 40:
        return None
    normals = _frame(base)
    # Round one, along the given line: find what disagrees, place the centre.
    low, low_weight = _binned(low_evidence, base, normals)
    high, high_weight = _binned(high_evidence, base, normals)
    if (low_weight > 0).sum() < 20 or (high_weight > 0).sum() < 20:
        return None
    low_course, low_kept = _robust(low, low_weight, FIRST_SIGMA)
    high_course, high_kept = _robust(high, high_weight, FIRST_SIGMA)
    middle = (low_course + high_course) / 2.0
    centre = base + normals * middle[:, None]
    # The centre line as a smooth curve in the picture. A local cubic follows the
    # omega's humps without flattening them, which a plain blur would.
    window = min(CENTRE_WINDOW, (count // 2) * 2 - 1)
    if window >= 7:
        centre = np.column_stack([savgol_filter(centre[:, axis], window, 3, mode='interp')
                                  for axis in (0, 1)])
    normals = _frame(centre)
    # Round two, along the smooth centre: the width.
    low, low_weight = _binned(low_evidence, centre, normals)
    high, high_weight = _binned(high_evidence, centre, normals)
    low_course, low_kept = _robust(low, low_weight, FIRST_SIGMA)
    high_course, high_kept = _robust(high, high_weight, FIRST_SIGMA)
    both = np.minimum(low_kept, high_kept)
    widths, _ = _smooth(np.nan_to_num(high_course - low_course), np.maximum(both, 1e-6), WIDTH_SIGMA)
    shift, _ = _smooth(np.nan_to_num((high_course + low_course) / 2.0), np.maximum(both, 1e-6), WIDTH_SIGMA)
    centre = centre + normals * shift[:, None]
    ring = np.vstack([centre - normals * (widths / 2.0)[:, None],
                      (centre + normals * (widths / 2.0)[:, None])[::-1]])

    low_edge, high_edge = shift - widths / 2.0, shift + widths / 2.0
    kept = np.r_[np.abs(low - low_edge)[low_kept > 0.3], np.abs(high - high_edge)[high_kept > 0.3]]
    every = np.r_[np.abs(low - low_edge)[low_weight > 0], np.abs(high - high_edge)[high_weight > 0]]
    # Where the evidence itself is wider or narrower than the ribbon by more than
    # its noise allows. Each source is looked at alone; when there are two, both
    # must say so at the place, since either alone is wrong in places.
    rule = dict(DEVIATION, **(deviation or {}))
    marked = np.ones(count, bool)
    excess = np.zeros(count)
    spread = 0.0
    sources = np.unique(np.r_[low_evidence[:, 3:].ravel(), high_evidence[:, 3:].ravel()]) \
        if low_evidence.shape[1] > 3 else [None]
    samples = np.arange(count)
    for source in sources:
        chosen = (lambda rows: rows if source is None else rows[rows[:, 3] == source])
        one_low, one_low_weight = _binned(chosen(low_evidence), centre, normals)
        one_high, one_high_weight = _binned(chosen(high_evidence), centre, normals)
        there = (one_low_weight > 0) & (one_high_weight > 0)
        if there.sum() < 20:
            continue
        found = np.interp(samples, samples[there], (one_high - one_low)[there])
        # Against the line's usual width taken as a long running median, which a
        # defect does not move. (Against the smoothed ribbon, the stretches beside
        # a wide spot would read as narrow, since the smoothing spreads it out.)
        usual = median_filter(found, min(int(rule['usual']) | 1, (count // 2) * 2 - 1), mode='nearest')
        apart = median_filter(found - usual, int(rule['median']) | 1, mode='nearest')
        fired = np.zeros(count, bool)
        wide = np.zeros(count)
        for window in rule['windows']:
            mean = uniform_filter1d(apart, int(window), mode='nearest')
            noise = 1.4826 * float(np.median(np.abs(mean - np.median(mean)))) + 1e-6
            over = (np.abs(mean) > rule['noise'] * noise) & (np.abs(mean) > rule['floor_px'])
            fired |= uniform_filter1d(over.astype(float), int(window), mode='nearest') > 0
            wide = np.where(np.abs(mean) > np.abs(wide), mean, wide)
            if window == rule['windows'][0]:
                spread = max(spread, noise)
        near = gaussian_filter1d(there.astype(float), 5.0, mode='nearest') > 0.3
        marked &= near & fired & ((excess == 0) | (np.sign(wide) == np.sign(excess)))
        excess = np.where(np.abs(wide) > np.abs(excess), wide, excess)
    deviations = []
    start = None
    for index in range(count + 1):
        inside = index < count and marked[index]
        if inside and start is None:
            start = index
        elif not inside and start is not None:
            if index - start >= 3:
                peak = excess[start:index]
                deviations.append((start, index - 1, float(peak[np.abs(peak).argmax()])))
            start = None
    return {'ring': ring, 'centre': centre, 'widths': widths,
            'backed': float((_smooth(both, np.ones(count), 3.0)[0] > 0.15).mean()),
            'misfit': float(np.median(kept)) if kept.size else float('nan'),
            'rough': float(np.sqrt(np.mean(every ** 2))) if every.size else float('nan'),
            'deviations': deviations, 'width_noise': float(spread)}

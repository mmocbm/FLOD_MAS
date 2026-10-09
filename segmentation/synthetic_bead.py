"""Glue lines of known edges, painted onto real fabric: a truth to measure error against.

No picture in this project has its glue line's true edges marked, so no method's
error could be measured, only its agreement with another method. Here a strip of
real fabric without a bead is taken from the same picture (same weave, lighting,
noise, blur) and a bead is painted into it whose edges are chosen, so they are
known exactly.

The painted bead follows what was measured on real ones (``GLUE_LINE_FINDINGS.md``,
*Accuracy*): a band darker than the fabric with edges blurred by the camera, the
weave's texture much weaker inside it (glue fills the weave), and in stretches a
thin bright core with a thin dark line each side. All of it is a model of the
glue line, not the glue line: a method that does well here has passed a necessary
test, not a sufficient one.

Everything is in strip coordinates: rows are samples along the line (2 px apart),
columns are pixels across it.
"""
import numpy as np

# Medians measured on the M1 captures, per fabric colour: how much darker the band
# is than the fabric (grey levels), the core's brightness over the band, and the
# share of the weave's texture left inside the bead.
MEASURED = {
    'dark': {'depth': 45.0, 'core': 120.0, 'texture': 0.21},
    'pink': {'depth': 7.0, 'core': 16.0, 'texture': 0.24},
    'white': {'depth': 7.0, 'core': 28.0, 'texture': 0.39},
}
EDGE_BLUR = 2.1          # pixels: the camera's blur of an edge, as measured
CORE_SIGMA = 2.2         # pixels: half-width of the shiny core
SIDE_LINE = 0.35         # depth of the thin dark lines beside the core, share of the core


def fabric_strip(gray, centre, region, shifts=(110.0, -110.0, 140.0, -140.0, 180.0, -180.0),
                 min_samples=300):
    """A strip of fabric with no bead: the picture straightened along the line
    moved sideways into the plain part of its ROI region. The longest stretch whose
    whole width lies inside the region is returned, as ``(strip, offsets)``; None
    if there is no stretch of ``min_samples`` samples."""
    from . import bead_edges
    lengthened = bead_edges._lengthened(np.asarray(centre, np.float64), 0.0)
    _, points, normals, _ = bead_edges._straighten(gray, lengthened)
    best = None
    for shift in shifts:
        moved = points + normals * shift
        clear = np.ones(len(moved), bool)
        for across in (-bead_edges.REACH - 8, 0, bead_edges.REACH + 8):
            side = moved + normals * across
            columns = np.round(side[:, 0]).astype(int)
            rows = np.round(side[:, 1]).astype(int)
            inside = (columns >= 0) & (columns < gray.shape[1]) & (rows >= 0) & (rows < gray.shape[0])
            clear &= inside
            clear[inside] &= region[rows[inside], columns[inside]] > 0
        kept = np.flatnonzero(clear)
        if len(kept) == 0:
            continue
        run = max(np.split(kept, np.flatnonzero(np.diff(kept) > 1) + 1), key=len)
        if len(run) >= min_samples and (best is None or len(run) > len(best[0])):
            best = (run, moved)
    if best is None:
        return None
    run, moved = best
    strip, _, _, offsets = bead_edges._straighten(gray, moved[run[0]:run[-1] + 1])
    return strip, offsets


def wave(count, rng, reach, period):
    """A slow random course along the line: smooth, within +-reach."""
    from scipy.ndimage import gaussian_filter1d
    course = gaussian_filter1d(rng.normal(0.0, 1.0, count + 4 * period), period, mode='wrap')[:count]
    return course / max(np.abs(course).max(), 1e-6) * reach


def paint(fabric, offsets, rng, colour='white', width=30.0, depth=None, core=None, texture=None,
          blur=EDGE_BLUR, wander=12.0, width_wobble=1.0, shiny_share=0.6, core_at_edge=0.0,
          defects=(), soften=0.0):
    """Paint a bead into a fabric strip. Returns ``(strip, truth)``.

    ``truth`` holds the true ``low`` and ``high`` edge per sample (the half-height
    points of the band), ``shiny`` (where the core is on) and ``defect`` (sample
    -> signed width change). ``defects`` are ``(start, length, pixels)`` width
    changes. ``core_at_edge`` is the share of shiny stretches where the core runs
    along one edge instead of the middle. ``soften`` blurs the finished strip
    (pixels), to imitate a soft capture.
    """
    from scipy.ndimage import gaussian_filter, gaussian_filter1d
    from scipy.special import erf
    measured = MEASURED[colour]
    depth = measured['depth'] if depth is None else depth
    core = measured['core'] if core is None else core
    texture = measured['texture'] if texture is None else texture
    count = len(fabric)
    samples = np.arange(count)
    widths = np.full(count, float(width)) + wave(count, rng, width_wobble, 150)
    defect = np.zeros(count)
    for start, length, pixels in defects:
        ramp = np.clip(np.minimum(samples - start, start + length - samples) / 4.0, 0.0, 1.0)
        defect += pixels * ramp
    widths = widths + defect
    middle = wave(count, rng, wander, 220)
    low, high = middle - widths / 2.0, middle + widths / 2.0
    across = offsets[None, :].astype(np.float64)
    inside = 0.5 * (erf((across - low[:, None]) / (np.sqrt(2) * blur))
                    - erf((across - high[:, None]) / (np.sqrt(2) * blur)))
    level = gaussian_filter(fabric.astype(np.float64), 2.5)
    weave = fabric - level
    strip = level + weave * (1.0 - (1.0 - texture) * inside) - depth * inside
    # Shiny stretches: the core comes and goes along the line.
    shiny = gaussian_filter1d(rng.normal(0, 1, count), 40, mode='wrap')
    shiny = shiny > np.quantile(shiny, 1.0 - shiny_share) if shiny_share > 0 else np.zeros(count, bool)
    at_edge = gaussian_filter1d(rng.normal(0, 1, count), 60, mode='wrap')
    at_edge = shiny & (at_edge > np.quantile(at_edge, 1.0 - core_at_edge)) if core_at_edge > 0 else np.zeros(count, bool)
    position = np.where(at_edge, high - 5.0, middle + wave(count, rng, 0.08 * width, 90))
    strength = core * gaussian_filter1d(shiny.astype(np.float64), 6, mode='nearest') \
        * (1.0 + 0.3 * wave(count, rng, 1.0, 30))
    bump = np.exp(-(across - position[:, None]) ** 2 / (2 * CORE_SIGMA ** 2))
    lines = sum(np.exp(-(across - (position[:, None] + side)) ** 2 / (2 * 1.6 ** 2)) for side in (-5.5, 5.5))
    strip = strip + strength[:, None] * (bump - SIDE_LINE * lines)
    if soften > 0:
        strip = gaussian_filter(strip, soften)
    truth = {'low': low, 'high': high, 'shiny': shiny, 'at_edge': at_edge, 'defect': defect}
    return np.clip(strip, 0, 255).astype(np.float32), truth

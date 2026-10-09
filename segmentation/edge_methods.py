"""Ways of finding the glue band's two edges in a straightened strip.

Each method takes the strip (rows = samples along the line, columns = pixels
across it), the column offsets and the baseline reading (``bead_edges.read_strip``,
which says where the band's middle runs) and returns ``(low, high)``: the two
edges per sample, in pixels across the line, sub-pixel where the method allows.

They are compared on painted glue lines of known edges by
``tools/accuracy_bench.py``; the figures are in ``GLUE_LINE_FINDINGS.md``,
*Accuracy*. ``METHODS`` lists them by name.
"""
import numpy as np

from . import bead_edges

ALONG = bead_edges.ALONG


def _averaged(strip, along=ALONG, across=1.0):
    from scipy.ndimage import gaussian_filter1d, uniform_filter1d
    return gaussian_filter1d(uniform_filter1d(strip.astype(np.float32), along, axis=0, mode='nearest'),
                             across, axis=1)


def _peak(values, index):
    """Sub-pixel position of an extremum by a parabola through three samples."""
    if index <= 0 or index >= len(values) - 1:
        return float(index)
    left, middle, right = values[index - 1], values[index], values[index + 1]
    bend = left - 2.0 * middle + right
    return float(index) if abs(bend) < 1e-9 else index + 0.5 * (left - right) / bend


def baseline(strip, offsets, read):
    """The edges as version 4 reads them: steepest slope each side, whole pixels."""
    return read['low'].copy(), read['high'].copy()


def slope_subpixel(strip, offsets, read, scale=2.0):
    """The same steepest slopes, placed between pixels by a parabola."""
    from scipy.ndimage import gaussian_filter1d
    slope = gaussian_filter1d(read['plain'], scale, axis=1, order=1)
    low, high = np.zeros(len(strip)), np.zeros(len(strip))
    for index, column in enumerate(read['middle']):
        first = max(1, column - bead_edges.EDGE_FAR)
        last = min(slope.shape[1] - 1, column + bead_edges.EDGE_FAR)
        a = first + int(np.argmin(slope[index, first:column - bead_edges.EDGE_NEAR + 1]))
        b = column + bead_edges.EDGE_NEAR + int(np.argmax(slope[index, column + bead_edges.EDGE_NEAR:last]))
        low[index] = offsets[0] + _peak(slope[index], a)
        high[index] = offsets[0] + _peak(slope[index], b)
    return low, high


def _steepest(channel, offsets, middle, scale, near=bead_edges.EDGE_NEAR, far=bead_edges.EDGE_FAR):
    """Sub-pixel steepest fall on the low side and rise on the high side of the
    band's middle, in a channel where the band is LOW."""
    from scipy.ndimage import gaussian_filter1d
    slope = gaussian_filter1d(channel, scale, axis=1, order=1)
    low, high = np.zeros(len(channel)), np.zeros(len(channel))
    strength = np.zeros((len(channel), 2))
    for index, column in enumerate(middle):
        first = max(1, column - far)
        last = min(slope.shape[1] - 1, column + far)
        a = first + int(np.argmin(slope[index, first:column - near + 1]))
        b = column + near + int(np.argmax(slope[index, column + near:last]))
        low[index] = offsets[0] + _peak(slope[index], a)
        high[index] = offsets[0] + _peak(slope[index], b)
        strength[index] = -slope[index, a], slope[index, b]
    return low, high, strength, slope


def slope_scale3(strip, offsets, read):
    """Steepest slope at a coarser scale (3 px): less noise, more rounding."""
    return _steepest(read['plain'], offsets, read['middle'], 3.0)[:2]


def weave_energy(strip):
    """How strong the weave's texture is at each place, as a logarithm. Glue fills
    the weave, so the bead is LOW here whatever its brightness."""
    return bead_edges._weave(strip)


def texture(strip, offsets, read):
    """Edges of the smooth (texture-free) stripe, from the weave's energy alone."""
    return _steepest(weave_energy(strip), offsets, read['middle'], 3.0)[:2]


def _noise(slope):
    """Spread of a slope image where nothing is: its median absolute value."""
    return float(np.median(np.abs(slope))) / 0.6745 + 1e-6


def fused(strip, offsets, read, scale=2.5):
    """Lightness and weave texture together: each channel's slope is divided by
    its own noise, the two are added, and the edge is the steepest of the sum. An
    edge weak in one channel is carried by the other."""
    from scipy.ndimage import gaussian_filter1d
    light = gaussian_filter1d(read['plain'], scale, axis=1, order=1)
    weave = gaussian_filter1d(weave_energy(strip), scale + 0.5, axis=1, order=1)
    both = light / _noise(light) + weave / _noise(weave)
    low, high = np.zeros(len(strip)), np.zeros(len(strip))
    for index, column in enumerate(read['middle']):
        first = max(1, column - bead_edges.EDGE_FAR)
        last = min(both.shape[1] - 1, column + bead_edges.EDGE_FAR)
        a = first + int(np.argmin(both[index, first:column - bead_edges.EDGE_NEAR + 1]))
        b = column + bead_edges.EDGE_NEAR + int(np.argmax(both[index, column + bead_edges.EDGE_NEAR:last]))
        low[index] = offsets[0] + _peak(both[index], a)
        high[index] = offsets[0] + _peak(both[index], b)
    return low, high


def profile_fit(strip, offsets, read, every=4):
    """Fit the band's cross-section as a model: two edges blurred by the camera,
    with the fabric level free on each side. The edges are parameters of the fit,
    so they come out between pixels and are not pulled by the core (which the
    opening has removed from ``plain``)."""
    from scipy.optimize import least_squares
    from scipy.special import erf
    plain = read['plain'].astype(np.float64)
    across = offsets.astype(np.float64)
    start_low, start_high = read['low_course'], read['high_course']
    chosen = np.arange(0, len(strip), every)
    low, high = np.zeros(len(chosen)), np.zeros(len(chosen))

    def model(values, x):
        edge_low, edge_high, depth, level, tilt, blur = values
        inside = 0.5 * (erf((x - edge_low) / (1.4142 * blur)) - erf((x - edge_high) / (1.4142 * blur)))
        return level + tilt * x - depth * inside

    for place, index in enumerate(chosen):
        middle = (start_low[index] + start_high[index]) / 2.0
        window = (across > middle - 42) & (across < middle + 42)
        x, y = across[window], plain[index, window]
        guess = [start_low[index], start_high[index], max(1.0, np.median(y[[0, -1]]) - y.min()),
                 float(np.median(y[[0, 1, -2, -1]])), 0.0, 2.5]
        try:
            found = least_squares(lambda values: model(values, x) - y, guess, loss='soft_l1', f_scale=1.5,
                                  bounds=([middle - 40, middle - 5, 0.0, 0.0, -1.0, 1.0],
                                          [middle + 5, middle + 40, 255.0, 255.0, 1.0, 6.0]), max_nfev=40).x
            low[place], high[place] = found[0], found[1]
        except ValueError:
            low[place], high[place] = start_low[index], start_high[index]
    samples = np.arange(len(strip))
    return np.interp(samples, chosen, low), np.interp(samples, chosen, high)


def phase(strip, offsets, read, wavelengths=(8.0, 14.0, 24.0)):
    """Edges by phase congruency: an edge is where the picture's components of
    several scales are all in step at a quarter turn, whatever the contrast. Each
    row is taken through a bank of one-dimensional log-Gabor filters; the edge is
    where the odd (step-like) responses add up most, relative to the total."""
    plain = read['plain'].astype(np.float64)
    count, width = plain.shape
    frequency = np.fft.fftfreq(width)
    spectrum = np.fft.fft(plain - plain.mean(axis=1, keepdims=True), axis=1)
    odd_sum = np.zeros_like(plain)
    energy = np.zeros_like(plain)
    for wavelength in wavelengths:
        with np.errstate(divide='ignore', invalid='ignore'):
            gain = np.exp(-(np.log(np.abs(frequency) * wavelength)) ** 2 / (2 * np.log(0.55) ** 2))
        gain[frequency <= 0] = 0.0                      # analytic: positive frequencies only
        response = np.fft.ifft(spectrum * (2.0 * gain)[None, :], axis=1)
        odd_sum += response.imag
        energy += np.abs(response)
    congruent = odd_sum / (energy + 1.0)                # signed: + rising, - falling (or the reverse)
    low, high = np.zeros(count), np.zeros(count)
    for index, column in enumerate(read['middle']):
        first = max(1, column - bead_edges.EDGE_FAR)
        last = min(width - 1, column + bead_edges.EDGE_FAR)
        left = congruent[index, first:column - bead_edges.EDGE_NEAR + 1]
        right = congruent[index, column + bead_edges.EDGE_NEAR:last]
        # Which sign is "falling" is fixed by the transform; found from the data once.
        a = first + int(np.argmax(np.abs(left)))
        b = column + bead_edges.EDGE_NEAR + int(np.argmax(np.abs(right)))
        low[index] = offsets[0] + _peak(np.abs(congruent[index]), a)
        high[index] = offsets[0] + _peak(np.abs(congruent[index]), b)
    return low, high


METHODS = {'baseline': baseline, 'slope_subpixel': slope_subpixel, 'slope_scale3': slope_scale3,
           'texture': texture, 'fused': fused, 'profile_fit': profile_fit, 'phase': phase}

"""How far each way of finding the glue line's edges is from the truth.

    python tools/accuracy_bench.py --runs RUNS.pkl ... [--methods a,b] [--json out.json]

Truth comes from glue lines painted onto strips of real fabric taken from the
frames themselves (``segmentation.synthetic_bead``): their edges are known. For
every method, fabric colour and condition it reports, in pixels:

    edge bias / RMS    signed and root-mean-square error of the two edges
    width bias / RMS   the same for the width
    ribbon ...         the same after the smooth ribbon is fitted to those edges
    defects            share of painted width defects found, and false alarms

``--runs`` are pickles of Pipeline Lab runs (``image`` path, packed ``roi``,
``shape``, ``rough`` lines), which say where fabric strips can be cut.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT), str(Path(__file__).resolve().parent)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import cv2  # noqa: E402 - after the path is prepared
import numpy as np  # noqa: E402

import pipeline_steps as steps  # noqa: E402
from segmentation import bead_edges, edge_methods, ribbon, synthetic_bead  # noqa: E402
import importlib  # noqa: E402

importlib.import_module('segmentation.bead_ml')        # registers the 'trees' method

MARGIN = 40          # samples left out at each end of a strip when scoring
CONDITIONS = {                 # the rough line is taken to be within 4 px unless stated
    'standard': {},
    'rough line off 12': {'wander': 12.0},
    'core at edge': {'core_at_edge': 0.5},
    'half contrast': {'depth_scale': 0.5},
    'soft 2.5': {'soften': 2.5},
    'soft 4': {'soften': 4.0},
    'no shine': {'shiny_share': 0.0},
}
DEFECTS = ((3.0, 30), (5.0, 30), (5.0, 60), (8.0, 30), (8.0, 60), (-5.0, 60), (-8.0, 30))


def fabric_strips(runs, per_colour, rng):
    """Bead-free strips cut from the frames, by fabric colour."""
    found = {}
    order = list(runs)
    rng.shuffle(order)
    for path in order:
        run = pickle.loads(Path(path).read_bytes())
        colour = run['colour']
        if len(found.get(colour, ())) >= per_colour:
            continue
        gray = cv2.imread(run['image'], cv2.IMREAD_GRAYSCALE)
        region = np.unpackbits(run['roi'])[:run['shape'][0] * run['shape'][1]].reshape(run['shape'])
        for line in run['rough']:
            strip = synthetic_bead.fabric_strip(gray, steps._centre(line['ring']), region)
            if strip is not None and len(found.get(colour, ())) < per_colour:
                found.setdefault(colour, []).append(strip)
    return found


def ribbon_edges(low, high, read, offsets, weights=ribbon.STATE_WEIGHT, deviation=None):
    """The smooth ribbon fitted to a method's edges, as edges per sample (NaN
    outside the stretch the ribbon covers)."""
    from scipy.ndimage import median_filter
    reading = dict(read, low=low, high=high, low_course=median_filter(low, 15, mode='nearest'),
                   high_course=median_filter(high, 15, mode='nearest'))
    told = bead_edges.strip_states(reading)
    count = len(low)
    nothing = np.full(count, np.nan)
    if told is None:
        return nothing, nothing.copy(), []
    state, steady, _, span, _ = told
    # Strip coordinates as a picture: x across the line, y along it. With that
    # frame's normal pointing to -x, the edges swap sides and change sign.
    points = np.column_stack([np.zeros(count), np.arange(count) * bead_edges.STEP])
    measured = {'points': points, 'normals': np.tile([-1.0, 0.0], (count, 1)), 'low': -high,
                'high': -low, 'steady': steady, 'state': state, 'span': span}
    evidence = ribbon.gather(measured, None, {'edges': weights['edges']})
    fitted = ribbon.fit_ribbon(points[span], *evidence, deviation=deviation)
    if fitted is None:
        return nothing, nothing.copy(), []
    centre, widths = fitted['centre'][:, 0], fitted['widths']
    fitted_low, fitted_high = nothing.copy(), nothing.copy()
    fitted_low[span], fitted_high[span] = centre - widths / 2.0, centre + widths / 2.0
    return fitted_low, fitted_high, [(span.start + first, span.start + last, pixels)
                                     for first, last, pixels in fitted['deviations']]


def score(low, high, truth):
    """Edge and width errors over the scored part of a strip."""
    keep = np.zeros(len(low), bool)
    keep[MARGIN:-MARGIN] = True
    keep &= np.isfinite(low) & np.isfinite(high)
    edge = np.r_[(truth['low'] - low)[keep], (high - truth['high'])[keep]]    # + = outside the truth
    width = ((high - low) - (truth['high'] - truth['low']))[keep]
    return edge, width, float(keep[MARGIN:-MARGIN].mean())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--methods', default=','.join(edge_methods.METHODS))
    parser.add_argument('--conditions', default=','.join(CONDITIONS))
    parser.add_argument('--strips', type=int, default=12, help='fabric strips per colour')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--track', default='dark', choices=('dark', 'dark+weave', 'pair', 'pair+weave'))
    parser.add_argument('--trees', help="model file for the 'trees' method (default: the installed one)")
    parser.add_argument('--json')
    args = parser.parse_args(argv)
    if args.trees:
        from segmentation import bead_ml
        bead_ml.MODEL_FILE, bead_ml._model = Path(args.trees), None
    rng = np.random.default_rng(args.seed)
    fabrics = fabric_strips(args.runs, args.strips, rng)
    print('fabric strips:', {colour: len(strips) for colour, strips in fabrics.items()})
    methods = args.methods.split(',')
    results = []
    for colour in ('dark', 'pink', 'white'):
        for condition in args.conditions.split(','):
            settings = dict(CONDITIONS[condition])
            depth_scale = settings.pop('depth_scale', 1.0)
            collected = {name: {'edge': [], 'width': [], 'cover': [], 'r_edge': [], 'r_width': [],
                                'r_cover': [], 'hit': 0, 'defects': 0, 'false': 0, 'length': 0}
                         for name in methods}
            for number, (fabric, offsets) in enumerate(fabrics.get(colour, ())):
                painter = np.random.default_rng(1000 * args.seed + number)
                width = painter.uniform(26.0, 38.0)
                count = len(fabric)
                # Two defects per strip, well apart, of sizes taken in turn.
                defects = []
                for place in (0.3, 0.7):
                    pixels, length = DEFECTS[(number * 2 + len(defects)) % len(DEFECTS)]
                    defects.append((int(count * place), length, pixels))
                strip, truth = synthetic_bead.paint(
                    fabric, offsets, painter, colour=colour, width=width, defects=defects,
                    depth=synthetic_bead.MEASURED[colour]['depth'] * depth_scale,
                    **dict({'wander': 4.0}, **settings))
                read = bead_edges.read_strip(strip, offsets, track=args.track)
                for name in methods:
                    low, high = edge_methods.METHODS[name](strip, offsets, read)
                    edge, wide, cover = score(low, high, truth)
                    entry = collected[name]
                    entry['edge'].append(edge)
                    entry['width'].append(wide)
                    entry['cover'].append(cover)
                    fitted_low, fitted_high, deviations = ribbon_edges(low, high, read, offsets)
                    edge, wide, cover = score(fitted_low, fitted_high, truth)
                    entry['r_edge'].append(edge)
                    entry['r_width'].append(wide)
                    entry['r_cover'].append(cover)
                    marked = np.zeros(count, bool)
                    for first, last, _ in deviations:
                        marked[first:last + 1] = True
                    for start, length, _ in defects:
                        entry['defects'] += 1
                        entry['hit'] += bool(marked[start:start + length].any())
                    clear = truth['defect'] == 0
                    grown = np.convolve(~clear, np.ones(41), 'same') > 0      # near a defect is not false
                    entry['false'] += int(np.count_nonzero(np.diff(np.r_[0, (marked & ~grown).astype(int)]) == 1))
                    entry['length'] += count * bead_edges.STEP
            for name, entry in collected.items():
                if not entry['edge']:
                    continue
                edge, wide = np.concatenate(entry['edge']), np.concatenate(entry['width'])
                r_edge, r_wide = np.concatenate(entry['r_edge']), np.concatenate(entry['r_width'])
                row = {'colour': colour, 'condition': condition, 'method': name,
                       'edge_bias': float(edge.mean()), 'edge_rms': float(np.sqrt((edge ** 2).mean())),
                       'width_bias': float(wide.mean()), 'width_rms': float(np.sqrt((wide ** 2).mean())),
                       'ribbon_edge_bias': float(r_edge.mean()) if r_edge.size else None,
                       'ribbon_edge_rms': float(np.sqrt((r_edge ** 2).mean())) if r_edge.size else None,
                       'ribbon_width_bias': float(r_wide.mean()) if r_wide.size else None,
                       'ribbon_width_rms': float(np.sqrt((r_wide ** 2).mean())) if r_wide.size else None,
                       'ribbon_cover': float(np.mean(entry['r_cover'])),
                       'defects_found': entry['hit'], 'defects': entry['defects'],
                       'false_per_metre_px': entry['false'] / max(entry['length'], 1) * 1000.0}
                results.append(row)
                print(f"{colour:5s} {condition:13s} {name:16s} edge {row['edge_bias']:+5.2f} / {row['edge_rms']:5.2f}"
                      f" | width {row['width_bias']:+5.2f} / {row['width_rms']:5.2f}"
                      + (f" || ribbon edge {row['ribbon_edge_bias']:+5.2f} / {row['ribbon_edge_rms']:5.2f}"
                         f" | width {row['ribbon_width_bias']:+5.2f} / {row['ribbon_width_rms']:5.2f}"
                         f" | cover {row['ribbon_cover']:.2f}" if r_edge.size else ' || no ribbon')
                      + f" | defects {row['defects_found']}/{row['defects']}, false {entry['false']}",
                      flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

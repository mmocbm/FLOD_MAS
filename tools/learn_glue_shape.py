"""Learn the wave shape of a product's glue line from sample frames.

The shape-conditioned detector (``segmentation.glue_line.shape_prior``) needs to
know what a whole glue line looks like on a product. This reads frames of ONE
product, keeps the lines that were found whole, and averages them into a
template in ``segmentation/glue_shapes.json``.

    .venv\\Scripts\\python.exe tools\\learn_glue_shape.py two_panel frames\\*.png

Use frames taken at the working camera height, with the lines as visible as the
product allows. It prints how many whole lines were used and how closely they
agree; a spread of more than about 1 % of the line's length means the frames mix
products or the lines were not found whole.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402 - after the path is prepared
import numpy as np  # noqa: E402

from segmentation import glue_line, glue_shape  # noqa: E402

JOIN_GAP_PX = 500.0      # generous joining: whole lines are wanted here, not caution
MIN_SEEN_SHARE = 0.6
MAX_BOW = 0.13           # a real line strays no further than this share of its chord
CHORD_AGREEMENT = 0.03   # whole lines of one product agree in length this closely


def frame_pieces(frame, background='white'):
    """Centreline pieces of every bead-like stretch in one frame."""
    options = dict(glue_line.DEFAULTS, background=background)
    fabric = glue_line.find_fabric(frame, options)
    margin = int(options['fabric_margin_px']) | 1
    interior = cv2.erode(fabric, np.ones((margin, margin), np.uint8))
    if not interior.any():
        return []
    dark, bright, direction = glue_line.line_strength(
        glue_line.flatten(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
    centres, spans = glue_line.bead_points(
        dark, bright, direction, interior,
        glue_line._threshold(dark, interior, options['line_sensitivity']),
        glue_line._threshold(bright, interior, options['line_sensitivity']),
        options['min_width_px'], options['max_width_px'])
    return glue_line.centrelines(centres, spans, dark.shape) if len(centres) else []


def whole_lines(pieces, min_length_px):
    """Joined pieces that look like one complete, gently waved line."""
    lines = []
    for points, _, seen in glue_line.join_pieces(pieces, JOIN_GAP_PX):
        length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        local, chord = glue_shape.to_chord_frame(points)
        if (length >= min_length_px and 0.90 <= chord / length <= 0.97
                and local[:, 1].max() < MAX_BOW and local[:, 1].min() > -0.02
                and seen.mean() > MIN_SEEN_SHARE):
            lines.append(points)
    return lines


def lines_from_shape(frame, background, start_from, scale_range, min_seen):
    """Whole lines found by fitting an existing template, then following the bead.

    For a new camera height or product the old template is close but not right.
    It is fitted loosely, the bead itself is followed, and the lines that were
    mostly seen are what the new template is learned from.
    """
    settings = {'glue_line': {'background': background, 'shape_prior': {
        'enabled': True, 'template': start_from, 'min_scale': scale_range[0],
        'max_scale': scale_range[1], 'min_seen_share': min_seen}}}
    lines = []
    for instance in glue_line.GlueLineSegmenter(settings).segment(frame).instances:
        half = len(instance.polygon) // 2
        lines.append((instance.polygon[:half] + instance.polygon[half:][::-1]) / 2.0)
    return lines


def agreeing(lines):
    """Drop lines whose length disagrees with the rest: they are not whole."""
    if not lines:
        return lines
    chords = np.array([glue_shape.to_chord_frame(line)[1] for line in lines])
    return [line for line, chord in zip(lines, chords)
            if abs(chord / np.median(chords) - 1.0) < CHORD_AGREEMENT]


def main(argv=None):
    parser = argparse.ArgumentParser(description='Learn a glue-line shape template.')
    parser.add_argument('name', help='template name, for example two_panel')
    parser.add_argument('images', nargs='+', help='frames of one product, or wildcard patterns')
    parser.add_argument('--min-length', type=float, default=1800.0,
                        help='shortest line, in pixels, accepted as whole')
    parser.add_argument('--background', choices=('white', 'any'), default='white',
                        help='table the frames were taken on: white, or any')
    parser.add_argument('--start-from', default='',
                        help='existing template to fit loosely and refine, for a new '
                             'camera height or a product whose lines are not found whole')
    parser.add_argument('--scale', type=float, nargs=2, default=(0.8, 1.6), metavar=('MIN', 'MAX'),
                        help='size range, relative to --start-from, searched for lines')
    parser.add_argument('--min-seen', type=float, default=0.6,
                        help='with --start-from: share of a line that must be measured to use it')
    parser.add_argument('--shapes', default=str(glue_shape.SHAPES_FILE))
    args = parser.parse_args(argv)

    paths = [match for pattern in args.images for match in (sorted(glob.glob(pattern)) or [pattern])]
    lines = []
    for path in paths:
        frame = cv2.imread(path)
        if frame is None:
            print(f'{path}: cannot be read, skipped')
            continue
        if args.start_from:
            found = lines_from_shape(frame, args.background, args.start_from,
                                     args.scale, args.min_seen)
        else:
            found = whole_lines(frame_pieces(frame, args.background), args.min_length)
        print(f'{Path(path).name}: {len(found)} whole line(s)')
        lines.extend(found)
    lines = agreeing(lines)
    if len(lines) < 3:
        print(f'Only {len(lines)} whole line(s) agree; at least 3 are needed. Nothing saved.')
        return 1

    template, spread = glue_shape.learn_template(args.name, lines)
    print(f'{args.name}: {len(lines)} lines, end-to-end {template.chord_px:.0f} px, '
          f'spread {100 * spread:.2f} % of that ({spread * template.chord_px:.0f} px)')
    shapes = Path(args.shapes)
    existing = glue_shape.load_templates(path=shapes) if shapes.is_file() else []
    glue_shape.save_templates([item for item in existing if item.name != args.name] + [template],
                              shapes)
    print(f'Saved to {shapes}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

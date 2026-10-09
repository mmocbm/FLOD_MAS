"""Learn what a glue line looks like from frames where it is seen perfectly.

    python tools/learn_bead_reference.py NAME dark_frame.jpg [more frames...]

On black fabric SAM outlines every line evenly. Those outlines give the reference
that outlines on pale fabric are checked against:

- the wave of the centre line (an omega template);
- the width along the line, as a share of the line's median width;
- how far the line's two ends lie from the ends of its ROI region;
- how smooth the outline's edges are.

It is written to ``segmentation/bead_reference.json`` under NAME. Each frame is
put through the Pipeline Lab's automatic SAM 3 run (image tiles are uploaded to
Roboflow); ``--results`` takes already computed runs instead (pickles holding
``frame`` path or name, ``lines`` and the packed ``roi``).
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
from segmentation import glue_shape  # noqa: E402

REFERENCE_FILE = ROOT / 'segmentation' / 'bead_reference.json'
PROFILE_POINTS = 201
MIN_SEEN = 0.95          # only lines SAM outlined this evenly are learned from
MIN_BACKED = 0.9


def end_gaps(centre, roi):
    """How far each end of the line is from where its own direction leaves the ROI."""
    gaps = []
    for end, inner in ((centre[0], centre[min(20, len(centre) - 1)]),
                       (centre[-1], centre[max(-21, -len(centre))])):
        direction = (end - inner) / max(np.linalg.norm(end - inner), 1e-6)
        distance = 0.0
        while distance < 600.0:
            x, y = np.round(end + direction * distance).astype(int)
            if not (0 <= x < roi.shape[1] and 0 <= y < roi.shape[0]) or roi[y, x] == 0:
                break
            distance += 1.0
        gaps.append(distance)
    return gaps


def line_facts(image, roi, line):
    """One evenly outlined line as a smooth ribbon: centre, widths, end gaps."""
    fitted = steps.finish_as_ribbon(image, roi, [dict(line)])[0]
    if fitted.get('backed', 0.0) < MIN_BACKED:
        return None
    ring = fitted['ring']
    half = len(ring) // 2
    centre = (ring[:half] + ring[half:][::-1]) / 2.0
    widths = np.linalg.norm(ring[:half] - ring[half:][::-1], axis=1)
    return {'centre': centre, 'widths': widths, 'gaps': end_gaps(centre, roi)}


def learn(name, facts):
    """The reference from a list of ``line_facts``."""
    template, spread = glue_shape.learn_template(name, [fact['centre'] for fact in facts])
    positions = np.linspace(0.0, 1.0, PROFILE_POINTS)
    profiles = []
    for fact in facts:
        widths = fact['widths'] / np.median(fact['widths'])
        profile = np.interp(positions, np.linspace(0.0, 1.0, len(widths)), widths)
        # A line may have been traced from either end, so the profile is made the
        # same read from both.
        profiles.append((profile + profile[::-1]) / 2.0)
    profiles = np.array(profiles)
    gaps = np.sort(np.array([fact['gaps'] for fact in facts]), axis=1)   # nearer end, farther end
    medians = np.array([np.median(fact['widths']) for fact in facts])
    lengths = np.array([np.linalg.norm(np.diff(fact['centre'], axis=0), axis=1).sum() for fact in facts])
    return template, {
        'name': name, 'lines': len(facts),
        'shape_spread': round(spread, 5),                    # RMS, share of the chord
        'chord_px': round(float(template.chord_px), 1),
        'shape': np.round(template.points, 5).tolist(),      # the wave, ends at (0, 0) and (1, 0)
        'length_px': [round(float(lengths.mean()), 1), round(float(lengths.std()), 1)],
        'width_px': [round(float(np.median(medians)), 2), round(float(medians.std()), 2)],
        'width_profile': np.round(np.median(profiles, axis=0), 4).tolist(),
        'width_profile_spread': np.round(profiles.std(axis=0), 4).tolist(),
        'end_gap_px': {'near': [round(float(gaps[:, 0].mean()), 1), round(float(gaps[:, 0].std()), 1)],
                       'far': [round(float(gaps[:, 1].mean()), 1), round(float(gaps[:, 1].std()), 1)]},
    }


def shape_residual(centre, template):
    """How far a centre line strays from the template's wave: RMS, pixels."""
    local, chord = glue_shape.to_chord_frame(centre)
    flipped = np.column_stack([1.0 - local[::-1, 0], local[::-1, 1]])
    return chord * float(min(np.sqrt(np.mean((each[:, 1] - template.points[:, 1]) ** 2))
                             for each in (local, flipped)))


def load_reference(name, path=REFERENCE_FILE):
    reference = json.loads(Path(path).read_text(encoding='utf-8'))[name]
    return reference, glue_shape.Template(name, np.asarray(reference['shape'], np.float64),
                                          float(reference['chord_px']))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('name')
    parser.add_argument('frames', nargs='*')
    parser.add_argument('--results', nargs='*', default=[], help='pickles of runs already made')
    parser.add_argument('--no-save', action='store_true')
    args = parser.parse_args(argv)
    facts = []
    runs = []
    for path in args.results:
        run = pickle.loads(Path(path).read_bytes())
        roi = np.unpackbits(run['roi'])[:run['shape'][0] * run['shape'][1]].reshape(run['shape'])
        runs.append((run['image'], roi, run['lines']))
    for path in args.frames:
        image = cv2.imread(path)
        roi = steps.model_roi(image)
        rough = steps.segment_local(image, roi, True)
        runs.append((path, roi, steps.segment_sam_from_lines(image, roi, rough, 'sam3')))
    for path, roi, lines in runs:
        image = cv2.imread(str(path))
        for line in lines:
            if line.get('seen', 0.0) < MIN_SEEN or 'evidence' not in line:
                continue
            fact = line_facts(image, roi, line)
            if fact is not None:
                facts.append(fact)
        print(f'{Path(path).name}: {len(facts)} line(s) so far', flush=True)
    if len(facts) < 5:
        raise SystemExit('Too few evenly outlined lines to learn from.')
    _, reference = learn(args.name, facts)
    print(json.dumps({key: value for key, value in reference.items()
                      if 'profile' not in key and key != 'shape'}, indent=1))
    if not args.no_save:
        stored = json.loads(REFERENCE_FILE.read_text(encoding='utf-8')) if REFERENCE_FILE.exists() else {}
        stored[args.name] = reference
        REFERENCE_FILE.write_text(json.dumps(stored) + '\n', encoding='utf-8')
        print(f'Saved {args.name} to {REFERENCE_FILE.name}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

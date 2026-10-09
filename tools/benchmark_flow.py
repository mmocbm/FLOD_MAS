"""Where the time goes in a Pipeline Lab run: every stage timed inside ``run_job``.

    python tools/benchmark_flow.py frame.jpg [more frames...] [--sam local --model sam2.1-large]

Prints one row per frame and the means, and can write the raw figures as JSON.
The first frame is run once beforehand and not counted: loading the models and
opening the API session are paid once per session and are reported separately.
``--sam cloud`` uploads image tiles to Roboflow.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT), str(Path(__file__).resolve().parent)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import cv2  # noqa: E402 - after the path is prepared
import numpy as np  # noqa: E402

import pipeline_lab as lab  # noqa: E402
import pipeline_steps as steps  # noqa: E402
from segmentation import bead_edges, glue_line, ribbon, sam_local, sam_refine  # noqa: E402

STAGES = (
    ('roi', 'ROI model', glue_line, 'glue_region'),
    ('cardboard', 'ROI cardboard mask', glue_line, 'cardboard_mask'),
    ('rough', 'Rough line', steps, 'segment_local'),
    ('sam', 'SAM (tiles, answers)', sam_refine, 'refine_line'),
    ('repair', 'Leak repair', sam_refine, 'trim_ring'),
    ('states', 'States + band edges', bead_edges, 'measure'),
    ('gather', 'Ribbon evidence', ribbon, 'gather'),
    ('ribbon', 'Ribbon fit', ribbon, 'fit_ribbon'),
    ('stats', 'Width and length', steps, 'ring_stats'),
)


def instrument(spent, counts):
    """Wrap each stage's function so its time and call count are recorded."""
    def wrap(owner, name, key):
        original = getattr(owner, name)

        def timed(*args, **kwargs):
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                spent[key] += time.perf_counter() - started
                counts[key] += 1
        setattr(owner, name, timed)
    for key, _, owner, name in STAGES:
        wrap(owner, name, key)
    wrap(sam_refine, 'segment_points', 'sam_calls')


def gpu_megabytes():
    try:
        import torch
        return torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0
    except ImportError:
        return 0.0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('frames', nargs='+')
    parser.add_argument('--sam', choices=('local', 'cloud', 'none'), default='local',
                        help="'none': rough line, band edges and ribbon only")
    parser.add_argument('--model', default=lab.DEFAULT_LOCAL_MODEL, choices=lab.LOCAL_MODELS)
    parser.add_argument('--ask-again', action='store_true', help='ask SAM up to three times per line')
    parser.add_argument('--no-ribbon', action='store_true')
    parser.add_argument('--json')
    args = parser.parse_args(argv)
    recipe = {'roi': 'model', 'pipeline': [], 'point_source': 'lines',
              'segmenter': 'local' if args.sam == 'none' else 'sam3',
              'ribbon': not args.no_ribbon, 'local_model': args.model,
              'sam_runs': 'cloud' if args.sam == 'cloud' else 'local',
              'ask_once': not args.ask_again}
    spent, counts = collections.defaultdict(float), collections.defaultdict(int)
    instrument(spent, counts)
    started = time.perf_counter()
    lab.run_job(cv2.imread(args.frames[0]), recipe, None, [])
    first = time.perf_counter() - started
    print(f'First run of the session (models loaded): {first:.1f} s')
    rows = []
    for path in args.frames:
        spent.clear()
        counts.clear()
        encodes = sam_local.get(args.model).encodes if args.sam == 'local' else 0
        started = time.perf_counter()
        image = cv2.imread(path)
        read = time.perf_counter() - started
        started = time.perf_counter()
        result = lab.run_job(image, recipe, None, [])
        total = time.perf_counter() - started
        row = {'frame': Path(path).name, 'fabric': result['fabric'], 'lines': len(result['lines']),
               'read': read, 'total': total, 'sam_calls': counts['sam_calls'],
               'tiles_encoded': (sam_local.get(args.model).encodes - encodes) if args.sam == 'local' else None,
               **{key: spent[key] for key, *_ in STAGES}}
        row['other'] = total - sum(row[key] for key, *_ in STAGES)
        rows.append(row)
        print(f"{row['frame'][:40]:40s} {row['fabric']:5s} lines {row['lines']} total {total:5.1f} s | "
              + ' | '.join(f"{key} {row[key]:.2f}" for key, *_ in STAGES)
              + f" | other {row['other']:.2f} | SAM asks {row['sam_calls']}"
              + (f", tiles encoded {row['tiles_encoded']}" if row['tiles_encoded'] is not None else ''),
              flush=True)
    for fabric in sorted({row['fabric'] for row in rows}):
        chosen = [row for row in rows if row['fabric'] == fabric]
        print(f"{fabric}: {len(chosen)} frame(s), mean {np.mean([row['total'] for row in chosen]):.1f} s | "
              + ' | '.join(f"{key} {np.mean([row[key] for row in chosen]):.2f}" for key, *_ in STAGES))
    print(f'Peak GPU memory: {gpu_megabytes():.0f} MB')
    if args.json:
        Path(args.json).write_text(json.dumps(
            {'first_run': first, 'recipe': recipe, 'gpu_mb': gpu_megabytes(), 'frames': rows}, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Train the gradient-boosted edge corrector (``segmentation.bead_ml``) on painted
glue lines.

    python tools/train_bead_ml.py --runs RUNS.pkl ... [--hold-out 0.3]

Fabric strips are cut from the frames named by the runs; frames are split by
capture time into a training and a held-out part, so the same piece of fabric
(the pieces were photographed several times in a row) is never on both sides.
Beads are painted with widths, contrast, blur, shine and core position varied
well beyond what was measured, and the trees learn the correction from the
steepest-slope edge to the true one. The held-out error is printed.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT), str(Path(__file__).resolve().parent)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import numpy as np  # noqa: E402

import accuracy_bench  # noqa: E402
from segmentation import bead_edges, bead_ml, synthetic_bead  # noqa: E402


def session_split(runs, hold_out):
    """Per fabric colour: runs in time order, cut into blocks, every third block
    held out. So every colour is in both parts, and frames taken one after the
    other (the same pieces, not moved) stay on one side."""
    by_colour = {}
    for path in runs:
        run = pickle.loads(Path(path).read_bytes())
        by_colour.setdefault(run['colour'], []).append((run['frame'], path))
    train, held = [], []
    for members in by_colour.values():
        ordered = [path for _, path in sorted(members)]
        blocks = np.array_split(np.arange(len(ordered)),
                                min(len(ordered), max(3, int(round(3 / max(hold_out, 0.05))))))
        for number, block in enumerate(blocks):
            (held if number % 3 == 1 else train).extend(ordered[index] for index in block)
    return train, held


def examples(fabrics, copies, rng):
    """Feature rows and true corrections from painted strips."""
    rows, targets = [], []
    for colour, strips in fabrics.items():
        measured = synthetic_bead.MEASURED[colour]
        for fabric, offsets in strips:
            for _ in range(copies):
                strip, truth = synthetic_bead.paint(
                    fabric, offsets, rng, colour=colour, width=rng.uniform(22.0, 42.0),
                    depth=measured['depth'] * rng.uniform(0.4, 1.6),
                    core=measured['core'] * rng.uniform(0.3, 1.8),
                    texture=float(np.clip(measured['texture'] * rng.uniform(0.5, 2.0), 0.1, 0.9)),
                    blur=rng.uniform(1.5, 3.2), wander=rng.uniform(2.0, 12.0),
                    shiny_share=rng.choice([0.0, 0.4, 0.7]), core_at_edge=rng.choice([0.0, 0.0, 0.3, 0.6]),
                    soften=rng.choice([0.0, 0.0, 1.5, 2.5, 4.0]),
                    defects=[(int(len(fabric) * rng.uniform(0.2, 0.8)), int(rng.integers(12, 40)),
                              float(rng.choice([-8, -5, -3, 3, 5, 8])))])
                read = bead_edges.read_strip(strip, offsets)
                low_rows, high_rows = bead_ml.features(strip, offsets, read)
                keep = np.arange(accuracy_bench.MARGIN, len(strip) - accuracy_bench.MARGIN, 3)
                rows += [low_rows[keep], high_rows[keep]]
                targets += [(read['low_course'] - truth['low'])[keep], (truth['high'] - read['high_course'])[keep]]
    return np.vstack(rows), np.clip(np.concatenate(targets), -bead_ml.LIMIT, bead_ml.LIMIT)


def main(argv=None):
    import lightgbm
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--hold-out', type=float, default=0.3)
    parser.add_argument('--strips', type=int, default=30)
    parser.add_argument('--copies', type=int, default=4)
    parser.add_argument('--out', default=str(bead_ml.MODEL_FILE))
    parser.add_argument('--list-held-out', help='write the held-out run paths here')
    args = parser.parse_args(argv)
    rng = np.random.default_rng(11)
    train_runs, held_runs = session_split(args.runs, args.hold_out)
    print(f'{len(train_runs)} frames to train on, {len(held_runs)} held out')
    train_x, train_y = examples(accuracy_bench.fabric_strips(train_runs, args.strips, rng), args.copies, rng)
    held_x, held_y = examples(accuracy_bench.fabric_strips(held_runs, args.strips // 2, rng), 2, rng)
    print(f'{len(train_x)} training rows, {len(held_x)} held-out rows, {train_x.shape[1]} features')
    model = lightgbm.LGBMRegressor(n_estimators=600, learning_rate=0.04, num_leaves=48, subsample=0.8,
                                   subsample_freq=1, colsample_bytree=0.7, min_child_samples=40,
                                   objective='huber', verbose=-1)
    model.fit(train_x, train_y, eval_set=[(held_x, held_y)], callbacks=[lightgbm.early_stopping(40, verbose=False)])
    before = float(np.sqrt(np.mean(held_y ** 2)))
    after = float(np.sqrt(np.mean((held_y - model.predict(held_x)) ** 2)))
    print(f'held-out edge error: {before:.2f} px RMS before correction, {after:.2f} px after '
          f'({model.best_iteration_} trees)')
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(args.out)
    print(f'saved {args.out}')
    if args.list_held_out:
        Path(args.list_held_out).write_text('\n'.join(held_runs) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

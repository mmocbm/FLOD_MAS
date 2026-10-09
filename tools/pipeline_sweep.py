"""Sweep pre-processing recipes against the local glue-line detector, and score them.

    python tools/pipeline_sweep.py [--quick] [--out DIR] [--runs NAME,...]

Every run takes one recipe of ``pipeline_steps`` steps, applies it to a frame,
hands the result to ``segmentation.glue_line`` with the panel ROI, and records
what came back. Frames come from ``enhancement_images`` in four groups:

- ``m1``       masked frames: how many of the marked panels got a line (scored);
- ``white``    the pale two-panel frames of ``10_05``: four beads are expected (scored);
- ``navy``     the navy frames of ``10_05``: counted only, nothing is marked on them;
- ``open``     the loose WIN/WhatsApp frames and the real captures: counted only,
               because nobody has marked how many beads they hold.

For each run the report gives the score, an overlay picture per frame for judging
by eye, and the winning recipe written in the JSON shape ``tools/pipeline_lab.py``
loads. Nothing here needs a display and nothing here is uploaded anywhere.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT), str(Path(__file__).resolve().parent)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import pipeline_steps as steps  # noqa: E402
from segmentation.glue_line import GlueLineSegmenter  # noqa: E402

IMAGES = ROOT / 'enhancement_images'
DEFAULT_OUT = Path.home() / 'flod_experiments' / 'out' / 'cristal_sweep'
PREVIEW_WIDTH = 1800


def entry(name, **params):
    item = steps.new_entry(name)
    item['params'].update(params)
    return item


RECIPES = {
    'P0_none': [],
    'P1_flatten': [entry('flatten', radius=25)],
    'P2_flat_bead': [entry('flatten', radius=25), entry('bead_enhance', largest=31)],
    'P3_denoise_flat_bead_sharp': [entry('denoise', strength=18, median=5),
                                   entry('flatten', radius=25),
                                   entry('bead_enhance', largest=31),
                                   entry('sharpen', amount=1.0, radius=2.0)],
    'P4_clahe_flat': [entry('clahe', clip=2.0, grid=8), entry('flatten', radius=25)],
    'P5_flat_shine_bead': [entry('flatten', radius=25), entry('shine_compress', knee=40),
                           entry('bead_enhance', largest=31)],
    'P6_flat_bead_big': [entry('flatten', radius=45), entry('bead_enhance', largest=51),
                         entry('sharpen', amount=1.5, radius=3.0)],
}

# name -> (recipe, detector options on top of the guided lab defaults)
RUNS = {
    'base': ('P0_none', {}),
    'bead': ('P0_none', {'enhance_bead': True}),
    'flat': ('P1_flatten', {}),
    'flat_bead': ('P2_flat_bead', {}),
    'denoise_flat_bead_sharp': ('P3_denoise_flat_bead_sharp', {}),
    'clahe_flat': ('P4_clahe_flat', {}),
    'flat_shine_bead': ('P5_flat_shine_bead', {}),
    'flat_bead_big': ('P6_flat_bead_big', {}),
    'unguided': ('P0_none', {'roi_guided': False, 'roi_strict': False}),
}


# --- frames -------------------------------------------------------------------

def frame_list(quick=False):
    frames = []

    def add(path, group, expect, mask=None):
        frames.append({'path': Path(path), 'group': group, 'expect': expect, 'mask': mask})

    masks = {int(re.search(r'\((\d+)\)', p.name).group(1)): p
             for p in sorted((IMAGES / 'M1_Masks' / 'full_mask').glob('*'))}
    captures = sorted((IMAGES / 'M1_Masks' / 'captures').glob('*.jpg'))
    m1_keys = sorted(masks)
    m1_step = 16 if quick else 5
    for key in m1_keys[::m1_step]:
        if key <= len(captures):
            add(captures[key - 1], 'm1', None, masks[key])

    white = [n for n in range(36, 53) if n not in (38, 39, 40)]
    white_step = 6 if quick else 2
    for number in white[::white_step]:
        path = IMAGES / '10_05' / f'{number:04d}_frame.png'
        if path.exists():
            add(path, 'white', 4)
    for number in (1, 15, 30):
        path = IMAGES / '10_05' / f'{number:04d}_frame.png'
        if path.exists():
            add(path, 'navy', None)

    for path in sorted(IMAGES.glob('*.jpeg')) + sorted(IMAGES.glob('*.jpg')):
        add(path, 'open', None)
    for path in sorted((IMAGES / 'real_captures').glob('*.jpg')):
        add(path, 'open', None)
    if quick:
        return frames[:8]
    return frames


def panels_from_mask(mask_path, shape):
    """Panel components of a marked frame, flipped to the capture's orientation."""
    marked = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    binary = np.ascontiguousarray((marked[..., 0] > 127)[::-1, ::-1]).astype(np.uint8)
    if binary.shape[:2] != shape:
        binary = cv2.resize(binary, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    keep = [index for index in range(1, count)
            if stats[index, cv2.CC_STAT_AREA] > 0.005 * binary.size]
    return labels, keep, binary


def assign_panels(labels, keep, ring_centre):
    """Which marked panel a strip's centreline belongs to, and whether it is inside."""
    rows = np.clip(ring_centre[:, 1].astype(int), 0, labels.shape[0] - 1)
    columns = np.clip(ring_centre[:, 0].astype(int), 0, labels.shape[1] - 1)
    found = labels[rows, columns]
    inside = np.isin(found, keep)
    if inside.mean() <= 0.7:
        return None, 1.0 - float(inside.mean())
    return int(np.bincount(found[inside]).argmax()), 0.0


# --- one run ------------------------------------------------------------------

def detect(image, roi, options):
    settings = {'glue_line': {'background': 'any', 'roi_strict': True,
                              'min_length_px': 400.0}}
    settings['glue_line'].update(options)
    segmenter = GlueLineSegmenter(settings)
    segmenter.region_override = roi
    return segmenter.segment(image)


def ring_centre(ring):
    half = len(ring) // 2
    return (ring[:half] + ring[half:][::-1]) / 2.0


def overlay(image, lines, title, scale=1.0):
    picture = image.copy()
    factor = picture.shape[1] / PREVIEW_WIDTH
    for number, line in enumerate(lines, 1):
        ring = np.round(line['ring'] * scale).astype(np.int32)
        cv2.polylines(picture, [ring], True, (0, 200, 255),
                      max(2, int(round(factor))))
        x, y = int(ring[:, 0].min()), int(ring[:, 1].min())
        cv2.putText(picture, str(number), (x, max(30, y - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    factor, (0, 200, 255), 3, cv2.LINE_AA)
    cv2.putText(picture, title, (int(12 * factor), int(34 * factor)),
                cv2.FONT_HERSHEY_SIMPLEX, factor, (0, 255, 0), 2, cv2.LINE_AA)
    if picture.shape[1] > PREVIEW_WIDTH:
        picture = cv2.resize(picture, (PREVIEW_WIDTH,
                                       int(round(picture.shape[0] * PREVIEW_WIDTH / picture.shape[1]))),
                             interpolation=cv2.INTER_AREA)
    return picture


def diagnostic(image):
    """``gray - GaussianBlur(gray, 25)`` scaled x7: every bead shows along its length."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    shown = np.clip((gray - cv2.GaussianBlur(gray, (0, 0), 25)) * 7.0 + 128, 0, 255)
    return cv2.cvtColor(shown.astype(np.uint8), cv2.COLOR_GRAY2BGR)


# --- main ---------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--quick', action='store_true', help='a handful of frames, for a smoke test')
    parser.add_argument('--out', default=str(DEFAULT_OUT), help='where pictures and the report go')
    parser.add_argument('--runs', default='', help='comma-separated run names, default all')
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    chosen = {name: RUNS[name] for name in args.runs.split(',') if name} or dict(RUNS)
    frames = frame_list(args.quick)
    print(f'{len(frames)} frame(s), {len(chosen)} run(s) -> {out}', flush=True)

    rows = []
    for number, frame in enumerate(frames, 1):
        image = cv2.imread(str(frame['path']))
        if image is None:
            print(f'  unreadable: {frame["path"]}', flush=True)
            continue
        started = time.perf_counter()
        roi = steps.model_roi(image)
        panels = panels_from_mask(frame['mask'], image.shape[:2]) if frame['mask'] else None
        cv2.imwrite(str(out / f'diag_{frame["path"].stem}.jpg'), diagnostic(image),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        for name, (recipe, options) in chosen.items():
            processed, roi_now, _ = steps.run_pipeline(image, roi, RECIPES[recipe])
            result = detect(processed, roi_now, options)
            lines = []
            for instance in result.instances:
                if instance.polygon is None:
                    continue
                ring = instance.polygon
                lines.append({'ring': ring, 'length': float(np.linalg.norm(
                    np.diff(np.round(ring).astype(np.int32), axis=0), axis=1).sum()),
                    'width': steps.ring_stats(ring)['width_px']})
            row = {'frame': frame['path'].name, 'group': frame['group'], 'run': name,
                   'recipe': recipe, 'lines': len(lines),
                   'length': round(sum(line['length'] for line in lines)), 'widths': [],
                   'panels_gt': 0, 'panels_found': 0, 'stray': 0, 'unseen_share': 0.0}
            if lines:
                row['widths'] = [round(line['width'], 1) for line in lines]
            if panels is not None:
                labels, keep, _ = panels
                row['panels_gt'] = len(keep)
                hit, stray, shares = set(), 0, []
                for line in lines:
                    panel, outside = assign_panels(labels, keep, ring_centre(line['ring']))
                    shares.append(outside)
                    if panel is None:
                        stray += 1
                    else:
                        hit.add(panel)
                row['panels_found'] = len(hit)
                row['stray'] = stray
                row['unseen_share'] = round(float(np.mean(shares)), 3) if shares else 0.0
            rows.append(row)
            title = f'{name}  {row["lines"]} line(s)'
            if panels is not None:
                title += f'  panels {row["panels_found"]}/{row["panels_gt"]}'
            cv2.imwrite(str(out / f'{name}_{frame["path"].stem}.jpg'),
                        overlay(image, lines, title), [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f'[{number}/{len(frames)}] {frame["path"].name} ({frame["group"]}) '
              f'{time.perf_counter() - started:.0f}s', flush=True)

    report = summarise(rows, frames)
    (out / 'results.json').write_text(
        json.dumps({'rows': rows, 'report': report}, indent=1), encoding='utf-8')
    print(report, flush=True)
    best = report['best']
    if best:
        recipe = RECIPES[RUNS[best][0]]
        (out / 'best_recipe.json').write_text(json.dumps(
            {'roi': 'model', 'pipeline': recipe, 'segmenter': 'local', 'run': best,
             'detector_options': RUNS[best][1]}, indent=2) + '\n', encoding='utf-8')
        print(f'best run: {best} -> {out / "best_recipe.json"}', flush=True)


def summarise(rows, frames):
    names = sorted({row['run'] for row in rows})
    header = (f'{"run":<22}{"recipe":<26}{"m1 panels":<16}{"white>=3":<12}'
              f'{"navy":<14}{"open":<14}{"width":<9}{"score":>6}')
    lines = [header, '-' * len(header)]
    scored = {}
    for name in names:
        mine = [row for row in rows if row['run'] == name]
        m1 = [row for row in mine if row['group'] == 'm1']
        white = [row for row in mine if row['group'] == 'white']
        navy = [row for row in mine if row['group'] == 'navy']
        open_rows = [row for row in mine if row['group'] == 'open']
        gt = sum(row['panels_gt'] for row in m1)
        found = sum(row['panels_found'] for row in m1)
        white_hit = sum(1 for row in white if row['lines'] >= 3)
        widths = [w for row in mine for w in row['widths']]
        score = 100.0 * (0.55 * (found / gt if gt else 0)
                         + 0.45 * (white_hit / len(white) if white else 0))
        scored[name] = round(score, 1)
        navy_rate = f'{np.mean([row["lines"] for row in navy]):.1f}/f' if navy else '-'
        open_rate = f'{np.mean([row["lines"] for row in open_rows]):.1f}/f' if open_rows else '-'
        width = f'{np.median(widths):.1f}' if widths else '-'
        lines.append(f'{name:<22}{RUNS[name][0]:<26}'
                     f'{f"{found}/{gt} ({100 * found // max(gt, 1)}%)":<16}'
                     f'{f"{white_hit}/{len(white)}":<12}'
                     f'{navy_rate:<14}{open_rate:<14}{width:<9}{scored[name]:>6.1f}')
    best = max(scored, key=scored.get) if scored else None
    return {'table': '\n'.join(lines), 'scores': scored, 'best': best}


if __name__ == '__main__':
    main()

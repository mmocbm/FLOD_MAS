"""Per-ROI recipe sweep: extract every region the trained model marks, then crop,
pre-process and detect inside each one, mapping the answer back to the frame.

    python tools/roi_sweep.py [--quick] [--out DIR] [--runs NAME,...]

Why per ROI: the pre-processing steps then take their statistics from one strip of
fabric rather than from the whole frame (table, cardboard and other panels included),
and the detector searches a small picture with the region it was built for.

ROI sources (axis of the sweep):

- ``strip``   ``model_for_segment_bra_roi places/.../strip_unet_resnet34.onnx``
              at 1152 px wide, one normal pass;
- ``strip_m`` the same with the mirrored second pass averaged in;
- ``glue``    ``segmentation/models/glue_roi.onnx`` at 768 px wide.

Frames, groups and scoring are the ones ``tools/pipeline_sweep.py`` uses: masked M1
frames (panel hits), pale ``10_05`` frames (four beads expected), navy frames and the
loose frames (counted only). Run that script first for the full-frame reference.
"""
from __future__ import annotations

import argparse
import json
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
from pipeline_sweep import (RECIPES, assign_panels, diagnostic, frame_list,  # noqa: E402
                            overlay, panels_from_mask, ring_centre)
from segmentation.glue_line import GlueLineSegmenter, glue_region  # noqa: E402

STRIP_MODEL = ('model_for_segment_bra_roi places/model_test_and_running_scripts/'
               'models/strip_unet_resnet34.onnx')
ROI_SOURCES = {
    'strip': {'model': STRIP_MODEL, 'width': 1152, 'mirror': False},
    'strip_m': {'model': STRIP_MODEL, 'width': 1152, 'mirror': True},
    'glue': {'model': 'models/glue_roi.onnx', 'width': 768, 'mirror': False},
}
RECIPES_USED = ('P0_none', 'P2_flat_bead', 'P3_denoise_flat_bead_sharp')
CROP_MARGIN = 90           # pixels added around each region before cropping
MIN_COMPONENT = 0.005      # share of the frame a region must cover to be kept

RUNS = {f'{source}|{recipe}': (source, recipe)
        for source in ROI_SOURCES for recipe in RECIPES_USED}
RUNS['whole|P0_none'] = ('strip', 'whole')     # the first sweep's arrangement, as a control


def region_components(image, source):
    """Full-frame region mask and its components as ``(x0, y0, x1, y1, mask)``."""
    options = ROI_SOURCES[source]
    mask = glue_region(image, options['model'], 0.0, options['width'], options['mirror'])
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    height, width = mask.shape
    parts = []
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] < MIN_COMPONENT * mask.size:
            continue
        x, y = stats[index, cv2.CC_STAT_LEFT], stats[index, cv2.CC_STAT_TOP]
        x0, y0 = max(0, x - CROP_MARGIN), max(0, y - CROP_MARGIN)
        x1 = min(width, x + stats[index, cv2.CC_STAT_WIDTH] + CROP_MARGIN)
        y1 = min(height, y + stats[index, cv2.CC_STAT_HEIGHT] + CROP_MARGIN)
        parts.append((x0, y0, x1, y1, (labels[y0:y1, x0:x1] == index).astype(np.uint8)))
    return mask, parts


def detect(image, region, guided=True, options=None):
    settings = {'glue_line': {'background': 'any', 'roi_strict': True, 'min_length_px': 400.0}}
    if not guided:
        settings['glue_line'].update(roi_guided=False, roi_strict=False)
    settings['glue_line'].update(options or {})
    segmenter = GlueLineSegmenter(settings)
    segmenter.region_override = region
    return segmenter.segment(image)


def lines_of(result, offset=(0.0, 0.0)):
    lines = []
    for instance in result.instances:
        if instance.polygon is None:
            continue
        ring = instance.polygon + np.asarray(offset, np.float64)
        lines.append({'ring': ring, 'length': float(np.linalg.norm(
            np.diff(np.round(ring).astype(np.int32), axis=0), axis=1).sum()),
            'width': steps.ring_stats(ring)['width_px']})
    return lines


def run_on_frame(image, source, recipe):
    """Crop each region, run the recipe and the detector inside it, back to frame."""
    mask, parts = region_components(image, source)
    if recipe == 'whole':                       # the first sweep's arrangement, as a control
        processed, roi, _ = steps.run_pipeline(image, mask, [])
        return lines_of(detect(processed, roi))
    lines = []
    for x0, y0, x1, y1, part in parts:
        crop = np.ascontiguousarray(image[y0:y1, x0:x1])
        processed, roi, _ = steps.run_pipeline(crop, part, RECIPES[recipe])
        lines.extend(lines_of(detect(processed, roi), (x0, y0)))
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--quick', action='store_true', help='a handful of frames')
    parser.add_argument('--out', default=str(Path.home() / 'flod_experiments' / 'out' / 'roi_sweep'))
    parser.add_argument('--runs', default='', help='comma-separated run names, default all')
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    chosen = {name: RUNS[name] for name in args.runs.split(',') if name} or dict(RUNS)
    frames = frame_list(args.quick)
    print(f'{len(frames)} frame(s), {len(chosen)} run(s) -> {out}', flush=True)

    rows = []
    for number, frame in enumerate(frames, 1):
        started = time.perf_counter()
        image = cv2.imread(str(frame['path']))
        if image is None:
            print(f'  unreadable: {frame["path"]}', flush=True)
            continue
        panels = panels_from_mask(frame['mask'], image.shape[:2]) if frame['mask'] else None
        if number == 1:
            cv2.imwrite(str(out / f'diag_{frame["path"].stem}.jpg'), diagnostic(image),
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
        for name, (source, recipe) in chosen.items():
            lines = run_on_frame(image, source, recipe)
            row = {'frame': frame['path'].name, 'group': frame['group'], 'run': name,
                   'source': source, 'recipe': recipe, 'lines': len(lines),
                   'length': round(sum(line['length'] for line in lines)),
                   'widths': [round(line['width'], 1) for line in lines],
                   'panels_gt': 0, 'panels_found': 0, 'stray': 0}
            if panels is not None:
                labels, keep, _ = panels
                row['panels_gt'] = len(keep)
                hit, stray = set(), 0
                for line in lines:
                    panel, _ = assign_panels(labels, keep, ring_centre(line['ring']))
                    if panel is None:
                        stray += 1
                    else:
                        hit.add(panel)
                row['panels_found'] = len(hit)
                row['stray'] = stray
            rows.append(row)
            title = f'{name}  {row["lines"]} line(s)'
            if panels is not None:
                title += f'  panels {row["panels_found"]}/{row["panels_gt"]}'
            cv2.imwrite(str(out / f'{name}_{frame["path"].stem}.jpg'),
                        overlay(image, lines, title), [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f'[{number}/{len(frames)}] {frame["path"].name} ({frame["group"]}) '
              f'{time.perf_counter() - started:.0f}s', flush=True)

    report = summarise(rows)
    (out / 'results.json').write_text(json.dumps({'rows': rows, 'report': report}, indent=1),
                                      encoding='utf-8')
    print(report['table'], flush=True)
    best = report['best']
    if best:
        source, recipe = RUNS[best]
        (out / 'best_recipe.json').write_text(json.dumps(
            {'roi': 'model', 'roi_source': source, 'per_roi': True,
             'pipeline': [] if recipe == 'whole' else RECIPES[recipe],
             'segmenter': 'local', 'run': best}, indent=2) + '\n', encoding='utf-8')
        print(f'best run: {best} -> {out / "best_recipe.json"}', flush=True)


def summarise(rows):
    names = sorted({row['run'] for row in rows})
    header = (f'{"run":<30}{"m1 panels":<16}{"white>=3":<12}{"navy":<12}'
              f'{"open":<12}{"stray":<8}{"width":<9}{"score":>6}')
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
        stray = sum(row['stray'] for row in m1)
        widths = [w for row in mine for w in row['widths']]
        score = 100.0 * (0.55 * (found / gt if gt else 0)
                         + 0.45 * (white_hit / len(white) if white else 0)) - 5.0 * stray
        scored[name] = round(score, 1)
        navy_rate = f'{np.mean([row["lines"] for row in navy]):.1f}/f' if navy else '-'
        open_rate = f'{np.mean([row["lines"] for row in open_rows]):.1f}/f' if open_rows else '-'
        width = f'{np.median(widths):.1f}' if widths else '-'
        lines.append(f'{name:<30}'
                     f'{f"{found}/{gt} ({100 * found // max(gt, 1)}%)":<16}'
                     f'{f"{white_hit}/{len(white)}":<12}{navy_rate:<12}{open_rate:<12}'
                     f'{stray:<8}{width:<9}{scored[name]:>6.1f}')
    best = max(scored, key=scored.get) if scored else None
    return {'table': '\n'.join(lines), 'scores': scored, 'best': best}


if __name__ == '__main__':
    main()

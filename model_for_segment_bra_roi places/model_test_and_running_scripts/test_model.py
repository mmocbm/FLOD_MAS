"""Test the model against hand masks, or compare the .onnx and .pt models.

Accuracy against masks (images and masks paired by file name; mask: white/bright = strip, any of
grey, RGB or RGBA):
    python test_model.py --images ../latest_captures/Machine4/M4_full_mask/rename_captures
                         --masks ../latest_captures/Machine4/M4_full_mask/full_mask --output test_m4

Same photos through both model files (they should agree to rounding):
    python test_model.py --compare-backends --images ../real_captures --output compare

Accuracy per photo: pixel IoU, Dice, precision, recall; per hand-painted strip the IoU of the
predicted region that overlaps it most (found = IoU >= 0.5); outline error: distance between the
predicted and hand outlines, both ways (mean and 95th percentile, px). Writes metrics.csv,
summary.json and overlays (hand outline green, prediction red). Note: frames the model was
trained on score higher than new ones; the held-out score of this model is in README.md.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from run_model import collect
from strip_model import clean_mask, load_model, predict_probability

HERE = Path(__file__).resolve().parent
MIN_STRIP = 0.002


def read_mask(path):
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return ((m[..., :3].max(-1) if m.ndim == 3 else m) > 127).astype(np.uint8) * 255


def outline(mask):
    m = (mask > 0).astype(np.uint8)
    return (m - cv2.erode(m, np.ones((3, 3), np.uint8))) > 0


def score(pred, hand):
    p, g = pred > 0, hand > 0
    tp, fp, fn = (p & g).sum(), (p & ~g).sum(), (~p & g).sum()
    res = dict(iou=tp / max(tp + fp + fn, 1), dice=2 * tp / max(2 * tp + fp + fn, 1),
               precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1))
    n_g, lab_g, st_g, _ = cv2.connectedComponentsWithStats(g.astype(np.uint8), 8)
    n_p, lab_p = cv2.connectedComponents(p.astype(np.uint8), 8)
    strips = []
    for k in range(1, n_g):
        if st_g[k, cv2.CC_STAT_AREA] < MIN_STRIP * g.size:
            continue
        gk = lab_g == k
        hits = np.bincount(lab_p[gk], minlength=n_p)
        hits[0] = 0
        pk = lab_p == int(hits.argmax()) if hits.max() else np.zeros_like(gk)
        strips.append(float((gk & pk).sum() / max((gk | pk).sum(), 1)))
    po, go = outline(pred), outline(hand)
    if po.any() and go.any():
        d = np.concatenate([cv2.distanceTransform((~go).astype(np.uint8), cv2.DIST_L2, 5)[po],
                            cv2.distanceTransform((~po).astype(np.uint8), cv2.DIST_L2, 5)[go]])
        mean, p95 = float(d.mean()), float(np.percentile(d, 95))
    else:
        mean = p95 = float('nan')
    res.update(strips=len(strips), strips_found=sum(s >= 0.5 for s in strips),
               strip_iou_min=min(strips) if strips else float('nan'), outline_mean_px=mean, outline_p95_px=p95)
    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else int(v)) for k, v in res.items()}


def accuracy(args):
    model = load_model(args.model, args.device)
    masks = {p.stem: p for p in Path(args.masks).iterdir() if p.suffix.lower() in {'.png', '.jpg', '.bmp', '.tif'}}
    pairs = [(f, masks[f.stem]) for f in collect([args.images]) if f.stem in masks]
    if not pairs:
        raise SystemExit('No image/mask pairs with the same file name')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for f, m in pairs:
        bgr, hand = cv2.imread(str(f)), read_mask(m)
        if hand.shape != bgr.shape[:2]:
            print(f'skip {f.name}: mask {hand.shape[::-1]} vs image {bgr.shape[1::-1]}')
            continue
        pred = clean_mask(predict_probability(model, bgr, flip=not args.no_flip), args.threshold)
        s = score(pred, hand)
        rows.append(dict(image=f.name, **s))
        vis = bgr.copy()
        for mask, colour in ((hand, (0, 255, 0)), (pred, (0, 0, 255))):
            cnts, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(vis, cnts, -1, colour, 6)
        cv2.imwrite(str(out / f'{f.stem}_check.jpg'), cv2.resize(vis, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA))
        print(f"{f.name}: IoU {s['iou']:.3f}, strips {s['strips_found']}/{s['strips']}, outline {s['outline_mean_px']:.1f} px mean,"
              f" {s['outline_p95_px']:.1f} px p95", flush=True)
    with (out / 'metrics.csv').open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    finite = lambda k: [r[k] for r in rows if np.isfinite(r[k])]
    summary = dict(model=Path(args.model).name, backend=model.backend, photos=len(rows),
                   iou_mean=round(float(np.mean([r['iou'] for r in rows])), 4), iou_min=round(float(min(r['iou'] for r in rows)), 4),
                   precision_mean=round(float(np.mean([r['precision'] for r in rows])), 4),
                   recall_mean=round(float(np.mean([r['recall'] for r in rows])), 4),
                   strips=sum(r['strips'] for r in rows), strips_found=sum(r['strips_found'] for r in rows),
                   outline_mean_px=round(float(np.mean(finite('outline_mean_px'))), 2) if finite('outline_mean_px') else None,
                   outline_p95_px=round(float(np.mean(finite('outline_p95_px'))), 2) if finite('outline_p95_px') else None)
    (out / 'summary.json').write_text(json.dumps(summary, indent=1), encoding='utf-8')
    print(json.dumps(summary, indent=1))


def compare_backends(args):
    onnx_path = HERE / 'models' / 'strip_unet_resnet34.onnx'
    pt_path = HERE / 'models' / 'strip_unet_resnet34.pt'
    a, b = load_model(onnx_path, args.device), load_model(pt_path, args.device)
    worst = 0.0
    for f in collect([args.images]):
        bgr = cv2.imread(str(f))
        times, probs = [], []
        for m in (a, b):
            t0 = time.perf_counter()
            probs.append(predict_probability(m, bgr, flip=not args.no_flip))
            times.append(time.perf_counter() - t0)
        ma, mb = clean_mask(probs[0]) > 0, clean_mask(probs[1]) > 0
        agree = (ma & mb).sum() / max((ma | mb).sum(), 1)
        worst = max(worst, float(np.abs(probs[0] - probs[1]).max()))
        print(f'{f.name}: mask agreement IoU {agree:.5f}, largest probability difference {np.abs(probs[0] - probs[1]).max():.1e}, '
              f'{a.backend} {times[0]:.2f} s, {b.backend} {times[1]:.2f} s', flush=True)
    print(f'largest probability difference over all photos: {worst:.1e}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default=str(HERE / 'models' / 'strip_unet_resnet34.onnx'), help='.onnx or .pt')
    ap.add_argument('--images', required=True, help='Folder (or glob) of photos')
    ap.add_argument('--masks', help='Folder of hand masks, same file names as the photos')
    ap.add_argument('--output', default=str(HERE / 'test_output'))
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--device', choices=['cuda', 'cpu'])
    ap.add_argument('--no-flip', action='store_true')
    ap.add_argument('--compare-backends', action='store_true', help='Run the .onnx and .pt models on the same photos')
    args = ap.parse_args()
    if args.compare_backends:
        compare_backends(args)
    elif args.masks:
        accuracy(args)
    else:
        raise SystemExit('Give --masks for an accuracy test, or --compare-backends')


if __name__ == '__main__':
    main()

"""Segment the crop part (paper strips) in photos with the trained model (.onnx or .pt).

    python run_model.py --input ../real_captures --output results
    python run_model.py --model models/strip_unet_resnet34.pt --input photo.jpg --output results
    python run_model.py --input "../latest_captures/Machine4/captures/*.jpg" --output results --device cpu

Per photo: <stem>_strip_mask.png (full size, 255 = strip), <stem>_strip_overlay.jpg (quarter size,
outline red) and <stem>_strips.json (one entry per separate strip region). Touching strips come out
as one region. The model expects panels on the board; empty boards are out of scope.
"""
import argparse
import glob
import json
import time
from pathlib import Path

import cv2

from strip_model import clean_mask, load_model, predict_probability, regions

HERE = Path(__file__).resolve().parent
EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}


def collect(specs):
    files = []
    for spec in specs:
        for src in map(Path, sorted(glob.glob(spec)) or [spec]):
            files += sorted(p for p in src.iterdir() if p.suffix.lower() in EXTS) if src.is_dir() else [src]
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default=str(HERE / 'models' / 'strip_unet_resnet34.onnx'), help='.onnx or .pt')
    ap.add_argument('--input', required=True, nargs='+', help='Photos, folders or glob patterns')
    ap.add_argument('--output', required=True)
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--device', choices=['cuda', 'cpu'], help='Default: GPU when available')
    ap.add_argument('--no-flip', action='store_true', help='Skip the mirrored second pass (about 2x faster)')
    ap.add_argument('--save-probability', action='store_true', help='Also write <stem>_strip_prob.png')
    args = ap.parse_args()
    model = load_model(args.model, args.device)
    files = collect(args.input)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    print(f'{len(files)} photos, model {Path(args.model).name}, backend {model.backend}')
    for f in files:
        bgr = cv2.imread(str(f))
        if bgr is None:
            print(f'skip {f}: unreadable')
            continue
        t0 = time.perf_counter()
        prob = predict_probability(model, bgr, flip=not args.no_flip)
        mask = clean_mask(prob, args.threshold)
        seconds = time.perf_counter() - t0
        found = regions(mask)
        cv2.imwrite(str(out / f'{f.stem}_strip_mask.png'), mask)
        if args.save_probability:
            cv2.imwrite(str(out / f'{f.stem}_strip_prob.png'), (prob * 255).astype('uint8'))
        vis = bgr.copy()
        cnts, _ = cv2.findContours((mask > 0).astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(vis, cnts, -1, (0, 0, 255), 6)
        cv2.imwrite(str(out / f'{f.stem}_strip_overlay.jpg'),
                    cv2.resize(vis, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 88])
        (out / f'{f.stem}_strips.json').write_text(json.dumps(dict(
            image=str(f), model=Path(args.model).name, backend=model.backend, threshold=args.threshold,
            seconds=round(seconds, 3), image_size=[bgr.shape[1], bgr.shape[0]], strips=found), indent=1), encoding='utf-8')
        sizes = ', '.join(f"{r['area_fraction']:.1%}" for r in found)
        print(f'{f.name}: {len(found)} strip regions ({sizes}), {seconds:.2f} s', flush=True)


if __name__ == '__main__':
    main()

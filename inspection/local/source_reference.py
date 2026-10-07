"""Run the ONNX strip model on photos and show the segmented crop part.

    python onnx_run.py ../real_captures/WIN_20261006_21_43_37_Pro.jpg
    python onnx_run.py ../real_captures                 # a folder: browse with keys
    python onnx_run.py photo.jpg --save out             # also save mask and overlay

Window keys: n / space = next photo, p = previous, m = toggle mask view, s = save, q / Esc = quit.
Needs only: numpy, opencv-python, onnxruntime (or onnxruntime-gpu).
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
MODEL = HERE / 'models' / 'strip_unet_resnet34.onnx'
MEAN = np.array([0.485, 0.456, 0.406], np.float32)   # ImageNet, RGB
STD = np.array([0.229, 0.224, 0.225], np.float32)
TRAIN_WIDTH = 1152                                   # the model was trained on photos 1152 px wide
THRESHOLD = 0.5
MIN_REGION = 0.002                                   # drop regions smaller than 0.2% of the photo
COLOURS = [(0, 0, 255), (0, 200, 0), (255, 120, 0), (0, 200, 255), (255, 0, 200), (200, 200, 0)]


def load_session(device):
    providers = ['CPUExecutionProvider']
    if device != 'cpu' and 'CUDAExecutionProvider' in ort.get_available_providers():
        try:
            ort.preload_dlls()
        except Exception:
            pass
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(str(MODEL), providers=providers)
    print(f"model {MODEL.name}, running on {session.get_providers()[0].replace('ExecutionProvider', '')}")
    return session


def segment(session, bgr, flip=False):
    """Photo -> strip mask (uint8, 255 = strip) at the photo's size."""
    h, w = bgr.shape[:2]
    # 1. scale to the training width, pad to a multiple of 32, normalise
    small = cv2.resize(bgr, (TRAIN_WIDTH, round(h * TRAIN_WIDTH / w)), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    padded = cv2.copyMakeBorder(small, 0, (32 - sh % 32) % 32, 0, (32 - sw % 32) % 32, cv2.BORDER_REFLECT)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    x = np.ascontiguousarray(((rgb - MEAN) / STD).transpose(2, 0, 1)[None], np.float32)
    # 2. run the model once (with --flip, also on the mirrored photo, and average)
    name = session.get_inputs()[0].name
    prob = 1 / (1 + np.exp(-session.run(None, {name: x})[0]))
    if flip:
        mirrored = session.run(None, {name: np.ascontiguousarray(x[..., ::-1])})[0][..., ::-1]
        prob = (prob + 1 / (1 + np.exp(-mirrored))) / 2
    # 3. back to the photo's size, threshold, drop specks, fill holes
    prob = cv2.resize(np.ascontiguousarray(prob[0, 0, :sh, :sw]), (w, h), interpolation=cv2.INTER_LINEAR)
    mask = (prob >= THRESHOLD).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, 8)
    keep = np.r_[False, st[1:, cv2.CC_STAT_AREA] >= MIN_REGION * mask.size]
    mask = keep[lab].astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(1 - mask, 4)
    for k in range(1, n):
        bx, by, bw, bh = st[k, :4]
        if bx > 0 and by > 0 and bx + bw < w and by + bh < h:
            mask[lab == k] = 1
    return mask * 255


def draw(bgr, mask):
    """Each strip region tinted in its own colour, outlined and numbered with its area."""
    vis = bgr.copy()
    n, lab, st, cen = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    for k in range(1, n):
        colour = COLOURS[(k - 1) % len(COLOURS)]
        region = lab == k
        vis[region] = (0.55 * vis[region] + 0.45 * np.array(colour)).astype(np.uint8)
        cnts, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(vis, cnts, -1, colour, 8)
        text = f'{k}: {st[k, cv2.CC_STAT_AREA] / mask.size:.1%}'
        pos = (int(cen[k][0]) - 90, int(cen[k][1]))
        cv2.putText(vis, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 14, cv2.LINE_AA)
        cv2.putText(vis, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5, cv2.LINE_AA)
    return vis, n - 1


def fit(img, max_w=1500, max_h=900):
    s = min(max_w / img.shape[1], max_h / img.shape[0], 1.0)
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input', help='Photo or folder of photos')
    ap.add_argument('--save', help='Folder to save <name>_mask.png and <name>_segment.jpg for every photo')
    ap.add_argument('--device', choices=['cuda', 'cpu'])
    ap.add_argument('--flip', action='store_true', help='Also predict the mirrored photo and average (2x time)')
    ap.add_argument('--no-window', action='store_true', help='Only save, do not show')
    args = ap.parse_args()
    src = Path(args.input)
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}) if src.is_dir() else [src]
    if not files:
        raise SystemExit(f'No photos in {src}')
    session = load_session(args.device)
    out = Path(args.save) if args.save else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    cache = {}
    i, show_mask = 0, False
    while True:
        f = files[i]
        if f not in cache:
            bgr = cv2.imread(str(f))
            if bgr is None:
                raise SystemExit(f'Cannot read {f}')
            mask = segment(session, bgr, flip=args.flip)
            vis, count = draw(bgr, mask)
            cache[f] = (mask, vis)
            print(f'{f.name}: {count} strip regions')
            if out:
                cv2.imwrite(str(out / f'{f.stem}_mask.png'), mask)
                cv2.imwrite(str(out / f'{f.stem}_segment.jpg'), vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
        mask, vis = cache[f]
        if args.no_window:
            if i + 1 == len(files):
                break
            i += 1
            continue
        view = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if show_mask else vis
        cv2.imshow('strip segmentation (n next, p previous, m mask, s save, q quit)', fit(view))
        key = cv2.waitKey(0) & 0xFF
        if key in (ord('q'), 27):
            break
        if key in (ord('n'), ord(' ')):
            i = (i + 1) % len(files)
        elif key == ord('p'):
            i = (i - 1) % len(files)
        elif key == ord('m'):
            show_mask = not show_mask
        elif key == ord('s'):
            target = out or (HERE / 'results' / 'onnx_run')
            target.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(target / f'{f.stem}_mask.png'), mask)
            cv2.imwrite(str(target / f'{f.stem}_segment.jpg'), vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
            print(f'saved to {target}')
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

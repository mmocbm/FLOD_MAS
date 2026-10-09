"""Convert the trained PyTorch checkpoint to ONNX and check that both give the same result.

    python convert_to_onnx.py                                   # models/strip_unet_resnet34.pt -> .onnx
    python convert_to_onnx.py --checkpoint other.pt --output other.onnx --check-image photo.jpg

The ONNX graph takes 'image' (1 x 3 x H x W float32, ImageNet-normalised RGB, H and W multiples of
32; any size) and returns 'logits' (1 x 1 x H x W; strip probability = sigmoid). The training
settings are stored in the model's metadata ('config'). Needs torch, segmentation-models-pytorch,
onnx and onnxruntime.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', default=str(HERE / 'models' / 'strip_unet_resnet34.pt'))
    ap.add_argument('--output', help='Default: the checkpoint path with .onnx')
    ap.add_argument('--opset', type=int, default=17)
    ap.add_argument('--check-image', help='Photo used to compare PyTorch and ONNX outputs')
    args = ap.parse_args()
    import torch
    import onnx
    import segmentation_models_pytorch as smp
    from strip_model import OnnxModel, TorchModel, clean_mask, predict_probability, preprocess

    out = Path(args.output) if args.output else Path(args.checkpoint).with_suffix('.onnx')
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = ck['config']
    net = smp.Unet(config['encoder'], encoder_weights=None, classes=1)
    net.load_state_dict(ck['model'])
    net.eval()
    dummy = torch.zeros(1, 3, 864, 1152)
    torch.onnx.export(net, dummy, str(out), input_names=['image'], output_names=['logits'], opset_version=args.opset,
                      dynamic_axes={'image': {2: 'height', 3: 'width'}, 'logits': {2: 'height', 3: 'width'}},
                      do_constant_folding=True, dynamo=False)
    model = onnx.load(str(out))
    meta = {'config': json.dumps({k: config[k] for k in ('encoder', 'train_width', 'epochs', 'crop') if k in config}
                                 | {'threshold': 0.5, 'input': 'image 1x3xHxW float32 ImageNet-normalised RGB, H,W multiple of 32',
                                    'output': 'logits 1x1xHxW, probability = sigmoid'}),
            'source_checkpoint': Path(args.checkpoint).name}
    for k, v in meta.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = k, v
    onnx.checker.check_model(model)
    onnx.save(model, str(out))
    print(f'wrote {out} ({out.stat().st_size / 1e6:.1f} MB, opset {args.opset})')

    # Same input through both: largest probability difference and agreement of the final masks.
    if args.check_image:
        bgr = cv2.imread(args.check_image)
    else:
        rng = np.random.default_rng(0)
        bgr = cv2.GaussianBlur(rng.integers(0, 255, (3456, 4608, 3), dtype=np.uint8), (0, 0), 3)
    tm, om = TorchModel(args.checkpoint, 'cpu'), OnnxModel(out, 'cpu')
    x, _ = preprocess(bgr)
    diff = float(np.abs(tm.logits(x) - om.logits(x)).max())
    pt, po = predict_probability(tm, bgr), predict_probability(om, bgr)
    mt, mo = clean_mask(pt) > 0, clean_mask(po) > 0
    agree = (mt & mo).sum() / max((mt | mo).sum(), 1)
    print(f'check on {"the given photo" if args.check_image else "a synthetic image"} (CPU): largest logit difference {diff:.2e}, '
          f'largest probability difference {np.abs(pt - po).max():.2e}, mask agreement IoU {agree:.6f}')
    for m in (OnnxModel(out), tm):
        t0 = time.perf_counter()
        for _ in range(3):
            predict_probability(m, bgr)
        print(f'  {m.backend}: {(time.perf_counter() - t0) / 3:.2f} s per photo (with mirrored pass)')


if __name__ == '__main__':
    main()

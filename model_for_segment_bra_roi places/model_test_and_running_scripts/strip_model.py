"""Crop-part (paper strip) segmentation model: loading and inference for PyTorch (.pt) and ONNX (.onnx).

Both backends use the same pre- and post-processing as training (train_segemnt_crop_part):
the photo is scaled to a width of 1152 px, padded to a multiple of 32, normalised with the
ImageNet mean/std and predicted, plus a mirrored second pass averaged in; the probability is
resized back to the photo's size, thresholded at 0.5, specks are removed and holes filled.
"""
import json
from pathlib import Path

import cv2
import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], np.float32)   # ImageNet, RGB
STD = np.array([0.229, 0.224, 0.225], np.float32)
TRAIN_WIDTH = 1152
MIN_REGION = 0.002                                   # regions smaller than this share of the image are dropped


def preprocess(bgr):
    """Photo -> (NCHW float32 batch, (scaled height, scaled width))."""
    h, w = bgr.shape[:2]
    small = cv2.resize(bgr, (TRAIN_WIDTH, round(h * TRAIN_WIDTH / w)), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    padded = cv2.copyMakeBorder(small, 0, (32 - sh % 32) % 32, 0, (32 - sw % 32) % 32, cv2.BORDER_REFLECT)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    x = ((rgb - MEAN) / STD).transpose(2, 0, 1)[None]
    return np.ascontiguousarray(x, np.float32), (sh, sw)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class TorchModel:
    """The trained checkpoint (.pt): needs torch and segmentation-models-pytorch."""

    def __init__(self, path, device=None):
        import torch
        import segmentation_models_pytorch as smp
        self.torch = torch
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.config = ck['config']
        self.net = smp.Unet(self.config['encoder'], encoder_weights=None, classes=1)
        self.net.load_state_dict(ck['model'])
        self.net.to(self.device).eval()
        self.backend = f'torch ({self.device.type})'

    def logits(self, x):
        with self.torch.no_grad(), self.torch.autocast(self.device.type, enabled=self.device.type == 'cuda'):
            return self.net(self.torch.from_numpy(x).to(self.device)).float().cpu().numpy()


class OnnxModel:
    """The exported model (.onnx): needs only onnxruntime (GPU if available, else CPU)."""

    def __init__(self, path, device=None):
        import onnxruntime as ort
        providers = ['CPUExecutionProvider']
        if device != 'cpu' and 'CUDAExecutionProvider' in ort.get_available_providers():
            try:
                ort.preload_dlls()          # CUDA/cuDNN from installed pip packages (e.g. torch), if any
            except Exception:
                pass
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(str(path), providers=providers)
        self.input = self.session.get_inputs()[0].name
        meta = self.session.get_modelmeta().custom_metadata_map
        self.config = json.loads(meta['config']) if 'config' in meta else {}
        self.backend = f"onnx ({self.session.get_providers()[0].replace('ExecutionProvider', '')})"

    def logits(self, x):
        return self.session.run(None, {self.input: x})[0]


def load_model(path, device=None):
    path = Path(path)
    if path.suffix.lower() == '.onnx':
        return OnnxModel(path, device)
    return TorchModel(path, device)


def predict_probability(model, bgr, flip=True):
    """Strip probability (0..1) at the photo's full resolution."""
    x, (sh, sw) = preprocess(bgr)
    p = sigmoid(model.logits(x))
    if flip:
        p = (p + sigmoid(model.logits(np.ascontiguousarray(x[..., ::-1])))[..., ::-1]) / 2
    p = np.ascontiguousarray(p[0, 0, :sh, :sw], np.float32)
    return cv2.resize(p, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR)


def clean_mask(prob, threshold=0.5):
    """Threshold, drop specks, fill holes. Returns uint8 0/255."""
    m = (prob >= threshold).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= MIN_REGION * m.size
    m = keep[lab].astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats((1 - m).astype(np.uint8), 4)
    for k in range(1, n):                       # holes: background not touching the border
        x, y, w, h = st[k, :4]
        if x > 0 and y > 0 and x + w < m.shape[1] and y + h < m.shape[0]:
            m[lab == k] = 1
    return m * 255


def regions(mask):
    """Separate strip regions: area, bounding box (x, y, w, h), centre."""
    n, lab, st, cen = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    return [dict(id=k, area_px=int(st[k, 4]), area_fraction=round(float(st[k, 4] / mask.size), 5),
                 bbox_xywh=[int(v) for v in st[k, :4]], centre=[round(float(v), 1) for v in cen[k]])
            for k in range(1, n)]

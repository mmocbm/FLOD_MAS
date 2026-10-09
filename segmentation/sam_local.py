"""SAM on this computer's own GPU: nothing is uploaded.

The cloud API runs the whole model for every question. A SAM model has two parts
of very different cost: an image encoder (heavy) and a prompt decoder (light).
Here a tile is encoded once and may then be asked any number of times, so the
second asking with "not this" points, or several candidate masks, cost a few
milliseconds instead of another round trip.

Models are loaded through ``transformers`` (needs ``torch``; a CUDA GPU is used
when there is one). Names:

    sam2.1-tiny, sam2.1-small, sam2.1-base-plus, sam2.1-large   open weights
    sam3                                                        gated: the licence of
        facebook/sam3 must be accepted on huggingface.co and this computer logged
        in (``hf auth login``) once, to download the weights
    mobile-sam, fastsam-s, fastsam-x                            through ``ultralytics``
        (AGPL-3.0 licence: check before using it in a product)

After the first download a model loads from the local cache; with the environment
variable ``HF_HUB_OFFLINE=1`` no network is touched at all.
"""
import threading
from pathlib import Path

import cv2
import numpy as np

MODELS = {
    'sam2.1-tiny': ('sam2', 'facebook/sam2.1-hiera-tiny'),
    'sam2.1-small': ('sam2', 'facebook/sam2.1-hiera-small'),
    'sam2.1-base-plus': ('sam2', 'facebook/sam2.1-hiera-base-plus'),
    'sam2.1-large': ('sam2', 'facebook/sam2.1-hiera-large'),
    'sam3': ('sam3', 'facebook/sam3'),
    'mobile-sam': ('ultralytics-sam', 'mobile_sam.pt'),
    'fastsam-s': ('ultralytics-fastsam', 'FastSAM-s.pt'),
    'fastsam-x': ('ultralytics-fastsam', 'FastSAM-x.pt'),
}
WEIGHTS = Path.home() / '.cache' / 'flod_sam'      # where the Ultralytics weights are kept
PREFIX = 'local:'          # a model name starting with this runs here, not in the cloud

_loaded = {}
_lock = threading.Lock()


class LocalSam:
    """One SAM model held on the GPU."""

    def __init__(self, name, device=None, half=True):
        import torch
        import transformers
        if name not in MODELS:
            raise ValueError(f'Unknown local SAM model {name!r}; known: {", ".join(MODELS)}')
        family, repository = MODELS[name]
        model_class, processor_class = {
            'sam2': (transformers.Sam2Model, transformers.Sam2Processor),
            'sam3': (transformers.Sam3TrackerModel, transformers.Sam3TrackerProcessor),
        }[family]
        self.torch = torch
        self.name = name
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.dtype = torch.float16 if half and self.device.type == 'cuda' else torch.float32
        self.processor = processor_class.from_pretrained(repository)
        self.model = model_class.from_pretrained(repository, torch_dtype=self.dtype)
        self.model.to(self.device).eval()
        self.encodes = 0
        self.asks = 0

    def open(self, tile):
        """Encode a tile (BGR) once. The result is passed to ``ask``."""
        torch = self.torch
        rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, return_tensors='pt')
        with torch.inference_mode():
            embeddings = self.model.get_image_embeddings(
                inputs['pixel_values'].to(self.device, self.dtype))
        self.encodes += 1
        return {'embeddings': embeddings, 'size': tile.shape[:2]}

    def ask(self, opened, points, multimask=False):
        """Masks for one object given points ``[{'x', 'y', 'positive'}, ...]``.

        Returns ``[(mask, score), ...]``, best first; one entry unless ``multimask``.
        """
        torch = self.torch
        height, width = opened['size']
        inputs = self.processor(
            input_points=[[[[float(point['x']), float(point['y'])] for point in points]]],
            input_labels=[[[1 if point.get('positive', True) else 0 for point in points]]],
            original_sizes=[[height, width]], return_tensors='pt')
        with torch.inference_mode():
            output = self.model(
                image_embeddings=opened['embeddings'],
                input_points=inputs['input_points'].to(self.device, self.dtype),
                input_labels=inputs['input_labels'].to(self.device),
                multimask_output=bool(multimask))
            masks = self.processor.post_process_masks(output.pred_masks.float().cpu(),
                                                      [[height, width]])[0][0]
            scores = output.iou_scores.float().cpu()[0, 0]
        self.asks += 1
        order = torch.argsort(scores, descending=True)
        return [(masks[index].numpy().astype(np.uint8), float(scores[index])) for index in order]


class UltralyticsSam:
    """MobileSAM or FastSAM through the ``ultralytics`` package. These run the
    whole model for every question (their encoders are small), so ``open`` only
    keeps the tile."""

    def __init__(self, name, device=None, half=True):
        import torch
        import ultralytics
        family, weights = MODELS[name]
        WEIGHTS.mkdir(parents=True, exist_ok=True)
        self.torch = torch
        self.name = name
        self.fast = family == 'ultralytics-fastsam'
        self.device = 0 if (device in (None, 'cuda') and torch.cuda.is_available()) else 'cpu'
        self.model = (ultralytics.FastSAM if self.fast else ultralytics.SAM)(str(WEIGHTS / weights))
        self.encodes = 0
        self.asks = 0

    def open(self, tile):
        self.encodes += 1
        return {'tile': tile, 'size': tile.shape[:2]}

    def ask(self, opened, points, multimask=False):
        xy = [[float(point['x']), float(point['y'])] for point in points]
        labels = [1 if point.get('positive', True) else 0 for point in points]
        if self.fast:        # segments everything, then picks by the points
            results = self.model(opened['tile'], points=xy, labels=labels, device=self.device,
                                 retina_masks=True, imgsz=1024, conf=0.4, iou=0.9, verbose=False)
        else:                # several points, one object
            results = self.model.predict(opened['tile'], points=[xy], labels=[labels],
                                         device=self.device, imgsz=1024, verbose=False)
        self.asks += 1
        found = results[0].masks
        if found is None or len(found.data) == 0:
            return [(np.zeros(opened['size'], np.uint8), 0.0)]
        masks = found.data.cpu().numpy().astype(np.uint8)
        scores = (results[0].boxes.conf.cpu().numpy() if results[0].boxes is not None
                  else np.ones(len(masks)))
        order = np.argsort(-scores)
        answers = []
        for index in order:
            mask = masks[index]
            if mask.shape != tuple(opened['size']):
                mask = cv2.resize(mask, (opened['size'][1], opened['size'][0]),
                                  interpolation=cv2.INTER_NEAREST)
            answers.append((mask, float(scores[index])))
        return answers


def get(name, **settings):
    """The model of that name, loaded once and kept."""
    with _lock:
        if name not in _loaded:
            if name not in MODELS:
                raise ValueError(f'Unknown local SAM model {name!r}; known: {", ".join(MODELS)}')
            kind = UltralyticsSam if MODELS[name][0].startswith('ultralytics') else LocalSam
            _loaded[name] = kind(name, **settings)
        return _loaded[name]


_last = {'tile': None, 'opened': None, 'model': None}


def segment_points(image, prompts, name):
    """One question to a local model, in the cloud reply's form.

    The tile last asked about stays encoded, so further questions about the same
    tile go to the decoder only. Each prediction carries its ``mask`` (an array)
    as well as nothing else to draw: readers use the mask directly.
    """
    model = get(name)
    with _lock:
        if _last['tile'] is not image or _last['model'] is not model:
            _last.update(tile=image, opened=model.open(image), model=model)
        opened = _last['opened']
        predictions = []
        for prompt in prompts:
            mask, score = model.ask(opened, prompt['points'])[0]
            predictions.append({'mask': mask, 'masks': [], 'confidence': score})
    return {'predictions': predictions}

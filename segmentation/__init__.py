"""Replaceable segmentation boundary. Input and output use undistorted pixels."""
from dataclasses import dataclass
import importlib
import cv2
import numpy as np


@dataclass(frozen=True)
class Instance:
    polygon: np.ndarray  # N x 2 float coordinates in the ORIGINAL input frame
    confidence: float = 1.0
    label: str = "glue strip"
    panel_box: tuple | None = None  # x1, y1, x2, y2 in original frame


@dataclass(frozen=True)
class SegmentationResult:
    instances: tuple[Instance, ...]
    frame_size: tuple[int, int]


def create_segmenter(settings):
    """A local provider implements Factory(settings).segment(BGR) -> Result.

    All resize, colour conversion, model loading and postprocessing belong in
    that provider. The UI and measurement layer never see model coordinates.
    """
    if settings['provider'] == 'workflow':
        from .provider import WorkflowSegmenter
        return WorkflowSegmenter(settings)
    module, factory = settings['provider'].split(':', 1)
    return getattr(importlib.import_module(module), factory)(settings)


def validate_result(result, frame):
    h, w = frame.shape[:2]
    if result.frame_size != (w, h):
        raise ValueError('Segmentation must return coordinates in the input frame')
    for instance in result.instances:
        ring = np.asarray(instance.polygon)
        if (ring.ndim != 2 or ring.shape[1] != 2 or len(ring) < 3
                or not np.isfinite(ring).all()
                or np.any(ring < 0) or np.any(ring[:, 0] > w - 1)
                or np.any(ring[:, 1] > h - 1)):
            raise ValueError('Invalid segmentation polygon')
    return result


def mask_to_polygons(mask):
    """Local-model adapter helper; mask MUST already match input-frame size."""
    contours, _ = cv2.findContours((np.asarray(mask) > 0).astype('uint8'),
                                  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c.reshape(-1, 2).astype(float) for c in contours if len(c) >= 3]

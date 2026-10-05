"""Reference workflow preprocessing and coordinate mapping, isolated from UI."""
import cv2
import numpy as np
from . import Instance, SegmentationResult
from .workflow import adhesive_strip_workflow as workflow


def roi_pixels(roi, width, height):
    """Normalised [x1, y1, x2, y2] -> pixel bounds; anything unusable is the full frame."""
    try:
        x1, y1, x2, y2 = (float(value) for value in roi)
    except (TypeError, ValueError):
        return 0, 0, width, height
    left, right = int(round(x1 * width)), int(round(x2 * width))
    top, bottom = int(round(y1 * height)), int(round(y2 * height))
    left, top = max(0, left), max(0, top)
    right, bottom = min(width, right), min(height, bottom)
    if right - left < 2 or bottom - top < 2:
        return 0, 0, width, height
    return left, top, right, bottom


def enhance_contrast(image, settings):
    """CLAHE on lightness only, so a pale strip gains local contrast without a colour shift.

    Applied to the image sent for segmentation and nothing else: measurement and
    the saved sources keep the camera's own pixels.
    """
    if not settings or not settings.get('enabled', False):
        return image
    grid = int(settings.get('tile_grid', 8))
    clahe = cv2.createCLAHE(clipLimit=float(settings.get('clip_limit', 2.0)),
                            tileGridSize=(grid, grid))
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


class WorkflowSegmenter:
    def __init__(self, settings, client=None):
        self.settings = settings
        self.client = client
        self.last_input = None  # BGR image most recently sent, for audit

    def segment(self, frame):
        if self.client is None:
            self.client = workflow.build_client(workflow.load_api_key())
        h, w = frame.shape[:2]
        width, height = self.settings['input_width'], self.settings['input_height']
        left, top, right, bottom = roi_pixels(self.settings.get('roi'), w, h)
        region = frame[top:bottom, left:right]
        image = cv2.resize(region, (width, height), interpolation=cv2.INTER_AREA)
        image = enhance_contrast(image, self.settings.get('enhance'))
        self.last_input = image
        if self.settings.get('convert_to_rgb', False):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = workflow.run_workflow(
            image, output_dir=None, save_name='inspection', client=self.client,
            timeout=self.settings['timeout_seconds'],
            max_retries=self.settings['max_retries'])
        source_w, source_h = result.image_width or width, result.image_height or height
        scale = np.array([(right - left) / source_w, (bottom - top) / source_h])
        offset = np.array([left, top])
        boxes = [tuple(np.asarray(box) * np.tile(scale, 2) + np.tile(offset, 2))
                 for box in result.panel_boxes]
        instances = []
        for strip in result.strips:
            ring = np.asarray(strip.points, dtype=float).reshape(-1, 2) * scale + offset
            if len(ring) < 3 or not np.isfinite(ring).all():
                continue
            ring = np.clip(ring, (0, 0), (w - 1, h - 1))
            centre = ring.mean(axis=0)
            containing = [b for b in boxes if b[0] <= centre[0] <= b[2]
                          and b[1] <= centre[1] <= b[3]]
            box = min(containing, key=lambda b: (b[2]-b[0])*(b[3]-b[1])) if containing else None
            instances.append(Instance(ring, strip.confidence, strip.class_name, box))
        if self.settings.get('report_missing_strips', False):
            # A matched panel with no strip is a fabric that was not inspected.
            # Returning it keeps that fabric visible instead of silently absent.
            used = {instance.panel_box for instance in instances}
            for box in dict.fromkeys(boxes):
                if box not in used and np.isfinite(box).all():
                    instances.append(Instance(None, 0.0, 'no strip', box))

        def position(item):
            if item.polygon is None:
                x1, y1, x2, y2 = item.panel_box
                return (float(x1 + x2) / 2.0, float(y1 + y2) / 2.0)
            return (float(item.polygon[:, 0].mean()), float(item.polygon[:, 1].mean()))
        # Stable spatial order is independent of the API's prediction ordering.
        instances.sort(key=position)
        return SegmentationResult(tuple(instances), (w, h))

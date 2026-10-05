"""Reference workflow preprocessing and coordinate mapping, isolated from UI."""
import cv2
import numpy as np
from . import Instance, SegmentationResult
from .workflow import adhesive_strip_workflow as workflow


class WorkflowSegmenter:
    def __init__(self, settings, client=None):
        self.settings = settings
        self.client = client

    def segment(self, frame):
        if self.client is None:
            self.client = workflow.build_client(workflow.load_api_key())
        h, w = frame.shape[:2]
        width, height = self.settings['input_width'], self.settings['input_height']
        image = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        if self.settings.get('convert_to_rgb', False):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = workflow.run_workflow(
            image, output_dir=None, save_name='inspection', client=self.client,
            timeout=self.settings['timeout_seconds'],
            max_retries=self.settings['max_retries'])
        source_w, source_h = result.image_width or width, result.image_height or height
        scale = np.array([w / source_w, h / source_h])
        boxes = [tuple(np.asarray(box) * np.tile(scale, 2)) for box in result.panel_boxes]
        instances = []
        for strip in result.strips:
            ring = np.asarray(strip.points, dtype=float).reshape(-1, 2) * scale
            if len(ring) < 3 or not np.isfinite(ring).all():
                continue
            ring = np.clip(ring, (0, 0), (w - 1, h - 1))
            centre = ring.mean(axis=0)
            containing = [b for b in boxes if b[0] <= centre[0] <= b[2]
                          and b[1] <= centre[1] <= b[3]]
            box = min(containing, key=lambda b: (b[2]-b[0])*(b[3]-b[1])) if containing else None
            instances.append(Instance(ring, strip.confidence, strip.class_name, box))
        # Stable spatial order is independent of the API's prediction ordering.
        instances.sort(key=lambda item: (float(item.polygon[:, 0].mean()),
                                         float(item.polygon[:, 1].mean())))
        return SegmentationResult(tuple(instances), (w, h))

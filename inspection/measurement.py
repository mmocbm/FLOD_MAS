"""Reuse the established strip algorithm for every automatically found fabric."""
from dataclasses import dataclass
import cv2
import numpy as np
import plane_scale
import sam_detection


@dataclass
class FabricResult:
    image: np.ndarray
    record: dict


def measure_instances(frame, segmentation, scale, settings, target_width, tolerance):
    results = []
    for index, instance in enumerate(segmentation.instances, 1):
        ring = np.asarray(instance.polygon, np.float64)
        x, y, w, h = cv2.boundingRect(ring.astype(np.float32))
        # Automatic display crops only: retain original pixels and scale. No warp.
        if instance.panel_box is not None:
            bx1, by1, bx2, by2 = instance.panel_box
            x1, y1 = min(x, bx1), min(y, by1)
            x2, y2 = max(x+w, bx2), max(y+h, by2)
        else:
            x1, y1, x2, y2 = x, y, x+w, y+h
        pad = max(32, round(max(x2-x1, y2-y1)*0.06))
        x1, y1 = max(0, int(x1)-pad), max(0, int(y1)-pad)
        x2, y2 = min(frame.shape[1], int(x2)+pad), min(frame.shape[0], int(y2)+pad)
        crop = frame[y1:y2, x1:x2].copy()
        polygon = sam_detection.Polygon(instance.label, instance.confidence,
                                         np.round(ring - (x1, y1)).astype(np.int32))
        local_scale = None
        if scale is not None:
            translation = np.array([[1, 0, x1], [0, 1, y1], [0, 0, 1]], np.float64)
            local_scale = plane_scale.PlaneScale(scale.homography @ translation, scale.mm_per_pixel)
        measurement = sam_detection.analyze_detection(
            [polygon], crop.shape, settings['strip_segments'], local_scale, target_width, tolerance)
        record = sam_detection.strip_record(measurement)
        grade = 'UNMEASURED'
        if record and record['metric']:
            grade = 'PASS' if record['within_tolerance'] else 'FAIL'
        results.append(FabricResult(sam_detection.draw_analysis(crop, [polygon], measurement), {
            'fabric': index, 'status': grade, 'polygon': ring.tolist(),
            'confidence': float(instance.confidence), 'label': instance.label,
            'display_box': [x1, y1, x2, y2],
            'measurement_coordinates': 'display_box pixels (add x1, y1 for full frame)',
            'measurement': record}))
    return results
